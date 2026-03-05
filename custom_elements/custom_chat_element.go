package custom_elements

import (
	"context"
	"fmt"
	"log"
	"encoding/json"
	"os"
	"strings"
	"sync"
	"time"
	"unicode/utf8"
	"path/filepath"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared"
	"github.com/realtime-ai/realtime-ai/pkg/pipeline"
)

// Make sure CustomChatElement implements pipeline.Element
var _ pipeline.Element = (*CustomChatElement)(nil)

// LLMJSONResponse mappa il formato di output richiesto nel System Prompt
type LLMJSONResponse struct {
    Reasoning        string `json:"reasoning"`
    SelectedStrategy string `json:"selected_strategy"`
    EmotionDetected  string `json:"emotion_detected"`
    TTSConfig        struct {
        Speed        float64 `json:"speed"`
		Language     string  `json:"language"`
        Instructions string  `json:"instructions"`
    } `json:"tts_config"`
    ResponseText string `json:"response_text"`
}

// CustomChatConfig holds configuration for the chat element
type CustomChatConfig struct {
	APIKey       string
	Model        string
	BaseURL      string
	SystemPrompt string
	MaxTokens    int
	Streaming    bool
	MaxHistory   int
	Temperature  float64
	InitialPrompt string 
	// --- NUOVI CAMPI PER MCP ---
	MCPServerURL   string            // Es: "http://localhost:3001/mcp"
	MCPServerLabel string            // Es: "test"
	MCPHeaders     map[string]string // Es: {"Bearer": "..."}
}
// CustomChatElement processes text input through OpenAI Chat Completion API
type CustomChatElement struct {
    *pipeline.BaseElement

    config  CustomChatConfig
    client  *openai.Client
    history []openai.ChatCompletionMessageParamUnion

    lastInput     string
    lastInputTime time.Time

    cancel context.CancelFunc
    wg     sync.WaitGroup
    mu     sync.RWMutex

    // --- NUOVI CAMPI PER L'INTERRUZIONE ---
    currentReqCancel context.CancelFunc
    reqMu            sync.Mutex
    interruptSub     chan pipeline.Event
}

func (e *CustomChatElement) getMCPTools() []map[string]interface{} {
	if e.config.MCPServerURL == "" {
		return nil
	}

	// Default label se non presente
	label := e.config.MCPServerLabel
	if label == "" {
		label = "default_mcp_server"
	}

	return []map[string]interface{}{
		{
			"type":         "mcp",
			"server_label": label,
			"server_url":   e.config.MCPServerURL,
			"headers":      e.config.MCPHeaders,
		},
	}
}

// NewCustomChatElement creates a new custom chat element
func NewCustomChatElement(config CustomChatConfig) (*CustomChatElement, error) {
	if config.APIKey == "" {
		return nil, fmt.Errorf("API key is required")
	}
	if config.Model == "" {
		config.Model = "gpt-4o-mini"
	}
	if config.SystemPrompt == "" {
		config.SystemPrompt = "You are a helpful voice assistant. Keep your responses concise and conversational."
	}
	if config.MaxHistory == 0 {
		config.MaxHistory = 20 // Default: keep last 20 messages
	}
	if config.Temperature == 0 {
		config.Temperature = 0.7
	}

	return &CustomChatElement{
		BaseElement: pipeline.NewBaseElement("custom-chat-element", 100),
		config:      config,
		history:     make([]openai.ChatCompletionMessageParamUnion, 0),
	}, nil
}

// Start initializes the chat element and begins processing
func (e *CustomChatElement) Start(ctx context.Context) error {
    ctx, cancel := context.WithCancel(ctx)
    e.cancel = cancel

    opts := []option.RequestOption{
        option.WithAPIKey(e.config.APIKey),
    }

    if e.config.BaseURL != "" {
        opts = append(opts, option.WithBaseURL(e.config.BaseURL))
    } else if baseURL := os.Getenv("OPENAI_BASE_URL"); baseURL != "" {
        opts = append(opts, option.WithBaseURL(baseURL))
    }

    client := openai.NewClient(opts...)
    e.client = &client

    // --- NUOVO: Listener per interruzioni ---
    if bus := e.BaseElement.Bus(); bus != nil {
        e.interruptSub = make(chan pipeline.Event, 10)
        bus.Subscribe(pipeline.EventInterrupted, e.interruptSub)
        bus.Subscribe(pipeline.EventVADSpeechStart, e.interruptSub)

        e.wg.Add(1)
        go func() {
            defer e.wg.Done()
            for {
                select {
                case <-ctx.Done():
                    return
                case <-e.interruptSub:
                    e.reqMu.Lock()
                    if e.currentReqCancel != nil {
                        log.Println("[CustomChat] 🛑 Interruzione ricevuta! Blocco la generazione AI.")
                        e.currentReqCancel() // Killa la richiesta HTTP/Streaming a OpenAI
                        e.currentReqCancel = nil
                    }
                    e.reqMu.Unlock()
                }
            }
        }()
    }
    // ----------------------------------------

    e.wg.Add(1)
    go func() {
        defer e.wg.Done()
        e.processLoop(ctx)
    }()

    if e.config.InitialPrompt != "" {
		e.wg.Add(1)
		go func() {
			defer e.wg.Done()
			
			// Attendiamo 200ms per assicurarci che la connessione SIP/WebRTC 
			// sia completamente stabilita prima di far generare l'audio
			time.Sleep(200 * time.Millisecond)
			
			log.Printf("[CustomChat] Simulazione prompt iniziale utente: %s", e.config.InitialPrompt)
			
			// Chiamiamo processMessage per inviare la frase all'LLM.
			// L'LLM la tratterà come una domanda dell'utente, genererà la risposta 
			// e la manderà automaticamente al TTS.
			err := e.processMessage(ctx, e.config.InitialPrompt, "session-init")
			if err != nil && ctx.Err() == nil {
				log.Printf("[CustomChat] Errore nel prompt iniziale: %v", err)
			}
		}()
	}

	log.Printf("[CustomChat] Started (model: %s, streaming: %v, max_history: %d)",
		e.config.Model, e.config.Streaming, e.config.MaxHistory)
	return nil
}

// Stop stops the chat element
func (e *CustomChatElement) Stop() error {
	// --- AGGIUNGI QUESTA RIGA QUI ---
	e.PrintAndSaveSessionTranscript()

	if e.cancel != nil {
		e.cancel()
		e.wg.Wait()
		e.cancel = nil
	}

	// --- Pulizia bus ---
	if e.interruptSub != nil && e.BaseElement.Bus() != nil {
		e.BaseElement.Bus().Unsubscribe(pipeline.EventInterrupted, e.interruptSub)
		e.BaseElement.Bus().Unsubscribe(pipeline.EventVADSpeechStart, e.interruptSub)
		close(e.interruptSub)
		e.interruptSub = nil
	}

	log.Println("[CustomChat] Stopped")
	return nil
}

// ClearHistory clears the conversation history
func (e *CustomChatElement) ClearHistory() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.history = make([]openai.ChatCompletionMessageParamUnion, 0)
	log.Println("[CustomChat] History cleared")
}

// GetHistoryLength returns the current number of messages in history
func (e *CustomChatElement) GetHistoryLength() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return len(e.history)
}

// processLoop handles incoming messages
func (e *CustomChatElement) processLoop(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case msg, ok := <-e.BaseElement.InChan:
			if !ok {
				return
			}
			if msg.Type == pipeline.MsgTypeData && msg.TextData != nil {
				text := strings.TrimSpace(string(msg.TextData.Data))
				if text == "" {
					continue
				}

				// Process the message
				if err := e.processMessage(ctx, text, msg.SessionID); err != nil {
					log.Printf("[CustomChat] Error processing message: %v", err)
					e.BaseElement.Bus().Publish(pipeline.Event{
						Type:      pipeline.EventError,
						Timestamp: time.Now(),
						Payload:   fmt.Sprintf("Chat error: %v", err),
					})
				}
			} else {
				// Pass through non-text messages
				e.BaseElement.OutChan <- msg
			}
		}
	}
}

// processMessage handles a single user message
func (e *CustomChatElement) processMessage(ctx context.Context, userText string, sessionID string) error {
    log.Printf("[CustomChat] User: %s", userText)

    e.addToHistory(openai.UserMessage(userText))

    e.BaseElement.Bus().Publish(pipeline.Event{
        Type:      pipeline.EventResponseStart,
        Timestamp: time.Now(),
        Payload:   sessionID,
    })

    // --- NUOVO: Crea un contesto specifico per questa generazione ---
    e.reqMu.Lock()
    msgCtx, cancelFunc := context.WithCancel(ctx)
    e.currentReqCancel = cancelFunc
    e.reqMu.Unlock()

    // Assicurati di rilasciare il context quando la funzione termina
    defer func() {
        e.reqMu.Lock()
        if e.currentReqCancel != nil {
            e.currentReqCancel()
            e.currentReqCancel = nil
        }
        e.reqMu.Unlock()
    }()
    // ----------------------------------------------------------------

    var response string
    var err error

    // Usa msgCtx invece di ctx!
    if e.config.Streaming {
        response, err = e.chatStreaming(msgCtx, sessionID)
    } else {
        response, err = e.chatNonStreaming(msgCtx, sessionID)
    }

    // Se c'è un errore ed è un context.Canceled, significa che l'abbiamo interrotto noi.
    if err != nil {
        if msgCtx.Err() == context.Canceled {
            log.Println("[CustomChat] Generazione interrotta correttamente dall'utente.")
            // Non considerarlo un errore di sistema, esci silenziosamente
            return nil 
        }

        e.BaseElement.Bus().Publish(pipeline.Event{
            Type:      pipeline.EventResponseEnd,
            Timestamp: time.Now(),
            Payload:   map[string]interface{}{"error": err.Error()},
        })
        return err
    }

    e.addToHistory(openai.AssistantMessage(response))

    e.BaseElement.Bus().Publish(pipeline.Event{
        Type:      pipeline.EventResponseEnd,
        Timestamp: time.Now(),
        Payload:   map[string]interface{}{"text": response},
    })

    log.Printf("[CustomChat] Assistant: %s", response)
    return nil
}

// chatStreaming performs streaming chat completion
func (e *CustomChatElement) chatStreaming(ctx context.Context, sessionID string) (string, error) {
	messages := e.buildMessages()

	params := openai.ChatCompletionNewParams{
		Messages: messages,
		Model:    shared.ChatModel(e.config.Model),
	}
	if e.config.MaxTokens > 0 {
		params.MaxTokens = openai.Int(int64(e.config.MaxTokens))
	}

	stream := e.client.Chat.Completions.NewStreaming(ctx, params)

	var builder strings.Builder
	var sentenceBuffer strings.Builder

	for stream.Next() {
		chunk := stream.Current()
		if len(chunk.Choices) == 0 {
			continue
		}

		delta := chunk.Choices[0].Delta.Content
		if delta == "" {
			continue
		}

		builder.WriteString(delta)
		sentenceBuffer.WriteString(delta)

		// Check if we have a complete sentence to send to TTS
		sentence := sentenceBuffer.String()
		if shouldFlushSentence(sentence) {
			e.sendToTTS(sentence, sessionID, false, nil)
			sentenceBuffer.Reset()

			// Publish partial result event
			e.BaseElement.Bus().Publish(pipeline.Event{
				Type:      pipeline.EventTextDelta,
				Timestamp: time.Now(),
				Payload:   sentence,
			})
		}
	}

	if err := stream.Err(); err != nil {
		return "", fmt.Errorf("streaming error: %w", err)
	}

	// Send remaining text
	remaining := sentenceBuffer.String()
	if remaining != "" {
		e.sendToTTS(remaining, sessionID, true, nil)
	}

	return builder.String(), nil
}

// chatNonStreaming performs non-streaming chat completion
func (e *CustomChatElement) chatNonStreaming(ctx context.Context, sessionID string) (string, error) {
	messages := e.buildMessages()

	params := openai.ChatCompletionNewParams{
		Messages: messages,
		Model:    shared.ChatModel(e.config.Model),
	}
	if e.config.MaxTokens > 0 {
		params.MaxTokens = openai.Int(int64(e.config.MaxTokens))
	}

	// --- MODIFICA QUI ---
	requestOptions := []option.RequestOption{}

	// Se abbiamo configurato MCP, iniettiamo il tool raw
	if mcpTools := e.getMCPTools(); mcpTools != nil {
		requestOptions = append(requestOptions, option.WithJSONSet("tools", mcpTools))
	}

	completion, err := e.client.Chat.Completions.New(ctx, params, requestOptions...)
	// --------------------

	if err != nil {
		return "", fmt.Errorf("completion error: %w", err)
	}

	if len(completion.Choices) == 0 {
		return "", fmt.Errorf("no response from model")
	}

	response := completion.Choices[0].Message.Content

	// 1. Pulizia: A volte gli LLM aggiungono ```json ... ``` ignorando le regole. Li rimuoviamo.
	cleanResponse := strings.TrimSpace(response)
	if strings.HasPrefix(cleanResponse, "```json") {
		cleanResponse = strings.TrimPrefix(cleanResponse, "```json")
		cleanResponse = strings.TrimSuffix(cleanResponse, "```")
		cleanResponse = strings.TrimSpace(cleanResponse)
	} else if strings.HasPrefix(cleanResponse, "```") {
		cleanResponse = strings.TrimPrefix(cleanResponse, "```")
		cleanResponse = strings.TrimSuffix(cleanResponse, "```")
		cleanResponse = strings.TrimSpace(cleanResponse)
	}

	// 2. Parsing del JSON
	var parsedResponse LLMJSONResponse
	err = json.Unmarshal([]byte(cleanResponse), &parsedResponse)
	
	if err != nil {
		log.Printf("[CustomChat] ⚠️ Fallback testo grezzo: %v", err)
		e.sendToTTS(response, sessionID, true, nil)
	} else {
		log.Printf("🧠 [Orchestratore] Emozione: %s | Strategia: %s", parsedResponse.EmotionDetected, parsedResponse.SelectedStrategy)
		
		// Costruiamo le istruzioni dinamiche per il tono della voce
		// dynamicInst := fmt.Sprintf("Speak with a %s tone and %s pitch. %s", 
		// 	parsedResponse.TTSConfig.Tone, 
		// 	parsedResponse.TTSConfig.Pitch,
		// 	"Concise response.")

		options := map[string]interface{}{
            "speed":        parsedResponse.TTSConfig.Speed,
            "language":     parsedResponse.TTSConfig.Language,
            "instructions": parsedResponse.TTSConfig.Instructions,
        }

		// Inviamo al TTS il testo + le opzioni di velocità e tono
		e.sendToTTS(parsedResponse.ResponseText, sessionID, true, options)
	}

	return response, nil
}

// buildMessages builds the message array for API call
func (e *CustomChatElement) buildMessages() []openai.ChatCompletionMessageParamUnion {
	e.mu.RLock()
	defer e.mu.RUnlock()

	messages := make([]openai.ChatCompletionMessageParamUnion, 0, len(e.history)+1)

	// Add system message
	messages = append(messages, openai.SystemMessage(e.config.SystemPrompt))

	// Add history
	messages = append(messages, e.history...)

	return messages
}

// addToHistory adds a message to history with limit enforcement
func (e *CustomChatElement) addToHistory(msg openai.ChatCompletionMessageParamUnion) {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.history = append(e.history, msg)

	// Enforce history limit (keep pairs of user/assistant messages)
	if e.config.MaxHistory > 0 && len(e.history) > e.config.MaxHistory {
		// Remove oldest messages, keeping pairs
		excess := len(e.history) - e.config.MaxHistory
		if excess%2 != 0 {
			excess++ // Keep pairs
		}
		e.history = e.history[excess:]
	}
}

// Modifica la firma per accettare le opzioni
func (e *CustomChatElement) sendToTTS(text string, sessionID string, isFinal bool, options map[string]interface{}) {
	if strings.TrimSpace(text) == "" {
		return
	}

	textType := "partial"
	if isFinal {
		textType = "final"
	}

	msg := &pipeline.PipelineMessage{
        Type:      pipeline.MsgTypeData,
        SessionID: sessionID,
        Timestamp: time.Now(),
        TextData: &pipeline.TextData{
            Data:      []byte(text),
            TextType:  textType,
            Timestamp: time.Now(),
        },
    }

    if options != nil {
        msg.Metadata = options
    }

    e.BaseElement.OutChan <- msg
}

// shouldFlushSentence checks if the buffer contains a complete sentence
func shouldFlushSentence(text string) bool {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return false
	}

	lastRune, _ := utf8.DecodeLastRuneInString(trimmed)
	if lastRune == utf8.RuneError {
		return false
	}

	sentenceEnders := ".!?;:。！？；："

	return strings.ContainsRune(sentenceEnders, lastRune)
}

// truncateForLog truncates text for logging
func truncateForLog(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// --- STRUTTURE PER L'ESPORTAZIONE JSON ---
type ConversationLog struct {
	Timestamp    string    `json:"timestamp"`
	SystemPrompt string    `json:"system_prompt"`
	Conversation []TurnLog `json:"conversation"`
}

type TurnLog struct {
	Turn      int           `json:"turn"`
	User      UserTurnLog   `json:"user"`
	Assistant AssistantTurn `json:"assistant"`
}

type UserTurnLog struct {
	Text  string    `json:"text"`
	Audio *AudioLog `json:"audio,omitempty"`
}

type AssistantTurn struct {
	Text             string  `json:"text"`
	Reasoning        string  `json:"reasoning,omitempty"`
	SelectedStrategy string  `json:"selected_strategy,omitempty"`
	EmotionDetected  string  `json:"emotion_detected,omitempty"`
	TTSConfig        *TTSLog `json:"tts_config,omitempty"`
}

type TTSLog struct {
    Speed        float64 `json:"speed"`
    Language     string  `json:"language"` 
    Instructions string  `json:"instructions"`
}

type AudioLog struct {
	Current       *AudioDataLog `json:"current,omitempty"`
	HistoricalAvg *AudioDataLog `json:"historical_avg,omitempty"`
}

// Creiamo una struttura specifica. Emozioni è un RawMessage: Go non lo mescolerà!
// step_corrente non è presente, quindi Go lo cancellerà automaticamente.
type AudioDataLog struct {
	Emozioni json.RawMessage        `json:"emozioni"`
	PAD      map[string]interface{} `json:"pad"`
}

// PrintAndSaveSessionTranscript stampa la chat e la salva in un file JSON
func (e *CustomChatElement) PrintAndSaveSessionTranscript() {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if len(e.history) == 0 {
		return
	}

	now := time.Now()
	timestampStr := now.Format("2006-01-02_15-04-05")
	
	chatLog := ConversationLog{
		Timestamp:    now.Format(time.RFC3339),
		SystemPrompt: e.config.SystemPrompt,
		Conversation: make([]TurnLog, 0),
	}

	fmt.Println("\n==================================================")
	fmt.Println("🎙️  RIEPILOGO CONVERSAZIONE (SESSIONE CONCLUSA) ")
	fmt.Println("==================================================")

	var currentTurn *TurnLog
	turnCounter := 1

	for _, msg := range e.history {
		var role, content string

		// Estrazione del messaggio da OpenAI-Go
		msgBytes, err := json.Marshal(msg)
		if err == nil {
			var msgMap map[string]interface{}
			json.Unmarshal(msgBytes, &msgMap)
			if r, ok := msgMap["role"].(string); ok { role = r }
			if cStr, ok := msgMap["content"].(string); ok { 
				content = cStr 
			} else if cArr, ok := msgMap["content"].([]interface{}); ok && len(cArr) > 0 {
				if first, ok := cArr[0].(map[string]interface{}); ok {
					if text, ok := first["text"].(string); ok { content = text }
				}
			}
		}

		if role == "user" {
			userText := content
			var audioLog *AudioLog

			// Destrutturiamo la stringa "User message: ... [Informazioni...]"
			if strings.Contains(content, "User message:") {
				parts := strings.Split(content, "[Informazioni Audio Attuale]:")
				userText = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(parts[0]), "User message:"))

				if len(parts) > 1 {
					audioLog = &AudioLog{}
					histParts := strings.Split(parts[1], "[Media Storica dei Valori]:")
					
					// 1. Parsing dati Attuali
					currJsonStr := strings.TrimSpace(histParts[0])
					var currData AudioDataLog
					if err := json.Unmarshal([]byte(currJsonStr), &currData); err == nil {
						audioLog.Current = &currData
					}

					// 2. Parsing dati Storici
					if len(histParts) > 1 {
						instrParts := strings.Split(histParts[1], "(Istruzione interna:")
						histJsonStr := strings.TrimSpace(instrParts[0])
						var histData AudioDataLog
						if err := json.Unmarshal([]byte(histJsonStr), &histData); err == nil {
							// La chiave "step_corrente" viene eliminata automaticamente
							// perché non esiste nella struct AudioDataLog!
							// E l'ordine delle Emozioni rimane intatto grazie a json.RawMessage!
							audioLog.HistoricalAvg = &histData
						}
					}
				}
			}

			// Inizializziamo il turno utente
			currentTurn = &TurnLog{
				Turn: turnCounter,
				User: UserTurnLog{
					Text:  userText,
					Audio: audioLog,
				},
			}

			fmt.Printf("\n👤 UTENTE (Turno %d):\n%s\n", turnCounter, userText)

		} else if role == "assistant" {
			// Struttura di default (se l'LLM non risponde in JSON)
			assistantData := AssistantTurn{Text: content}

			// Tentiamo di decodificare il JSON testuale dell'LLM
			cleanContent := strings.TrimSpace(content)
			if strings.HasPrefix(cleanContent, "```json") {
				cleanContent = strings.TrimPrefix(cleanContent, "```json")
				cleanContent = strings.TrimSuffix(cleanContent, "```")
				cleanContent = strings.TrimSpace(cleanContent)
			} else if strings.HasPrefix(cleanContent, "```") {
				cleanContent = strings.TrimPrefix(cleanContent, "```")
				cleanContent = strings.TrimSuffix(cleanContent, "```")
				cleanContent = strings.TrimSpace(cleanContent)
			}

			var parsed LLMJSONResponse
			if err := json.Unmarshal([]byte(cleanContent), &parsed); err == nil {
                assistantData.Text = parsed.ResponseText
                assistantData.Reasoning = parsed.Reasoning
                assistantData.SelectedStrategy = parsed.SelectedStrategy
                assistantData.EmotionDetected = parsed.EmotionDetected
                assistantData.TTSConfig = &TTSLog{
                    Speed:        parsed.TTSConfig.Speed,
					Language:     parsed.TTSConfig.Language, 
                    Instructions: parsed.TTSConfig.Instructions, 
                }
            }

			// Aggiungiamo la risposta dell'assistente al turno corrente
			if currentTurn != nil {
				currentTurn.Assistant = assistantData
				chatLog.Conversation = append(chatLog.Conversation, *currentTurn)
				currentTurn = nil
			} else {
				chatLog.Conversation = append(chatLog.Conversation, TurnLog{
					Turn:      turnCounter,
					Assistant: assistantData,
				})
			}

			// Nel terminale stampiamo solo la risposta effettiva pronunciata a voce
			fmt.Printf("\n🤖 ASSISTENTE (Turno %d):\n%s\n", turnCounter, assistantData.Text)
			fmt.Println(strings.Repeat("-", 50))
			turnCounter++
		}
	}

	// Se l'utente ha interrotto bruscamente, salviamo l'ultima domanda rimasta in sospeso
	if currentTurn != nil {
		chatLog.Conversation = append(chatLog.Conversation, *currentTurn)
	}

	fmt.Println("==================================================\n")

	// --- SALVATAGGIO SU FILE ---
	folderName := "conversation"
	os.MkdirAll(folderName, 0755)
	fileName := filepath.Join(folderName, fmt.Sprintf("%s.json", timestampStr))
	
	// Formattazione JSON con 2 spazi (Indentazione standard pulita)
	fileData, err := json.MarshalIndent(chatLog, "", "  ")
	if err == nil {
		os.WriteFile(fileName, fileData, 0644)
		log.Printf("[CustomChat] 💾 Conversazione salvata con successo in: %s\n", fileName)
	} else {
		log.Printf("[CustomChat] ❌ Errore salvataggio file: %v\n", err)
	}
}