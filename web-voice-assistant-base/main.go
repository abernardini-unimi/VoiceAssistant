// Usage:
//
//	go run web-voice-assistant-base/main.go
//	open http://localhost:8082

package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"time"

	"github.com/joho/godotenv"
	"github.com/realtime-ai/realtime-ai/pkg/elements"
	"github.com/realtime-ai/realtime-ai/pkg/pipeline"
	"github.com/realtime-ai/realtime-ai/pkg/realtimeapi"
	"github.com/realtime-ai/realtime-ai/pkg/server"
	"github.com/realtime-ai/realtime-ai/pkg/tts"
)

const (
	defaultHTTPPort = ":8082"
	defaultUDPPort  = 9001
)

func main() {
	// Load environment variables
	godotenv.Load()

	// Validate required environment variables
	openaiKey := os.Getenv("OPENAI_API_KEY")
	if openaiKey == "" {
		log.Fatal("OPENAI_API_KEY is required")
	}

	// Find VAD model
	vadModelPath := findVADModel()
	if vadModelPath == "" {
		log.Println("Warning: VAD model not found, interrupt feature will be limited")
	}

	// Get configuration from environment
	httpPort := getEnv("VOICE_ASSISTANT_PORT", defaultHTTPPort)
	udpPort := getEnvInt("VOICE_ASSISTANT_UDP_PORT", defaultUDPPort)
	voice := getEnv("VOICE_ASSISTANT_VOICE", "Coral")
	systemPrompt := getEnv("VOICE_ASSISTANT_SYSTEM_PROMPT",
		`
		Your name is Sara and you are a voice assistant for the customer service department of Increso.
		Your role is to assist customers in a professional, friendly, and efficient manner, using clear, simple, and natural language suitable for voice interactions.

		Behavior guidelines:
			- Introduce yourself as Sara from Increso customer service at the beginning of the conversation.
			- Always maintain a polite, empathetic, and reassuring tone, even when dealing with frustrated or dissatisfied customers.
			- Carefully listen to the user s request and ask clarifying questions only when strictly necessary.
			- Provide answers that are concise, direct, and complete, avoiding unnecessary details or technical jargon.
			- If a request is outside your scope or you lack sufficient information, explain this politely and suggest an alternative (for example, speaking with a human agent).
			- Do not invent information. If you are unsure, clearly state it.
			- Use short, natural sentences suitable for spoken dialogue.

		Main objective:
			- Ensure a positive customer experience by resolving issues as quickly and clearly as possible, while representing Increso professionally.

		Language rules:
			- Always respond in the same user language as the input.
			- Always respond concisely.
		`)

	// Create server configuration
	config := server.DefaultWebRTCRealtimeConfig()
	config.RTCUDPPort = udpPort
	config.ICELite = false

	// Create WebRTC server
	srv := server.NewWebRTCRealtimeServer(config)

	// Set pipeline factory
	srv.SetPipelineFactory(func(ctx context.Context, session *realtimeapi.Session) (*pipeline.Pipeline, error) {
		return createPipeline(ctx, session, PipelineConfig{
			OpenAIKey:     openaiKey,
			VADModelPath:  vadModelPath,
			Voice:         voice,
			SystemPrompt:  systemPrompt,
		})
	})

	// Start WebRTC server
	if err := srv.Start(); err != nil {
		log.Fatalf("Failed to start WebRTC server: %v", err)
	}

	// HTTP handlers
	http.HandleFunc("/session", srv.HandleNegotiate)
	http.Handle("/", http.FileServer(http.Dir("static")))

	log.Println("===========================================")
	log.Println("  Web Voice Assistant")
	log.Println("===========================================")
	log.Printf("  HTTP: http://localhost%s", httpPort)
	log.Printf("  UDP:  %d", udpPort)
	log.Printf("  Voice: %s", voice)
	log.Printf("  VAD:  %v", vadModelPath != "")
	log.Println("===========================================")

	if err := http.ListenAndServe(httpPort, nil); err != nil {
		log.Fatalf("Failed to start HTTP server: %v", err)
	}
}

// PipelineConfig holds configuration for pipeline creation
type PipelineConfig struct {
	OpenAIKey     string
	VADModelPath  string
	Voice         string
	SystemPrompt  string
}

// createPipeline creates the voice assistant pipeline using OpenAI TTS
func createPipeline(ctx context.Context, session *realtimeapi.Session, cfg PipelineConfig) (*pipeline.Pipeline, error) {
	pipe := pipeline.NewPipeline("voice-assistant-" + session.ID)

	// Enable interrupt manager with hybrid mode
	interruptConfig := pipeline.DefaultInterruptConfig()
	interruptConfig.EnableHybridMode = true
	interruptConfig.MinSpeechForConfirmMs = 400
	interruptConfig.InterruptCooldownMs = 500
	pipe.EnableInterruptManager(interruptConfig)

	interruptManager := pipeline.NewInterruptManager(pipe.Bus(), interruptConfig)
	if err := interruptManager.Start(ctx); err != nil {
		log.Printf("Failed to start interrupt manager: %v", err)
	}

	// Create elements
	var elems []pipeline.Element
	var prevElem pipeline.Element

	// 1. Input resample: 48kHz → 16kHz (WebRTC to processing)
	inputResample := elements.NewAudioResampleElement(48000, 16000, 1, 1)
	elems = append(elems, inputResample)
	prevElem = inputResample

	// 2. VAD (optional, but recommended for interrupt)
	if cfg.VADModelPath != "" {
		vadConfig := elements.SileroVADConfig{
			ModelPath:       cfg.VADModelPath,
			Threshold:       0.4,
			MinSilenceDurMs: 500, 	
			SpeechPadMs:     20,
			PreRollMs:       300, 	
			Mode:            elements.VADModePassthrough,
		}
		vadElem, err := elements.NewSileroVADElement(vadConfig)
		if err != nil {
			log.Printf("[Pipeline] Warning: Failed to create VAD element: %v", err)
		} else {
			if err := vadElem.Init(ctx); err != nil {
				log.Printf("[Pipeline] Warning: Failed to init VAD element: %v", err)
			} else {
				elems = append(elems, vadElem)
				pipe.Link(prevElem, vadElem)
				prevElem = vadElem
				log.Printf("[Pipeline] VAD enabled")
			}
		}
	}

	// 3. Whisper STT
	asrConfig := elements.WhisperSTTConfig{
		APIKey:               cfg.OpenAIKey,
		Language:             "it",
		Model:                "gpt-4o-transcribe",
		EnablePartialResults: false,
		VADEnabled:           true,
		SampleRate:           16000,
		Channels:             1,
	}
	asrElem, err := elements.NewWhisperSTTElement(asrConfig)
	if err != nil {
		return nil, err
	}
	
	elems = append(elems, asrElem)
	pipe.Link(prevElem, asrElem)
	prevElem = asrElem

	// 4. Chat (OpenAI gpt-4o-mini)
	chatConfig := elements.ChatConfig{
		APIKey:       cfg.OpenAIKey,
		Model:        "gpt-4o-mini",
		SystemPrompt: cfg.SystemPrompt,
		MaxTokens:    100,
		MaxHistory:   20,
		Temperature:  0.7,
		Streaming:    false,
	}

	chatElem, err := elements.NewChatElement(chatConfig)
	if err != nil {
		return nil, err
	}
	elems = append(elems, chatElem)
	pipe.Link(prevElem, chatElem)
	prevElem = chatElem

	// 5. OpenAI TTS (gpt-4o-mini-tts)
	ttsProvider := tts.NewOpenAITTSProvider(cfg.OpenAIKey)
	ttsProvider.SetInstructions("Speak in a friendly and engaging tone in the same user language. Coincise responses.")

	ttsElem := elements.NewUniversalTTSElement(ttsProvider)
	ttsElem.SetVoice("coral")         // scegli la voce: coral, alloy, ash, etc.
	ttsElem.SetLanguage("it-IT")      // lingua
	ttsElem.SetOption("speed", 1.2)   // velocità

	elems = append(elems, ttsElem)
	pipe.Link(prevElem, ttsElem)
	prevElem = ttsElem

	// 6. Nuova soluzione: Simple Resampler manuale
	customResample := NewSimpleResamplerElement()
	elems = append(elems, customResample)
	pipe.Link(prevElem, customResample)
	prevElem = customResample

	// Add all elements to pipeline
	pipe.AddElements(elems)

	log.Printf("[Pipeline] Created voice assistant pipeline for session %s", session.ID)
	log.Printf("[Pipeline] Flow: Resample(48k→16k) → VAD → ASR(11labs) → Chat(gpt-4o-mini) → TTS(OpenAI gpt-4o-mini-tts) → Resample(16k→48k)")

	return pipe, nil
}


// findVADModel looks for the VAD model in common locations
func findVADModel() string {
	paths := []string{
		"models/silero_vad.onnx",      // Standard path from root
		"./models/silero_vad.onnx",    // Explicit relative
		"/app/models/silero_vad.onnx", // Absolute path for Docker
	}

	// Try relative to executable
	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		paths = append(paths, filepath.Join(dir, "models", "silero_vad.onnx"))
	}

	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			abs, _ := filepath.Abs(p)
			return abs
		}
	}

	return ""
}

// getEnv returns environment variable value or default
func getEnv(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// getEnvInt returns environment variable as int or default
func getEnvInt(key string, defaultValue int) int {
	if value := os.Getenv(key); value != "" {
		var result int
		if _, err := os.Stdout.Write([]byte("")); err == nil {
			// Just return default if parsing fails
		}
		if n, err := parseInt(value); err == nil {
			return n
		}
		return result
	}
	return defaultValue
}

func parseInt(s string) (int, error) {
	var n int
	for _, c := range s {
		if c < '0' || c > '9' {
			return 0, nil
		}
		n = n*10 + int(c-'0')
	}
	return n, nil
}

type SimpleResamplerElement struct {
	*pipeline.BaseElement
}

func NewSimpleResamplerElement() *SimpleResamplerElement {
	return &SimpleResamplerElement{
		BaseElement: pipeline.NewBaseElement("simple-resampler", 100),
	}
}

func (e *SimpleResamplerElement) Start(ctx context.Context) error {
	go func() {
		// 48000 Hz * 2 byte (16-bit) * 0.02 secondi = 1920 byte
		const chunkSize = 1920 
		
		for {
			select {
			case <-ctx.Done():
				return

			case msg := <-e.InChan:
				if msg.Type == pipeline.MsgTypeAudio && len(msg.AudioData.Data) > 0 {
					// 1. Upsampling 24k -> 48k
					input := msg.AudioData.Data
					output := make([]byte, len(input)*2)

					for i := 0; i < len(input); i += 2 {
						if i+1 < len(input) {
							output[i*2] = input[i]
							output[i*2+1] = input[i+1]
							output[i*2+2] = input[i]
							output[i*2+3] = input[i+1]
						}
					}

					// 2. Suddivisione in chunk da 20ms
					for i := 0; i < len(output); i += chunkSize {
						end := i + chunkSize
						if end > len(output) {
							end = len(output)
						}

						// Creiamo una copia sicura del chunk
						chunk := make([]byte, end-i)
						copy(chunk, output[i:end])

						// Creiamo un nuovo messaggio di output per evitare data races
						outMsg := &pipeline.PipelineMessage{
							Type:      pipeline.MsgTypeAudio,
							SessionID: msg.SessionID,
							AudioData: &pipeline.AudioData{
								Data:       chunk,
								SampleRate: 48000,
								Channels:   1, // Assumiamo audio mono
							},
						}

						// Manteniamo eventuali metadati originali se presenti
						if msg.AudioData != nil {
							outMsg.AudioData.MediaType = msg.AudioData.MediaType
							outMsg.AudioData.Timestamp = msg.AudioData.Timestamp
						}

						// 3. Invio controllato
						select {
						case e.OutChan <- outMsg:
						case <-ctx.Done():
							return
						}

						// 4. Pacing: aspettiamo quasi 20ms prima di mandare il prossimo blocco.
						// Usiamo un valore leggermente inferiore per compensare i tempi di esecuzione
						time.Sleep(18 * time.Millisecond)
					}

				} else {
					// Inoltro diretto per messaggi non audio (es. eventi di controllo)
					e.OutChan <- msg
				}
			}
		}
	}()
	return nil
}



