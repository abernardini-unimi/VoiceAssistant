package custom_elements

import (
	"context"
	"fmt"
	"log"
	"os"
	"reflect"
	"sync"
	"time"

	"github.com/realtime-ai/realtime-ai/pkg/asr"
	"github.com/realtime-ai/realtime-ai/pkg/pipeline"
)

// Ensure CustomWhisperElement implements pipeline.Element
var _ pipeline.Element = (*CustomWhisperElement)(nil)

// CustomWhisperElement implements speech-to-text using OpenAI or Groq Whisper API.
type CustomWhisperElement struct {
	*pipeline.BaseElement

	provider asr.Provider

	// ASR configuration
	language             string
	model                string
	enablePartialResults bool
	prompt               string
	temperature          float32

	// Audio configuration
	sampleRate    int
	channels      int
	bitsPerSample int

	// VAD integration
	vadEnabled    bool
	vadEventsSub  chan pipeline.Event
	isSpeaking    bool
	speakingMutex sync.Mutex

	// Audio buffering
	audioBuffer     []byte
	audioBufferLock sync.Mutex

	// Streaming recognizer
	recognizer     asr.StreamingRecognizer
	recognizerLock sync.Mutex

	// Lifecycle management
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// CustomWhisperConfig holds configuration for CustomWhisperElement.
type CustomWhisperConfig struct {
	// APIKey is the API key (OpenAI or Groq)
	APIKey string

	// BaseURL allows overriding the API endpoint (e.g. for Groq)
	// Default: "https://api.openai.com/v1"
	BaseURL string

	// Language code (e.g., "en", "it")
	Language string

	// Model to use (e.g. "whisper-1" or "whisper-large-v3")
	Model string

	EnablePartialResults bool
	Prompt               string
	Temperature          float32
	VADEnabled           bool
	SampleRate           int
	Channels             int
	BitsPerSample        int
}

// NewCustomWhisperElement creates a new Custom Whisper STT element (compatible with OpenAI and Groq).
func NewCustomWhisperElement(config CustomWhisperConfig) (*CustomWhisperElement, error) {
	// Get API key from config or environment
	apiKey := config.APIKey
	if apiKey == "" {
		// Fallback to specific env vars depending on usage, or generic OPENAI_API_KEY
		apiKey = os.Getenv("GROQ_API_KEY")
		if apiKey == "" {
			apiKey = os.Getenv("OPENAI_API_KEY")
		}
	}

	if apiKey == "" {
		return nil, fmt.Errorf("API key is required")
	}

	// Set Default BaseURL
	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}

	// Set Default Model based on provider
	if config.Model == "" {
		if baseURL == "https://api.openai.com/v1" {
			config.Model = "whisper-1"
		} else {
			// Default logic for Groq or others
			config.Model = "whisper-large-v3"
		}
	}

	provider, err := NewCustomWhisperProvider(apiKey, baseURL)
	if err != nil {
		return nil, fmt.Errorf("failed to create Whisper provider: %w", err)
	}

	if config.SampleRate == 0 {
		config.SampleRate = 16000
	}
	if config.Channels == 0 {
		config.Channels = 1
	}
	if config.BitsPerSample == 0 {
		config.BitsPerSample = 16
	}

	elem := &CustomWhisperElement{
		BaseElement:          pipeline.NewBaseElement("custom-whisper-stt", 100),
		provider:             provider,
		language:             config.Language,
		model:                config.Model,
		enablePartialResults: config.EnablePartialResults,
		prompt:               config.Prompt,
		temperature:          config.Temperature,
		vadEnabled:           config.VADEnabled,
		sampleRate:           config.SampleRate,
		channels:             config.Channels,
		bitsPerSample:        config.BitsPerSample,
		audioBuffer:          make([]byte, 0, 16000*2*10),
	}

	elem.registerProperties()

	return elem, nil
}

// registerProperties sets up the property system for runtime configuration.
func (e *CustomWhisperElement) registerProperties() {
	e.RegisterProperty(pipeline.PropertyDesc{
		Name:     "language",
		Type:     reflect.TypeOf(""),
		Writable: true,
		Readable: true,
		Default:  e.language,
	})
	e.RegisterProperty(pipeline.PropertyDesc{
		Name:     "model",
		Type:     reflect.TypeOf(""),
		Writable: true,
		Readable: true,
		Default:  e.model,
	})
	e.RegisterProperty(pipeline.PropertyDesc{
		Name:     "enable_partial_results",
		Type:     reflect.TypeOf(false),
		Writable: true,
		Readable: true,
		Default:  e.enablePartialResults,
	})
	e.RegisterProperty(pipeline.PropertyDesc{
		Name:     "vad_enabled",
		Type:     reflect.TypeOf(false),
		Writable: true,
		Readable: true,
		Default:  e.vadEnabled,
	})
}

// Start starts the Whisper STT element.
func (e *CustomWhisperElement) Start(ctx context.Context) error {
	ctx, cancel := context.WithCancel(ctx)
	e.cancel = cancel

	log.Printf("[CustomWhisperSTT] Starting element (VAD: %v, Language: %s, Model: %s)",
		e.vadEnabled, e.language, e.model)

	if e.vadEnabled && e.BaseElement.Bus() != nil {
		e.vadEventsSub = make(chan pipeline.Event, 10)
		e.BaseElement.Bus().Subscribe(pipeline.EventVADSpeechStart, e.vadEventsSub)
		e.BaseElement.Bus().Subscribe(pipeline.EventVADSpeechEnd, e.vadEventsSub)
		log.Printf("[CustomWhisperSTT] Subscribed to VAD events")
	}

	e.wg.Add(1)
	go e.processAudio(ctx)

	if e.vadEnabled {
		e.wg.Add(1)
		go e.handleVADEvents(ctx)
	}

	if err := e.startRecognizer(ctx); err != nil {
		cancel()
		e.wg.Wait()
		return fmt.Errorf("failed to start recognizer: %w", err)
	}

	e.wg.Add(1)
	go e.handleResults(ctx)

	return nil
}

// Stop stops the Whisper STT element.
func (e *CustomWhisperElement) Stop() error {
	log.Printf("[CustomWhisperSTT] Stopping element")
	if e.cancel != nil {
		e.cancel()
		e.wg.Wait()
		e.cancel = nil
	}
	e.recognizerLock.Lock()
	if e.recognizer != nil {
		e.recognizer.Close()
		e.recognizer = nil
	}
	e.recognizerLock.Unlock()
	if e.provider != nil {
		e.provider.Close()
	}
	if e.vadEventsSub != nil {
		if e.BaseElement.Bus() != nil {
			e.BaseElement.Bus().Unsubscribe(pipeline.EventVADSpeechStart, e.vadEventsSub)
			e.BaseElement.Bus().Unsubscribe(pipeline.EventVADSpeechEnd, e.vadEventsSub)
		}
		close(e.vadEventsSub)
		e.vadEventsSub = nil
	}
	log.Printf("[CustomWhisperSTT] Stopped")
	return nil
}

// startRecognizer creates and starts a streaming recognizer.
func (e *CustomWhisperElement) startRecognizer(ctx context.Context) error {
	e.recognizerLock.Lock()
	defer e.recognizerLock.Unlock()

	audioConfig := asr.AudioConfig{
		SampleRate:    e.sampleRate,
		Channels:      e.channels,
		Encoding:      "pcm",
		BitsPerSample: e.bitsPerSample,
	}

	recognitionConfig := asr.RecognitionConfig{
		Language:             e.language,
		Model:                e.model,
		EnablePartialResults: e.enablePartialResults,
		Prompt:               e.prompt,
		Temperature:          e.temperature,
	}

	recognizer, err := e.provider.StreamingRecognize(ctx, audioConfig, recognitionConfig)
	if err != nil {
		return fmt.Errorf("failed to create streaming recognizer: %w", err)
	}

	e.recognizer = recognizer
	log.Printf("[CustomWhisperSTT] Streaming recognizer started")
	return nil
}

// processAudio processes incoming audio messages.
func (e *CustomWhisperElement) processAudio(ctx context.Context) {
	defer e.wg.Done()
	for {
		select {
		case <-ctx.Done():
			return
		case msg, ok := <-e.BaseElement.InChan:
			if !ok {
				return
			}
			if msg.Type != pipeline.MsgTypeAudio || msg.AudioData == nil {
				continue
			}

			// SEMPRE bufferizzare l'audio, non inviarlo mai direttamente
			e.audioBufferLock.Lock()
			e.audioBuffer = append(e.audioBuffer, msg.AudioData.Data...)
			e.audioBufferLock.Unlock()

			// Se il VAD è disabilitato, dobbiamo mantenere il comportamento originale (streaming)
			if !e.vadEnabled {
				e.sendAudioToRecognizer(ctx, msg.AudioData.Data)
				// Svuotiamo il buffer perché lo stiamo inviando in streaming
				e.audioBufferLock.Lock()
				e.audioBuffer = e.audioBuffer[:0]
				e.audioBufferLock.Unlock()
			}
		}
	}
}

// handleVADEvents processes VAD speech start/end events.
func (e *CustomWhisperElement) handleVADEvents(ctx context.Context) {
	defer e.wg.Done()
	for {
		select {
		case <-ctx.Done():
			return
		case event, ok := <-e.vadEventsSub:
			if !ok {
				return
			}

			switch event.Type {
			case pipeline.EventVADSpeechStart:
				log.Printf("[CustomWhisperSTT] VAD Speech START received")

				// Gestione robusta del payload (Pointer vs Value)
				var payload pipeline.VADPayload
				if p, ok := event.Payload.(pipeline.VADPayload); ok {
					payload = p
				} else if p, ok := event.Payload.(*pipeline.VADPayload); ok {
					payload = *p
				} else {
					log.Printf("[CustomWhisperSTT] Error: Unknown payload type for SpeechStart: %T", event.Payload)
					// Resettiamo comunque il buffer per sicurezza
					e.audioBufferLock.Lock()
					e.audioBuffer = e.audioBuffer[:0]
					e.audioBufferLock.Unlock()
					continue
				}

				// Reset buffer e aggiungi pre-roll
				e.audioBufferLock.Lock()
				e.audioBuffer = e.audioBuffer[:0] // Reset totale
				if len(payload.PreRollAudio) > 0 {
					e.audioBuffer = append(e.audioBuffer, payload.PreRollAudio...)
				}
				e.audioBufferLock.Unlock()

				e.speakingMutex.Lock()
				e.isSpeaking = true
				e.speakingMutex.Unlock()

			case pipeline.EventVADSpeechEnd:
				log.Printf("[CustomWhisperSTT] VAD Speech END received - Flushing Audio")
				e.speakingMutex.Lock()
				e.isSpeaking = false
				e.speakingMutex.Unlock()

				// Lanciamo in background per non bloccare la ricezione di nuovi eventi
				go func() {
					e.recognizeBufferedAudio(ctx)
				}()
			}
		}
	}
}

// sendAudioToRecognizer sends audio data to the streaming recognizer.
func (e *CustomWhisperElement) sendAudioToRecognizer(ctx context.Context, audioData []byte) {
	e.recognizerLock.Lock()
	recognizer := e.recognizer
	e.recognizerLock.Unlock()
	if recognizer == nil {
		return
	}
	if err := recognizer.SendAudio(ctx, audioData); err != nil {
		log.Printf("[CustomWhisperSTT] Error sending audio: %v", err)
	}
}

// recognizeBufferedAudio processes all buffered audio.
func (e *CustomWhisperElement) recognizeBufferedAudio(ctx context.Context) {
	e.audioBufferLock.Lock()
	if len(e.audioBuffer) == 0 {
		e.audioBufferLock.Unlock()
		return
	}
	audioData := make([]byte, len(e.audioBuffer))
	copy(audioData, e.audioBuffer)
	e.audioBufferLock.Unlock()
	e.sendAudioToRecognizer(ctx, audioData)
}

// handleResults processes recognition results.
func (e *CustomWhisperElement) handleResults(ctx context.Context) {
	defer e.wg.Done()
	e.recognizerLock.Lock()
	recognizer := e.recognizer
	e.recognizerLock.Unlock()
	if recognizer == nil {
		return
	}

	resultsChan := recognizer.Results()
	for {
		select {
		case <-ctx.Done():
			return
		case result, ok := <-resultsChan:
			if !ok {
				return
			}
			if result == nil {
				continue
			}
			if result.Text == "" && !result.IsFinal {
				continue
			}
			textType := "text/partial"
			eventType := pipeline.EventPartialResult
			if result.IsFinal {
				textType = "text/final"
				eventType = pipeline.EventFinalResult
			}
			log.Printf("[CustomWhisperSTT] Result (%s): %s", textType, result.Text)
			textMsg := &pipeline.PipelineMessage{
				Type:      pipeline.MsgTypeData,
				Timestamp: time.Now(),
				TextData: &pipeline.TextData{
					Data:      []byte(result.Text),
					TextType:  textType,
					Timestamp: result.Timestamp,
				},
			}
			select {
			case e.BaseElement.OutChan <- textMsg:
			case <-ctx.Done():
				return
			}
			if e.BaseElement.Bus() != nil {
				e.BaseElement.Bus().Publish(pipeline.Event{
					Type:      eventType,
					Timestamp: result.Timestamp,
					Payload:   result.Text,
				})
			}
		}
	}
}

// SetProperty sets a property value at runtime.
func (e *CustomWhisperElement) SetProperty(name string, value interface{}) error {
	switch name {
	case "language":
		if lang, ok := value.(string); ok {
			e.language = lang
			return nil
		}
	case "model":
		if model, ok := value.(string); ok {
			e.model = model
			return nil
		}
	case "enable_partial_results":
		if enable, ok := value.(bool); ok {
			e.enablePartialResults = enable
			return nil
		}
	case "vad_enabled":
		if enable, ok := value.(bool); ok {
			e.vadEnabled = enable
			return nil
		}
	}
	return e.BaseElement.SetProperty(name, value)
}

// GetProperty gets a property value.
func (e *CustomWhisperElement) GetProperty(name string) (interface{}, error) {
	switch name {
	case "language":
		return e.language, nil
	case "model":
		return e.model, nil
	case "enable_partial_results":
		return e.enablePartialResults, nil
	case "vad_enabled":
		return e.vadEnabled, nil
	}
	return e.BaseElement.GetProperty(name)
}