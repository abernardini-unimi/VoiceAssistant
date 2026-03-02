package custom_elements

import (
	"bytes"
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

var _ pipeline.Element = (*CustomWhisperElement2)(nil)

type CustomWhisperElement2 struct {
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

	// Lifecycle management
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

type NewCustomWhisperConfig struct {
	APIKey               string
	BaseURL              string
	Language             string
	Model                string
	EnablePartialResults bool
	Prompt               string
	Temperature          float32
	VADEnabled           bool
	SampleRate           int
	Channels             int
	BitsPerSample        int
}

func NewCustomWhisperElement2(config NewCustomWhisperConfig) (*CustomWhisperElement2, error) {
	apiKey := config.APIKey
	if apiKey == "" {
		apiKey = os.Getenv("GROQ_API_KEY")
		if apiKey == "" {
			apiKey = os.Getenv("OPENAI_API_KEY")
		}
	}

	if apiKey == "" {
		return nil, fmt.Errorf("API key is required")
	}

	baseURL := config.BaseURL
	if baseURL == "" {
		baseURL = "https://api.openai.com/v1"
	}

	if config.Model == "" {
		if baseURL == "https://api.openai.com/v1" {
			config.Model = "whisper-1"
		} else {
			config.Model = "whisper-large-v3"
		}
	}

	provider, err := NewCustomWhisperProvider(apiKey, baseURL)
	if err != nil {
		return nil, fmt.Errorf("failed to create Whisper provider: %w", err)
	}

	if config.SampleRate == 0 { config.SampleRate = 16000 }
	if config.Channels == 0 { config.Channels = 1 }
	if config.BitsPerSample == 0 { config.BitsPerSample = 16 }

	elem := &CustomWhisperElement2{
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

func (e *CustomWhisperElement2) registerProperties() {
	e.RegisterProperty(pipeline.PropertyDesc{Name: "language", Type: reflect.TypeOf(""), Writable: true, Readable: true, Default: e.language})
	e.RegisterProperty(pipeline.PropertyDesc{Name: "model", Type: reflect.TypeOf(""), Writable: true, Readable: true, Default: e.model})
	e.RegisterProperty(pipeline.PropertyDesc{Name: "vad_enabled", Type: reflect.TypeOf(false), Writable: true, Readable: true, Default: e.vadEnabled})
}

func (e *CustomWhisperElement2) Start(ctx context.Context) error {
	ctx, cancel := context.WithCancel(ctx)
	e.cancel = cancel

	log.Printf("[CustomWhisperSTT] Starting (VAD: %v, Model: %s)", e.vadEnabled, e.model)

	if e.vadEnabled && e.BaseElement.Bus() != nil {
		e.vadEventsSub = make(chan pipeline.Event, 10)
		e.BaseElement.Bus().Subscribe(pipeline.EventVADSpeechStart, e.vadEventsSub)
		e.BaseElement.Bus().Subscribe(pipeline.EventVADSpeechEnd, e.vadEventsSub)
	}

	e.wg.Add(1)
	go e.processAudio(ctx)

	if e.vadEnabled {
		e.wg.Add(1)
		go e.handleVADEvents(ctx)
	}

	return nil
}

func (e *CustomWhisperElement2) Stop() error {
	if e.cancel != nil {
		e.cancel()
		e.wg.Wait()
		e.cancel = nil
	}
	if e.provider != nil {
		e.provider.Close()
	}
	if e.vadEventsSub != nil && e.BaseElement.Bus() != nil {
		e.BaseElement.Bus().Unsubscribe(pipeline.EventVADSpeechStart, e.vadEventsSub)
		e.BaseElement.Bus().Unsubscribe(pipeline.EventVADSpeechEnd, e.vadEventsSub)
		close(e.vadEventsSub)
		e.vadEventsSub = nil
	}
	return nil
}

// processAudio accumula l'audio nel buffer.
func (e *CustomWhisperElement2) processAudio(ctx context.Context) {
	defer e.wg.Done()
	for {
		select {
		case <-ctx.Done():
			return
		case msg, ok := <-e.BaseElement.InChan:
			if !ok { return }
			if msg.Type != pipeline.MsgTypeAudio || msg.AudioData == nil { continue }

			// Bufferizziamo sempre
			e.audioBufferLock.Lock()
			e.audioBuffer = append(e.audioBuffer, msg.AudioData.Data...)
			e.audioBufferLock.Unlock()

			// Se VAD disabilitato, qui dovresti implementare una logica di chunking manuale
			// ma per il tuo caso d'uso VAD è attivo, quindi ignoriamo il caso NO-VAD per semplicità
		}
	}
}

// handleVADEvents gestisce gli eventi VAD e innesca la trascrizione IMMEDIATA
func (e *CustomWhisperElement2) handleVADEvents(ctx context.Context) {
	defer e.wg.Done()
	for {
		select {
		case <-ctx.Done():
			return
		case event, ok := <-e.vadEventsSub:
			if !ok { return }

			switch event.Type {
			case pipeline.EventVADSpeechStart:
				log.Printf("[CustomWhisperSTT] 🟢 Speech START")

				// Gestione payload per recuperare il pre-roll
				var payload pipeline.VADPayload
				if p, ok := event.Payload.(pipeline.VADPayload); ok {
					payload = p
				} else if p, ok := event.Payload.(*pipeline.VADPayload); ok {
					payload = *p
				}

				e.audioBufferLock.Lock()
				e.audioBuffer = e.audioBuffer[:0] // Reset Buffer
				if len(payload.PreRollAudio) > 0 {
					e.audioBuffer = append(e.audioBuffer, payload.PreRollAudio...)
				}
				e.audioBufferLock.Unlock()

				e.speakingMutex.Lock()
				e.isSpeaking = true
				e.speakingMutex.Unlock()

			case pipeline.EventVADSpeechEnd:
				log.Printf("[CustomWhisperSTT] 🔴 Speech END - Transcribing immediately...")
				e.speakingMutex.Lock()
				e.isSpeaking = false
				e.speakingMutex.Unlock()

				// Copia buffer e lancia trascrizione in goroutine
				e.audioBufferLock.Lock()
				dataToTranscribe := make([]byte, len(e.audioBuffer))
				copy(dataToTranscribe, e.audioBuffer)
				e.audioBuffer = e.audioBuffer[:0] // Pulisci buffer per il prossimo turno
				e.audioBufferLock.Unlock()

				// QUI IL FIX: Chiamata diretta, bypassando lo StreamingRecognizer lento
				go e.transcribeDirectly(ctx, dataToTranscribe)
			}
		}
	}
}

// transcribeDirectly fa una chiamata REST diretta a Groq/OpenAI senza timer
func (e *CustomWhisperElement2) transcribeDirectly(ctx context.Context, audioData []byte) {
	if len(audioData) == 0 { return }

	startTime := time.Now()

	audioConfig := asr.AudioConfig{
		SampleRate:    e.sampleRate,
		Channels:      e.channels,
		Encoding:      "pcm",
		BitsPerSample: e.bitsPerSample,
	}

	recConfig := asr.RecognitionConfig{
		Language:    e.language,
		Model:       e.model,
		Prompt:      e.prompt,
		Temperature: e.temperature,
	}

	// Usa il metodo Recognize (che è one-shot) invece di StreamingRecognize
	result, err := e.provider.Recognize(ctx, bytes.NewReader(audioData), audioConfig, recConfig)
	if err != nil {
		log.Printf("[CustomWhisperSTT] Error during transcription: %v", err)
		return
	}

	latency := time.Since(startTime)
	log.Printf("[CustomWhisperSTT] Transcribed in %v: %q", latency, result.Text)

	if result.Text != "" {
		// Invia output
		textMsg := &pipeline.PipelineMessage{
			Type:      pipeline.MsgTypeData,
			Timestamp: time.Now(),
			TextData: &pipeline.TextData{
				Data:       []byte(result.Text),
				TextType:   "text/final",
				Timestamp:  result.Timestamp,
			},
		}
		e.BaseElement.OutChan <- textMsg

		// Notifica anche sul Bus per debug o altri componenti
		if e.BaseElement.Bus() != nil {
			e.BaseElement.Bus().Publish(pipeline.Event{
				Type:      pipeline.EventFinalResult,
				Timestamp: result.Timestamp,
				Payload:   result.Text,
			})
		}
	}
}

// SetProperty e GetProperty rimangono uguali (omessi per brevità se non li cambi)
func (e *CustomWhisperElement2) SetProperty(name string, value interface{}) error {
	return e.BaseElement.SetProperty(name, value)
}
func (e *CustomWhisperElement2) GetProperty(name string) (interface{}, error) {
	return e.BaseElement.GetProperty(name)
}