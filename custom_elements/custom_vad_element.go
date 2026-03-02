package custom_elements

import (
	"context"
	"encoding/binary"
	"fmt"
	"log"
	"reflect"
	"sync"
	"sync/atomic"
	"time"

	"github.com/realtime-ai/realtime-ai/pkg/audio"
	"github.com/realtime-ai/realtime-ai/pkg/pipeline"
	"github.com/realtime-ai/realtime-ai/pkg/vad"
)

// VADMode defines the operating mode of the VAD element
type VADMode int

const (
	// VADModePassthrough passes all audio through and emits events
	VADModePassthrough VADMode = iota
	// VADModeFilter only passes audio segments containing speech
	VADModeFilter
)

// VADEventPayload contains information about VAD events
type VADEventPayload struct {
	SessionID  string
	Confidence float32
	Timestamp  time.Time
	// AudioMs is approximate audio position (ms) when the event is emitted.
	// It is derived from processed samples and has ~chunk-level resolution.
	AudioMs int
}

// SileroVADConfig holds configuration for Silero VAD
type SileroVADConfig struct {
	ModelPath       string
	Threshold       float32
	MinSilenceDurMs int
	SpeechPadMs     int
	PreRollMs       int // Pre-roll buffer duration in ms (default 300ms)
	Mode            VADMode
}

// CustomSileroVADElement implements voice activity detection using Silero VAD
type CustomSileroVADElement struct {
	*pipeline.BaseElement

	// Configuration
	modelPath       string
	threshold       float32
	minSilenceDurMs int
	speechPadMs     int
	preRollMs       int
	mode            VADMode

	// VAD detector (interface for testability)
	detector vad.DetectorInterface

	// State management
	isSpeaking       atomic.Bool
	audioBuffer      []float32
	stateLock        sync.Mutex
	processedSamples int64

	// Pre-roll buffer for capturing audio before speech detection
	preRollBuffer *audio.RingBuffer

	// [MODIFICA] Canale per ascoltare quando l'AI inizia a rispondere
	responseStartCh chan pipeline.Event

	// Detection state (moved from vad.Detector)
	currSample int
	triggered  bool
	tempEnd    int

	// Lifecycle management
	cancel context.CancelFunc
	wg     sync.WaitGroup
}

// NewSileroVADElement creates a new Silero VAD element
func CustomNewSileroVADElement(config SileroVADConfig) (*CustomSileroVADElement, error) {
	if config.ModelPath == "" {
		return nil, fmt.Errorf("model path is required")
	}

	if config.Threshold == 0 {
		config.Threshold = 0.5
	}
	if config.MinSilenceDurMs == 0 {
		config.MinSilenceDurMs = 100
	}
	if config.SpeechPadMs == 0 {
		config.SpeechPadMs = 30
	}
	if config.PreRollMs == 0 {
		config.PreRollMs = 300
	}

	elem := &CustomSileroVADElement{
		BaseElement:      pipeline.NewBaseElement("silero-vad-element", 100),
		modelPath:        config.ModelPath,
		threshold:        config.Threshold,
		minSilenceDurMs:  config.MinSilenceDurMs,
		speechPadMs:      config.SpeechPadMs,
		preRollMs:        config.PreRollMs,
		mode:             config.Mode,
		audioBuffer:      make([]float32, 0, 1024),
		processedSamples: 0,
		preRollBuffer:    audio.NewRingBuffer(16000, config.PreRollMs),
	}

	if err := elem.registerProperties(); err != nil {
		return nil, fmt.Errorf("failed to register properties: %w", err)
	}

	return elem, nil
}

// registerProperties registers configurable properties
func (e *CustomSileroVADElement) registerProperties() error {
	props := []pipeline.PropertyDesc{
		{
			Name:     "threshold",
			Type:     reflect.TypeOf(float32(0)),
			Writable: true,
			Readable: true,
			Default:  e.threshold,
		},
		{
			Name:     "mode",
			Type:     reflect.TypeOf(int(0)),
			Writable: true,
			Readable: true,
			Default:  int(e.mode),
		},
		{
			Name:     "min-silence-ms",
			Type:     reflect.TypeOf(int(0)),
			Writable: true,
			Readable: true,
			Default:  e.minSilenceDurMs,
		},
		{
			Name:     "speech-pad-ms",
			Type:     reflect.TypeOf(int(0)),
			Writable: true,
			Readable: true,
			Default:  e.speechPadMs,
		},
	}

	for _, prop := range props {
		if err := e.RegisterProperty(prop); err != nil {
			return err
		}
	}

	return nil
}

// Init initializes the VAD detector
func (e *CustomSileroVADElement) Init(ctx context.Context) error {
	if e.detector == nil {
		detector, err := vad.NewDetector(vad.DetectorConfig{
			ModelPath:  e.modelPath,
			SampleRate: 16000, // Only support 16kHz
			LogLevel:   vad.LogLevelWarn,
		})
		if err != nil {
			return fmt.Errorf("failed to create VAD detector: %w", err)
		}
		e.detector = detector
	}

	e.currSample = 0
	e.triggered = false
	e.tempEnd = 0

	log.Printf("[SileroVAD] Initialized with threshold=%.2f, minSilence=%dms, preRoll=%dms",
		e.threshold, e.minSilenceDurMs, e.preRollMs)

	return nil
}

// Start starts the VAD element processing
func (e *CustomSileroVADElement) Start(ctx context.Context) error {
	ctx, cancel := context.WithCancel(ctx)
	e.cancel = cancel

	// [MODIFICA] Sottoscrizione agli eventi di Reset (Risposta AI o Interruzione)
	if bus := e.BaseElement.Bus(); bus != nil {
		e.responseStartCh = make(chan pipeline.Event, 10)
		// Quando l'AI inizia a parlare, il vecchio audio nel buffer non serve più
		bus.Subscribe(pipeline.EventResponseStart, e.responseStartCh)
		// Anche se c'è un interrupt manuale, puliamo per sicurezza
		bus.Subscribe(pipeline.EventInterrupted, e.responseStartCh)
		log.Printf("[SileroVAD] Subscribed to ResponseStart/Interrupted events for buffer flushing")
	}

	e.wg.Add(1)
	go func() {
		defer e.wg.Done()
		e.processAudio(ctx)
	}()

	return nil
}

// Stop stops the VAD element and cleans up resources
func (e *CustomSileroVADElement) Stop() error {
	if e.cancel != nil {
		e.cancel()
		e.wg.Wait()
		e.cancel = nil
	}

	// [MODIFICA] Pulizia sottoscrizioni
	if e.responseStartCh != nil && e.BaseElement.Bus() != nil {
		e.BaseElement.Bus().Unsubscribe(pipeline.EventResponseStart, e.responseStartCh)
		e.BaseElement.Bus().Unsubscribe(pipeline.EventInterrupted, e.responseStartCh)
		close(e.responseStartCh)
		e.responseStartCh = nil
	}

	if e.detector != nil {
		e.detector.Destroy()
		e.detector = nil
	}

	return nil
}

// processAudio is the main audio processing loop
func (e *CustomSileroVADElement) processAudio(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return

		// [MODIFICA] Logica di Flush del Buffer
		case <-e.responseStartCh:
			// L'AI ha preso la parola: tutto ciò che è successo "prima" è storia vecchia.
			// Puliamo il PreRoll e lo stato interno per evitare di riattivare il VAD con audio vecchio.
			e.preRollBuffer.Clear()

			e.stateLock.Lock()
			// Reset stato rilevazione
			e.audioBuffer = e.audioBuffer[:0]
			e.currSample = 0
			e.triggered = false
			e.tempEnd = 0
			e.stateLock.Unlock()

			// log.Printf("[SileroVAD] 🧹 Buffer Flushed (AI turn started)")
			continue

		case msg := <-e.BaseElement.InChan:
			if msg.Type != pipeline.MsgTypeAudio {
				continue
			}
			if msg.AudioData == nil || len(msg.AudioData.Data) == 0 {
				continue
			}
			if msg.AudioData.MediaType != pipeline.AudioMediaTypeRaw {
				log.Printf("[SileroVAD] Skipping non-raw audio: %s", msg.AudioData.MediaType)
				continue
			}
			if msg.AudioData.SampleRate != 16000 {
				log.Printf("[SileroVAD] Warning: Expected 16kHz audio, got %dHz.", msg.AudioData.SampleRate)
				continue
			}

			e.handleAudioData(ctx, msg)
		}
	}
}

// handleAudioData processes a single audio message
func (e *CustomSileroVADElement) handleAudioData(ctx context.Context, msg *pipeline.PipelineMessage) {
	// Write raw audio to pre-roll buffer (before any processing)
	e.preRollBuffer.Write(msg.AudioData.Data)

	// Convert byte data to normalized float32 samples in [-1, 1]
	samples := e.bytesToFloat32(msg.AudioData.Data)

	// Add samples to buffer
	e.stateLock.Lock()
	e.audioBuffer = append(e.audioBuffer, samples...)
	e.stateLock.Unlock()

	// Process audio in windows using Infer
	const windowSize = 512 // Window size for 16kHz
	const sampleRate = 16000

	minSilenceSamples := e.minSilenceDurMs * sampleRate / 1000
	speechPadSamples := e.speechPadMs * sampleRate / 1000

	for {
		e.stateLock.Lock()
		bufLen := len(e.audioBuffer)
		if bufLen < windowSize {
			e.stateLock.Unlock()
			break
		}

		// Extract window for processing
		window := make([]float32, windowSize)
		copy(window, e.audioBuffer[:windowSize])
		e.audioBuffer = e.audioBuffer[windowSize:]
		e.processedSamples += int64(windowSize)
		threshold := e.threshold
		e.stateLock.Unlock()

		// Run inference to get speech probability
		speechProb, err := e.detector.Infer(window)
		if err != nil {
			log.Printf("[SileroVAD] Infer error: %v", err)
			continue
		}

		e.currSample += windowSize

		// Speech detection logic
		if speechProb >= threshold && e.tempEnd != 0 {
			e.tempEnd = 0
		}

		if speechProb >= threshold && !e.triggered {
			e.triggered = true
			speechStartSample := e.currSample - windowSize - speechPadSamples
			if speechStartSample < 0 {
				speechStartSample = 0
			}
			speechStartMs := speechStartSample * 1000 / sampleRate

			if !e.isSpeaking.Load() {
				e.isSpeaking.Store(true)
				e.emitEvent(pipeline.EventVADSpeechStart, msg.SessionID, speechProb, speechStartMs)
				log.Printf("[SileroVAD] Speech started (startMs=%d, prob=%.3f)", speechStartMs, speechProb)
			}
		}

		if speechProb < (threshold - 0.15) && e.triggered {
			if e.tempEnd == 0 {
				e.tempEnd = e.currSample
			}

			// Check if enough silence has passed
			if e.currSample-e.tempEnd >= minSilenceSamples {
				speechEndSample := e.tempEnd + speechPadSamples
				speechEndMs := speechEndSample * 1000 / sampleRate
				e.tempEnd = 0
				e.triggered = false

				if e.isSpeaking.Load() {
					e.isSpeaking.Store(false)
					e.emitEvent(pipeline.EventVADSpeechEnd, msg.SessionID, speechProb, speechEndMs)
					log.Printf("[SileroVAD] Speech ended (endMs=%d, prob=%.3f)", speechEndMs, speechProb)
				}
			}
		}
	}

	// Handle output based on mode
	switch e.mode {
	case VADModePassthrough:
		select {
		case e.BaseElement.OutChan <- msg:
		case <-ctx.Done():
			return
		}

	case VADModeFilter:
		if e.isSpeaking.Load() {
			select {
			case e.BaseElement.OutChan <- msg:
			case <-ctx.Done():
				return
			}
		}
	}
}

// emitEvent emits a VAD event to the bus
func (e *CustomSileroVADElement) emitEvent(eventType pipeline.EventType, sessionID string, confidence float32, audioMs int) {
	if e.Bus() == nil {
		return
	}

	payload := pipeline.VADPayload{
		AudioMs:    audioMs,
		ItemID:     sessionID,
		Confidence: confidence,
	}

	// For speech start events, include pre-roll audio and clear buffer
	if eventType == pipeline.EventVADSpeechStart {
		payload.PreRollAudio = e.preRollBuffer.ReadAll()
		payload.SampleRate = 16000
		payload.Channels = 1
		// Clear pre-roll buffer after use to avoid including old audio in next speech segment
		e.preRollBuffer.Clear()
	}

	event := pipeline.Event{
		Type:      eventType,
		Timestamp: time.Now(),
		Payload:   payload,
	}

	e.Bus().Publish(event)
}

// bytesToFloat32 converts 16-bit PCM (little-endian) to normalized float32 in [-1, 1].
func (e *CustomSileroVADElement) bytesToFloat32(data []byte) []float32 {
	n := len(data) / 2
	samples := make([]float32, n)
	for i := 0; i < n; i++ {
		v := int16(binary.LittleEndian.Uint16(data[i*2 : i*2+2]))
		samples[i] = float32(v) / 32768.0
	}
	return samples
}

// SetThreshold updates the VAD threshold
func (e *CustomSileroVADElement) SetThreshold(threshold float32) error {
	if threshold < 0 || threshold > 1 {
		return fmt.Errorf("threshold must be between 0 and 1")
	}
	e.stateLock.Lock()
	e.threshold = threshold
	e.currSample = 0
	e.triggered = false
	e.tempEnd = 0
	e.stateLock.Unlock()

	if e.detector != nil {
		e.detector.Reset()
	}
	return nil
}

// GetIsSpeaking returns whether speech is currently detected
func (e *CustomSileroVADElement) GetIsSpeaking() bool {
	return e.isSpeaking.Load()
}

// SetDetector sets a custom detector (useful for testing).
func (e *CustomSileroVADElement) SetDetector(detector vad.DetectorInterface) {
	e.detector = detector
}

// GetDetector returns the current detector (useful for testing).
func (e *CustomSileroVADElement) GetDetector() vad.DetectorInterface {
	return e.detector
}