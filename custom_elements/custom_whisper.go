package custom_elements

import (
	"bytes"
	"context"
	"encoding/binary"
	"io"
	"log"
	"sync"
	"time"

	"github.com/realtime-ai/realtime-ai/pkg/asr"
	"github.com/sashabaranov/go-openai"
)

// CustomWhisperProvider implements the asr.Provider interface using OpenAI's Whisper API.
type CustomWhisperProvider struct {
	client *openai.Client
	mu     sync.RWMutex
}

// NewCustomWhisperProvider creates a new OpenAI Whisper ASR provider.
func NewCustomWhisperProvider(apiKey string, baseURL string) (*CustomWhisperProvider, error) {
	if apiKey == "" {
		return nil, &asr.Error{
			Code:    asr.ErrCodeInvalidConfig,
			Message: "OpenAI API key is required",
		}
	}

	clientConfig := openai.DefaultConfig(apiKey)
	if baseURL != "" {
		clientConfig.BaseURL = baseURL
		log.Printf("[Whisper STT] Using BaseURL: %s", clientConfig.BaseURL)
	}
	client := openai.NewClientWithConfig(clientConfig)

	return &CustomWhisperProvider{
		client: client,
	}, nil
}

// Name returns the provider name.
func (w *CustomWhisperProvider) Name() string {
	return "custom-openai-whisper"
}

// Recognize performs speech recognition on a complete audio segment.
func (w *CustomWhisperProvider) Recognize(ctx context.Context, audio io.Reader, audioConfig asr.AudioConfig, config asr.RecognitionConfig) (*asr.RecognitionResult, error) {
	w.mu.RLock()
	defer w.mu.RUnlock()

	// Read all audio data
	audioData, err := io.ReadAll(audio)
	if err != nil {
		return nil, &asr.Error{
			Code:    asr.ErrCodeInvalidAudio,
			Message: "failed to read audio data",
			Err:     err,
		}
	}

	if len(audioData) == 0 {
		return nil, &asr.Error{
			Code:    asr.ErrCodeInvalidAudio,
			Message: "audio data is empty",
		}
	}

	// Convert audio to WAV format if it's raw PCM
	var fileBytes []byte
	if audioConfig.Encoding == "pcm" || audioConfig.Encoding == "" {
		fileBytes, err = convertPCMToWAV(audioData, audioConfig)
		if err != nil {
			return nil, &asr.Error{
				Code:    asr.ErrCodeInvalidAudio,
				Message: "failed to convert PCM to WAV",
				Err:     err,
			}
		}
	} else {
		fileBytes = audioData
	}

	// Prepare Whisper API request
	req := openai.AudioRequest{
		Model:    config.Model,
		FilePath: "audio.wav", // Filename hint for API
		Reader:   bytes.NewReader(fileBytes),
		Prompt:   config.Prompt,
		Language: config.Language,
	}

	if req.Model == "" {
		req.Model = openai.Whisper1 // Default to whisper-1
	}

	// Set temperature if specified
	if config.Temperature > 0 {
		req.Temperature = config.Temperature
	}

	// Call Whisper API
	startTime := time.Now()
	resp, err := w.client.CreateTranscription(ctx, req)
	if err != nil {
		return nil, &asr.Error{
			Code:    asr.ErrCodeProviderError,
			Message: "Whisper API request failed",
			Err:     err,
		}
	}

	duration := time.Since(startTime)

	result := &asr.RecognitionResult{
		Text:       resp.Text,
		IsFinal:    true, // Whisper always returns final results
		Confidence: -1,   // Whisper API doesn't provide confidence scores
		Language:   config.Language,
		Duration:   duration,
		Timestamp:  time.Now(),
		Metadata: map[string]interface{}{
			"model": req.Model,
		},
	}

	if config.Language == "" || config.Language == "auto" {
		result.Metadata["language_detection"] = "not available in basic mode"
	}

	return result, nil
}

// StreamingRecognize creates a streaming recognizer for continuous audio input.
func (w *CustomWhisperProvider) StreamingRecognize(ctx context.Context, audioConfig asr.AudioConfig, config asr.RecognitionConfig) (asr.StreamingRecognizer, error) {
	w.mu.RLock()
	defer w.mu.RUnlock()

	recognizer := &customWhisperStreamingRecognizer{
		provider:    w,
		audioConfig: audioConfig,
		config:      config,
		resultsChan: make(chan *asr.RecognitionResult, 10),
		audioChan:   make(chan []byte, 100),
		ctx:         ctx,
	}

	go recognizer.processAudio()

	return recognizer, nil
}

// SupportsStreaming indicates if the provider supports streaming recognition.
func (w *CustomWhisperProvider) SupportsStreaming() bool {
	return true
}

// SupportedLanguages returns a list of supported language codes.
func (w *CustomWhisperProvider) SupportedLanguages() []string {
	return []string{}
}

// Close releases any resources held by the provider.
func (w *CustomWhisperProvider) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	return nil
}

type customWhisperStreamingRecognizer struct {
	provider    *CustomWhisperProvider
	audioConfig asr.AudioConfig
	config      asr.RecognitionConfig
	resultsChan chan *asr.RecognitionResult
	audioChan   chan []byte
	audioBuffer []byte
	ctx         context.Context
	cancel      context.CancelFunc
	mu          sync.Mutex
	closed      bool
}

func (r *customWhisperStreamingRecognizer) SendAudio(ctx context.Context, audioData []byte) error {
	r.mu.Lock()
	if r.closed {
		r.mu.Unlock()
		return &asr.Error{
			Code:    asr.ErrCodeProviderError,
			Message: "recognizer is closed",
		}
	}
	r.mu.Unlock()

	select {
	case r.audioChan <- audioData:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	case <-r.ctx.Done():
		return r.ctx.Err()
	}
}

func (r *customWhisperStreamingRecognizer) Results() <-chan *asr.RecognitionResult {
	return r.resultsChan
}

func (r *customWhisperStreamingRecognizer) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if r.closed {
		return nil
	}

	r.closed = true
	close(r.audioChan)
	if r.cancel != nil {
		r.cancel()
	}

	return nil
}

func (r *customWhisperStreamingRecognizer) processAudio() {
	defer close(r.resultsChan)

	ctx, cancel := context.WithCancel(r.ctx)
	r.cancel = cancel
	defer cancel()

	const maxBufferSize = 16000 * 2 * 10

	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			r.processBufferedAudio(ctx)
			return

		case audioData, ok := <-r.audioChan:
			if !ok {
				r.processBufferedAudio(ctx)
				return
			}

			r.mu.Lock()
			r.audioBuffer = append(r.audioBuffer, audioData...)
			bufferSize := len(r.audioBuffer)
			r.mu.Unlock()

			if bufferSize >= maxBufferSize {
				r.processBufferedAudio(ctx)
			}

		case <-ticker.C:
			r.mu.Lock()
			hasAudio := len(r.audioBuffer) > 0
			r.mu.Unlock()

			if hasAudio {
				r.processBufferedAudio(ctx)
			}
		}
	}
}

func (r *customWhisperStreamingRecognizer) processBufferedAudio(ctx context.Context) {
	r.mu.Lock()
	if len(r.audioBuffer) == 0 {
		r.mu.Unlock()
		return
	}

	minSamples := r.audioConfig.SampleRate * r.audioConfig.Channels * (r.audioConfig.BitsPerSample / 8) / 10
	if len(r.audioBuffer) < minSamples {
		r.mu.Unlock()
		return
	}

	audioData := make([]byte, len(r.audioBuffer))
	copy(audioData, r.audioBuffer)
	r.audioBuffer = r.audioBuffer[:0]
	r.mu.Unlock()

	if r.config.EnablePartialResults && len(audioData) > 0 {
		select {
		case r.resultsChan <- &asr.RecognitionResult{
			Text:       "",
			IsFinal:    false,
			Confidence: -1,
			Language:   r.config.Language,
			Timestamp:  time.Now(),
			Metadata: map[string]interface{}{
				"processing": true,
			},
		}:
		case <-ctx.Done():
			return
		}
	}

	reader := bytes.NewReader(audioData)
	result, err := r.provider.Recognize(ctx, reader, r.audioConfig, r.config)
	if err != nil {
		log.Printf("Whisper recognition error: %v", err)
		return
	}

	select {
	case r.resultsChan <- result:
	case <-ctx.Done():
		return
	}
}

func convertPCMToWAV(pcmData []byte, config asr.AudioConfig) ([]byte, error) {
	var buf bytes.Buffer

	buf.WriteString("RIFF")
	fileSize := uint32(36 + len(pcmData))
	binary.Write(&buf, binary.LittleEndian, fileSize)
	buf.WriteString("WAVE")

	buf.WriteString("fmt ")
	subChunk1Size := uint32(16)
	binary.Write(&buf, binary.LittleEndian, subChunk1Size)
	audioFormat := uint16(1)
	binary.Write(&buf, binary.LittleEndian, audioFormat)
	binary.Write(&buf, binary.LittleEndian, uint16(config.Channels))
	binary.Write(&buf, binary.LittleEndian, uint32(config.SampleRate))

	bitsPerSample := config.BitsPerSample
	if bitsPerSample == 0 {
		bitsPerSample = 16
	}

	byteRate := uint32(config.SampleRate * config.Channels * bitsPerSample / 8)
	binary.Write(&buf, binary.LittleEndian, byteRate)

	blockAlign := uint16(config.Channels * bitsPerSample / 8)
	binary.Write(&buf, binary.LittleEndian, blockAlign)
	binary.Write(&buf, binary.LittleEndian, uint16(bitsPerSample))

	buf.WriteString("data")
	binary.Write(&buf, binary.LittleEndian, uint32(len(pcmData)))
	buf.Write(pcmData)

	return buf.Bytes(), nil
}