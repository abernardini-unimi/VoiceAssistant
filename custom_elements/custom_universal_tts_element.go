package custom_elements

import (
	"context"
	"fmt"
	"log"
	"reflect"
	"sync"
	"time"

	"github.com/realtime-ai/realtime-ai/pkg/pipeline"
	"github.com/realtime-ai/realtime-ai/pkg/tts"
)

// CustomUniversalTTSElement is a TTS element that can use any TTSProvider
type CustomUniversalTTSElement struct {
    *pipeline.BaseElement

    provider tts.TTSProvider
    voice    string
    language string
    options  map[string]interface{}

    streaming bool

    cancel context.CancelFunc
    wg     sync.WaitGroup

    // --- NUOVI CAMPI PER L'INTERRUZIONE ---
    currentReqCancel context.CancelFunc
    reqMu            sync.Mutex
    interruptSub     chan pipeline.Event
}

// drainInChan svuota i messaggi in sospeso nel canale di input
func (e *CustomUniversalTTSElement) drainInChan() {
    for {
        select {
        case <-e.BaseElement.InChan:
            // Scarta silenziosamente il vecchio messaggio
        default:
            // Il canale è vuoto, esci
            return
        }
    }
}

// NewCustomUniversalTTSElement creates a new custom universal TTS element
func NewCustomUniversalTTSElement(provider tts.TTSProvider) *CustomUniversalTTSElement {
	elem := &CustomUniversalTTSElement{
		BaseElement: pipeline.NewBaseElement(fmt.Sprintf("%s-custom-tts-element", provider.Name()), 100),
		provider:    provider,
		voice:       provider.GetDefaultVoice(),
		language:    "en-US", // Default language
		options:     make(map[string]interface{}),
		streaming:   false,
	}

	// Register properties
	elem.registerProperties()

	return elem
}

// registerProperties registers the element's configurable properties
func (e *CustomUniversalTTSElement) registerProperties() {
	e.RegisterProperty(pipeline.PropertyDesc{
		Name:     "voice",
		Type:     reflect.TypeOf(""),
		Writable: true,
		Readable: true,
		Default:  e.provider.GetDefaultVoice(),
	})

	e.RegisterProperty(pipeline.PropertyDesc{
		Name:     "language",
		Type:     reflect.TypeOf(""),
		Writable: true,
		Readable: true,
		Default:  "en-US",
	})
}

func (e *CustomUniversalTTSElement) SetStreaming(enable bool) {
	e.streaming = enable
}

// Start starts the TTS element
func (e *CustomUniversalTTSElement) Start(ctx context.Context) error {
    if err := e.provider.ValidateConfig(); err != nil {
        return fmt.Errorf("TTS provider validation failed: %w", err)
    }

    ctx, cancel := context.WithCancel(ctx)
    e.cancel = cancel

    // --- NUOVO: Sottoscrizione per fermare la generazione in corso ---
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
                    // 1. Ferma la chiamata API in corso
                    e.reqMu.Lock()
                    if e.currentReqCancel != nil {
                        log.Printf("[%s] 🛑 Interruzione! Blocco lo streaming audio verso OpenAI.", e.provider.Name())
                        e.currentReqCancel()
                        e.currentReqCancel = nil
                    }
                    e.reqMu.Unlock()

                    // 2. Svuota eventuali testi in coda
                    e.drainInChan()
                }
            }
        }()
    }
    // ----------------------------------------------------------------

    e.wg.Add(1)
    go func() {
        defer e.wg.Done()
        e.processMessages(ctx)
    }()

    log.Printf("[%s] Custom TTS element started with voice: %s", e.provider.Name(), e.voice)
    return nil
}

func (e *CustomUniversalTTSElement) Stop() error {
    if e.cancel != nil {
        e.cancel()
        e.wg.Wait()
        e.cancel = nil
    }

    // --- NUOVO: Pulizia eventi ---
    if e.interruptSub != nil && e.BaseElement.Bus() != nil {
        e.BaseElement.Bus().Unsubscribe(pipeline.EventInterrupted, e.interruptSub)
        e.BaseElement.Bus().Unsubscribe(pipeline.EventVADSpeechStart, e.interruptSub)
        close(e.interruptSub)
        e.interruptSub = nil
    }

    log.Printf("[%s] Custom TTS element stopped", e.provider.Name())
    return nil
}

// processMessages processes incoming text messages and synthesizes speech
func (e *CustomUniversalTTSElement) processMessages(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case msg := <-e.BaseElement.InChan:
			if msg.Type == pipeline.MsgTypeData && msg.TextData != nil {
				text := string(msg.TextData.Data)

				// --- NUOVO: AGGIORNAMENTO DINAMICO PARAMETRI DA CHAT ---
				if msg.Metadata != nil {
					// Trasformiamo Metadata in una mappa leggibile
					if meta, ok := msg.Metadata.(map[string]interface{}); ok {
						
						// 1. Applica la velocità (speed)
						if speed, ok := meta["speed"].(float64); ok {
							// Usiamo il metodo dell'elemento, non del provider direttamente
							e.SetOption("speed", speed) 
							log.Printf("[%s] ⚡ Velocità aggiornata: %.2f", e.provider.Name(), speed)
						}
						
						// 2. Applica le istruzioni (tone/pitch)
						if inst, ok := meta["instructions"].(string); ok {
							// Qui dobbiamo fare un controllo: il provider supporta le istruzioni?
							// Se stai usando OpenAI TTS del package realtime-ai, di solito è un puntatore
							// che espone SetInstructions. Proviamo il type casting:
							type instructionSetter interface {
								SetInstructions(string)
							}
							if setter, ok := e.provider.(instructionSetter); ok {
								setter.SetInstructions(inst)
								log.Printf("[%s] 🎭 Tono aggiornato: %s", e.provider.Name(), inst)
							}
						}
					}
				}
				// -------------------------------------------------------

				// --- Creazione del contesto specifico per questa sintesi ---
				e.reqMu.Lock()
				msgCtx, cancelFunc := context.WithCancel(ctx)
				e.currentReqCancel = cancelFunc
				e.reqMu.Unlock()

				handled := false
				if e.streaming {
					if streamProvider, ok := e.provider.(tts.StreamingTTSProvider); ok {
						if err := e.streamAndOutput(msgCtx, streamProvider, text); err != nil {
							if msgCtx.Err() == context.Canceled {
								log.Printf("[%s] Streaming saltato/interrotto correttamente.", e.provider.Name())
							} else {
								log.Printf("[%s] Streaming failed: %v", e.provider.Name(), err)
								e.publishError(fmt.Sprintf("Streaming failed: %v", err))
							}
						}
						handled = true
					}
				}

				if !handled {
					if err := e.synthesizeAndOutput(msgCtx, text); err != nil {
						if msgCtx.Err() == context.Canceled {
							log.Printf("[%s] Sintesi saltata/interrotta correttamente.", e.provider.Name())
						} else {
							log.Printf("[%s] Failed to synthesize: %v", e.provider.Name(), err)
							e.publishError(fmt.Sprintf("Failed to synthesize: %v", err))
						}
					}
				}

				e.reqMu.Lock()
				if e.currentReqCancel != nil {
					e.currentReqCancel()
					e.currentReqCancel = nil
				}
				e.reqMu.Unlock()

			} else {
				select {
				case e.BaseElement.OutChan <- msg:
				case <-ctx.Done():
					return
				}
			}
		}
	}
}

// streamAndOutput handles chunk-by-chunk reception
func (e *CustomUniversalTTSElement) streamAndOutput(ctx context.Context, provider tts.StreamingTTSProvider, text string) error {
	req := &tts.SynthesizeRequest{
		Text:     text,
		Voice:    e.voice,
		Language: e.language,
		Options:  e.options,
	}

	log.Printf("[%s] Starting stream synthesis...", e.provider.Name())

	// --- Send PRE-ROLL SILENCE (Fix firts word distorted)---
    silenceDuration := 100 * time.Millisecond
    sampleRate := 24000 // Default OpenAI
    if sr, ok := e.options["sample_rate"].(int); ok {
        sampleRate = sr
    }
    
    numBytes := int(float64(sampleRate) * 1 * 2 * silenceDuration.Seconds())
    silenceData := make([]byte, numBytes) 

    silenceMsg := &pipeline.PipelineMessage{
        Type: pipeline.MsgTypeAudio,
        AudioData: &pipeline.AudioData{
            Data:       silenceData,
            SampleRate: sampleRate,
            Channels:   1,
            MediaType:  pipeline.AudioMediaTypePCM,
            Timestamp:  time.Now(),
        },
    }

    select {
    case e.BaseElement.OutChan <- silenceMsg:
    case <-ctx.Done():
        return ctx.Err()
    }

	audioChan, errChan := provider.StreamSynthesize(ctx, req)

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()

		case err := <-errChan:
			if err != nil {
				return err
			}

		case chunk, ok := <-audioChan:
			if !ok {
				log.Printf("[%s] Stream finished.", e.provider.Name())
				return nil
			}

			msg := &pipeline.PipelineMessage{
				Type: pipeline.MsgTypeAudio,
				AudioData: &pipeline.AudioData{
					Data:       chunk,
					SampleRate: sampleRate,
					Channels:   1,
					MediaType:  pipeline.AudioMediaTypePCM,
					Timestamp:  time.Now(),
				},
			}
			e.BaseElement.OutChan <- msg
		}
	}
}

// synthesizeAndOutput synthesizes speech from text and outputs audio data
func (e *CustomUniversalTTSElement) synthesizeAndOutput(ctx context.Context, text string) error {
	req := &tts.SynthesizeRequest{
		Text:     text,
		Voice:    e.voice,
		Language: e.language,
		Options:  e.options,
	}

	resp, err := e.provider.Synthesize(ctx, req)
	if err != nil {
		return err
	}

	var mediaType pipeline.AudioMediaType
	if amt, ok := resp.AudioFormat.MediaType.(pipeline.AudioMediaType); ok {
		mediaType = amt
	} else if str, ok := resp.AudioFormat.MediaType.(string); ok {
		mediaType = pipeline.AudioMediaType(str)
	} else {
		mediaType = pipeline.AudioMediaTypeRaw
	}

	msg := &pipeline.PipelineMessage{
		Type: pipeline.MsgTypeAudio,
		AudioData: &pipeline.AudioData{
			Data:       resp.AudioData,
			SampleRate: resp.AudioFormat.SampleRate,
			Channels:   resp.AudioFormat.Channels,
			MediaType:  mediaType,
			Timestamp:  time.Now(),
		},
	}

	e.BaseElement.OutChan <- msg

	log.Printf("[%s] Synthesized %d bytes of audio (voice: %s)",
		e.provider.Name(), len(resp.AudioData), e.voice)

	return nil
}

// publishError publishes an error event to the pipeline bus
func (e *CustomUniversalTTSElement) publishError(message string) {
	if e.BaseElement.Bus() != nil {
		e.BaseElement.Bus().Publish(pipeline.Event{
			Type:      pipeline.EventError,
			Timestamp: time.Now(),
			Payload:   message,
		})
	}
}

// SetVoice sets the voice to use for synthesis
func (e *CustomUniversalTTSElement) SetVoice(voice string) {
	e.voice = voice
}

// SetLanguage sets the language for synthesis
func (e *CustomUniversalTTSElement) SetLanguage(language string) {
	e.language = language
}

// SetOption sets a provider-specific option
func (e *CustomUniversalTTSElement) SetOption(key string, value interface{}) {
	if e.options == nil {
		e.options = make(map[string]interface{})
	}
	e.options[key] = value
}

// GetProvider returns the underlying TTS provider
func (e *CustomUniversalTTSElement) GetProvider() tts.TTSProvider {
	return e.provider
}

// GetSupportedVoices returns the list of supported voices
func (e *CustomUniversalTTSElement) GetSupportedVoices() []string {
	return e.provider.GetSupportedVoices()
}