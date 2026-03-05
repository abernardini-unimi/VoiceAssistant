// Usage:
//
//  go run examples/web-voice-assistant-groq_v3/main.go
//  open http://localhost:8082

package main

import (
    "context"
    "log"
    "os"
	"net/http"
    "path/filepath"
    "sync"
    "time"

    "github.com/joho/godotenv"
    "github.com/realtime-ai/realtime-ai/pkg/elements"
    "github.com/realtime-ai/realtime-ai/pkg/pipeline"
    "github.com/realtime-ai/realtime-ai/pkg/realtimeapi"
    "github.com/realtime-ai/realtime-ai/pkg/server"
    "github.com/realtime-ai/realtime-ai/pkg/tts"
    "inx-voice-assistant-inxide/custom_elements"
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

    var padManager *custom_elements.PADManager
    
    padManager = custom_elements.NewPADManager(
        "models/pad/model.onnx",          // 1. Percorso ONNX PAD
        "models/pad/config.json",         // 2. Percorso Config PAD
        "models/emotion/model.onnx",      // 3. Percorso ONNX Emotion
        "models/emotion/config.json",     // 4. Percorso Config Emotion
    )
    if padManager != nil {
        log.Println("[Init] PAD Manager pronto (In attesa dell'avvio ONNX da parte del VAD).")
    } else {
        log.Println("[Init] WARNING: Impossibile caricare il PAD Manager.")
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
    
    promptPath := "prompt/base_v6.txt"
    promptBytes, err := os.ReadFile(promptPath)
    var systemPrompt string
    if err != nil {
        log.Printf("[Init] ⚠️ Impossibile leggere %s: %v. Uso prompt di default.", promptPath, err)
        systemPrompt = "Sei un assistente vocale. Rispondi in modo conciso."
    } else {
        systemPrompt = string(promptBytes)
        log.Printf("[Init] 📝 Prompt caricato con successo da %s (%d bytes)", promptPath, len(systemPrompt))
    }

    // Create server configuration
    config := server.DefaultWebRTCRealtimeConfig()
    config.RTCUDPPort = udpPort
    config.ICELite = false

    // Create WebRTC server
    srv := server.NewWebRTCRealtimeServer(config)

    // Set pipeline factory
    srv.SetPipelineFactory(func(ctx context.Context, session *realtimeapi.Session) (*pipeline.Pipeline, error) {
        return createPipeline(ctx, session, PipelineConfig{
            OpenAIKey:    openaiKey,
            VADModelPath: vadModelPath,
            Voice:        voice,
            SystemPrompt: systemPrompt,
            PADManager:   padManager,
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
    OpenAIKey    string
    VADModelPath string
    Voice        string
    SystemPrompt string
    PADManager   *custom_elements.PADManager
}

// createPipeline creates the voice assistant pipeline using OpenAI TTS
func createPipeline(ctx context.Context, session *realtimeapi.Session, cfg PipelineConfig) (*pipeline.Pipeline, error) {
    pipe := pipeline.NewPipeline("voice-assistant-" + session.ID)

    interruptConfig := pipeline.DefaultInterruptConfig()
    interruptConfig.EnableHybridMode = true
    interruptConfig.InterruptCooldownMs = 500
    interruptConfig.MinSpeechForConfirmMs = 400

    interruptManager := pipeline.NewInterruptManager(pipe.Bus(), interruptConfig)
    if err := interruptManager.Start(ctx); err != nil {
        log.Printf("Failed to start interrupt manager: %v", err)
    }

    // Create elements
    var elems []pipeline.Element
    var prevElem pipeline.Element

    // 1. Input resample: 48kHz → 16kHz
    inputResample := elements.NewAudioResampleElement(48000, 16000, 1, 1)
    elems = append(elems, inputResample)
    prevElem = inputResample

    // 2. VAD
	if cfg.VADModelPath != "" {
		vadConfig := custom_elements.SileroVADConfig{
			ModelPath:       cfg.VADModelPath,
			Threshold:       0.75,
			MinSilenceDurMs: 700,
			SpeechPadMs:     20,
			PreRollMs:       99999,
			Mode:            custom_elements.VADModePassthrough,
		}
		vadElem, err := custom_elements.CustomNewSileroVADElement(vadConfig)
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
				
				// --- NUOVO: IL VAD HA ACCESO ONNX, CARICHIAMO IL PAD MANAGER! ---
				if cfg.PADManager != nil {
					cfg.PADManager.InitSession()
				}
			}
		}
	}

    // --- NUOVO: PAD Collector (Raccoglie l'audio e lancia l'inferenza al termine) ---
    if cfg.PADManager != nil {
        padCollector := custom_elements.NewPADCollectorElement(cfg.PADManager)
        elems = append(elems, padCollector)
        pipe.Link(prevElem, padCollector)
        prevElem = padCollector
    }

    // 3. ASR (Groq Whisper)
    config := custom_elements.NewCustomWhisperConfig{
        APIKey:                 os.Getenv("GROQ_API_KEY"),
        BaseURL:                "https://api.groq.com/openai/v1",
        Model:                  "whisper-large-v3",                
        Language:               "it",
        EnablePartialResults:   false,
        VADEnabled:             true,
        SampleRate:             16000,
        Channels:               1,                              
    }

    asrElem, err := custom_elements.NewCustomWhisperElement(config)
    if err != nil {
        log.Fatal(err)
    }

    elems = append(elems, asrElem)
    pipe.Link(prevElem, asrElem)
    prevElem = asrElem

    // --- NUOVO: PAD Enricher (Aggiunge il JSON PAD al testo trascritto) ---
    if cfg.PADManager != nil {
        padEnricher := custom_elements.NewPADEnricherElement(cfg.PADManager)
        elems = append(elems, padEnricher)
        pipe.Link(prevElem, padEnricher)
        prevElem = padEnricher
    }

    // 4. Chat (Groq gpt-oss-120)
    chatConfig := custom_elements.CustomChatConfig{
        APIKey:       os.Getenv("GROQ_API_KEY"),
        Model:        "openai/gpt-oss-120b",
        BaseURL:      "https://api.groq.com/openai/v1",  
        SystemPrompt: cfg.SystemPrompt,
        MaxHistory:   20,
        Temperature:  0.7,
        Streaming:    false,
        InitialPrompt:   "Ciao, chi sei?",
    }

    chatElem, err := custom_elements.NewCustomChatElement(chatConfig)
    if err != nil {
        return nil, err
    }
    elems = append(elems, chatElem)
    pipe.Link(prevElem, chatElem)
    prevElem = chatElem
    
    // 5. OpenAI TTS (gpt-4o-mini-tts)
    ttsProvider := tts.NewOpenAITTSProvider(cfg.OpenAIKey)
    ttsProvider.SetInstructions("Speak in a friendly and engaging tone in the same user language. Coincise responses.")
    useStreaming := true

    ttsElem := custom_elements.NewCustomUniversalTTSElement(ttsProvider)
    ttsElem.SetVoice("coral")         
    ttsElem.SetLanguage("it-IT")      
    // ttsElem.SetOption("speed", 1.1)   
    ttsElem.SetStreaming(useStreaming)

    elems = append(elems, ttsElem)
    pipe.Link(prevElem, ttsElem)
    prevElem = ttsElem

    // 6. Simple Resampler manuale
    customResample := NewSimpleResamplerElement()
    elems = append(elems, customResample)
    pipe.Link(prevElem, customResample)
    prevElem = customResample

    // Add all elements to pipeline
    pipe.AddElements(elems)

    log.Printf("[Pipeline] Flow aggiornato: Resample → VAD → PAD_Collector → ASR → PAD_Enricher → Chat → TTS → Resample")
    return pipe, nil
}

// findVADModel looks for the VAD model in common locations
func findVADModel() string {
	paths := []string{
		"models/vad/silero_vad.onnx",
		"../models/vad/silero_vad.onnx",
		"../../models/vad/silero_vad.onnx",
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
    cancel context.CancelFunc
    wg     sync.WaitGroup
}

func NewSimpleResamplerElement() *SimpleResamplerElement {
    return &SimpleResamplerElement{
        BaseElement: pipeline.NewBaseElement("simple-resampler", 100),
    }
}

func (e *SimpleResamplerElement) Start(ctx context.Context) error {
    ctx, cancel := context.WithCancel(ctx)
    e.cancel = cancel

    e.wg.Add(1)
    go func() {
        defer e.wg.Done()
        defer close(e.OutChan)

        interruptChan := make(chan pipeline.Event, 10)
        if bus := e.BaseElement.Bus(); bus != nil {
            bus.Subscribe(pipeline.EventInterrupted, interruptChan)
            bus.Subscribe(pipeline.EventVADSpeechStart, interruptChan)
            defer func() {
                bus.Unsubscribe(pipeline.EventInterrupted, interruptChan)
                bus.Unsubscribe(pipeline.EventVADSpeechStart, interruptChan)
            }()
        }

        const (
            targetSampleRate = 48000
            chunkSamples     = 960              // 20ms @ 48kHz mono
            chunkBytes       = chunkSamples * 2 // 16-bit = 2 bytes/sample
            chunkDuration    = 20 * time.Millisecond
        )

        // Buffer residuo per garantire chunk esatti (evita scatti da chunk irregolari)
        var residual []byte

        for {
            select {
            case <-ctx.Done():
                return

            case msg, ok := <-e.InChan:
                if !ok {
                    return
                }

                if msg.Type != pipeline.MsgTypeAudio || len(msg.AudioData.Data) == 0 {
                    select {
                    case e.OutChan <- msg:
                    case <-ctx.Done():
                        return
                    }
                    continue
                }

                // Svuota segnali di interrupt "vecchi" prima di processare nuovo audio TTS
                drainChan(interruptChan)

                // Upsampling con interpolazione lineare
                inputData := msg.AudioData.Data
                inputRate := msg.AudioData.SampleRate
                var pcm48k []byte

                switch inputRate {
                case 24000:
                    pcm48k = upsample24to48(inputData)
                case 48000:
                    pcm48k = inputData
                default:
                    log.Printf("[Resampler] Unexpected sample rate: %d, passthrough", inputRate)
                    pcm48k = inputData
                }

                // Prepend residuo precedente
                data := append(residual, pcm48k...)
                residual = nil

                // ── PACING BASATO SU WALL-CLOCK ──
                // Calcola quando dovrebbe partire il primo chunk
                nextTick := time.Now()

                numChunks := len(data) / chunkBytes
                residual = data[numChunks*chunkBytes:] // Salva il resto

                for i := 0; i < numChunks; i++ {
                    // Controlla interruzione
                    select {
                    case <-interruptChan:
                        log.Println("[Resampler] 🛑 Interruzione durante riproduzione.")
                        residual = nil // Scarta anche il residuo
                        drainInChan(e.InChan, interruptChan)
                        goto NextMessage
                    case <-ctx.Done():
                        return
                    default:
                    }

                    chunk := data[i*chunkBytes : (i+1)*chunkBytes]
                    outMsg := &pipeline.PipelineMessage{
                        Type:      pipeline.MsgTypeAudio,
                        SessionID: msg.SessionID,
                        AudioData: &pipeline.AudioData{
                            Data:       chunk,
                            SampleRate: targetSampleRate,
                            Channels:   1,
                            MediaType:  msg.AudioData.MediaType,
                            Timestamp:  time.Now(),
                        },
                    }

                    select {
                    case e.OutChan <- outMsg:
                    case <-ctx.Done():
                        return
                    }
					
                    // Avanza il tick di 20ms e attendi fino a quel momento esatto
                    nextTick = nextTick.Add(chunkDuration)
                    delay := time.Until(nextTick)
                    if delay > 0 {
                        select {
                        case <-interruptChan:
                            log.Println("[Resampler] 🛑 Interruzione durante attesa.")
                            residual = nil
                            drainInChan(e.InChan, interruptChan)
                            goto NextMessage
                        case <-time.After(delay):
                        case <-ctx.Done():
                            return
                        }
                    }
                }
            }
        NextMessage:
        }
    }()
    return nil
}

// upsample24to48 converte PCM 16-bit 24kHz → 48kHz con interpolazione lineare
func upsample24to48(in []byte) []byte {
    numSamples := len(in) / 2
    out := make([]byte, numSamples*2*2) // doppio dei sample

    for i := 0; i < numSamples; i++ {
        s0 := int16(in[i*2]) | int16(in[i*2+1])<<8

        var s1 int16
        if i+1 < numSamples {
            s1 = int16(in[(i+1)*2]) | int16(in[(i+1)*2+1])<<8
        } else {
            s1 = s0
        }

        // Sample originale
        out[i*4] = byte(s0)
        out[i*4+1] = byte(s0 >> 8)

        // Sample interpolato (media)
        mid := int16((int32(s0) + int32(s1)) / 2)
        out[i*4+2] = byte(mid)
        out[i*4+3] = byte(mid >> 8)
    }
    return out
}

func drainChan(ch chan pipeline.Event) {
    for {
        select {
        case <-ch:
        default:
            return
        }
    }
}

func drainInChan(in chan *pipeline.PipelineMessage, interrupt chan pipeline.Event) {
    for {
        select {
        case <-in:
        case <-interrupt:
        default:
            return
        }
    }
}

func (e *SimpleResamplerElement) Stop() error {
    if e.cancel != nil {
        e.cancel()
        e.wg.Wait()
        e.cancel = nil
    }
    return nil
}