package custom_elements

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strconv"
	"sync"

	"github.com/realtime-ai/realtime-ai/pkg/pipeline"
	ort "github.com/yalue/onnxruntime_go"
)

// ModelConfig mappa la struttura del config.json (valida sia per PAD che per Emotion)
type ModelConfig struct {
	Id2Label map[string]string `json:"id2label"`
}

// Struttura di supporto per tenere in memoria lo storico accoppiato
type AcousticData struct {
	PAD     map[string]float64
	Emotion map[string]float64
}

// Funzione che bypassa il riordino alfabetico di Go, ordina per valore e arrotonda a 2 cifre
func buildOrderedEmotionJSON(emoMap map[string]float64) json.RawMessage {
	type kv struct {
		Key   string
		Value float64
	}
	var scores []kv
	for k, v := range emoMap {
		scores = append(scores, kv{k, math.Round(v*100) / 100}) // Arrotonda a 2 cifre decimali
	}

	// Ordina in modo decrescente
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].Value > scores[j].Value
	})

	// Costruisce la stringa JSON manualmente per bloccare il riordino alfabetico
	jsonStr := "{"
	for i, s := range scores {
		jsonStr += fmt.Sprintf(`"%s": %.2f`, s.Key, s.Value)
		if i < len(scores)-1 {
			jsonStr += ", "
		}
	}
	jsonStr += "}"

	return json.RawMessage(jsonStr)
}

// PADManager ora gestisce SIA l'analisi PAD che l'analisi Sentiment (Emotion)
type PADManager struct {
	audioMu          sync.Mutex
	audioBuf         []float32
	samplesSinceLast int
	isSpeaking       bool

	stateMu        sync.Mutex
	padResults     []map[string]float64
	emotionResults []map[string]float64

	wg sync.WaitGroup

	stepCounter int
	history     []AcousticData // Taccuino invisibile che tiene in memoria i due modelli

	padOnnxPath   string
	padConfigPath string
	padConfig     ModelConfig
	padSession    *ort.DynamicAdvancedSession

	emoOnnxPath   string
	emoConfigPath string
	emoConfig     ModelConfig
	emoSession    *ort.DynamicAdvancedSession

	options  *ort.SessionOptions
	initOnce sync.Once
	isReady  bool
}

func NewPADManager(padOnnx, padConf, emoOnnx, emoConf string) *PADManager {
	// Carica Config PAD
	filePad, err := os.ReadFile(padConf)
	if err != nil { log.Printf("[AudioAnalysis] Errore lettura config PAD: %v", err); return nil }
	var pConf ModelConfig
	json.Unmarshal(filePad, &pConf)

	// Carica Config Emotion
	fileEmo, err := os.ReadFile(emoConf)
	if err != nil { log.Printf("[AudioAnalysis] Errore lettura config Emotion: %v", err); return nil }
	var eConf ModelConfig
	json.Unmarshal(fileEmo, &eConf)

	return &PADManager{
		padOnnxPath:    padOnnx,
		padConfigPath:  padConf,
		padConfig:      pConf,
		emoOnnxPath:    emoOnnx,
		emoConfigPath:  emoConf,
		emoConfig:      eConf,
		isSpeaking:     false,
		padResults:     make([]map[string]float64, 0),
		emotionResults: make([]map[string]float64, 0),
		stepCounter:    0,
		history:        make([]AcousticData, 0),
	}
}

func (m *PADManager) InitSession() {
	m.initOnce.Do(func() {
		opts, err := ort.NewSessionOptions()
		if err != nil {
			log.Printf("[AudioAnalysis] Errore opzioni ONNX: %v", err)
			return
		}

		// Avvia Sessione PAD
		pSession, err := ort.NewDynamicAdvancedSession(
			m.padOnnxPath,
			[]string{"input_values", "attention_mask"},
			[]string{"logits"},
			opts,
		)
		if err != nil { log.Printf("[AudioAnalysis] Errore avvio PAD: %v", err); return }
		m.padSession = pSession

		// Avvia Sessione Emotion
		eSession, err := ort.NewDynamicAdvancedSession(
			m.emoOnnxPath,
			[]string{"input_values", "attention_mask"},
			[]string{"logits"},
			opts,
		)
		if err != nil { log.Printf("[AudioAnalysis] Errore avvio Emotion: %v", err); return }
		m.emoSession = eSession

		m.options = opts
		m.isReady = true
		log.Println("[AudioAnalysis] 🚀 Doppio Motore ONNX (PAD + Sentiment) Inizializzato!")
	})
}

func (m *PADManager) SetSpeaking(speaking bool) {
	m.audioMu.Lock()
	defer m.audioMu.Unlock()
	m.isSpeaking = speaking
}

func (m *PADManager) ClearState() {
	m.audioMu.Lock()
	m.audioBuf = nil
	m.samplesSinceLast = 0
	m.audioMu.Unlock()

	m.stateMu.Lock()
	m.padResults = make([]map[string]float64, 0)
	m.emotionResults = make([]map[string]float64, 0)
	m.stateMu.Unlock()
}

func (m *PADManager) ProcessAudioChunk(pcm []float32) {
	if !m.isReady { return }

	m.audioMu.Lock()
	m.audioBuf = append(m.audioBuf, pcm...)
	m.samplesSinceLast += len(pcm)

	if len(m.audioBuf) > 48000 {
		m.audioBuf = m.audioBuf[len(m.audioBuf)-48000:]
	}

	shouldRun := false
	var chunk []float32

	if m.isSpeaking && m.samplesSinceLast >= 48000 && len(m.audioBuf) == 48000 {
		shouldRun = true
		m.samplesSinceLast = 0
		chunk = make([]float32, len(m.audioBuf))
		copy(chunk, m.audioBuf)
	}
	m.audioMu.Unlock()

	if shouldRun {
		m.wg.Add(1)
		go m.runInference(chunk)
	}
}

func (m *PADManager) FlushAudio() {
	if !m.isReady { return }

	m.audioMu.Lock()
	chunk := make([]float32, len(m.audioBuf))
	copy(chunk, m.audioBuf)
	m.audioBuf = nil
	m.samplesSinceLast = 0
	m.audioMu.Unlock()

	if len(chunk) >= 8000 {
		m.wg.Add(1)
		go m.runInference(chunk)
	}
}

// Funzione di utilità per trasformare i logits di Emotion in probabilità (0.0 - 1.0)
func softmax(logits []float32) []float64 {
	if len(logits) == 0 { return nil }
	max := logits[0]
	for _, v := range logits {
		if v > max { max = v }
	}
	sum := 0.0
	res := make([]float64, len(logits))
	for i, v := range logits {
		res[i] = math.Exp(float64(v - max))
		sum += res[i]
	}
	for i := range res {
		res[i] /= sum
	}
	return res
}

func (m *PADManager) runInference(audioFloat []float32) {
	defer func() {
		if r := recover(); r != nil {
			log.Printf("[AudioAnalysis] ❌ PANIC recuperato in runInference: %v", r)
		}
		m.wg.Done()
	}()

	// --- Normalizzazione Audio Singola ---
	var sum float64
	for _, val := range audioFloat { sum += float64(val) }
	mean := sum / float64(len(audioFloat))

	var variance float64
	for _, val := range audioFloat {
		diff := float64(val) - mean
		variance += diff * diff
	}
	variance /= float64(len(audioFloat))
	stdDev := float32(math.Sqrt(variance + 1e-7))

	for i := range audioFloat {
		audioFloat[i] = (audioFloat[i] - float32(mean)) / stdDev
	}

	// --- Tensori Condivisi ---
	inputShape := ort.NewShape(1, int64(len(audioFloat)))
	inputTensor, err := ort.NewTensor(inputShape, audioFloat)
	if err != nil { return }
	defer inputTensor.Destroy()

	maskData := make([]int64, len(audioFloat))
	for i := range maskData { maskData[i] = 1 }
	maskShape := ort.NewShape(1, int64(len(maskData)))
	maskTensor, err := ort.NewTensor(maskShape, maskData)
	if err != nil { return }
	defer maskTensor.Destroy()

	// --- 1. Esecuzione PAD ---
	outShapePAD := ort.NewShape(1, 3)
	outTensorPAD, err := ort.NewEmptyTensor[float32](outShapePAD)
	if err == nil {
		defer outTensorPAD.Destroy()
		if err := m.padSession.Run([]ort.ArbitraryTensor{inputTensor, maskTensor}, []ort.ArbitraryTensor{outTensorPAD}); err == nil {
			valoriPAD := outTensorPAD.GetData()
			risultatoPAD := make(map[string]float64)
			for i, val := range valoriPAD {
				label := m.padConfig.Id2Label[strconv.Itoa(i)]
				risultatoPAD[label] = float64(val)
			}
			m.stateMu.Lock()
			m.padResults = append(m.padResults, risultatoPAD)
			m.stateMu.Unlock()
		}
	}

	// --- 2. Esecuzione Emotion ---
	outShapeEmo := ort.NewShape(1, 7) // 7 Emozioni
	outTensorEmo, err := ort.NewEmptyTensor[float32](outShapeEmo)
	if err == nil {
		defer outTensorEmo.Destroy()
		if err := m.emoSession.Run([]ort.ArbitraryTensor{inputTensor, maskTensor}, []ort.ArbitraryTensor{outTensorEmo}); err == nil {
			valoriEmo := outTensorEmo.GetData()
			probEmo := softmax(valoriEmo) // Trasforma i logit in probabilità
			risultatoEmo := make(map[string]float64)
			for i, val := range probEmo {
				label := m.emoConfig.Id2Label[strconv.Itoa(i)]
				risultatoEmo[label] = val
			}
			m.stateMu.Lock()
			m.emotionResults = append(m.emotionResults, risultatoEmo)
			m.stateMu.Unlock()
		}
	}

	log.Printf("🎛️ [AudioAnalysis] Blocco 3s analizzato (PAD + Emotion)")
}

func (m *PADManager) GetAndClearCurrentState() (string, string) {
	log.Println("[AudioAnalysis] ⏳ Attendo la fine dei calcoli ONNX...")
	m.wg.Wait()
	log.Println("[AudioAnalysis] ✅ Tutti i calcoli terminati. Genero le medie.")

	m.stateMu.Lock()
	defer m.stateMu.Unlock()

	// Se non c'è audio, restituiamo due stringhe vuote
	if len(m.padResults) == 0 { return "", "" }

	// --- 1. Medie Attuali ---
	padAttuale := make(map[string]float64)
	for _, res := range m.padResults {
		for k, v := range res { padAttuale[k] += v }
	}
	numPAD := float64(len(m.padResults))
	for k := range padAttuale { padAttuale[k] = math.Round((padAttuale[k]/numPAD)*10000) / 10000 }

	emoAttuale := make(map[string]float64)
	for _, res := range m.emotionResults {
		for k, v := range res { emoAttuale[k] += v }
	}
	numEmo := float64(len(m.emotionResults))
	for k := range emoAttuale { emoAttuale[k] = math.Round((emoAttuale[k]/numEmo)*10000) / 10000 }

	// --- 2. SALVATAGGIO IMMEDIATO (Il fix è qui!) ---
	// Salviamo subito il dato attuale nello storico prima di calcolare la media
	m.stepCounter++
	m.history = append(m.history, AcousticData{PAD: padAttuale, Emotion: emoAttuale})

	// --- 3. Medie Storiche Cumulabili ---
	// Ora lo storico calcola la media su tutti i messaggi, incluso quello appena detto
	padStorico := make(map[string]float64)
	emoStorico := make(map[string]float64)

	for _, oldRes := range m.history {
		for k, v := range oldRes.PAD { padStorico[k] += v }
		for k, v := range oldRes.Emotion { emoStorico[k] += v }
	}
	numStorico := float64(len(m.history)) // Ora sarà sempre almeno 1
	for k := range padStorico { padStorico[k] = math.Round((padStorico[k]/numStorico)*10000) / 10000 }
	for k := range emoStorico { emoStorico[k] = math.Round((emoStorico[k]/numStorico)*10000) / 10000 }

	// --- 4. Creazione JSON ---
	// attualeData := map[string]interface{}{
	// 	"pad":      padAttuale,
	// 	"emozioni": buildOrderedEmotionJSON(emoAttuale),
	// }

	delete(padAttuale, "valence") 
	delete(padStorico, "valence")

	attualeData := map[string]interface{}{
		"pad":      padAttuale,
		"emozioni": buildOrderedEmotionJSON(emoAttuale),
	}

	storicoData := map[string]interface{}{
		"step_corrente": m.stepCounter,
		"pad":           padStorico,
		"emozioni":      buildOrderedEmotionJSON(emoStorico),
	}

	// Formattazione JSON
	jsonAttuale, _ := json.MarshalIndent(attualeData, "", "    ")
	jsonStorico, _ := json.MarshalIndent(storicoData, "", "    ")

	m.padResults = make([]map[string]float64, 0)
	m.emotionResults = make([]map[string]float64, 0)
	
	log.Printf("[AudioAnalysis] 📊 Dati estratti (Step %d).", m.stepCounter)
	
	return string(jsonAttuale), string(jsonStorico)
}

// --- ELEMENTI PIPELINE (INVARIATI) ---

func bytesToFloat32(data []byte) []float32 {
	numSamples := len(data) / 2
	floats := make([]float32, numSamples)
	for i := 0; i < numSamples; i++ {
		s := int16(data[i*2]) | int16(data[i*2+1])<<8
		floats[i] = float32(s) / 32768.0
	}
	return floats
}

type PADCollectorElement struct {
	*pipeline.BaseElement
	manager *PADManager
}

func NewPADCollectorElement(m *PADManager) *PADCollectorElement {
	return &PADCollectorElement{
		BaseElement: pipeline.NewBaseElement("pad-collector", 100),
		manager:     m,
	}
}

func (e *PADCollectorElement) Start(ctx context.Context) error {
	startChan := make(chan pipeline.Event, 10)
	stopChan := make(chan pipeline.Event, 10)

	bus := e.BaseElement.Bus()
	if bus != nil {
		log.Println("[AudioAnalysis] ✅ Bus connesso, iscrizione eventi VAD...")
		bus.Subscribe(pipeline.EventVADSpeechStart, startChan)
		bus.Subscribe(pipeline.EventVADSpeechEnd, stopChan)
	}

	go func() {
		defer close(e.OutChan)
		for {
			select {
			case <-ctx.Done(): return
			case <-startChan:
				log.Println("[AudioAnalysis] 🟢 Rubinetto Aperto")
				e.manager.ClearState()
				e.manager.SetSpeaking(true)
			case <-stopChan:
				log.Println("[AudioAnalysis] 🔴 Rubinetto Chiuso (Flush in corso...)")
				e.manager.SetSpeaking(false)
				e.manager.FlushAudio()
			case msg, ok := <-e.InChan:
				if !ok { return }
				
				if msg.Type == pipeline.MsgTypeAudio && msg.AudioData != nil {
					floats := bytesToFloat32(msg.AudioData.Data)
					e.manager.ProcessAudioChunk(floats)
				}
				
				select {
				case e.OutChan <- msg:
				case <-ctx.Done(): return
				}
			}
		}
	}()
	return nil
}

type PADEnricherElement struct {
	*pipeline.BaseElement
	manager *PADManager
}

func NewPADEnricherElement(m *PADManager) *PADEnricherElement {
	return &PADEnricherElement{
		BaseElement: pipeline.NewBaseElement("pad-enricher", 100),
		manager:     m,
	}
}

func (e *PADEnricherElement) Start(ctx context.Context) error {
	go func() {
		defer close(e.OutChan)
		for {
			select {
			case <-ctx.Done(): return
			case msg, ok := <-e.InChan:
				if !ok { return }

				if msg.TextData != nil && len(msg.TextData.Data) > 0 {
					// Ora riceviamo le due stringhe separate
					jsonAttuale, jsonStorico := e.manager.GetAndClearCurrentState()
					
					if jsonAttuale != "" && jsonStorico != "" {
						originalText := string(msg.TextData.Data)
						
						// Costruiamo il prompt completo direttamente nell'Enricher
						nuovoTesto := fmt.Sprintf("User message: %s\n\n[Informazioni Audio Attuale]:\n%s\n\n[Media Storica dei Valori]:\n%s", originalText, jsonAttuale, jsonStorico)
						msg.TextData.Data = []byte(nuovoTesto)
						log.Printf("[AudioAnalysis] Inserito Profilo Psicologico diviso nel prompt!")
					}
				}

				select {
				case e.OutChan <- msg:
				case <-ctx.Done(): return
				}
			}
		}
	}()
	return nil
}

func (m *PADManager) ResetFullHistory() {
	m.stateMu.Lock()
	defer m.stateMu.Unlock()
	m.stepCounter = 0
	m.history = make([]AcousticData, 0)
	log.Println("[AudioAnalysis] 🧹 Storico globale azzerato per nuova sessione.")
}

// Stop viene chiamato automaticamente quando si chiude la pagina web
func (e *PADEnricherElement) Stop() error {
	e.manager.ResetFullHistory()
	return nil
}