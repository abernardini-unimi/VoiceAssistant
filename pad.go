package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"

	"github.com/go-audio/wav"
	ort "github.com/yalue/onnxruntime_go"
)

// Config mappa la struttura di config.json per estrarre id2label
type Config struct {
	Id2Label map[string]string `json:"id2label"`
}

func main() {
	// --- Configurazione Percorsi ---
	percorsoConfig := "./models/pad/config.json"
	percorsoONNX := "./models/pad/model.onnx"
	fileAudio := "registrazione_29.wav"
	
	// Inserisci qui il percorso assoluto alla libreria estratta
	percorsoLibreriaONNX := os.Getenv("ONNXRUNTIME_LIB")
	if percorsoLibreriaONNX == "" {
		log.Fatalf("ERRORE: La variabile d'ambiente ONNXRUNTIME_LIB non è impostata. Esegui l'export prima di avviare.")
	}
	// Nomi dei nodi di input/output (standard per i modelli HF Wav2Vec2 ONNX)
	inputName := "input_values"
	outputName := "logits"

	// --- 1. Setup: Lettura configurazione JSON ---
	fileConfig, err := os.ReadFile(percorsoConfig)
	if err != nil {
		log.Fatalf("Errore lettura config.json: %v", err)
	}
	var config Config
	if err := json.Unmarshal(fileConfig, &config); err != nil {
		log.Fatalf("Errore parsing config.json: %v", err)
	}

	// --- 2. Audio: Lettura e Preprocessing del file WAV ---
	f, err := os.Open(fileAudio)
	if err != nil {
		log.Fatalf("Errore apertura file audio: %v", err)
	}
	defer f.Close()

	decoder := wav.NewDecoder(f)
	if !decoder.IsValidFile() {
		log.Fatalf("Il file audio non è un WAV valido")
	}

	formato := decoder.Format()
	if formato == nil {
		log.Fatalf("Errore: impossibile determinare il formato audio")
	}
	if formato.SampleRate != 16000 {
		log.Printf("ATTENZIONE: Il sample rate è %d. Il modello richiede 16000Hz.", formato.SampleRate)
	}

	buf, err := decoder.FullPCMBuffer()
	if err != nil {
		log.Fatalf("Errore lettura buffer PCM: %v", err)
	}

	// Conversione audio in float32
	audioFloat := make([]float32, len(buf.Data))
	for i, val := range buf.Data {
		audioFloat[i] = float32(val) / 32768.0 // Assumendo 16-bit PCM
	}

	// Normalizzazione (Simula AutoFeatureExtractor con do_normalize=true)
	var sum float64
	for _, val := range audioFloat {
		sum += float64(val)
	}
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

	// --- 3. ONNX: Setup ed Esecuzione ---
	ort.SetSharedLibraryPath(percorsoLibreriaONNX)

	err = ort.InitializeEnvironment()
	if err != nil {
		log.Fatalf("Errore inizializzazione ONNX Runtime: %v", err)
	}
	defer ort.DestroyEnvironment()

	// Creazione Tensore di Input: [1, sequence_length]
	inputShape := ort.NewShape(1, int64(len(audioFloat)))
	inputTensor, err := ort.NewTensor(inputShape, audioFloat)
	if err != nil {
		log.Fatalf("Errore creazione tensore input: %v", err)
	}
	defer inputTensor.Destroy()

	// Creazione Tensore di Output: [1, 3]
	outputShape := ort.NewShape(1, 3)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		log.Fatalf("Errore creazione tensore output: %v", err)
	}
	defer outputTensor.Destroy()

	// Creazione ed esecuzione della sessione
	session, err := ort.NewAdvancedSession(percorsoONNX,
		[]string{inputName}, []string{outputName},
		[]ort.ArbitraryTensor{inputTensor}, []ort.ArbitraryTensor{outputTensor}, nil)
	if err != nil {
		log.Fatalf("Errore creazione sessione ONNX: %v", err)
	}
	defer session.Destroy()

	err = session.Run()
	if err != nil {
		log.Fatalf("Errore esecuzione modello: %v", err)
	}

	// --- 4. Mappatura Dinamica e Output JSON ---
	valori := outputTensor.GetData()
	risultatoFinale := make(map[string]float64)

	for i, val := range valori {
		chiaveStr := strconv.Itoa(i)
		label, ok := config.Id2Label[chiaveStr]
		if !ok {
			label = fmt.Sprintf("classe_%d", i) // Fallback se la label non esiste
		}
		// Arrotonda a 4 decimali come in Python
		risultatoFinale[label] = math.Round(float64(val)*10000) / 10000
	}

	// Generazione JSON formattato correttamente con indentazione a 4 spazi
	jsonOutput, err := json.MarshalIndent(risultatoFinale, "", "    ")
	if err != nil {
		log.Fatalf("Errore generazione JSON: %v", err)
	}

	fmt.Println(string(jsonOutput))
}