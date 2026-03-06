# inx-voice-assistant-inxide
AI voice assistant that connects to the sip controller/connector.

# Fisrt Installation

```bash
go mod init inx-voice-assistant
go build
```

```bash
chmod +x setup-ffmpeg.sh
./setup-ffmpeg.sh

eval "$(./setup-ffmpeg.sh --env)"
sudo apt-get install pkg-config libopus-dev libavcodec-dev libavformat-dev libavutil-dev libswresample-dev
```

# System Dependencies

1. **ONNX Runtime v1.20.1** (or compatible version)
   ```bash
   wget https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz
   tar -xzf onnxruntime-linux-x64-1.20.1.tgz
   export ONNXRUNTIME_LIB=$(pwd)/onnxruntime-linux-x64-1.20.1/lib/libonnxruntime.so
   export LD_LIBRARY_PATH=$(pwd)/onnxruntime-linux-x64-1.20.1/lib:$LD_LIBRARY_PATH
   ```

2. **Silero VAD Model**
   ```bash
   mkdir -p models
   wget https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx -O models/silero_vad.onnx
   ```

# Usage

```bash
eval "$(./setup-ffmpeg.sh --env)"
export ONNXRUNTIME_LIB=$(pwd)/onnxruntime-linux-x64-1.20.1/lib/libonnxruntime.so
export LD_LIBRARY_PATH=$(pwd)/onnxruntime-linux-x64-1.20.1/lib:$LD_LIBRARY_PATH
```

```bash
go run main.go
```

# Configuration .env

Rename `.env-example` in `.env` and insert api_key.

# Docker

```
docker build -t voice-assistant .
docker run -p 8082:8082 -p 9001:9001/udp --env-file .env voice-assistant
```

---

# 🚀 Fly.io Cheat Sheet - EVA Voice Assistant

Questo documento contiene tutti i comandi utili di Fly.io per gestire, monitorare, potenziare e fare debug dell'applicazione.

## 📦 1. Deploy e Stato Base

I comandi principali per aggiornare e controllare l'applicazione.

* `fly deploy`
Compila il codice, crea il container Docker e pubblica la nuova versione online. *Da usare ogni volta che modifichi il codice (`main.go`, `index.html`, ecc).*
* `fly status`
Mostra se la macchina è accesa, spenta o in errore.
* `fly apps list`
Mostra tutte le app attive sul tuo account Fly.io.

---

## ⚙️ 2. Gestione Risorse (Risparmio e Potenza)

Comandi fondamentali per non sprecare soldi o per dare la massima potenza ai modelli ONNX.

* `fly scale count 0`
**Congela l'app**. Spegne i server istantaneamente e azzera i costi di calcolo (ideale a fine giornata).
* `fly scale count 1`
**Riaccende l'app** dopo averla congelata.
* `fly scale vm performance-8x`
Imposta un server potentissimo (8 CPU Dedicate, 32GB RAM) ideale per far girare modelli AI in tempo reale senza latenza.
* `fly scale vm shared-cpu-4x --memory 4096`
Imposta un server intermedio (4 CPU Condivise, 4GB RAM).
* `fly scale memory 2048`
Aumenta solo la RAM a 2GB (utile per evitare l'errore `OOM: Out Of Memory`).

---

## 🔍 3. Debug e Terminale

Strumenti per capire cosa succede "sotto il cofano".

* `fly logs`
Mostra i log in tempo reale del server (gli stessi che vedi sul terminale locale). Ottimo per monitorare errori o il flusso della conversazione.
* `fly ssh console`
Apre un terminale SSH sicuro direttamente **dentro** il server Linux che sta ospitando la tua app. Per uscire digita `exit`.

---

## 💾 4. Gestione File e Salvataggi (SFTP)

Siccome il disco di Fly.io è volatile, questi comandi ti servono per estrarre i dati raccolti (come i feedback o l'audio) prima di spegnere la macchina.

* `fly sftp get -r /go/src/app/feedback/ .`
Scarica l'intera cartella dei **feedback** dal server sul tuo computer locale (il `.` finale indica la cartella corrente del tuo PC).
* `fly sftp get -r /go/src/app/conversation/ .`
Scarica l'intera cartella delle **conversazioni salvate** sul tuo PC.

---

## 🌐 5. Rete e Variabili di Sicurezza

Gestione degli IP e dei segreti (API Key, IP Pubblici).

* `fly ips list`
Mostra gli indirizzi IP associati alla tua applicazione (fondamentale per trovare l'IP pubblico IPv4 per WebRTC).
* `fly secrets set NOME_VARIABILE="valore"`
Salva una variabile d'ambiente sicura e crittografata sul server (es: `fly secrets set FLY_PUBLIC_IP="77.x.x.x"`). Per le chiavi API è il metodo più sicuro.
* `fly secrets list`
Mostra i nomi di tutti i segreti salvati (i valori rimangono oscurati).

---

## 🗑️ 6. Danger Zone (Pulizia Definitiva)

Se vuoi fare pulizia nel tuo account.

* `fly apps destroy NOME_DELL_APP`
**Azione irreversibile**. Cancella definitivamente l'applicazione, libera l'indirizzo IP pubblico e rimuove ogni traccia del progetto dai server di Fly.io.