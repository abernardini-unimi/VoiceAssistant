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

Rename `.env-example` in `.env` and insert api_key:

```
OPENAI_API_KEY=''
GROQ_API_KEY=''
```

# Docker

```
docker build -t voice-assistant .

docker run -it --rm \
  -p 8080:8080 \
  -p 9001:9001/udp \
  -e OPENAI_API_KEY=sk-xxxxxxxxxxxx \
  -e GROQ_API_KEY=gsk-xxxxxxxxxxxx \
  voice-assistant
```
