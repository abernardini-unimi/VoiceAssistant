############################
# BUILD STAGE
############################
FROM golang:1.23-bookworm AS builder

RUN apt-get update && apt-get install -y \
        wget \
        pkg-config \
        xz-utils \
        libopusfile-dev \
        libopus-dev \
        libavcodec-dev \
        libavformat-dev \
        libavdevice-dev \
        libavutil-dev \
        libswresample-dev \
        ca-certificates \
        git \
        tzdata \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root

# Setup FFmpeg custom
COPY setup-ffmpeg.sh /root/
RUN chmod +x setup-ffmpeg.sh && ./setup-ffmpeg.sh

# Scarica ONNX Runtime manualmente (come da README)
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz \
    && tar -xzf onnxruntime-linux-x64-1.20.1.tgz \
    && cp onnxruntime-linux-x64-1.20.1/lib/libonnxruntime.so* /usr/local/lib/ \
    && cp -r onnxruntime-linux-x64-1.20.1/include/* /usr/local/include/ \
    && ldconfig

WORKDIR /go/src/app

# Copia tutto il sorgente (incluse sottocartelle custom_elements/ ecc.)
COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Build con tutte le variabili necessarie
RUN export FFMPEG_DIR=/root/ffmpeg && \
    export PKG_CONFIG_PATH="$FFMPEG_DIR/lib/pkgconfig:$PKG_CONFIG_PATH" && \
    export CGO_CFLAGS="-I$FFMPEG_DIR/include -I/usr/local/include" && \
    export CGO_LDFLAGS="-L$FFMPEG_DIR/lib -L/usr/local/lib" && \
    export LD_LIBRARY_PATH="$FFMPEG_DIR/lib:/usr/local/lib:$LD_LIBRARY_PATH" && \
    export PATH="$FFMPEG_DIR/bin:$PATH" && \
    CGO_ENABLED=1 go build -o /app-binary .

############################
# RUNTIME STAGE
############################
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
        libopus0 \
        libopusfile0 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia il binario
COPY --from=builder /app-binary ./app

# Copia le librerie dinamiche necessarie
COPY --from=builder /root/ffmpeg/lib/ /usr/local/lib/ffmpeg/
COPY --from=builder /usr/local/lib/libonnxruntime.so* /usr/local/lib/
RUN ldconfig

# Copia assets necessari a runtime
COPY static/ ./static/
COPY models/ ./models/
COPY prompt/ ./prompt/
COPY conversation/ ./conversation/

# Porta configurabile (Fly.io / Railway usano $PORT o 8080)
ENV VOICE_ASSISTANT_PORT=:8080
EXPOSE 8080

# Imposta LD_LIBRARY_PATH per trovare FFmpeg e ONNX a runtime
ENV LD_LIBRARY_PATH=/usr/local/lib/ffmpeg:/usr/local/lib

CMD ["./app"]