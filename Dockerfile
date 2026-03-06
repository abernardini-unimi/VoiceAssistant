############################
# BUILD STAGE
############################
FROM golang AS builder

RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "10001" \
    "app" && \
    apt update && \
    apt install -y \
        wget \
        pkg-config \
        xz-utils \
        libopusfile-dev \
        libopus-dev \
        libavcodec-dev \
        libavformat-dev \
        libavdevice-dev \
        libavutil-dev \
        libswresample-dev && \
    apt install -y ca-certificates git tzdata upx && \
    update-ca-certificates

WORKDIR /root
COPY setup-ffmpeg.sh /root
RUN chmod +x setup-ffmpeg.sh && \
    ./setup-ffmpeg.sh && \
    apt-get update && apt-get install -y \
    libonnxruntime-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /go/src/app
COPY models/ ./models/
COPY conversation/ ./conversation/
COPY custom_elements/ ./custom_elements/
COPY prompt/ ./prompt/
COPY static/ ./static/
COPY onnxruntime-linux-x64-1.20.1/ ./onnxruntime-linux-x64-1.20.1
COPY *.go go.mod go.sum /go/src/app/
RUN export FFMPEG_DIR=/root/ffmpeg && \
    export PKG_CONFIG_PATH="$FFMPEG_DIR/lib/pkgconfig:$PKG_CONFIG_PATH" && \
    export LD_LIBRARY_PATH="$FFMPEG_DIR/lib:$LD_LIBRARY_PATH" && \
    export DYLD_LIBRARY_PATH="$FFMPEG_DIR/lib:$DYLD_LIBRARY_PATH" && \
    export CGO_CFLAGS="-I$FFMPEG_DIR/include" && \
    export CGO_LDFLAGS="-L$FFMPEG_DIR/lib" && \
    export PATH="$FFMPEG_DIR/bin:$PATH" && \
    go get && \
    LC_ALL=C go build -o /inx-voice-assistant-inxide

ENV ONNX_DIR=/go/src/app/onnxruntime-linux-x64-1.20.1/lib

# QUESTA È LA PARTE CHE RISOLVE IL PROBLEMA SUL CLOUD:
RUN echo "$ONNX_DIR" > /etc/ld.so.conf.d/onnxruntime.conf && \
    # Creiamo a mano il link se è andato perso nel caricamento
    ln -s $ONNX_DIR/libonnxruntime.so.1.20.1 $ONNX_DIR/libonnxruntime.so || true && \
    # Aggiorniamo il database di sistema
    ldconfig

# Manteniamo le variabili per sicurezza
ENV LD_LIBRARY_PATH=$ONNX_DIR:$LD_LIBRARY_PATH

CMD ["/bin/bash", "-c", "eval \"$(/root/setup-ffmpeg.sh --env)\" && /inx-voice-assistant-inxide"]