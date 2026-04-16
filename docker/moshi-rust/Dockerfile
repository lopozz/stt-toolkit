FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    build-essential curl git pkg-config libssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app
RUN git clone https://github.com/kyutai-labs/moshi.git .
WORKDIR /app/rust

RUN cargo build --release --features cuda --bin moshi-backend

EXPOSE 8998
VOLUME ["/root/.cache/huggingface"]

# CMD ["/app/rust/target/release/moshi-backend", "--config", "/app/rust/moshi-backend/config-q8.json", "standalone"]