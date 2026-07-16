# Ballista scheduler + executor + CLI, patched for gs:// (GCS) object-store support.
#
# The stock Ballista binaries register only s3/http/file object stores; this image
# builds them from a pinned upstream tag with a small patch that adds a `gs` arm to
# CustomObjectStoreRegistry (see ../ballista/gcs-support.patch and
# .agents/projects/smallquery/ballista-poc-phase0.md). GCS credentials resolve from
# the environment via Application Default Credentials, so on Iris GCP workers the
# binaries read gs:// with no key material.

ARG BALLISTA_REF=54.0.0

# ── builder ──────────────────────────────────────────────────────────
# Run builder natively on host architecture (fast compile, no QEMU emulation)
FROM --platform=$BUILDPLATFORM rust:1.88-bookworm AS builder

ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Install build dependencies + x86_64 cross-compiler if building for amd64 on non-amd64 host
RUN apt-get update && apt-get install -y --no-install-recommends \
    protobuf-compiler \
    cmake \
    gcc-x86-64-linux-gnu \
    g++-x86-64-linux-gnu \
    libc6-dev-amd64-cross \
    && rm -rf /var/lib/apt/lists/*

RUN rustup target add x86_64-unknown-linux-gnu

ARG BALLISTA_REF
WORKDIR /build
RUN git clone --depth 1 --branch "${BALLISTA_REF}" \
    https://github.com/apache/datafusion-ballista

COPY ballista/gcs-support.patch /build/gcs-support.patch
WORKDIR /build/datafusion-ballista
RUN git apply --verbose /build/gcs-support.patch
RUN rustup target add x86_64-unknown-linux-gnu

ENV CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=x86_64-linux-gnu-gcc \
    CC_x86_64_unknown_linux_gnu=x86_64-linux-gnu-gcc \
    CXX_x86_64_unknown_linux_gnu=x86_64-linux-gnu-g++


RUN cargo build --release --target x86_64-unknown-linux-gnu \
    -p ballista-scheduler --bin ballista-scheduler \
    -p ballista-executor  --bin ballista-executor \
    -p ballista-cli       --bin ballista-cli

# ── runtime ──────────────────────────────────────────────────────────
FROM --platform=linux/amd64 debian:bookworm-slim AS runtime

# bash: the Iris worker execs `bash /app/_run.sh`. ca-certificates: TLS to GCS.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/datafusion-ballista/target/x86_64-unknown-linux-gnu/release/ballista-scheduler /usr/local/bin/
COPY --from=builder /build/datafusion-ballista/target/x86_64-unknown-linux-gnu/release/ballista-executor  /usr/local/bin/
COPY --from=builder /build/datafusion-ballista/target/x86_64-unknown-linux-gnu/release/ballista-cli       /usr/local/bin/

LABEL org.opencontainers.image.source="https://github.com/marin-community/marin"

# Default scheduler gRPC port; overridden per-task by the launcher.
EXPOSE 50050
