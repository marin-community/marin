# Ballista scheduler + executor + CLI, patched for gs:// (GCS) object-store support.
#
# The stock Ballista binaries register only s3/http/file object stores; this image
# builds them from a pinned upstream tag with a small patch that adds a `gs` arm to
# CustomObjectStoreRegistry (see ../ballista/gcs-support.patch and
# .agents/projects/smallquery/ballista-poc-phase0.md). GCS credentials resolve from
# the environment via Application Default Credentials, so on Iris GCP workers the
# binaries read gs:// with no key material.
#
# Build context is lib/smallquery/:
#   docker buildx build -t ghcr.io/marin-community/smallquery-ballista:<tag> \
#     -f lib/smallquery/docker/Dockerfile lib/smallquery

ARG BALLISTA_REF=54.0.0

# ── builder ──────────────────────────────────────────────────────────
FROM rust:1.88-bookworm AS builder

# protoc + cmake are required by Ballista's tonic/prost build and object_store's
# rustls/aws-lc-rs TLS stack.
RUN apt-get update && apt-get install -y --no-install-recommends \
    protobuf-compiler \
    cmake \
    && rm -rf /var/lib/apt/lists/*

ARG BALLISTA_REF
WORKDIR /build
RUN git clone --depth 1 --branch "${BALLISTA_REF}" \
    https://github.com/apache/datafusion-ballista

COPY ballista/gcs-support.patch /build/gcs-support.patch
WORKDIR /build/datafusion-ballista
RUN git apply --verbose /build/gcs-support.patch

# build-binary is a default feature, and it enables object_store with the patched
# gcp feature, so no --features flag is needed.
RUN cargo build --release \
    -p ballista-scheduler --bin ballista-scheduler \
    -p ballista-executor  --bin ballista-executor \
    -p ballista-cli       --bin ballista-cli

# ── runtime ──────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

# bash: the Iris worker execs `bash /app/_run.sh`. ca-certificates: TLS to GCS.
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/datafusion-ballista/target/release/ballista-scheduler /usr/local/bin/
COPY --from=builder /build/datafusion-ballista/target/release/ballista-executor  /usr/local/bin/
COPY --from=builder /build/datafusion-ballista/target/release/ballista-cli       /usr/local/bin/

# Default scheduler gRPC port; overridden per-task by the launcher.
EXPOSE 50050
