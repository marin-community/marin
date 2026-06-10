# finelog server image — structured-log store + push/fetch RPC server.
#
# The server is the Rust `finelog-server` binary (lib/finelog/rust): it serves the
# Connect/gRPC RPCs and the dashboard SPA on a single port. Python is no longer
# in the runtime image.
#
# Build context must be the marin repo root:
#   docker build -f lib/finelog/deploy/Dockerfile -t finelog:dev .
#
# The server is unauthenticated by design — restrict at the network layer
# (k8s NetworkPolicy, GCP firewall). Do NOT expose it to the public internet.

# ── Stage: dashboard ─────────────────────────────────────────────────
# Build the Vue SPA. Output is consumed by the runtime stage at
# /app/dashboard/dist; the Rust server resolves and serves it from there (see
# `DOCKER_VUE_DIST_DIR` in lib/finelog/rust/src/server/spa.rs).
FROM node:22-slim AS dashboard

WORKDIR /build/dashboard

# Copy manifests first so dependency install is cached across source edits.
COPY lib/finelog/dashboard/package.json lib/finelog/dashboard/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci

COPY lib/finelog/dashboard/ ./
RUN npm run build


# ── Stage: rustbuild ─────────────────────────────────────────────────
# Compile the release `finelog-server` binary from finelog's own Rust
# workspace at lib/finelog/rust (no top-level rust/ tree). `connectrpc-build`
# fetches a vendored protoc from the crate registry at build time, so no
# system protoc is needed.
FROM rust:1-bookworm AS rustbuild

# Cargo profile to build. Defaults to the fat-LTO `release` profile for
# production images. Dev/test deploys pass `--build-arg CARGO_PROFILE=fast`
# (no LTO, parallel codegen) for a far faster final link; see `[profile.fast]`
# in lib/finelog/rust/Cargo.toml. The profile name is also the target/<profile>/ subdir
# (true for `release` and any custom profile), so the cp below resolves it.
ARG CARGO_PROFILE=release

# mold: a fast, multi-threaded linker. The cargo build runs under `mold -run`,
# which redirects the default linker to mold without touching RUSTFLAGS — so the
# cargo build fingerprint is unchanged and the cached dependency rlibs stay valid
# (no cache-busting full rebuild). Speeds the final binary link on both profiles;
# the win is largest with `fast` (no LTO), where linking dominates. Installed
# before the source COPY so the apt layer caches across source edits.
RUN apt-get update && apt-get install -y --no-install-recommends mold \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build/rust
COPY lib/finelog/rust/ ./
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/rust/target \
    mold -run cargo build --profile ${CARGO_PROFILE} -p finelog --bin finelog-server \
    && cp target/${CARGO_PROFILE}/finelog-server /finelog-server


# ── Stage: runtime ───────────────────────────────────────────────────
# `debian:bookworm-slim` matches the glibc the `rust:1-bookworm` build stage
# links against. curl/ca-certificates aid debugging; gdb supports in-prod
# profiling (the k8s pod adds SYS_PTRACE).
FROM debian:bookworm-slim AS runtime

LABEL org.opencontainers.image.source="https://github.com/marin-community/marin"
LABEL org.opencontainers.image.description="finelog log server image (Rust)"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    gdb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The server binary + the pre-built dashboard SPA. The Rust server reads the
# SPA from /app/dashboard/dist (see spa.rs); the binary is on PATH.
COPY --from=rustbuild /finelog-server /usr/local/bin/finelog-server
COPY --from=dashboard /build/dashboard/dist ./dashboard/dist

# Local segment cache / durable store. Declared as a volume so k8s/Docker mount
# a PV here. The recursive venv chown the Python image needed is gone; only the
# cache dir must be writable by the finelog user.
RUN groupadd --system --gid 1000 finelog \
    && useradd --system --uid 1000 --gid finelog --home-dir /app --no-create-home finelog \
    && mkdir -p /var/cache/finelog \
    && chown finelog:finelog /var/cache/finelog

VOLUME ["/var/cache/finelog"]

# The k8s manifest sets FINELOG_PORT + FINELOG_REMOTE_DIR but NOT
# FINELOG_LOG_DIR; the Python launcher defaulted port=10001 and
# log-dir=/var/cache/finelog in its CLI. The Rust clap CLI defaults port 8080
# and has NO log-dir default — an unset log-dir runs the store in-memory and
# would silently lose all data. Pin the durable path + port here so the image
# matches the prior deployment regardless of binary defaults.
ENV FINELOG_PORT=10001
ENV FINELOG_LOG_DIR=/var/cache/finelog

USER finelog

EXPOSE 10001

# All server flags read from FINELOG_* envvars (see lib/finelog/rust/src/main.rs).
# Defaults set above; FINELOG_REMOTE_DIR="" (no offload) unless overridden.
CMD ["finelog-server"]
