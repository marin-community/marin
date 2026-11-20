# SWE-rebench V2.1 repair — rust group base.
# Covers rust_1.77 / 1.78 / 1.79 / 1.81 / 1.83 / 1.84 rows (1,129 tasks).
# Upstream anchor: base_dockerfiles/Dockerfile_rust_* (FROM rust:<version> + common apt layer).
# The official rust images ship rustup with the tag's toolchain as default; the repair
# pre-installs the six declared toolchains side by side and the task setup pins the row's
# declared one with `rustup default` (offline).
# Interpreter provisioning (repair, disclosed): apt python3 (bookworm ships 3.11).

FROM rust:1.84@sha256:738ae99a3d75623f41e6882566b4ef37e38a9840244a47efd4a0ca22e9628b88

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Disclosed repair normalization: a stale dpkg statoverride entry (e.g. a group the image no
# longer ships, as observed for 'messagebus') aborts every dpkg unpack. Drop entries whose
# owner and group no longer resolve before any package install; a clean file is a no-op.
RUN apt-get update \
    && if [ -f /var/lib/dpkg/statoverride ]; then \
         while read -r mode owner group path; do \
           [ -n "$path" ] || continue; \
           if ! getent passwd "$owner" >/dev/null 2>&1 && ! getent group "$group" >/dev/null 2>&1; then \
             dpkg-statoverride --remove "$path" || true; \
           fi; \
         done < /var/lib/dpkg/statoverride; \
       fi \
    && apt-get install -y --no-install-recommends wget git build-essential python3 \
    && rm -rf /var/lib/apt/lists/*

# Declared interpreter provisioning guard.
RUN test "$(python3 -c 'import sys; print("%d%02d" % sys.version_info[:2])')" -ge 309 \
    || { echo "python3 >= 3.9 required on PATH" >&2; exit 1; }

# Per-version rustup toolchains (offline default switching at task setup).
RUN rustup toolchain install 1.77.0 1.78.0 1.79.0 1.81.0 1.83.0 1.84.0 \
    && rustup default 1.84.0
RUN rustup toolchain list | grep -q 1.77.0 \
    || { echo "rust toolchains missing" >&2; exit 1; }

RUN mkdir -p /logs /output && chmod 777 /logs /output
