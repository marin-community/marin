# SWE-rebench V2.1 repair — elixir group base.
# Covers elixir_base rows (186 tasks).
# Upstream anchor: base_dockerfiles/Dockerfile_elixir (verbatim recipe below).
# Interpreter provisioning (repair, disclosed): apt python3 (the elixir image is debian-based).

FROM elixir:1.16@sha256:6d37f5a4a7c7412f66428445412c2cc5e38d5d3c6fb30fabbbba51fa0fab60bc

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Disclosed repair normalization: apt runs non-interactively with --force-confdef/--force-confold
# (keeps the image's existing conffiles; an upgraded media-types prompted on mime.types and
# aborted non-interactive builds).

RUN apt-get update && apt-get -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold install -y --no-install-recommends \
    git curl wget unzip zip ca-certificates build-essential pkg-config cmake gnupg \
    libssl-dev zlib1g-dev libgmp-dev libffi-dev \
    python3 \
    && rm -rf /var/lib/apt/lists/*

RUN test "$(python3 -c 'import sys; print("%d%02d" % sys.version_info[:2])')" -ge 309 \
    || { echo "python3 >= 3.9 required on PATH" >&2; exit 1; }

RUN elixir --version | grep -q '1.16' || { echo "elixir 1.16 required" >&2; exit 1; }

RUN mkdir -p /logs /output && chmod 777 /logs /output
