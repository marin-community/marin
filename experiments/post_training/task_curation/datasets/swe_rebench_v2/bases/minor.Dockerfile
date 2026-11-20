# SWE-rebench V2.1 repair — minor-polyglot group base.
# Covers c (15) / cpp (2, declared base 'c') / csharp_base (22) / dart_base (69+... see census) /
# r_base (69) / ocaml_base (11) / lua_base (16) rows (224 tasks).
# Upstream anchors, unioned:
#   c: ubuntu:22.04 + C/C++ build deps (Dockerfile_c, verbatim layer below)
#   csharp: mcr.microsoft.com/dotnet/sdk:8.0 (dotnet via the Microsoft jammy repo)
#   dart: dart:stable-sdk (dart via the dart repo, stable channel)
#   r: rocker/r-ver:4.4.1 (R 4.4.1 via the CRAN jammy repo)
#   ocaml: debian:bookworm + opam (Dockerfile_ocaml layer)
#   lua: nickblah/lua:5.4-bookworm (lua 5.4 built from source, upstream recipe pattern)
# Interpreter provisioning (repair, disclosed): apt python3 (jammy ships 3.10).
# Designated fidelity-gate pressure valve: if this group splits, c/cpp stay here and the
# remaining languages move to authored per-language bases folded with the php/elixir/swift/julia
# misc merge (snapshot cap is 10).

# R 4.4.1 from the upstream declared image (rocker/r-ver:4.4.1, jammy-based) — CRAN's apt
# repo no longer serves 4.4.x, so the repair copies the pinned R tree instead.
FROM rocker/r-ver:4.4.1@sha256:78cb94ce2db23aaaf7b546450fcf70b5a3f2ace5a9b5fa1f87217da329211312 AS r

FROM ubuntu:22.04@sha256:9abcaffc92a5279fff3a6a41635c0800ea33e528425fb389658806476e7defde

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Disclosed repair normalization: apt runs non-interactively with --force-confdef/--force-confold
# (keeps the image's existing conffiles; an upgraded media-types prompted on mime.types and
# aborted non-interactive builds).

# C/C++ build deps (upstream Dockerfile_c layer, verbatim package set).
RUN sed -i 's/^# deb-src/deb-src/' /etc/apt/sources.list \
    && apt-get update && if [ -f /var/lib/dpkg/statoverride ]; then while read -r mode owner group path; do [ -n "$path" ] || continue; if ! getent passwd "$owner" >/dev/null 2>&1 && ! getent group "$group" >/dev/null 2>&1; then dpkg-statoverride --remove "$path" || true; fi; done < /var/lib/dpkg/statoverride; fi && apt-get -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold install -y --no-install-recommends \
    wget git build-essential libtool automake autoconf tcl bison flex cmake \
    python3 python3-pip python3-venv python-is-python3 libffi-dev pkg-config \
    curl gnupg ca-certificates unzip zip software-properties-common \
    zlib1g-dev libpcre3-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN test "$(python3 -c 'import sys; print("%d%02d" % sys.version_info[:2])')" -ge 309 \
    || { echo "python3 >= 3.9 required on PATH" >&2; exit 1; }

# libclang 19 (the dart ffigen rows link libclang at setup; the upstream dart image ships
# libclang-19.1.7.so — read from swerebenchv2/dart-lang-ffigen@75329b59 — and apt.llvm.org's
# jammy llvm-toolchain-jammy-19 repo serves the same 19.1.7 point release; URL+install
# validated locally in a jammy container).
RUN set -eux; \
    apt-get update && if [ -f /var/lib/dpkg/statoverride ]; then while read -r mode owner group path; do [ -n "$path" ] || continue; if ! getent passwd "$owner" >/dev/null 2>&1 && ! getent group "$group" >/dev/null 2>&1; then dpkg-statoverride --remove "$path" || true; fi; done < /var/lib/dpkg/statoverride; fi; \
    apt-get install -y --no-install-recommends gnupg curl ca-certificates; \
    curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor -o /usr/share/keyrings/llvm-archive-keyring.gpg; \
    echo 'deb [signed-by=/usr/share/keyrings/llvm-archive-keyring.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-19 main' > /etc/apt/sources.list.d/llvm19.list; \
    apt-get update && apt-get install -y --no-install-recommends libclang-19-dev; \
    rm -rf /var/lib/apt/lists/*; \
    test -e /usr/lib/llvm-19/lib/libclang-19.so || { echo "libclang-19 missing" >&2; exit 1; }

# dotnet SDK 8.0 (upstream csharp base image).
RUN set -eux; \
    curl -fsSL https://packages.microsoft.com/config/ubuntu/22.04/packages-microsoft-prod.deb -o /tmp/ms.deb; \
    dpkg -i /tmp/ms.deb; rm /tmp/ms.deb; \
    apt-get update && if [ -f /var/lib/dpkg/statoverride ]; then while read -r mode owner group path; do [ -n "$path" ] || continue; if ! getent passwd "$owner" >/dev/null 2>&1 && ! getent group "$group" >/dev/null 2>&1; then dpkg-statoverride --remove "$path" || true; fi; done < /var/lib/dpkg/statoverride; fi && apt-get -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold install -y --no-install-recommends dotnet-sdk-8.0; \
    rm -rf /var/lib/apt/lists/*
RUN dotnet --version | grep -q '^8\.' || { echo "dotnet 8 required" >&2; exit 1; }

# Dart stable SDK (upstream dart base channel).
RUN set -eux; \
    wget -qO- https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -; \
    echo "deb https://storage.googleapis.com/download.dartlang.org/linux/debian stable main" \
      > /etc/apt/sources.list.d/dart_stable.list; \
    apt-get update && if [ -f /var/lib/dpkg/statoverride ]; then while read -r mode owner group path; do [ -n "$path" ] || continue; if ! getent passwd "$owner" >/dev/null 2>&1 && ! getent group "$group" >/dev/null 2>&1; then dpkg-statoverride --remove "$path" || true; fi; done < /var/lib/dpkg/statoverride; fi && apt-get -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold install -y --no-install-recommends dart; \
    rm -rf /var/lib/apt/lists/*
ENV PUB_CACHE=/workspace/.pub-cache \
    CI=true
RUN dart --version || { echo "dart required" >&2; exit 1; }

# R 4.4.1: the pinned tree from the upstream declared image (first stage above; R lives under
# /usr/local/lib/R with the launcher scripts in /usr/local/bin), plus the runtime libraries
# its binaries link against.
COPY --from=r /usr/local/lib/R /usr/local/lib/R
COPY --from=r /usr/local/bin/R /usr/local/bin/R
COPY --from=r /usr/local/bin/Rscript /usr/local/bin/Rscript
RUN set -eux; \
    apt-get update && if [ -f /var/lib/dpkg/statoverride ]; then while read -r mode owner group path; do [ -n "$path" ] || continue; if ! getent passwd "$owner" >/dev/null 2>&1 && ! getent group "$group" >/dev/null 2>&1; then dpkg-statoverride --remove "$path" || true; fi; done < /var/lib/dpkg/statoverride; fi && apt-get -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold install -y --no-install-recommends \
      gfortran libssl-dev libcurl4-openssl-dev libxml2-dev libicu-dev libbz2-dev liblzma-dev \
      libpcre2-dev libreadline-dev libblas-dev liblapack-dev libdeflate0 libopenblas0-pthread make; \
    rm -rf /var/lib/apt/lists/*
RUN R --version | head -1 | grep -q '4.4' || { echo "R 4.4 required" >&2; exit 1; }

# OCaml + opam (upstream ocaml recipe layer; bookworm's apt compiler, opam for project envs).
RUN apt-get update && if [ -f /var/lib/dpkg/statoverride ]; then while read -r mode owner group path; do [ -n "$path" ] || continue; if ! getent passwd "$owner" >/dev/null 2>&1 && ! getent group "$group" >/dev/null 2>&1; then dpkg-statoverride --remove "$path" || true; fi; done < /var/lib/dpkg/statoverride; fi && apt-get -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold install -y --no-install-recommends \
    m4 opam libgmp-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*
# opam root initialized BARE (config + repos, no compiler switch): the ocaml rows create
# their own switches at setup (`opam switch create yaml-switch 4.13.0` — the compilers build
# from source with the base's gcc — 10-20 min, within the 1800 s setup timeout). The earlier
# `opam init -a` auto-setup tried to build an ocaml-compiler switch and failed (no ocaml
# bootstrap in the base) — verified in-region (bases-v16).
RUN opam init --disable-shell-hook --bare --root /opt/opam \
    && test -f /opt/opam/config \
    && opam --version | grep -q . \
    || { echo "opam root init failed" >&2; exit 1; }
ENV OPAMROOT="/opt/opam"

# Lua 5.4 built from source (upstream lua base version).
RUN set -eux; \
    curl -fsSL https://www.lua.org/ftp/lua-5.4.6.tar.gz -o /tmp/lua.tar.gz; \
    tar -xzf /tmp/lua.tar.gz -C /tmp; \
    make -C /tmp/lua-5.4.6 linux MYCFLAGS="-fPIC" >/dev/null; \
    make -C /tmp/lua-5.4.6 INSTALL_TOP=/opt/lua-5.4 install; \
    ln -sfn /opt/lua-5.4/bin/lua /usr/local/bin/lua; \
    ln -sfn /opt/lua-5.4/bin/luac /usr/local/bin/luac; \
    rm -rf /tmp/lua.tar.gz /tmp/lua-5.4.6
RUN lua -v | grep -q '5.4' || { echo "lua 5.4 required" >&2; exit 1; }

# Luarocks (upstream lua base carries luarocks for rock installs).
RUN set -eux; \
    curl -fsSL https://luarocks.org/releases/luarocks-3.11.1.tar.gz -o /tmp/luarocks.tar.gz; \
    tar -xzf /tmp/luarocks.tar.gz -C /tmp; \
    cd /tmp/luarocks-3.11.1 \
    && ./configure --with-lua=/opt/lua-5.4 --prefix=/opt/luarocks >/dev/null \
    && make install >/dev/null \
    && ln -sfn /opt/luarocks/bin/luarocks /usr/local/bin/luarocks; \
    rm -rf /tmp/luarocks.tar.gz /tmp/luarocks-3.11.1 || true

RUN mkdir -p /logs /output /workspace && chmod 777 /logs /output /workspace
WORKDIR /workspace
