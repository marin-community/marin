#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# Install the pinned, checksum-verified runtime binaries a silo host needs.
#
# Runs on the DEFAULT Iris task image, which the Phase 0 spike showed can fetch
# these in seconds on cw-us-east-02a (egress is unrestricted there). That is why a
# v1 host needs no custom task image: code ships in the ordinary Iris bundle and
# only these three artifacts are fetched, each pinned and verified.
#
# Idempotent: a present, verified binary is not fetched again.
set -euo pipefail

SILO_HOME="${SILO_HOME:?SILO_HOME must be set}"
BIN="$SILO_HOME/bin"
mkdir -p "$BIN" "$SILO_HOME/dl"

NERDCTL_VERSION=2.1.2
NERDCTL_SHA256=b3ab8564c8fa6feb89d09bee881211b700b047373c767bec38256d0d68f93074
# Same gVisor pin Iris itself uses (lib/iris/.../gcp/worker_bootstrap.py).
RUNSC_VERSION=20260714.0
RUNSC_BASE="https://storage.googleapis.com/gvisor/releases/release/${RUNSC_VERSION}/x86_64"
BUSYBOX_URL=https://busybox.net/downloads/binaries/1.35.0-x86_64-linux-musl/busybox
BUSYBOX_SHA256=6e123e7f3202a8c1e9b1f94d8941580a25135382b99e8d3e34fb858bba311348

log() { printf '[silo-bootstrap] %s\n' "$*" >&2; }

# Every pin below is x86_64. On an arm64 node (cw-us-east-08a's GB200 pool) they
# would fail later with an opaque exec-format error; say what is wrong instead.
if [ "$(uname -m)" != "x86_64" ]; then
  log "unsupported architecture $(uname -m): silo hosts are x86_64-only (pinned amd64 binaries)"
  exit 2
fi

if [ ! -x "$BIN/containerd" ]; then
  log "fetching nerdctl-full ${NERDCTL_VERSION}"
  curl -fsSL --retry 3 -o "$SILO_HOME/dl/nerdctl-full.tgz" \
    "https://github.com/containerd/nerdctl/releases/download/v${NERDCTL_VERSION}/nerdctl-full-${NERDCTL_VERSION}-linux-amd64.tar.gz"
  echo "${NERDCTL_SHA256}  $SILO_HOME/dl/nerdctl-full.tgz" | sha256sum -c - >&2
  mkdir -p "$SILO_HOME/nerdctl"
  tar -C "$SILO_HOME/nerdctl" -xzf "$SILO_HOME/dl/nerdctl-full.tgz"
  cp "$SILO_HOME/nerdctl/bin/"* "$BIN/"
  # CNI plugins live under libexec in the full bundle; nerdctl looks for them there.
  mkdir -p "$SILO_HOME/cni"
  cp -r "$SILO_HOME/nerdctl/libexec/cni/"* "$SILO_HOME/cni/" 2>/dev/null || true
fi

if [ ! -x "$BIN/runsc" ]; then
  log "fetching runsc ${RUNSC_VERSION}"
  (
    cd "$BIN"
    curl -fsSL --retry 3 -O "$RUNSC_BASE/runsc"
    curl -fsSL --retry 3 -O "$RUNSC_BASE/runsc.sha512"
    sha512sum -c runsc.sha512 >&2
    chmod 0755 runsc
  )
fi

TOOLS="$SILO_HOME/tools"
if [ ! -x "$TOOLS/busybox" ]; then
  log "fetching static busybox"
  mkdir -p "$TOOLS"
  curl -fsSL --retry 3 -o "$TOOLS/busybox" "$BUSYBOX_URL"
  echo "${BUSYBOX_SHA256}  $TOOLS/busybox" | sha256sum -c - >&2
  chmod 0755 "$TOOLS/busybox"
fi

log "ready: $("$BIN/nerdctl" --version) / $("$BIN/runsc" --version | head -1)"
