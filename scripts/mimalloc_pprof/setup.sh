#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

: "${IRIS_VENV:?IRIS_VENV must identify the persistent task virtualenv}"
: "${IRIS_WORKDIR:?IRIS_WORKDIR must identify the persistent task workdir}"
if [ "$(uname -m)" != "aarch64" ]; then
  echo "mimalloc-pprof skipped: host architecture is not aarch64"
  exit 0
fi

mimalloc_pprof_commit=39a1a25b992f430efa87ae52d64c83bb8d9fbd50
mimalloc_pprof_sha256=fe3c935b70548025d5dbf7d4d018cc038aea699d3517baf873251e6b21810406
install_root="$IRIS_VENV/mimalloc-pprof"
build_prefix="$install_root/build-$mimalloc_pprof_commit"
build_jobs="${MIMALLOC_PPROF_BUILD_JOBS:-8}"

mkdir -p "$install_root/lib"
if [ ! -f "$build_prefix/lib/libmimalloc.so" ]; then
  build_root="$(mktemp -d "$IRIS_WORKDIR/.mimalloc-pprof-build.XXXXXX")"
  trap 'rm -rf "$build_root"' EXIT
  curl --fail --location --retry 5 \
    "https://github.com/zackees/mimalloc-pprof/archive/$mimalloc_pprof_commit.tar.gz" \
    --output "$build_root/mimalloc-pprof.tar.gz"
  printf '%s  %s\n' "$mimalloc_pprof_sha256" "$build_root/mimalloc-pprof.tar.gz" | sha256sum --check -
  mkdir -p "$build_root/source"
  tar -xzf "$build_root/mimalloc-pprof.tar.gz" -C "$build_root/source" --strip-components=1
  uvx --from cmake cmake \
    -S "$build_root/source" \
    -B "$build_root/build" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_C_FLAGS_RELWITHDEBINFO="-O2 -g -fno-omit-frame-pointer" \
    -DCMAKE_INSTALL_PREFIX="$build_prefix" \
    -DMI_BUILD_OBJECT=OFF \
    -DMI_BUILD_SHARED=ON \
    -DMI_BUILD_STATIC=OFF \
    -DMI_BUILD_TESTS=OFF \
    -DMI_EXTRA_CPPDEFS=MI_STAT=2 \
    -DMI_NO_OPT_ARCH=ON \
    -DMI_OVERRIDE=ON \
    -DMI_PPROF=ON
  uvx --from cmake cmake --build "$build_root/build" --parallel "$build_jobs"
  uvx --from cmake cmake --install "$build_root/build"
fi

ln -sfn "$build_prefix/lib/libmimalloc.so" "$install_root/lib/libmimalloc.so"
echo "mimalloc-pprof installed under $install_root"
