#!/usr/bin/env bash
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

if [ "$(uname -m)" = "aarch64" ]; then
  export LD_PRELOAD="$IRIS_VENV/mimalloc-pprof/lib/libmimalloc.so${LD_PRELOAD:+:$LD_PRELOAD}"
  export MIMALLOC_SHOW_STATS=0
  export PYTHONPATH="$IRIS_WORKDIR/scripts/mimalloc_pprof${PYTHONPATH:+:$PYTHONPATH}"
  unset MALLOC_CONF
fi
