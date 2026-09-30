#!/usr/bin/env bash
# Fused CE at the hero per-GPU shape: production tiles vs larger vocab tiles (GB200x4 fanout).
set -uo pipefail
export XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_memory_limit_slop_factor=85"
python lib/levanter/scripts/bench/bench_ce_hero_shape.py --fanout --num-gpus 4 --steps 10 \
  --variants fast-fwdfull,hero-bwdv8192,fast-max,fast-max32k,hero-v8192,hero-v8192-bwdv16384,hero-v16384,fast-fwdfull \
  --out /tmp/ce.json
