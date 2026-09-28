# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exec one bench_ragged_vs_pooled child with its variant env, streaming its output (debugging kills)."""

import os
import sys

from scratch_lc1.bench_ragged_vs_pooled import MODULE, VARIANTS, _child_env

variant = sys.argv[1]
env = _child_env(VARIANTS[variant][1])
env["NCCL_DEBUG"] = "WARN"
env["PYTHONUNBUFFERED"] = "1"
cmd = [sys.executable, "-u", "-m", MODULE, "--child", variant, "--iters", "10", "--warmup", "3", "--fused-relu2"]
print("exec:", " ".join(cmd), "XLA_FLAGS=", env.get("XLA_FLAGS"), flush=True)
os.execve(sys.executable, cmd, env)
