# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Exp 1a: per-statement DPO, lr=1e-7, 140 steps (4 epochs)."""
import os
from experiments.posttrain.per_stmt_dpo.common import run_exp1a

if __name__ == "__main__":
    run_exp1a(lr=1e-7, steps=140, tpu=os.environ.get("TPU_TYPE", "v6e-8"))
