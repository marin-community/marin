# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Exp 1b: SFT base, 3 statements, lr=1e-6, 210 steps (2 epochs)."""
import os
from experiments.posttrain.per_stmt_dpo.common import run_exp1b

if __name__ == "__main__":
    run_exp1b(lr=1e-6, steps=210, tpu=os.environ.get("TPU_TYPE", "v6e-8"))
