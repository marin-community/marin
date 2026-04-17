# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Exp 2a: DPO base (continual), 1 statement, lr=1e-6, 70 steps (2 epochs)."""
import os
from experiments.posttrain.per_stmt_dpo.common import run_exp2a

if __name__ == "__main__":
    run_exp2a(lr=1e-6, steps=70, tpu=os.environ.get("TPU_TYPE", "v6e-8"))
