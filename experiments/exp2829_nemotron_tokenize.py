# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Nemotron CC: download raw data and tokenize all 7 quality splits.

Usage:
  # Run locally (dry run)
  uv run python experiments/exp2829_nemotron_tokenize.py --dry_run

  # Submit to Ray cluster
  uv run lib/marin/src/marin/run/ray_run.py -- python experiments/exp2829_nemotron_tokenize.py
"""

from experiments.pretraining_datasets.nemotron import tokenize_nemotron
from marin.execution.executor import executor_main

# download_step = downloads["nemotron_cc"]
tokenize_steps = tokenize_nemotron(max_workers=512)

if __name__ == "__main__":
    executor_main(
        steps=list(tokenize_steps.values()),
        description="Nemotron CC: download + tokenize all splits",
    )
