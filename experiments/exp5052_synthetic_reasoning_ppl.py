# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Issue #5052: opt-in synthetic reasoning PPL dev slices."""

from experiments.evals.synthetic_reasoning_ppl import (
    synthetic_reasoning_ppl_raw,
    synthetic_reasoning_raw_validation_sets,
)
from marin.execution.executor import executor_main

RAW_SYNTHETIC_REASONING_PPL = synthetic_reasoning_ppl_raw
RAW_SYNTHETIC_REASONING_VALIDATION_SETS = synthetic_reasoning_raw_validation_sets(
    synthetic_reasoning_raw=RAW_SYNTHETIC_REASONING_PPL
)


if __name__ == "__main__":
    executor_main(steps=[RAW_SYNTHETIC_REASONING_PPL])
