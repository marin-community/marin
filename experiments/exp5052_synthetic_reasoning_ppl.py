# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Issue #5052: HF-backed synthetic reasoning PPL dev slices."""

from marin.execution.executor import executor_main

from experiments.evals.synthetic_reasoning_ppl import synthetic_reasoning_raw_validation_sets

RAW_SYNTHETIC_REASONING_VALIDATION_SETS = synthetic_reasoning_raw_validation_sets()


if __name__ == "__main__":
    executor_main(steps=[])
