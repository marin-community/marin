# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the smallest Europe Stack cache with historical tokenizer semantics."""

from marin.execution.executor import executor_main

from experiments.domain_phase_mix.prepare_delphi_tpp40_europe_historical_stack_caches import (
    historical_stack_partition_step,
)

CANARY_PARTITION = "stack_edu/Ruby"


def main() -> None:
    executor_main(
        steps=[historical_stack_partition_step(CANARY_PARTITION)],
        description="Canary Europe Stack historical full-document tokenization",
    )


if __name__ == "__main__":
    main()
