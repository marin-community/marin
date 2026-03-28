# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Download the pinned Mixtral 8x7B checkpoints via Executor."""

from experiments.models import mixtral_8x7b, mixtral_8x7b_instruct
from marin.execution.executor import executor_main

if __name__ == "__main__":
    executor_main(
        steps=[mixtral_8x7b, mixtral_8x7b_instruct],
        description="Download Mixtral 8x7B base and instruct checkpoints",
    )
