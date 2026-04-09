# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run only the data-preparation steps upstream of the Nemotron canary ferry.

Tokenizes all Nemotron CC splits using the kitoken backend.
"""

from levanter.tokenizers import TokenizerBackend

from experiments.pretraining_datasets.nemotron import tokenize_nemotron
from marin.execution.executor import executor_main

S3_PREFIX = "s3://marin-na/marin/tmp/rav"


def main() -> None:
    steps = [
        step.with_output_path(f"{S3_PREFIX}/{step.name}")
        for step in tokenize_nemotron(
            tokenizer_backend=TokenizerBackend.KITOKEN,
            max_workers=50,
            cache_copy_max_workers=50,
        ).values()
    ]

    executor_main(steps=steps)


if __name__ == "__main__":
    main()
