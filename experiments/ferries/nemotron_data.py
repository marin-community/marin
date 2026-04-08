# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run only the data-preparation steps upstream of the Nemotron canary ferry.

Tokenizes only the Nemotron CC ``hq_actual`` split using the kitoken backend.
"""

from levanter.tokenizers import TokenizerBackend

from experiments.pretraining_datasets.nemotron import tokenize_nemotron_subset
from marin.execution.executor import executor_main

S3_PREFIX = "s3://marin-na/marin/tmp/rav"


def main() -> None:
    step = tokenize_nemotron_subset(
        "hq_actual",
        tokenizer_backend=TokenizerBackend.KITOKEN,
    ).with_output_path(f"{S3_PREFIX}/tokenized/nemotron_cc/hq_actual")

    executor_main(steps=[step])


if __name__ == "__main__":
    main()
