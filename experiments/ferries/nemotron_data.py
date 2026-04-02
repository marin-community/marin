# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run only the data-preparation steps upstream of the Nemotron canary ferry.

Tokenizes all Nemotron CC splits, the DCLM code/math components, and the
default validation sets — everything that NEMOTRON_MIX_WITH_DEFAULT_VALIDATION
depends on.
"""


from experiments.pretraining_datasets.nemotron import tokenize_nemotron
from marin.execution.executor import executor_main

S3_PREFIX = "s3://marin-na/marin/tmp/rav"


def main() -> None:
    nemotron_steps = [
        step.with_output_path(f"{S3_PREFIX}/{step.name}")
        for step in tokenize_nemotron(max_workers=50, cache_copy_max_workers=50).values()
    ]
    # dclm_steps = [
    #     step.with_output_path(f"{S3_PREFIX}/{step.name}")
    #     for step in [dclm_components_llama3["starcoderdata"], dclm_components_llama3["proofpile_2"]]
    # ]
    # validation_steps = [
    #     step.with_output_path(f"{S3_PREFIX}/{step.name}")
    #     for step in default_validation_sets(tokenizer=nemotron_mix_block_shuffle.tokenizer).values()
    # ]
    steps = nemotron_steps

    executor_main(steps=steps)


if __name__ == "__main__":
    main()
