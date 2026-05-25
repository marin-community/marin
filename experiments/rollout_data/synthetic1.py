# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PrimeIntellect/SYNTHETIC-1 pretraining dataset.

Usage:
    uv run iris --cluster=marin job run --cpu=1 --memory=2G --extra=cpu \
      -- python -m experiments.rollout_data.synthetic1
"""

from marin.datakit.download.synthetic1 import download_synthetic1_step
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize import TokenizeConfig, tokenize

from experiments.marin_models import marin_tokenizer


def build_steps() -> list[StepSpec]:
    processed = download_synthetic1_step()

    tokenized = StepSpec(
        name="tokenized/synthetic-1",
        deps=[processed],
        fn=lambda output_path: tokenize(
            TokenizeConfig(
                train_paths=[processed.output_path],
                validation_paths=[],
                cache_path=output_path,
                tokenizer=marin_tokenizer,
            )
        ),
        hash_attrs={"tokenizer": marin_tokenizer},
    )

    return [*processed.deps, processed, tokenized]


if __name__ == "__main__":
    StepRunner().run(build_steps())
