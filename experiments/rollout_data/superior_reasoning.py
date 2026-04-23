# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b rollout dataset.

Usage:
    uv run iris --cluster=marin job run --cpu=1 --memory=2G --extra=cpu \
      -- python -m experiments.rollout_data.superior_reasoning
"""

from fray.v2 import ResourceConfig
from marin.datakit.download.superior_reasoning import download_superior_reasoning_step
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize import TokenizeConfig, tokenize
from experiments.marin_models import marin_tokenizer


def build_steps() -> list[StepSpec]:
    processed = download_superior_reasoning_step()

    tokenized = StepSpec(
        name="tokenized/superior-reasoning-sft",
        deps=[processed],
        fn=lambda output_path: tokenize(
            TokenizeConfig(
                train_paths=[processed.output_path],
                validation_paths=[],
                cache_path=output_path,
                tokenizer=marin_tokenizer,
                worker_resources=ResourceConfig(ram="80g", disk="5g"),
            )
        ),
        hash_attrs={"tokenizer": marin_tokenizer},
    )

    return [*processed.deps, processed, tokenized]


if __name__ == "__main__":
    StepRunner().run(build_steps())
