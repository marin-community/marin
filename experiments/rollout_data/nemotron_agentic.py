# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""nvidia/Nemotron-SFT-Agentic-v2 rollout dataset.

Usage:
    uv run iris --cluster=marin job run --cpu=1 --memory=2G --extra=cpu \
      -- python -m experiments.rollout_data.nemotron_agentic
"""

from fray import ResourceConfig
from marin.datakit.download.nemotron_agentic import NemotronAgenticSubset, download_nemotron_agentic_step
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize import TokenizeConfig, tokenize

from experiments.marin_models import marin_tokenizer

ACTIVE_SUBSETS = (NemotronAgenticSubset.TOOL_CALLING,)


def build_steps() -> list[StepSpec]:
    steps: list[StepSpec] = []
    for subset in ACTIVE_SUBSETS:
        processed = download_nemotron_agentic_step(subset)
        tokenized = StepSpec(
            name=f"tokenized/nemotron-sft-agentic-v2/{subset.value}",
            deps=[processed],
            fn=lambda output_path, processed=processed: tokenize(
                TokenizeConfig(
                    train_paths=[processed.output_path],
                    validation_paths=[],
                    cache_path=output_path,
                    tokenizer=marin_tokenizer,
                    worker_resources=ResourceConfig(ram="48g", disk="5g"),
                )
            ),
            hash_attrs={"tokenizer": marin_tokenizer},
        )
        steps.extend([*processed.deps, processed, tokenized])
    return steps


if __name__ == "__main__":
    StepRunner().run(build_steps())
