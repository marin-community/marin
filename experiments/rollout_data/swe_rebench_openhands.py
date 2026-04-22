# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""nebius/SWE-rebench-openhands-trajectories rollout dataset.

Usage:
    uv run iris --cluster=marin job run --cpu=1 --memory=2G --extra=cpu \
      -- python -m experiments.rollout_data.swe_rebench_openhands
"""

from fray.v2 import ResourceConfig
from marin.datakit.download.swe_rebench_openhands import download_swe_rebench_openhands_step
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize import TokenizeConfig, tokenize
from experiments.marin_models import marin_tokenizer


def build_steps() -> list[StepSpec]:
    processed = download_swe_rebench_openhands_step()

    tokenized = StepSpec(
        name="tokenized/swe-rebench-openhands-trajectories",
        deps=[processed],
        fn=lambda output_path: tokenize(
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

    return [*processed.deps, processed, tokenized]


if __name__ == "__main__":
    StepRunner().run(build_steps())
