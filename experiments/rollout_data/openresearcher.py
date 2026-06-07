# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""OpenResearcher/OpenResearcher-Dataset rollout dataset.

Usage:
    uv run iris --cluster=marin job run --cpu=1 --memory=2G --extra=cpu \
      -- python -m experiments.rollout_data.openresearcher
"""

from collections.abc import Sequence

from fray import ResourceConfig
from marin.datakit.download.openresearcher import SEED_CONFIGS, download_openresearcher_step
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.processing.tokenize import TokenizeConfig, tokenize

from experiments.marin_models import marin_tokenizer


def build_steps(seed_configs: Sequence[str] = SEED_CONFIGS, sample_count: int | None = None) -> list[StepSpec]:
    seed_configs = tuple(seed_configs)
    is_sample = sample_count is not None or seed_configs != SEED_CONFIGS
    step_suffix = "sample" if is_sample else ""

    processed = download_openresearcher_step(seed_configs=seed_configs, step_suffix=step_suffix)

    tokenized = StepSpec(
        name="tokenized/openresearcher-dataset-sample" if is_sample else "tokenized/openresearcher-dataset",
        deps=[processed],
        fn=lambda output_path: tokenize(
            TokenizeConfig(
                train_paths=[processed.output_path],
                validation_paths=[],
                cache_path=output_path,
                tokenizer=marin_tokenizer,
                sample_count=sample_count,
                num_shards=64,
                worker_resources=ResourceConfig(ram="80g", disk="10g"),
            )
        ),
        hash_attrs={"tokenizer": marin_tokenizer, "sample_count": sample_count, "seed_configs": list(seed_configs)},
    )

    return [*processed.deps, processed, tokenized]


if __name__ == "__main__":
    StepRunner().run(build_steps())
