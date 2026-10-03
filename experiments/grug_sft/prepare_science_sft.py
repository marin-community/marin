# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the proof and science chat stores used by the 100B-token curricula."""

import logging
import math
from dataclasses import dataclass

from marin.datakit.sft_sources import all_sft_sources
from marin.datakit.sft_text import SftInput, SftTokenStore, build_sft_store
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner
from marin.experiment.cli import experiment_main
from rigging.filesystem.storage_path import prefix_join

from experiments.grug_sft.regional_pool import snowball_model_path

logger = logging.getLogger(__name__)

CONTEXT_LENGTH = 262_144
SHUFFLE_SEED = 0
TARGET_TOKENS_PER_SHARD = 62_500_000
SCIENCE_SFT_SOURCES = (
    "nemotron_sft_v3/math_proofs_v1/lean",
    "nemotron_sft_v3/math_proofs_v2/train",
    "nemotron_sft_v3/science_v2/rqa",
    "nemotron_sft_v3/science_v2/so",
    "nemotron_sft_v3/science_v2/syn_mcq",
    "nemotron_sft_v3/science_v2/vendor",
    "megascience/textbook-reasoning",
)


@dataclass(frozen=True)
class SourceConfig:
    output_path: str
    name: str
    normalized_version: str
    shards: int
    workers: int


def prepare_source(config: SourceConfig) -> SftTokenStore:
    """Materialize and pack one pinned Datakit chat source."""
    source = all_sft_sources()[config.name]
    if source.normalized.name_with_hash != config.normalized_version:
        raise ValueError(f"Source definition changed for {config.name}")
    StepRunner().run([source.normalized], max_concurrent=1)
    result = build_sft_store(
        [SftInput(config.name, prefix_join(source.normalized.output_path, "outputs/main"))],
        output_path=config.output_path,
        tokenizer=snowball_model_path(),
        max_length=CONTEXT_LENGTH,
        seed=SHUFFLE_SEED,
        num_shards=config.shards,
        max_workers=config.workers,
    )
    logger.info("Prepared %s: %s", config.name, result.model_dump_json())
    return result


def build() -> dict[str, ArtifactStep[SftTokenStore]]:
    """Build one independently resumable store per selected chat source."""
    registry = all_sft_sources()
    handles = {}
    for name in SCIENCE_SFT_SOURCES:
        source = registry[name]
        artifact_name = f"grug_sft/science_sft/{name}"
        normalized_version = source.normalized.name_with_hash
        shards = max(1, math.ceil(source.rough_token_count_b * 1e9 / TARGET_TOKENS_PER_SHARD))

        def config(ctx: StepContext, name=name, normalized_version=normalized_version, shards=shards) -> SourceConfig:
            return SourceConfig(
                output_path=ctx.output_path,
                name=name,
                normalized_version=normalized_version,
                shards=shards,
                workers=ctx.runtime_arg("workers"),
            )

        handles[name] = ArtifactStep(
            name=artifact_name,
            version=resolve_version(artifact_name, None),
            artifact_type=SftTokenStore,
            run=prepare_source,
            build_config=config,
            runtime_args={"workers": 32},
        )
    return handles


if __name__ == "__main__":
    experiment_main(build)()
