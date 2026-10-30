# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build versioned, bounded replay caches on a region-local CPU worker."""

from dataclasses import dataclass

import draccus
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.step_runner import StepRunner

from experiments.grug.snowball_sft.replay_subset import ReplaySubsetConfig, prepare_replay_subset


@dataclass(frozen=True)
class ReplayCell:
    name: str
    source: str
    source_ledger_sha256: str
    shards: tuple[str, ...]
    short_tokens: int
    long_tokens: int


@dataclass(frozen=True)
class ReplayPreparation:
    name: str
    version: str
    tokenizer: str
    cells: tuple[ReplayCell, ...]
    max_concurrent: int = 2
    dry_run: bool = True


def materialize(config: ReplaySubsetConfig) -> None:
    prepare_replay_subset(config)


def replay_step(config: ReplayPreparation, cell: ReplayCell) -> ArtifactStep[Artifact]:
    source = ArtifactStep.adopt(
        name=f"{config.name}/sources/{cell.name}",
        version=config.version,
        source=cell.source,
        config={"ledger_sha256": cell.source_ledger_sha256},
    )

    def build_config(ctx: StepContext) -> ReplaySubsetConfig:
        source_path = ctx.artifact_path(source)
        return ReplaySubsetConfig(
            source_shards=tuple(f"{source_path}/{shard}" for shard in cell.shards),
            output_path=ctx.output_path,
            short_tokens=cell.short_tokens,
            long_tokens=cell.long_tokens,
            tokenizer=config.tokenizer,
        )

    return ArtifactStep(
        name=f"{config.name}/{cell.name}",
        version=config.version,
        artifact_type=Artifact,
        run=materialize,
        build_config=build_config,
        deps=(source,),
    )


@draccus.wrap()
def main(config: ReplayPreparation) -> None:
    if len({cell.name for cell in config.cells}) != len(config.cells):
        raise ValueError("Replay cell names must be unique")
    steps = [replay_step(config, cell).lower() for cell in config.cells]
    StepRunner().run(steps, dry_run=config.dry_run, max_concurrent=config.max_concurrent, force_run_failed=False)


if __name__ == "__main__":
    main()
