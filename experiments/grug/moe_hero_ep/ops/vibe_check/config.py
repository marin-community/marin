# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The production lineage and versioned prompt bank for native hero completions."""

import dataclasses
import json
import logging
from pathlib import Path

import draccus
from fray.types import GpuConfig, ResourceConfig
from levanter.checkpoint import discover_checkpoint_candidates
from rigging.filesystem.storage_path import prefix_join

from experiments.grug.moe_hero_ep import hero_recipe
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import (
    Checkpoint,
    Prompt,
    SampleRequest,
    SamplingSpec,
    digest,
)
from experiments.grug.moe_hero_ep.train import DEFAULT_DROPLESS_MOE_IMPLEMENTATION

logger = logging.getLogger(__name__)

CONFIG_DIRECTORY = Path(__file__).parent
CHECKPOINT_ROOT = "s3://marin-us-east-02a/marin/grug"
STORE_ROOT = "s3://marin-us-east-02a/marin/users/rav/hero-completions/"
TARGET_CLUSTER = "cw-us-east-08a"
SAMPLING_NODES = 8
SAMPLING_GPUS_PER_NODE = 4


@dataclasses.dataclass(frozen=True)
class CheckpointRun:
    run_id: str
    version: str
    max_step: int | None = None


CHECKPOINT_RUNS = (
    CheckpointRun("hero-12d8b6f0-dee637", "2026.08.19.2", max_step=58014),
    CheckpointRun("hero-wd-gate-router-p02-step58k", "2026.08.19.2", max_step=81716),
    CheckpointRun("hero-ragged_a2a-nccl2307-ep-step81k", "2026.08.19.2"),
)


def sampling_spec() -> SamplingSpec:
    # The local dropless backend matches held-out evaluation and cannot drop another prompt's tokens.
    model = dataclasses.replace(
        hero_recipe.HERO_MODEL_CONFIG, moe_implementation=DEFAULT_DROPLESS_MOE_IMPLEMENTATION, expert_chunks=1
    )
    return SamplingSpec(
        release="hero-native-v4-reference-eos",
        completions_per_prompt=3,
        batch_size=SAMPLING_NODES * SAMPLING_GPUS_PER_NODE,
        prompts=tuple(Prompt.model_validate(row) for row in json.loads((CONFIG_DIRECTORY / "prompts.json").read_text())),
        tokenizer="marin-community/marin-tokenizer",
        tokenizer_revision="a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2",
        model=draccus.encode(model),
        temperature=0.2,
        max_new_tokens=1024,
        context_length=4096,
    )


def sampling_resources() -> ResourceConfig:
    return ResourceConfig(
        cpu=120,
        ram="890g",
        disk="1t",
        device=GpuConfig(variant="GB200", count=SAMPLING_GPUS_PER_NODE),
        replicas=SAMPLING_NODES,
    )


def discover_requests(
    runs: tuple[CheckpointRun, ...], spec: SamplingSpec, revision: str, *, target_cluster: str
) -> list[SampleRequest]:
    """Read committed permanent checkpoints in the declared production lineage, without tensor reads."""
    checkpoints: dict[str, Checkpoint] = {}
    for run in runs:
        candidates = discover_checkpoint_candidates(
            prefix_join(CHECKPOINT_ROOT, f"{run.run_id}/{run.version}/checkpoints"),
            max_step=run.max_step,
        )
        if not candidates:
            logger.warning("No complete checkpoints found for run %s at version %s", run.run_id, run.version)
        for candidate in candidates:
            if candidate.metadata.get("is_temporary") is not False:
                continue
            checkpoints[candidate.path] = Checkpoint(
                uri=candidate.path,
                run_id=run.run_id,
                step=candidate.step,
                timestamp=candidate.timestamp.isoformat(),
                metadata_digest=digest(candidate.metadata),
            )
    return [
        SampleRequest(checkpoint=checkpoint, spec=spec, source_revision=revision, target_cluster=target_cluster)
        for checkpoint in checkpoints.values()
    ]
