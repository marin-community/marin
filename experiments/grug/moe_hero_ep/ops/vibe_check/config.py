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
    Record,
    SampleRequest,
    SamplingSpec,
    digest,
)
from experiments.grug.moe_hero_ep.train import DEFAULT_DROPLESS_MOE_IMPLEMENTATION

CONFIG_DIRECTORY = Path(__file__).parent
CHECKPOINT_ROOT = "s3://marin-us-east-02a/marin/grug"
STORE_ROOT = "s3://marin-us-east-02a/marin/hero-completions/v1"
logger = logging.getLogger(__name__)


class Ancestor(Record):
    run_id: str
    version: str
    max_step: int


class ProductionRun(Record):
    run_id: str
    version: str
    target_cluster: str
    handoff_checkpoint: str
    handoff_run_id: str
    ancestors: tuple[Ancestor, ...]


def production_run() -> ProductionRun:
    return ProductionRun.model_validate_json(Path(hero_recipe.__file__).with_name("production_run.json").read_bytes())


def sampling_spec() -> SamplingSpec:
    # The local dropless backend matches held-out evaluation and cannot drop another prompt's tokens.
    model = dataclasses.replace(
        hero_recipe.HERO_MODEL_CONFIG, moe_implementation=DEFAULT_DROPLESS_MOE_IMPLEMENTATION, expert_chunks=1
    )
    return SamplingSpec(
        release="hero-native-v1",
        batch_size=hero_recipe.HERO_EP_EXPERT_AXIS_SIZE,
        prompts=tuple(Prompt.model_validate(row) for row in json.loads((CONFIG_DIRECTORY / "prompts.json").read_text())),
        tokenizer="marin-community/marin-tokenizer",
        tokenizer_revision="a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2",
        model=draccus.encode(model),
        temperature=0.2,
        max_new_tokens=200,
        context_length=4096,
    )


def sampling_resources() -> ResourceConfig:
    return ResourceConfig(
        cpu=hero_recipe.HERO_NODE_CPU,
        ram=hero_recipe.HERO_NODE_RAM,
        disk=hero_recipe.HERO_NODE_DISK,
        device=GpuConfig(variant="GB200", count=hero_recipe.HERO_GPUS_PER_NODE),
        replicas=hero_recipe.HERO_EP_NODES,
    )


def discover_requests(run: ProductionRun, spec: SamplingSpec, revision: str) -> list[SampleRequest]:
    """Read committed permanent checkpoints in the declared production lineage, without tensor reads."""
    roots = [(ancestor.run_id, ancestor.version, ancestor.max_step) for ancestor in run.ancestors]
    roots.append((run.run_id, run.version, None))
    if run.handoff_run_id not in {ancestor.run_id for ancestor in run.ancestors}:
        raise ValueError("The handoff checkpoint must name an accepted ancestor")
    checkpoints: dict[str, Checkpoint] = {}
    for run_id, version, max_step in roots:
        additional = (run.handoff_checkpoint,) if run_id == run.handoff_run_id else ()
        candidates = discover_checkpoint_candidates(
            prefix_join(CHECKPOINT_ROOT, f"{run_id}/{version}/checkpoints"), *additional, max_step=max_step
        )
        if not candidates:
            logger.warning("No complete checkpoints found for declared run %s at version %s", run_id, version)
        if additional and not any(candidate.path == run.handoff_checkpoint for candidate in candidates):
            logger.warning("Declared handoff checkpoint is unavailable: %s", run.handoff_checkpoint)
        for candidate in candidates:
            if candidate.metadata.get("is_temporary") is not False:
                continue
            checkpoints[candidate.path] = Checkpoint(
                uri=candidate.path,
                run_id=run_id,
                step=candidate.step,
                timestamp=candidate.timestamp.isoformat(),
                metadata_digest=digest(candidate.metadata),
            )
    return [
        SampleRequest(checkpoint=checkpoint, spec=spec, source_revision=revision, target_cluster=run.target_cluster)
        for checkpoint in checkpoints.values()
    ]
