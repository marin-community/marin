# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint requests and the versioned prompt bank for native hero completions."""

import dataclasses
import json
from datetime import datetime
from pathlib import Path

import draccus
from fray.types import GpuConfig, ResourceConfig
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.moe_hero_ep import hero_recipe
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import (
    Checkpoint,
    Prompt,
    SampleRequest,
    SamplingSpec,
    digest,
)
from experiments.grug.moe_hero_ep.train import DEFAULT_DROPLESS_MOE_IMPLEMENTATION

CONFIG_DIRECTORY = Path(__file__).parent
STORE_ROOT = "s3://marin-us-east-02a/marin/users/rav/hero-completions/"
TARGET_CLUSTER = "cw-us-east-08a"
# One NVL72 rack block. The sampler puts one prompt row on each GPU, so 64 rows cover the whole
# prompt bank in a single pass per sample instead of two.
SAMPLING_NODES = 16
SAMPLING_GPUS_PER_NODE = 4


def sampling_spec() -> SamplingSpec:
    return SamplingSpec(
        release="hero-native-v4-reference-eos",
        completions_per_prompt=3,
        prompts=tuple(Prompt.model_validate(row) for row in json.loads((CONFIG_DIRECTORY / "prompts.json").read_text())),
        tokenizer="marin-community/marin-tokenizer",
        tokenizer_revision="a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2",
        temperature=0.2,
        max_new_tokens=1024,
        context_length=4096,
    )


def sampling_model() -> dict:
    """The hero architecture used to restore weights, recorded with each request.

    This is provenance, not identity. A training-side change re-pins the architecture for new
    requests and leaves completed results addressable.
    """
    # The local dropless backend matches held-out evaluation and cannot drop another prompt's tokens.
    model = dataclasses.replace(
        hero_recipe.HERO_MODEL_CONFIG, moe_implementation=DEFAULT_DROPLESS_MOE_IMPLEMENTATION, expert_chunks=1
    )
    return draccus.encode(model)


def sampling_resources() -> ResourceConfig:
    return ResourceConfig(
        cpu=120,
        ram="890g",
        disk="1t",
        device=GpuConfig(variant="GB200", count=SAMPLING_GPUS_PER_NODE),
        replicas=SAMPLING_NODES,
    )


def discover_requests(
    checkpoint_paths: list[str], spec: SamplingSpec, model: dict, revision: str, *, target_cluster: str
) -> list[SampleRequest]:
    """Build sample requests from permanent checkpoint paths and metadata."""
    requests = []
    for uri in checkpoint_paths:
        path = StoragePath(uri)
        metadata = json.loads((path / "metadata.json").read_text())
        checkpoint = Checkpoint(
            uri=uri,
            run_id=path.parent.parent.parent.name,
            step=metadata["step"],
            timestamp=datetime.fromisoformat(metadata["timestamp"]).isoformat(),
            metadata_digest=digest(metadata),
        )
        requests.append(
            SampleRequest(
                checkpoint=checkpoint,
                spec=spec,
                model=model,
                source_revision=revision,
                target_cluster=target_cluster,
            )
        )
    return requests
