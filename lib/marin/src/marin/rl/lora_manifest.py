# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Literal

import fsspec
from levanter.lora import LoraConfig

from marin.utilities.json_encoder import CustomJsonEncoder
from marin.utils import fsspec_mkdirs

RolloutPolicyFormat = Literal["merged", "adapter"]
ReferenceMode = Literal["base"]
RUN_MANIFEST_FILENAME = "rl_run_manifest.json"


@dataclass(frozen=True)
class RLRunManifest:
    """Trainer-side manifest for LoRA RL configuration and artifacts."""

    manifest_version: int
    initial_checkpoint: str | None
    model_config_type: str
    model_config_fingerprint: str
    lora_config: LoraConfig | None
    lora_config_fingerprint: str | None
    rollout_policy_format: RolloutPolicyFormat
    reference_mode: ReferenceMode
    inference_type: Literal["levanter", "vllm"]


def config_fingerprint(config: object) -> str:
    """Return a stable fingerprint for a serializable config object."""
    serialized = json.dumps(config, sort_keys=True, cls=CustomJsonEncoder)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def build_rl_run_manifest(
    *,
    initial_checkpoint: str | None,
    model_config: object,
    lora_config: LoraConfig | None,
    rollout_policy_format: RolloutPolicyFormat,
    inference_type: Literal["levanter", "vllm"],
) -> RLRunManifest:
    """Build the run manifest recorded alongside LoRA RL training artifacts."""
    return RLRunManifest(
        manifest_version=1,
        initial_checkpoint=initial_checkpoint,
        model_config_type=f"{type(model_config).__module__}.{type(model_config).__qualname__}",
        model_config_fingerprint=config_fingerprint(model_config),
        lora_config=lora_config,
        lora_config_fingerprint=None if lora_config is None else config_fingerprint(lora_config),
        rollout_policy_format=rollout_policy_format,
        reference_mode="base",
        inference_type=inference_type,
    )


def write_rl_run_manifest(path: str, manifest: RLRunManifest) -> None:
    """Write a manifest file describing the LoRA RL run configuration."""
    fsspec_mkdirs(os.path.dirname(path), exist_ok=True)
    with fsspec.open(path, "wt") as f:
        json.dump(dataclasses.asdict(manifest), f, indent=2, sort_keys=True, cls=CustomJsonEncoder)
