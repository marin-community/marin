# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the long-loop's prepared sources to the shared SFT launcher."""

import copy
import hashlib
import json
import math
from dataclasses import dataclass, replace

import draccus
from fray.types import GpuConfig, ResourceConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.lazy import ArtifactStep, StepContext
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.long_loop.sft import fresh_sft_config, read_json, sft_run
from experiments.sft.configs.snowball_h100 import SPEC as SNOWBALL_SFT
from experiments.sft.launcher import DeprecatedSFTSpec, SFTSpec, sft_step


@dataclass(frozen=True)
class LegacyGrugRun:
    output: str
    initializer: str
    recipe: dict
    run_id: str
    resources: ResourceConfig


def run_legacy_grug(config: LegacyGrugRun) -> None:
    sft_run(**vars(config))


@dataclass(frozen=True)
class DeprecatedGrugBackend:
    """Keep the historical Grug recipe intact behind the shared launch boundary."""

    recipe: dict
    initializer: ArtifactStep
    run_id: str

    @property
    def run(self):
        return run_legacy_grug

    def build_train_config(self, ctx: StepContext, resources: ResourceConfig) -> LegacyGrugRun:
        recipe = copy.deepcopy(self.recipe)
        initializer = ctx.artifact_path(self.initializer)
        fresh_sft_config(output=ctx.output_path, initializer=initializer, recipe=recipe, run_id=self.run_id)
        return LegacyGrugRun(ctx.output_path, initializer, recipe, self.run_id, resources)


def legacy_sft_step(
    name: str, version: str, recipe: dict, initializer: ArtifactStep, *, acknowledge: bool
) -> ArtifactStep:
    spec = DeprecatedSFTSpec(
        name=name,
        version=version,
        backend=DeprecatedGrugBackend(recipe, initializer, f"{name.replace('/', '-')}-{version}"),
        deps=(initializer,),
    )
    return sft_step(
        spec,
        draccus.decode(ResourceConfig, recipe["resources"]),
        acknowledge_deprecated_config_format=acknowledge,
    )


def shared_sft_spec(name: str, version: str, recipe: dict) -> SFTSpec:
    """Use the checked-in Snowball defaults with the pinned historical sources."""
    if recipe["profile"] != "snowball_h100":
        raise ValueError("The long-loop shared SFT adapter requires the snowball_h100 profile")
    mesh = draccus.decode(MeshConfig, recipe["mesh"])
    resources = draccus.decode(ResourceConfig, recipe["resources"])
    if not isinstance(resources.device, GpuConfig):
        raise ValueError("The Snowball H100 profile requires GPU resources")
    # Iris presents one eight-GPU node as a JAX slice. Validate the actual
    # hybrid layout, rather than a flat virtual CPU mesh, before allocation.
    mesh.axis_shapes(resources.device.count * resources.replicas, resources.replicas)
    data = copy.deepcopy(recipe["data"])
    steps = min(
        math.ceil(recipe["target_positions"] / (recipe["seq_len"] * recipe["batch_size"])),
        recipe["stop_after_steps"],
    )
    new_step_positions = recipe["seq_len"] * recipe["batch_size"]
    data["train_weights"] = [
        (step * recipe["source_step_positions"] // new_step_positions, weights)
        for step, weights in data["train_weights"]
    ]
    if any(step * recipe["source_step_positions"] % new_step_positions for step, _ in recipe["data"]["train_weights"]):
        raise ValueError("Source schedule boundaries do not align with the shared SFT batch")
    for name_, component in data["components"].items():
        if not name_.startswith("sft/"):
            continue
        children = component["children"].values() if component["type"] == "concat" else (component,)
        for child in children:
            child["format"] = {"type": "serialized_chat", **recipe["serialized_chat"]}
            child["pack"] = recipe["pack"]
    data["block_cross_document_attention"] = bool(recipe["pack"])
    data_config = draccus.decode(LmDataConfig, data)
    return replace(
        SNOWBALL_SFT,
        name=name,
        version=version,
        datasets=(),
        model=replace(
            SNOWBALL_SFT.model, tokenizer_path=data["tokenizer"], task_timeout_seconds=recipe["task_timeout_seconds"]
        ),
        prepared_data=data_config,
        num_train_steps=steps,
        seq_len=recipe["seq_len"],
        batch_size=recipe["batch_size"],
        per_device_parallelism=recipe["per_device_parallelism"],
        mesh=mesh,
        pack=bool(recipe["pack"]),
        wandb_project=recipe["wandb_project"],
        xla_flags=tuple(recipe["xla_flags"]),
    )


def native_sft_step(name: str, version: str, recipe: dict) -> ArtifactStep:
    spec = shared_sft_spec(name, version, recipe)
    trained = sft_step(spec, draccus.decode(ResourceConfig, recipe["resources"]))
    run = trained.run

    def execute(config):
        run(config)
        select_shared_sft(output=config.output_path, completed_updates=spec.num_train_steps)

    return replace(trained, run=execute)


def select_shared_sft(*, output: str, completed_updates: int) -> None:
    """Bind native zero-based checkpoint indices to the completed update count."""
    step = completed_updates - 1
    checkpoint = prefix_join(output, f"checkpoints/step-{step}")
    metadata = read_json(prefix_join(checkpoint, "metadata.json"))
    if metadata["step"] != step or metadata["is_temporary"]:
        raise ValueError("Shared SFT did not commit its final native checkpoint")
    (StoragePath(output) / "selected.json").write_text(
        json.dumps({"checkpoint": checkpoint, "step": step, "completed_updates": completed_updates, "format": "SFTSpec"})
    )


def export_shared_sft(*, output: str, trained: str, context_limit: int, source_revision: str) -> None:
    """Bind the native trainer's HF export to the existing RL handoff layout."""
    selected = read_json(prefix_join(trained, "selected.json"))
    source = prefix_join(trained, f"hf/step-{selected['step']}")
    fs, root = filesystem_for(source)
    destination_fs, destination = filesystem_for(prefix_join(output, "hf"))
    destination_fs.makedirs(destination, exist_ok=True)
    files = []
    for key in sorted(fs.find(root)):
        relative = str(StoragePath(key).relative_to(StoragePath(root)))
        target = prefix_join(destination, relative)
        if relative == "config.json":
            config = json.loads(fs.cat_file(key))
            config.update(max_position_embeddings=context_limit)
            destination_fs.pipe_file(target, json.dumps(config, indent=2).encode())
        else:
            # One server-side copy when both outputs use the same S3 filesystem.
            if fs is not destination_fs:
                raise ValueError("SFT handoff must remain on the same filesystem")
            fs.copy(key, target)
        files.append({"filename": relative, "size": destination_fs.size(target)})
    if not {"config.json", "model.safetensors.index.json"} <= {item["filename"] for item in files}:
        raise ValueError("Shared SFT produced no complete HF checkpoint")
    receipt = {
        "format": "SFTSpec",
        "source": source,
        "native_checkpoint": selected["checkpoint"],
        "source_revision": source_revision,
        "files": files,
        "consumed_config_sha256": (
            hashlib.sha256((StoragePath(trained) / "consumed-config.json").read_bytes()).hexdigest()
        ),
    }
    (StoragePath(output) / "handoff.json").write_text(json.dumps(receipt, indent=2))
