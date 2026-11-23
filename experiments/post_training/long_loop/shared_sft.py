# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the long-loop's prepared sources to the shared SFT launcher."""

import copy
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace

import draccus
from fray.types import GpuConfig, ResourceConfig
from iris.rpc.proto_display import priority_band_value
from levanter.data.text.datasets import LmDataConfig
from levanter.training_budget import TrainingBudgetConfig, TrainingStopReason, read_training_completion
from levanter.utils.mesh import MeshConfig
from marin.execution.lazy import ArtifactStep, StepContext
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.long_loop.execution_policy import SFT_CHECKPOINT_INTERVAL
from experiments.post_training.long_loop.exp3_recipe import exp3_sft_spec
from experiments.post_training.long_loop.sft import fresh_sft_config, read_json, sft_run
from experiments.post_training.long_loop.sft_epoch import step_exposure
from experiments.post_training.long_loop.stage_guard import StageGuard, run_guarded_stage
from experiments.sft.configs.snowball_h100 import SPEC as SNOWBALL_SFT
from experiments.sft.launcher import DeprecatedSFTSpec, HFModel, SFTSpec, levanter_train_request, sft_step


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
    if recipe["profile"] == "snowball_exp3":
        return exp3_sft_spec(name, version, recipe)
    if recipe["profile"] != "snowball_h100":
        raise ValueError("The long-loop shared SFT adapter requires the snowball_h100 profile")
    if not recipe.get("wandb_entity") or not recipe.get("wandb_project"):
        raise ValueError("Long-loop SFT requires explicit W&B entity and project")
    mesh = draccus.decode(MeshConfig, recipe["mesh"])
    resources = draccus.decode(ResourceConfig, recipe["resources"])
    if not isinstance(resources.device, GpuConfig):
        raise ValueError("The Snowball H100 profile requires GPU resources")
    # Iris presents one eight-GPU node as a JAX slice. Validate the actual
    # hybrid layout, rather than a flat virtual CPU mesh, before allocation.
    mesh.axis_shapes(resources.device.count * resources.replicas, resources.replicas)
    data = copy.deepcopy(recipe["data"])
    steps = recipe["stop_after_steps"]
    if recipe["target_positions"] != steps * recipe["seq_len"] * recipe["batch_size"]:
        raise ValueError("SFT target_positions must equal the declared updates times batch and context")
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
    assert isinstance(SNOWBALL_SFT.model, HFModel)
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
        checkpointer=replace(SNOWBALL_SFT.checkpointer, save_interval=SFT_CHECKPOINT_INTERVAL),
        pack=bool(recipe["pack"]),
        wandb_project=recipe["wandb_project"],
        wandb_entity=recipe["wandb_entity"],
        xla_flags=tuple(recipe["xla_flags"]),
    )


def native_sft_step(
    name: str, version: str, recipe: dict, *, spec_builder: Callable[[str, str, dict], SFTSpec] = shared_sft_spec
) -> ArtifactStep:
    spec = spec_builder(name, version, recipe)
    trained = sft_step(spec, draccus.decode(ResourceConfig, recipe["resources"]))
    builder = trained.build_config

    def build_config(context):
        config = builder(context)
        return replace(
            config,
            train_config=replace(
                config.train_config,
                trainer=replace(
                    config.train_config.trainer, native_telemetry_root=prefix_join(context.output_path, "telemetry")
                ),
            ),
        )

    run = trained.run

    def execute(config):
        for key in ("sft_capacity", "sft_exposure"):
            if key in recipe:
                (StoragePath(config.output_path) / f"{key}.json").write_text(json.dumps(recipe[key], indent=2))
        if recipe["profile"] == "snowball_exp3" and recipe["stage_guard"] is not None:
            request = levanter_train_request(
                config, task_timeout_seconds=recipe["task_timeout_seconds"], priority=priority_band_value("production")
            )
            receipt = run_guarded_stage(
                request, StageGuard(**recipe["stage_guard"]), prefix_join(config.output_path, "guard")
            )
            (StoragePath(config.output_path) / "guard-result.json").write_text(json.dumps(receipt, indent=2))
        else:
            run(config)
        budget = config.train_config.trainer.training_budget
        if budget is None:
            select_shared_sft(output=config.output_path, completed_updates=spec.num_train_steps)
        else:
            select_time_limited_sft(output=config.output_path, maximum_updates=spec.num_train_steps, budget=budget)
        if "sft_capacity" in recipe:
            selected = read_json(prefix_join(config.output_path, "selected.json"))
            actual = step_exposure(recipe, recipe["sft_capacity"]["sources"], selected["completed_updates"])
            actual["completed_updates"] = selected["completed_updates"]
            actual["epoch_target_complete"] = selected["completed_updates"] == spec.num_train_steps
            actual["stop_reason"] = selected.get("training_completion", {}).get("stop_reason")
            (StoragePath(config.output_path) / "sft_actual_exposure.json").write_text(json.dumps(actual, indent=2))
            if recipe["sft_exposure"]["stopping"]["type"] == "mixture_epochs":
                validate_epoch_completion(actual)

    return replace(trained, run=execute, build_config=build_config)


def validate_epoch_completion(actual: dict) -> None:
    """Accept a partial epoch only when the native receipt certifies its time limit."""
    if actual["epoch_target_complete"] and not actual["full_sft_coverage"]:
        raise ValueError("Completed SFT epoch lacks full training coverage; see sft_actual_exposure.json")
    if not actual["epoch_target_complete"] and actual["stop_reason"] != TrainingStopReason.TIME_BOUND:
        raise ValueError("SFT stopped before the declared epoch without a native time-bound completion")


def select_shared_sft(*, output: str, completed_updates: int) -> None:
    """Bind native zero-based checkpoint indices to the completed update count."""
    selection = shared_sft_selection(output=output, completed_updates=completed_updates)
    (StoragePath(output) / "selected.json").write_text(json.dumps(selection))


def shared_sft_selection(*, output: str, completed_updates: int) -> dict:
    """Return a selection only for a committed native checkpoint."""
    step = completed_updates - 1
    checkpoint = prefix_join(output, f"checkpoints/step-{step}")
    metadata = read_json(prefix_join(checkpoint, "metadata.json"))
    if metadata["step"] != step or metadata["is_temporary"]:
        raise ValueError("Shared SFT did not commit its final native checkpoint")
    return {"checkpoint": checkpoint, "step": step, "completed_updates": completed_updates, "format": "SFTSpec"}


def select_time_limited_sft(*, output: str, maximum_updates: int, budget: TrainingBudgetConfig) -> None:
    """Select the final committed update recorded by the time-limited trainer."""
    completion = read_training_completion(
        telemetry_root=prefix_join(output, "telemetry"),
        checkpoint_root=prefix_join(output, "checkpoints"),
        maximum_updates=maximum_updates,
        budget=budget,
    )
    selection = shared_sft_selection(output=output, completed_updates=completion["completed_updates"])
    (StoragePath(output) / "selected.json").write_text(json.dumps({**selection, "training_completion": completion}))


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
