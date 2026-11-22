# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Explicit Exp3 SFT controls, independent of the historical shared defaults."""

import hashlib
import json
import math
from dataclasses import dataclass, replace

import draccus
from fray.types import GpuConfig, ResourceConfig
from iris.rpc.proto_display import priority_band_value
from levanter.data.text.datasets import LmDataConfig
from levanter.eval_audit import InitialEvalMatchConfig
from levanter.models.snowball import SnowballConfig
from levanter.models.snowball_sft import SnowballSftConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.training_budget import budget_learning_rate_scale
from levanter.utils.mesh import MeshConfig
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.fsutil.verified_copy import COMPLETION_MANIFEST
from rolloutengine.chat_template import MARIN_CHAT_TEMPLATE

from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig
from experiments.post_training.long_loop.sft import BASE_MODEL, BASE_REVISION
from experiments.post_training.long_loop.source_loss import prepared_supervision_data, source_loss_metadata
from experiments.post_training.long_loop.stage_guard import StageGuard
from experiments.sft.launcher import HFModel, SFTSpec

# Exp1's LR anchor and schedule; the ratio and regularization restore the
# 67B pretrain recipe at 92514c5358e9cd99c89bdd4a6600ac7bbeddc147.
TOKEN_IDS = (128006, 128007, 128002, 128003, 128005, 128011, 128009)
ANCHOR_IDS = (3560, 1984, 1781, 4320, 5507, 842, 128001)


def verified_initializer_model(receipt_uri: str, receipt_sha256: str) -> str:
    """Resolve a staged initializer only after its source and completed copy agree."""
    raw = StoragePath(receipt_uri).read_bytes()
    if hashlib.sha256(raw).hexdigest() != receipt_sha256:
        raise ValueError("Initializer source receipt SHA256 mismatch")
    receipt = json.loads(raw)
    if receipt["repo"] != BASE_MODEL or receipt["revision"] != BASE_REVISION:
        raise ValueError("Staged initializer is not the pinned Exp3 source")
    model_ref = receipt["model_ref"]
    manifest_uri = prefix_join(model_ref, COMPLETION_MANIFEST)
    if receipt["verified_copy_manifest"] != manifest_uri:
        raise ValueError("Initializer completion manifest belongs to another destination")
    manifest_raw = StoragePath(manifest_uri).read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != receipt["verified_copy_manifest_sha256"]:
        raise ValueError("Initializer completion manifest SHA256 mismatch")
    manifest = json.loads(manifest_raw)
    expected = {item["path"]: (item["size"], item["sha256"]) for item in receipt["files"]}
    actual = {item["path"]: (item["size"], item["sha256"]) for item in manifest["files"]}
    if manifest["destination"] != model_ref or actual != expected:
        raise ValueError("Regional initializer differs from the verified Hub inventory")
    for item in receipt["files"]:
        if item["lfs"] is not None and item["sha256"] != item["lfs"]["sha256"]:
            raise ValueError(f"Initializer LFS identity mismatch: {item['path']}")
    return model_ref


def exp3_optimizer(*, weight_decay: float) -> GrugMoeMuonHConfig:
    return GrugMoeMuonHConfig(
        learning_rate=5e-5,
        adam_lr=5e-5 * 3 / 13,
        beta1=0.9062,
        beta2=0.95,
        epsilon=3.8339433005718795e-15,
        weight_decay=weight_decay,
        max_grad_norm=None,
        warmup=0,
        decay=475,
        min_lr_ratio=0.1,
        lr_schedule="linear",
    )


def exp3_controls() -> SnowballSftConfig:
    return SnowballSftConfig(TOKEN_IDS, ANCHOR_IDS, math.sqrt(32), True, True)


@dataclass(frozen=True)
class SftLrSchedule:
    horizon: int
    warmup: int
    decay: int
    min_lr_ratio: float
    lr_schedule: str


def whole_run_lr_receipt(spec: SFTSpec, *, schedule: SftLrSchedule) -> dict:
    """Verify declared optimizer fields and sample the configured LR controls."""
    if spec.trainer_config is None or spec.trainer_config.num_train_steps != spec.num_train_steps:
        raise ValueError("SFT trainer maximum differs from the declared update maximum")
    if spec.optimizer_num_train_steps != schedule.horizon:
        raise ValueError("SFT optimizer horizon differs from its declared schedule")
    optimizer = spec.optimizer
    if not isinstance(optimizer, GrugMoeMuonHConfig):
        raise ValueError("This two-group SFT receipt requires the declared GrugMoeMuonH optimizer")
    actual = (optimizer.warmup, optimizer.decay, optimizer.min_lr_ratio, optimizer.lr_schedule)
    declared = (schedule.warmup, schedule.decay, schedule.min_lr_ratio, schedule.lr_schedule)
    if actual != declared:
        raise ValueError("SFT optimizer settings differ from the declared LR schedule")
    steps = spec.num_train_steps
    indices = {min(int(steps * fraction), steps - 1) for fraction in (0, 0.25, 0.5, 0.75, 0.85, 1)}
    indices.update((min(schedule.warmup // 2, steps - 1), min(schedule.warmup, steps - 1)))
    samples = []
    for index in sorted(indices):
        rates = {}
        for name, peak in (("muonh", optimizer.learning_rate), ("adam", optimizer.adam_lr)):
            curve = optimizer.lr_scheduler(schedule.horizon, override_lr=peak)
            rates[name] = float(curve(index))
            if index == steps - 1 and schedule.lr_schedule != "constant" and steps >= schedule.horizon:
                floor = peak * schedule.min_lr_ratio
                if abs(rates[name] - floor) > max(floor * 0.1, 1e-12):
                    raise ValueError(f"SFT declared anneal final-update {name} LR is not within 10% of min_lr")
        samples.append({"fraction": index / steps, "optimizer_update_index": index, "learning_rates": rates})
    receipt = {
        "num_train_steps": steps,
        "optimizer_num_train_steps": schedule.horizon,
        "warmup_steps": schedule.warmup,
        "decay_steps": schedule.decay,
        "lr_schedule": schedule.lr_schedule,
        "samples": samples,
    }
    budget = spec.trainer_config.training_budget
    if budget is not None and budget.cooldown is not None:
        usable = budget.training_duration.total_seconds()
        window = usable * budget.cooldown.fraction
        receipt["wall_clock_cooldown"] = {
            "trigger": "remaining_wall_time_only",
            "duration_seconds": budget.duration.total_seconds(),
            "usable_training_seconds": usable,
            "cooldown_seconds": window,
            "start_after_seconds": usable - window,
            "fraction": budget.cooldown.fraction,
            "min_lr_ratio": budget.cooldown.min_lr_ratio,
            "samples": [
                {
                    "remaining_training_seconds": remaining,
                    "learning_rate_scale": budget_learning_rate_scale(budget, remaining),
                }
                for remaining in (usable, window, window / 2, 0)
            ],
            "final_update_check": (
                "Wall-clock stops require the final applied LR scale near the declared minimum; "
                "the update safety cap never triggers cooldown"
            ),
        }
    return receipt


def exp3_sft_spec(name: str, version: str, recipe: dict) -> SFTSpec:
    """Bind a pinned candidate and required gate, trainer and budget before allocation."""
    if recipe["stage_guard"] is not None:
        StageGuard(**recipe["stage_guard"])
    metadata = source_loss_metadata(recipe["source_loss_metadata"], recipe["source_loss_metadata_sha256"])
    data = prepared_supervision_data(
        draccus.decode(LmDataConfig, recipe["data"]),
        metadata,
        packing_cap=recipe["packing_cap"],
        lookahead=recipe["pack_lookahead_window"],
    )
    evaluation_data = prepared_supervision_data(
        draccus.decode(LmDataConfig, recipe["evaluation_data"]),
        metadata,
        packing_cap=recipe["packing_cap"],
        lookahead=recipe["pack_lookahead_window"],
    )
    trainer = draccus.decode(TrainerConfig, recipe["trainer"])
    if trainer.training_budget is not None:
        if trainer.training_budget.duration.total_seconds() != recipe["task_timeout_seconds"]:
            raise ValueError("Exp3 training duration must match its unchanged hard execution deadline")
    if not recipe["wandb_entity"] or not recipe["wandb_project"]:
        raise ValueError("Exp3 requires an explicit W&B entity and project")
    if recipe["packing_cap"] != 256:
        raise ValueError("Exp3 retains the 256-document packing cap")
    resources = draccus.decode(ResourceConfig, recipe["resources"])
    if not isinstance(resources.device, GpuConfig) or resources.device.variant.upper() != "H100":
        raise ValueError("Exp3 requires explicit H100 resources")
    mesh = draccus.decode(MeshConfig, recipe["mesh"])
    mesh.axis_shapes(resources.device.count * resources.replicas, resources.replicas)
    gate = draccus.decode(InitialEvalMatchConfig, recipe["initial_eval_match"])
    spec = SFTSpec(
        name=name,
        version=version,
        model=HFModel(
            model_ref=verified_initializer_model(recipe["initializer_receipt"], recipe["initializer_receipt_sha256"]),
            tokenizer_path=data.tokenizer,
            model_type="snowball",
            eos_token_ids=(128001, 128009),
            task_timeout_seconds=recipe["task_timeout_seconds"],
            priority=priority_band_value("production"),
        ),
        chat_template=MARIN_CHAT_TEMPLATE,
        datasets=(),
        prepared_data=data,
        model_config=draccus.decode(SnowballConfig, recipe["model"]),
        trainer_config=replace(
            trainer,
            tracker=WandbConfig(entity=recipe["wandb_entity"], project=recipe["wandb_project"], tags=["exp3", "sft"]),
        ),
        optimizer=exp3_optimizer(weight_decay=recipe["weight_decay"]),
        optimizer_num_train_steps=4750,
        z_loss_weight=1e-4,
        global_gradient_clip=1,
        snowball_sft=exp3_controls(),
        token_weighted_accumulation=True,
        initial_eval_match=gate,
        evaluation_data=evaluation_data,
        data_seed=recipe["data_seed"],
        tensorstore_cache_bytes=recipe["tensorstore_cache_bytes"],
        mesh=mesh,
        seq_len=recipe["seq_len"],
        batch_size=recipe["batch_size"],
        per_device_parallelism=recipe["per_device_parallelism"],
        num_train_steps=recipe["stop_after_steps"],
        checkpointer=trainer.checkpointer,
        pack=True,
        hf_save_dtype="bfloat16",
        wandb_project=recipe["wandb_project"],
        wandb_entity=recipe["wandb_entity"],
        xla_flags=tuple(recipe["xla_flags"]),
    )
    if "optimizer_schedule" not in recipe:
        return spec
    schedule = draccus.decode(SftLrSchedule, recipe["optimizer_schedule"])
    spec = replace(
        spec,
        optimizer=replace(
            exp3_optimizer(weight_decay=recipe["weight_decay"]),
            warmup=schedule.warmup,
            decay=schedule.decay,
            min_lr_ratio=schedule.min_lr_ratio,
            lr_schedule=schedule.lr_schedule,
        ),
        optimizer_num_train_steps=schedule.horizon,
    )
    whole_run_lr_receipt(spec, schedule=schedule)
    return spec
