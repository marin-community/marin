# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind the hero's native SFT recipe to a sealed long-loop mixture."""

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from datetime import timedelta
from pathlib import Path

import draccus
import jax
from levanter.data.text.datasets import LmDataConfig
from levanter.optim.config import OptimizerConfig
from levanter.training_budget import TrainingBudgetConfig
from marin.execution.lazy import ArtifactStep, StepContext
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.grug.moe_hero_ep.hero_recipe import HeroThroughputResult, validated_batch_axis_size
from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe_hero_ep.train import GrugRunConfig, run_grug
from experiments.post_training.long_loop.execution_policy import SFT_TRAINING_DURATION
from experiments.post_training.long_loop.exp3_production import native_sft_data, read_data_release
from experiments.post_training.long_loop.run_spec import FilePin, StorageConfig, read_pin
from experiments.post_training.long_loop.sft_epoch import SftStopping, bind_sft_stopping, step_exposure
from experiments.post_training.long_loop.source_loss import prepared_supervision_data, source_loss_metadata


@dataclass(frozen=True)
class MochiSftSchedule:
    updates: int
    warmup_steps: int
    decay_steps: int


@dataclass(frozen=True)
class MochiSftPacking:
    cap: int = 256
    lookahead: int | None = 128


@dataclass(frozen=True)
class MochiSftControls:
    gpu_count: int
    batch_size: int
    microbatch_size: int | None
    schedule: MochiSftSchedule
    stopping: SftStopping
    epoch_capacity: FilePin
    training_budget: TrainingBudgetConfig
    task_timeout_seconds: int
    packing: MochiSftPacking = field(default_factory=MochiSftPacking)


@dataclass(frozen=True)
class MochiSftInitializer:
    source: str
    destination: str
    copy_manifest: str
    metadata_sha256: str


@dataclass(frozen=True)
class MochiWeightsInitializer:
    source: str
    metadata_sha256: str
    step: int

    @property
    def destination(self) -> str:
        return self.source


@dataclass(frozen=True)
class MochiSftRound:
    schema: int
    format: str
    name: str
    version: str
    slug: str
    cluster: str
    storage: StorageConfig
    hero_reference: FilePin
    data_release: FilePin
    data_seal: FilePin
    initializer: MochiSftInitializer | MochiWeightsInitializer
    sft: MochiSftControls


@dataclass(frozen=True)
class NativeMochiSftRun:
    output: str
    config: GrugRunConfig
    receipt: dict


@dataclass(frozen=True)
class ResolvedMochiSft:
    hero_reference: dict
    prepared_data: dict
    receipt: dict


def run_mochi_sft(run: NativeMochiSftRun) -> None:
    """Run native hero SFT and record actual epoch coverage after permanent finalization."""
    root = StoragePath(run.output)
    (root / "inputs.json").write_text(json.dumps(run.receipt, indent=2))
    run_grug(run.config)
    completed = json.loads((root / "telemetry/completion.json").read_bytes())
    assert run.config.trainer.special_tokens is not None
    updates = completed["final_step"] - run.config.trainer.special_tokens.start_step
    if not 0 < updates <= run.receipt["schedule"]["updates"]:
        raise ValueError("Native SFT completion has no valid number of complete updates")
    metadata = StoragePath(prefix_join(completed["checkpoint"], "metadata.json")).read_bytes()
    if json.loads(metadata)["step"] != completed["final_step"] or json.loads(metadata)["is_temporary"]:
        raise ValueError("Native SFT final checkpoint is not permanently committed")
    data_key, _ = jax.random.split(jax.random.PRNGKey(run.config.trainer.trainer.seed))
    if run.config.trainer.data_seed is not None:
        data_key = jax.random.PRNGKey(run.config.trainer.data_seed)
    exposure = step_exposure(
        run.receipt["exposure_recipe"], run.receipt["epoch"]["components"], updates, data_key=data_key
    )
    (root / "actual-exposure.json").write_text(json.dumps(exposure, indent=2))
    (root / "selected.json").write_text(
        json.dumps(
            {
                **completed,
                "kind": "final_native_sft",
                "completed_updates": updates,
                "metadata_sha256": hashlib.sha256(metadata).hexdigest(),
            },
            indent=2,
        )
    )


def _native_config_values(value):
    # Artifact records encode durations as fingerprint objects; native configs take duration strings.
    if isinstance(value, dict):
        if set(value) == {"days", "seconds", "microseconds"}:
            return f"{timedelta(**value).total_seconds()}s"
        return {key: _native_config_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_native_config_values(item) for item in value]
    return value


def hero_native_config(reference: dict, data: dict) -> dict:
    """Decode recorded hero configuration without interpreting its old data classes."""
    raw = _native_config_values(copy.deepcopy(reference["config"]))
    raw["data"] = copy.deepcopy(data)
    raw["resources"] = {key: value for key, value in raw["resources"].items() if value is not None}
    raw["optimizer"]["type"] = OptimizerConfig.get_choice_name(GrugMoeMuonHConfig)
    precision = raw["trainer"]["trainer"]["mp"]
    if precision != {
        "param_dtype": {"__dtype__": "float32"},
        "compute_dtype": {"__dtype__": "bfloat16"},
        "output_dtype": {"__dtype__": "bfloat16"},
    }:
        raise ValueError("Hero reference no longer uses FP32 master weights and BF16 compute")
    raw["trainer"]["trainer"]["mp"] = "p=f32,c=bfloat16"
    return raw


def bind_mochi_sft_config(
    spec: MochiSftRound, reference: dict, data: dict, *, updates: int, output: str
) -> GrugRunConfig:
    """Keep hero hyperparameters while rebinding data, epoch, fleet and execution clock."""
    raw = hero_native_config(reference, data)
    if data["tokenizer"] != reference["config"]["data"]["tokenizer"]:
        raise ValueError("Sealed mixture and hero SFT use different tokenizers")
    controls = spec.sft
    baseline = reference["config"]
    fresh = isinstance(spec.initializer, MochiWeightsInitializer)
    if controls.gpu_count != 128:
        raise ValueError("This Mochi round requires 128 GB200")
    if fresh:
        if controls.batch_size != baseline["trainer"]["trainer"][
            "train_batch_size"
        ] or controls.microbatch_size != baseline["trainer"].get("microbatch_size"):
            raise ValueError("Fresh-optimizer SFT must preserve the reference batch and accumulation")
    elif controls.batch_size != 176 or controls.microbatch_size != 32:
        raise ValueError("Restored hero SFT requires a logical batch of 176 in 32-wide rounds")
    if controls.training_budget.duration != SFT_TRAINING_DURATION:
        raise ValueError("Mochi SFT must use the canonical 20-hour duration")
    if controls.task_timeout_seconds != int(SFT_TRAINING_DURATION.total_seconds()):
        raise ValueError("SFT dispatch timeout differs from its execution budget")
    start = baseline["trainer"]["special_tokens"]["start_step"]
    horizon = baseline["trainer"]["trainer"]["num_train_steps"] if fresh else baseline["stop_after_steps"]
    hero_updates = horizon - start
    schedule = controls.schedule
    if not 0 < schedule.updates <= updates:
        raise ValueError("Mochi SFT step target must fit within the planned SFT epoch")
    hero_optimizer = baseline["optimizer"]
    expected_warmup = (
        hero_optimizer["warmup"] if fresh else round(hero_optimizer["warmup"] * schedule.updates / hero_updates)
    )
    expected_decay = round(hero_optimizer["decay"] * schedule.updates / hero_updates)
    if (schedule.warmup_steps, schedule.decay_steps) != (expected_warmup, expected_decay):
        raise ValueError("Mochi SFT warmup and decay must scale the hero's SFT-step fractions")
    updates = schedule.updates
    raw["optimizer"].update(warmup=schedule.warmup_steps, decay=schedule.decay_steps)
    raw["stop_after_steps"] = start + updates
    raw["resources"]["replicas"] = controls.gpu_count // raw["resources"]["device"]["count"]
    raw["resources"]["target_cluster"] = None  # The canonical coordinator is already on 08a.
    raw["resources"]["excluded_nvlink_domains"] = ["dh1-r136-us-east-08a"]
    grug = raw["trainer"]
    if not fresh:
        grug["replica_axis_size"] = controls.gpu_count // 64
    grug["microbatch_size"] = controls.microbatch_size
    if not fresh:
        grug["data_seed"] = 0
    grug["expert_placement"]["source_checkpoint"] = spec.initializer.destination
    trainer = grug["trainer"]
    trainer.update(
        id=f"{spec.slug}-{spec.version}",
        train_batch_size=controls.batch_size,
        allow_nondivisible_batch_size=baseline["trainer"]["trainer"]["allow_nondivisible_batch_size"] if fresh else True,
        num_train_steps=start + updates,
        training_budget=draccus.encode(controls.training_budget),
        native_telemetry_root=prefix_join(output, "telemetry"),
    )
    # Mixture indices are relative to this SFT's first update, despite restoring pretraining's step counter.
    raw["data"]["mixture_start_step"] = start
    permanent = prefix_join(output, "checkpoints")
    temporary = prefix_join(spec.storage.temporary_prefix, f"ttl=3d/{spec.slug}/{spec.version}/checkpoints")
    if fresh:
        if trainer["initialize_from"] != spec.initializer.source:
            raise ValueError("Fresh-optimizer SFT initializer differs from its reference")
        # The normal checkpointer roots handle own-run resume; the base is weights-only.
        trainer["load_checkpoint_path"] = None
    else:
        trainer["load_checkpoint_path"] = [permanent, temporary, spec.initializer.destination]
    trainer["checkpointer"].update(base_path=permanent, temporary_base_path=temporary)
    if not fresh:
        trainer["tracker"].update(
            name=trainer["id"],
            id=None,
            group=spec.slug,
            replicate_path=output,
            tags=[tag for tag in trainer["tracker"]["tags"] if tag not in {"full-ab-2026.09.29", "racks-11"}]
            + ["exp3-v5", "racks-2"],
        )
    # Keep the hero's validation corpus separate from the sealed V5 training catalog and its reservations.
    if not fresh:
        old_data = baseline["data"]
        trained = {name for _, weights in old_data["train_weights"] for name, weight in weights.items() if weight > 0}
        raw["evaluation_data"] = _native_config_values(
            {
                **old_data,
                "components": {name: value for name, value in old_data["components"].items() if name not in trained},
                "train_weights": None,
                "mixture_start_step": None,
            }
        )
        raw["eval"]["eval_batch_size"] = controls.microbatch_size
    config = draccus.decode(GrugRunConfig, raw)
    batch_axes = validated_batch_axis_size(
        device_count=config.resources.chip_count(),
        dp_racks=grug["replica_axis_size"],
        batch_size=controls.microbatch_size or controls.batch_size,
        context_axis_size=grug["context_axis_size"],
        expert_axis_size=grug["expert_axis_size"],
    )
    replace(config.trainer.trainer).validate_batch_sizes(batch_axes)
    if config.model != draccus.decode(type(config.model), baseline["model"]):
        raise ValueError("Mochi SFT changed the hero architecture")
    expected_optimizer = {
        **baseline["optimizer"],
        "type": raw["optimizer"]["type"],
        "warmup": schedule.warmup_steps,
        "decay": schedule.decay_steps,
    }
    if config.optimizer != draccus.decode(OptimizerConfig, expected_optimizer):
        raise ValueError("Mochi SFT changed the hero optimizer beyond its scaled schedule")
    return config


def resolve_mochi_sft(spec: MochiSftRound) -> ResolvedMochiSft:
    """Check the release, regional initializer and native one-epoch capacities on CPU."""
    if spec.cluster != "cw-us-east-08a":
        raise ValueError("The Mochi V5 round must run in its data region on cw-us-east-08a")
    reference = {"config": read_pin(spec.hero_reference, Path.cwd())["config"]}
    if not spec.storage.output_prefix.startswith("s3://marin-us-east-08a/"):
        raise ValueError("Mochi SFT outputs must remain in us-east-08a")
    if spec.initializer.source != reference["config"]["trainer"]["expert_placement"]["source_checkpoint"]:
        raise ValueError("Mochi SFT initializer differs from the hero initializer")
    seal = read_pin(spec.data_seal, Path.cwd())
    release = read_data_release(spec.data_seal.path, require_ready=True)
    if release.document != seal or release.pin.sha256 != spec.data_seal.sha256:
        raise ValueError("Sealed V5 release changed")
    if {"uri": spec.data_release.path, "sha256": spec.data_release.sha256} != seal["publisher_manifest"]:
        raise ValueError("V5 publisher manifest is not the sealed publisher")
    read_pin(spec.data_release, Path.cwd())
    initializer_receipt = {}
    if isinstance(spec.initializer, MochiSftInitializer):
        copy_bytes = StoragePath(spec.initializer.copy_manifest).read_bytes()
        copied = json.loads(copy_bytes)
        if copied["source"] != spec.initializer.source or copied["destination"] != spec.initializer.destination:
            raise ValueError("Initializer copy receipt names another checkpoint")
        if copied["total_bytes"] <= 0 or len(copied["files"]) != copied["total_files"]:
            raise ValueError("Initializer copy receipt has no complete payload inventory")
        fs, root = filesystem_for(spec.initializer.destination)
        actual = fs.find(root, detail=True)
        expected = {prefix_join(root, entry["path"]): entry["size"] for entry in copied["files"]}
        if len(expected) != copied["total_files"] or sum(expected.values()) != copied["total_bytes"]:
            raise ValueError("Initializer copy inventory is inconsistent")
        actual.pop(prefix_join(root, ".verified-copy-manifest.json"), None)
        if set(actual) != set(expected) or any(actual[path]["size"] != size for path, size in expected.items()):
            raise ValueError("Staged initializer is missing or changed since verified copy")
        initializer_receipt = {
            "initializer_bytes": copied["total_bytes"],
            "initializer_copy_sha256": hashlib.sha256(copy_bytes).hexdigest(),
        }
    elif not spec.initializer.source.startswith("s3://marin-us-east-08a/"):
        raise ValueError("Weights-only initializer must already be in us-east-08a")
    metadata = StoragePath(prefix_join(spec.initializer.destination, "metadata.json")).read_bytes()
    if hashlib.sha256(metadata).hexdigest() != spec.initializer.metadata_sha256:
        raise ValueError("Staged initializer metadata changed")
    expected_step = (
        spec.initializer.step
        if isinstance(spec.initializer, MochiWeightsInitializer)
        else reference["config"]["trainer"]["special_tokens"]["start_step"]
    )
    if json.loads(metadata)["step"] != expected_step:
        raise ValueError("Initializer step differs from the hero SFT start")
    data = native_sft_data(release.data)
    if isinstance(spec.initializer, MochiWeightsInitializer):
        reference_tokenizer = reference["config"]["data"]["tokenizer"]
        if data["tokenizer"] != reference_tokenizer:
            if (StoragePath(data["tokenizer"]) / "tokenizer.json").read_bytes() != (
                StoragePath(reference_tokenizer) / "tokenizer.json"
            ).read_bytes():
                raise ValueError("Regional reference tokenizer differs from the sealed mixture tokenizer")
            data["tokenizer"] = reference_tokenizer
    recipe = {
        "data": data,
        "seq_len": reference["config"]["model"]["max_seq_len"],
        "batch_size": spec.sft.batch_size,
        "packing_cap": spec.sft.packing.cap,
        "pack_lookahead_window": spec.sft.packing.lookahead,
    }
    bind_sft_stopping(recipe, spec.sft.stopping, spec.sft.epoch_capacity)
    source_pin = release.artifacts["source_loss.json"]
    prepared = prepared_supervision_data(
        draccus.decode(LmDataConfig, recipe["data"]),
        source_loss_metadata(source_pin.uri, source_pin.sha256),
        packing_cap=spec.sft.packing.cap,
        lookahead=spec.sft.packing.lookahead,
    )
    vocabulary = len(prepared.the_tokenizer)
    if vocabulary != reference["config"]["model"]["vocab_size"]:
        raise ValueError("Hero model and sealed mixture tokenizer have different vocabularies")
    data = draccus.encode(prepared)
    receipt = {
        "schema": 1,
        "status": "native_cpu_preflight_pass",
        "round": draccus.encode(spec),
        "epoch": recipe["sft_exposure"],
        "schedule": draccus.encode(spec.sft.schedule),
        **initializer_receipt,
        "initializer_metadata_sha256": hashlib.sha256(metadata).hexdigest(),
        "initializer_mode": "weights_only" if isinstance(spec.initializer, MochiWeightsInitializer) else "full_state",
        "logical_batch": spec.sft.batch_size,
        "physical_batch": (
            math.ceil(spec.sft.batch_size / (spec.sft.microbatch_size or spec.sft.batch_size))
            * (spec.sft.microbatch_size or spec.sft.batch_size)
        ),
        "microbatches": math.ceil(spec.sft.batch_size / (spec.sft.microbatch_size or spec.sft.batch_size)),
        "packing": draccus.encode(spec.sft.packing),
        "vocabulary": vocabulary,
    }
    receipt["exposure_recipe"] = {
        "batch_size": spec.sft.batch_size,
        "data_seed": 0,
        "data": {
            "components": {name: {} for name in data["components"]},
            **{key: data[key] for key in ("train_weights", "mixture_block_size", "mixture_length")},
        },
    }
    return ResolvedMochiSft(reference, data, receipt)


def mochi_sft_step(spec: MochiSftRound, resolved: ResolvedMochiSft) -> ArtifactStep:
    def build_config(context: StepContext) -> NativeMochiSftRun:
        config = bind_mochi_sft_config(
            spec,
            resolved.hero_reference,
            resolved.prepared_data,
            updates=resolved.receipt["epoch"]["updates"],
            output=context.output_path,
        )
        return NativeMochiSftRun(context.output_path, config, resolved.receipt)

    return ArtifactStep(
        name=spec.name,
        version=spec.version,
        artifact_type=HeroThroughputResult,
        run=run_mochi_sft,
        build_config=build_config,
    )
