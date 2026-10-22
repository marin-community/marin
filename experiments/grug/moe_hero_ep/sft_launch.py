# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared SFTSpec/ModelSource binding and bounded current-Hero adapter runtime."""

import dataclasses
import hashlib
import json
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import partial

import draccus
import jax
import jax.numpy as jnp
import levanter.tracker
import numpy as np
from fray.cluster import ResourceConfig
from jax.experimental import multihost_utils
from levanter.data.text.datasets import LmDataConfig
from levanter.data.text.examples import grug_lm_example_from_named
from levanter.grug.attention import token_validity_from_attention_mask
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.lm_model import LmExample
from levanter.schedule import BatchSchedule
from levanter.trainer import TrainerConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from pydantic import BaseModel, Field, TypeAdapter
from rigging.filesystem.conditional_object import conditional_object
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe_hero_ep.model import GrugModelConfig
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import digest
from experiments.grug.moe_hero_ep.sft import (
    BaseReference,
    SftConfig,
    compact_optimizer_report,
    export_sft,
    initialize_from_checkpoint,
    make_sft_gradients,
    make_sft_optimizer_step,
    restore_sft_checkpoint,
    save_sft_checkpoint,
    sft_manifest,
)
from experiments.grug.moe_hero_ep.train import (
    _apply_hero_ep_runtime_defaults,
    build_train_dataset,
    build_train_loader,
    verify_ragged_pjrt,
)
from experiments.sft.launcher import SFTSpec, _trainer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PilotInspection:
    """Original pilot boundaries and the last evaluated segment's actual counters."""

    early_update: int
    non_padding_tokens: int
    policy_digest: str
    budget_digest: str
    reviewed_step: int = 0
    reviewed_tokens: int = 0
    reviewed_inspection_digest: str | None = None
    reviewed_inspection_path: str | None = None
    reviewed_evaluation_id: str | None = None


class PilotStop(StrEnum):
    EARLY_UPDATE = "early_update"
    TOKEN_INSPECTION = "token_inspection"
    UPDATE_CAP = "update_cap"
    TOKEN_CAP = "token_cap"


def pilot_inspection_due(step: int, processed_tokens: int, inspection: PilotInspection) -> PilotStop | None:
    """Pause before another update once an unevaluated original boundary is reached."""
    if inspection.reviewed_step < inspection.early_update <= step:
        return PilotStop.EARLY_UPDATE
    if inspection.reviewed_tokens < inspection.non_padding_tokens <= processed_tokens:
        return PilotStop.TOKEN_INSPECTION
    return None


def sft_stop_reason(
    step: int,
    processed_tokens: int,
    next_batch_tokens: int,
    *,
    num_steps: int,
    max_processed_tokens: int,
    inspection: PilotInspection | None,
) -> PilotStop | None:
    """Apply strict total caps before allowing an unevaluated inspection or update."""
    if step >= num_steps:
        return PilotStop.UPDATE_CAP
    if processed_tokens >= max_processed_tokens or processed_tokens + next_batch_tokens > max_processed_tokens:
        return PilotStop.TOKEN_CAP
    return pilot_inspection_due(step, processed_tokens, inspection) if inspection is not None else None


class HeroSftHandoff(BaseModel):
    native_checkpoint: str
    native_metadata_digest: str
    compact_checkpoint: str
    step: int = Field(ge=1)
    processed_tokens: int = Field(ge=1)
    supervised_tokens: int = Field(ge=1)
    base: BaseReference


class HeroSftInspection(BaseModel):
    handoff: HeroSftHandoff
    compact_metadata_digest: str
    sft: SftConfig
    measurement_identity_digest: str
    inspection: PilotInspection
    max_processed_tokens: int
    stop_reason: PilotStop

    @property
    def status(self) -> str:
        return (
            "awaiting_evaluation"
            if self.stop_reason in (PilotStop.EARLY_UPDATE, PilotStop.TOKEN_INSPECTION)
            else "budget_exhausted"
        )


def read_pilot_inspection(path: str, expected_digest: str) -> HeroSftInspection:
    """Recheck an immutable inspection handoff against both permanent checkpoints."""
    raw = json.loads(StoragePath(path).read_text())
    if digest(raw) != expected_digest:
        raise ValueError("Pilot inspection handoff differs from its pin")
    record = HeroSftInspection.model_validate(raw)
    handoff = record.handoff
    compact = json.loads((StoragePath(handoff.compact_checkpoint) / "metadata.json").read_text())
    native = json.loads((StoragePath(handoff.native_checkpoint) / "metadata.json").read_text())
    manifest = sft_manifest(handoff.base, record.sft)
    counters = {
        "sft_step": handoff.step,
        "processed_tokens": handoff.processed_tokens,
        "supervised_tokens": handoff.supervised_tokens,
    }
    if (
        digest(compact) != record.compact_metadata_digest
        or digest(native) != handoff.native_metadata_digest
        or any(
            metadata.get("is_temporary") is not False
            or metadata.get("step") != handoff.step
            or metadata.get("hero_sft") != json.loads(json.dumps(manifest))
            for metadata in (compact, native)
        )
        or compact.get("hero_sft_counters") != counters
        or native.get("merge_base") != "BF16-rounded candidate C"
        or handoff.processed_tokens > record.max_processed_tokens
        or handoff.step > record.sft.num_steps
    ):
        raise ValueError("Pilot inspection differs from its actual compact/export state or schedule")
    if record.status == "awaiting_evaluation" and (
        handoff.step >= record.sft.num_steps or handoff.processed_tokens >= record.max_processed_tokens
    ):
        raise ValueError("Pilot inspection exhausted its total update/token budget")
    if (
        record.status == "awaiting_evaluation"
        and pilot_inspection_due(handoff.step, handoff.processed_tokens, record.inspection) != record.stop_reason
    ):
        raise ValueError("Pilot inspection did not reach its declared boundary")
    return record


def save_pilot_inspection(output_path: str, record: HeroSftInspection) -> None:
    """Publish one immutable segment handoff; evaluation is a separate bounded job."""
    payload = record.model_dump(mode="json")
    payload["status"] = record.status
    conditional_object(str(StoragePath(output_path) / "inspection.json")).write(
        (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode(), expected_version=None
    )


class HeroSftCheckpoint(Artifact):
    """Completed Hero SFT output, resolved through its persisted final handoff."""

    @property
    def handoff(self) -> HeroSftHandoff:
        return HeroSftHandoff.model_validate_json((StoragePath(self.path) / "handoff.json").read_text())

    @property
    def native_checkpoint(self) -> str:
        return self.handoff.native_checkpoint

    @property
    def metadata_digest(self) -> str:
        return self.handoff.native_metadata_digest

    @property
    def compact_checkpoint(self) -> str:
        return self.handoff.compact_checkpoint


def write_sft_handoff(output_path: str, native_path: str, compact_path: str, state, reference: BaseReference):
    metadata = json.loads((StoragePath(native_path) / "metadata.json").read_text())
    if metadata.get("is_temporary") is not False or metadata["step"] != int(state.step):
        raise ValueError("Final Hero SFT export is incomplete or belongs to another update")
    handoff = HeroSftHandoff(
        native_checkpoint=native_path,
        native_metadata_digest=digest(metadata),
        compact_checkpoint=compact_path,
        step=int(state.step),
        processed_tokens=int(state.processed_tokens),
        supervised_tokens=int(state.supervised_tokens),
        base=reference,
    )
    if jax.process_index() == 0:
        (StoragePath(output_path) / "handoff.json").write_text(handoff.model_dump_json(indent=2) + "\n")
    return handoff


@dataclass(frozen=True)
class HeroSftRunConfig:
    reference: BaseReference
    model: GrugModelConfig
    sft: SftConfig
    data: LmDataConfig
    trainer: TrainerConfig
    resources: ResourceConfig
    output_path: str
    expert_axis_size: int
    context_axis_size: int
    replica_axis_size: int
    max_processed_tokens: int
    save_every_steps: int
    maximum_metrics: tuple[tuple[str, float], ...]
    diagnostic: Callable
    data_manifest_digest: str
    acquisition_panel_digest: str
    resume_checkpoint: str | None = None
    resume_sft_config: SftConfig | None = None
    pilot_inspection: PilotInspection | None = None


def sft_measurement_identity(
    *,
    reference: BaseReference,
    model: GrugModelConfig,
    sft: SftConfig,
    data_manifest_digest: str,
    acquisition_panel_digest: str,
    batch_size: int,
    expert_axis_size: int,
    context_axis_size: int,
    replica_axis_size: int,
    resources: ResourceConfig,
) -> str:
    """Bind smoke measurements to the pilot's weights, data, shapes and resources."""
    parameter_config = dataclasses.asdict(sft)
    for field in ("num_steps", "warmup_steps", "adapter_learning_rate", "vocab_learning_rate"):
        parameter_config.pop(field)
    return digest(
        {
            "base": dataclasses.asdict(reference),
            "model": draccus.encode(model),
            "trainable_configuration": parameter_config,
            "data_manifest_digest": data_manifest_digest,
            "acquisition_panel_digest": acquisition_panel_digest,
            "batch_size": batch_size,
            "expert_axis_size": expert_axis_size,
            "context_axis_size": context_axis_size,
            "replica_axis_size": replica_axis_size,
            "resources": draccus.encode(resources),
            "precision": "BF16 frozen weights, FP32 trainable parameters and routing control",
        }
    )


def _checkpoint_timing(path: str, state, execution_id: str, started: float) -> dict:
    metadata = json.loads((StoragePath(path) / "metadata.json").read_text())
    return {
        "update": int(state.step),
        "checkpoint_seconds": time.perf_counter() - started,
        "checkpoint": path,
        "metadata_digest": digest(metadata),
        "execution_id": execution_id,
    }


class PendingAcquisition(BaseModel):
    core_path: str
    core_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def pending_acquisition(config: HeroSftRunConfig, step: int) -> list[PendingAcquisition]:
    """Recover the queue persisted before the exact resumed compact checkpoint."""
    if config.resume_checkpoint is None:
        return []
    telemetry = StoragePath(config.resume_checkpoint).parent.parent / "telemetry"
    path = telemetry / f"diagnostics-step-{step}.json"
    if not path.exists():
        # Checkpoints written before deferred acquisition have no pending queue.
        return []
    records = TypeAdapter(list[PendingAcquisition]).validate_json(path.read_text())
    if len(records) > config.sft.num_steps:
        raise ValueError("Deferred acquisition queue exceeds the bounded update schedule")
    return records


def complete_acquisition(
    config: HeroSftRunConfig, frozen, final_state, mesh, pending: list[PendingAcquisition]
) -> list[str]:
    """Finalize saved-update diagnostics after the last gradient in this segment.

    The caller must never execute another training gradient after this function:
    ragged training and shorter acquisition executables share NCCL windows.
    """
    completed = []
    for item in pending:
        path = StoragePath(item.core_path)
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != item.core_sha256:
            raise ValueError("Deferred acquisition core telemetry changed")
        core = json.loads(raw)
        if (
            core["kind"] != "hero_sft_core_update"
            or core["checkpoint_metadata_digest"] != config.reference.metadata_digest
            or core["acquisition_panel_digest"] != config.acquisition_panel_digest
            or core["data_manifest_digest"] != config.data_manifest_digest
            or not 0 < core["update"] <= int(final_state.step)
        ):
            raise ValueError("Deferred acquisition changed its base, panel or schedule")
        output = path.parent / f"update-{core['update']:06d}.json"
        if output.exists():
            record = json.loads(output.read_text())
            if (
                record.get("core_update_sha256") != item.core_sha256
                or record.get("kind") != "hero_sft_raw_smoke_update"
                or any(record.get(key) != value for key, value in core.items() if key != "kind")
                or record["acquisition_diagnostic"]["panel_digest"] != config.acquisition_panel_digest
            ):
                raise ValueError("Finalized acquisition report differs from its exact core telemetry")
            completed.append(str(output))
            continue
        if TypeAdapter(SftConfig).validate_python(core["sft_config"]) != config.sft:
            raise ValueError("Unfinished acquisition changed its exact SFT schedule")
        state = restore_sft_checkpoint(core["checkpoint"], final_state, config.reference, config.sft, mesh)
        if (
            int(state.step) != core["update"]
            or int(state.processed_tokens) != core["processed_tokens_total"]
            or int(state.supervised_tokens) != core["supervised_tokens_total"]
        ):
            raise ValueError("Acquisition checkpoint differs from the measured update counters")
        diagnostic = config.diagnostic(frozen, state.params, config.sft, int(state.step))
        if (
            diagnostic["panel_digest"] != config.acquisition_panel_digest
            or diagnostic["cases"] <= 0
            or not 0 <= diagnostic["successes"] <= diagnostic["cases"]
        ):
            raise ValueError("Deferred acquisition differs from the pinned diagnostic panel")
        record = {
            **core,
            "kind": "hero_sft_raw_smoke_update",
            "core_update_sha256": item.core_sha256,
            "diagnostic_successes": diagnostic["successes"],
            "diagnostic_cases": diagnostic["cases"],
            "acquisition_diagnostic": diagnostic,
        }
        if jax.process_index() == 0:
            conditional_object(str(output)).write((json.dumps(record, indent=2) + "\n").encode(), expected_version=None)
        completed.append(str(output))
    return completed


def _run_sft_local(config: HeroSftRunConfig) -> None:
    """Train only compact leaves; a token cap is checked before each update."""
    if config.save_every_steps != 1:
        raise ValueError("Deferred per-update acquisition requires a compact checkpoint after every update")
    trainer = config.trainer
    trainer.initialize()
    levanter.tracker.log_configuration(config)
    if config.model.moe_implementation == "ragged_all_to_all":
        verify_ragged_pjrt()
    mesh = compact_grug_mesh(
        expert_axis_size=config.expert_axis_size,
        context_axis_size=config.context_axis_size,
        replica_axis_size=config.replica_axis_size,
    )
    with jax.set_mesh(mesh):
        frozen, state = initialize_from_checkpoint(config.reference, config.model, mesh, config.sft)
        if config.resume_checkpoint is not None:
            previous = config.resume_sft_config
            if (
                previous is None
                or dataclasses.replace(
                    config.sft,
                    num_steps=previous.num_steps,
                    warmup_steps=previous.warmup_steps,
                    adapter_learning_rate=previous.adapter_learning_rate,
                    vocab_learning_rate=previous.vocab_learning_rate,
                )
                != previous
            ):
                raise ValueError("Resume changes the compact parameters or optimizer beyond the reviewed schedule")
            state = restore_sft_checkpoint(config.resume_checkpoint, state, config.reference, previous, mesh)
        elif config.resume_sft_config is not None:
            raise ValueError("A prior SFT configuration requires an explicit compact resume checkpoint")
        if config.pilot_inspection is not None and config.pilot_inspection.reviewed_evaluation_id is not None:
            inspection = config.pilot_inspection
            if inspection.reviewed_inspection_path is None or inspection.reviewed_inspection_digest is None:
                raise ValueError("Evaluated pilot continuation lacks its immutable inspection pin")
            reviewed = read_pilot_inspection(inspection.reviewed_inspection_path, inspection.reviewed_inspection_digest)
            if (
                reviewed.handoff.compact_checkpoint != config.resume_checkpoint
                or reviewed.sft != config.sft
                or int(state.step) != inspection.reviewed_step
                or int(state.processed_tokens) != inspection.reviewed_tokens
                or int(state.supervised_tokens) != reviewed.handoff.supervised_tokens
            ):
                raise ValueError("Restored pilot state differs from its evaluated inspection and data cursor")
        pending = pending_acquisition(config, int(state.step))
        gradient_step = make_sft_gradients(config.sft)
        optimizer_step = make_sft_optimizer_step(config.sft)
        compiled_gradient = None
        compiled_optimizer = None
        compile_seconds = None
        input_batches_digest = digest(
            {"data": draccus.encode(config.data), "seed": config.sft.seed, "batch_size": trainer.train_batch_size}
        )
        measurement_identity = sft_measurement_identity(
            reference=config.reference,
            model=config.model,
            sft=config.sft,
            data_manifest_digest=config.data_manifest_digest,
            acquisition_panel_digest=config.acquisition_panel_digest,
            batch_size=trainer.train_batch_size,
            expert_axis_size=config.expert_axis_size,
            context_axis_size=config.context_axis_size,
            replica_axis_size=config.replica_axis_size,
            resources=config.resources,
        )
        optimizer_audit = compact_optimizer_report(state)
        optimizer_audit.update(
            execution_id=trainer.id,
            checkpoint_metadata_digest=config.reference.metadata_digest,
            measurement_identity_digest=measurement_identity,
        )
        state_bytes = optimizer_audit["optimizer_state_bytes"]
        trainable_bytes = optimizer_audit["trainable_parameter_bytes"]
        # Every segment owns a compact checkpoint with its full current schedule,
        # including a reused smoke that already reached the first inspection.
        last_saved_step = None
        last_saved_checkpoint = None
        stop_reason = None
        telemetry_path = StoragePath(config.output_path) / "telemetry"
        if jax.process_index() == 0:
            telemetry_path.mkdirs()
            (telemetry_path / "compact-optimizer-report.json").write_text(json.dumps(optimizer_audit, indent=2) + "\n")
        if not optimizer_audit["compact_optimizer_only"]:
            raise RuntimeError(f"SFT optimizer violates the compact state boundary: {optimizer_audit['errors']}")
        schedule = BatchSchedule(trainer.train_batch_size)
        dataset = build_train_dataset(
            config.data,
            max_seq_len=config.model.max_seq_len,
            batch_schedule=schedule,
            key=jax.random.PRNGKey(config.sft.seed),
        )
        loader = build_train_loader(dataset, batch_schedule=schedule, mesh=mesh)
        stop = partial(
            sft_stop_reason,
            num_steps=config.sft.num_steps,
            max_processed_tokens=config.max_processed_tokens,
            inspection=config.pilot_inspection,
        )
        stop_reason = stop(int(state.step), int(state.processed_tokens), 0)
        batches = () if stop_reason is not None else loader.iter_from_step(int(state.step))
        for batch in batches:
            if isinstance(batch, LmExample):
                batch = grug_lm_example_from_named(batch)
            valid = token_validity_from_attention_mask(
                batch.attn_mask, batch_size=batch.tokens.shape[0], sequence_length=batch.tokens.shape[1]
            )
            batch_tokens = int(jnp.sum(valid))
            stop_reason = stop(int(state.step), int(state.processed_tokens), batch_tokens)
            if stop_reason is not None:
                break
            if compiled_gradient is None:
                compile_start = time.perf_counter()
                args = (state.params, frozen, batch.tokens, batch.loss_weight, valid, batch.attn_mask)
                compiled_gradient = gradient_step.lower(*args).compile()
                gradient_shape, metrics_shape = jax.eval_shape(gradient_step, *args)
                compiled_optimizer = optimizer_step.lower(
                    state, gradient_shape, metrics_shape["processed_tokens"], metrics_shape["supervised_tokens"]
                ).compile()
                compile_seconds = time.perf_counter() - compile_start
                # Record the actual compact gradient interface without parsing
                # the executable: a general HLO absence certificate is not a
                # runtime acceptance gate for this bounded learning experiment.
                report = {
                    "schema_version": 1,
                    "kind": "hero_sft_gradient_structure",
                    "execution_id": trainer.id,
                    "checkpoint_metadata_digest": config.reference.metadata_digest,
                    "measurement_identity_digest": measurement_identity,
                    "general_compiler_certificate_required": False,
                    "scope": "compiled gradient output tree; no claim about internal kernel allocations",
                    "gradients": [
                        {"path": jax.tree_util.keystr(path), "shape": value.shape, "dtype": str(value.dtype)}
                        for path, value in jax.tree_util.tree_flatten_with_path(gradient_shape)[0]
                    ],
                }
                if jax.process_index() == 0:
                    (telemetry_path / "gradient-structure-report.json").write_text(json.dumps(report, indent=2) + "\n")
            started = time.perf_counter()
            grads, metrics = compiled_gradient(
                state.params, frozen, batch.tokens, batch.loss_weight, valid, batch.attn_mask
            )
            jax.block_until_ready((grads, metrics))
            optimizer_started = time.perf_counter()
            state = compiled_optimizer(state, grads, metrics["processed_tokens"], metrics["supervised_tokens"])
            jax.block_until_ready(state)
            optimizer_seconds = time.perf_counter() - optimizer_started
            duration = time.perf_counter() - started
            if not bool(jnp.isfinite(metrics["loss"])) or not bool(jnp.isfinite(metrics["gradient_norm"])):
                raise RuntimeError(f"Nonfinite SFT loss or gradient at step {int(state.step)}")
            activation_gradient_norm = float(jnp.linalg.norm(grads.input_delta))
            memory = [device.memory_stats() for device in jax.local_devices()]
            local_peaks = [
                item["peak_bytes_in_use"] for item in memory if item is not None and "peak_bytes_in_use" in item
            ]
            local_peak = max(local_peaks, default=0)
            # Collect exact byte counts as two uint32 words: JAX commonly disables
            # int64, while a rack GPU's HBM count exceeds signed int32.
            memory_samples = np.asarray(
                multihost_utils.process_allgather(
                    np.array([len(local_peaks), len(memory), local_peak >> 32, local_peak & 0xFFFFFFFF], dtype=np.uint32)
                )
            ).reshape(-1, 4)
            memory_bytes = (
                max((int(row[2]) << 32) | int(row[3]) for row in memory_samples)
                if np.all(memory_samples[:, 0] == memory_samples[:, 1])
                else None
            )
            for metric, maximum in config.maximum_metrics:
                if metric not in metrics:
                    raise RuntimeError(f"SFT is missing its predeclared diagnostic metric: {metric}")
                value = float(jnp.max(metrics[metric]))
                if not math.isfinite(value) or value > maximum:
                    raise RuntimeError(f"SFT crossed predeclared diagnostic limit: {metric} <= {maximum}")
            if jax.process_index() == 0:
                record = {
                    "schema_version": 1,
                    "kind": "hero_sft_core_update",
                    "execution_id": trainer.id,
                    "checkpoint_metadata_digest": config.reference.metadata_digest,
                    "measurement_identity_digest": measurement_identity,
                    "sft_config": dataclasses.asdict(config.sft),
                    "configuration_digest": digest(
                        {
                            "sft": dataclasses.asdict(config.sft),
                            "model": draccus.encode(config.model),
                            "base": dataclasses.asdict(config.reference),
                            "batch_size": trainer.train_batch_size,
                            "expert_axis_size": config.expert_axis_size,
                            "context_axis_size": config.context_axis_size,
                            "replica_axis_size": config.replica_axis_size,
                            "diagnostic_panel_digest": config.acquisition_panel_digest,
                        }
                    ),
                    "update": int(state.step),
                    "elapsed_seconds": duration,
                    "compile_seconds": compile_seconds,
                    "executed_tokens": batch.tokens.size,
                    "non_padding_tokens": batch_tokens,
                    "supervised_tokens": int(metrics["supervised_tokens"]),
                    "peak_hbm_bytes": memory_bytes,
                    "optimizer_seconds": optimizer_seconds,
                    "loss": float(metrics["loss"]),
                    "gradient_norm": float(metrics["gradient_norm"]),
                    "acquisition_panel_digest": config.acquisition_panel_digest,
                    "data_manifest_digest": config.data_manifest_digest,
                    "checkpoint": f"{config.output_path}/adapters/step-{int(state.step)}",
                    "processed_tokens_total": int(state.processed_tokens),
                    "supervised_tokens_total": int(state.supervised_tokens),
                    "routing_metrics": {
                        name: float(value)
                        for name, value in metrics.items()
                        if name.startswith("train/router/") and isinstance(value, jax.Array) and value.ndim == 0
                    },
                    "input_batches_digest": input_batches_digest,
                    "input_batches_digest_kind": "pinned data config, data seed, and global batch",
                    "input_delta_gradient_norm": activation_gradient_norm,
                    "preserves_activation_gradients": activation_gradient_norm > 0,
                    "trainable_parameter_bytes": trainable_bytes,
                    "optimizer_state_bytes": state_bytes,
                    "gpu_count": jax.device_count(),
                    "hardware": str(jax.devices()[0].device_kind),
                    "sequence_length": config.model.max_seq_len,
                    "missing_smoke_measurements": [] if memory_bytes is not None else ["peak_hbm_bytes"],
                }
                core_path = telemetry_path / f"core-update-{int(state.step):06d}.json"
                core_raw = (json.dumps(record, indent=2) + "\n").encode()
                conditional_object(str(core_path)).write(core_raw, expected_version=None)
                pending.append(
                    PendingAcquisition(core_path=str(core_path), core_sha256=hashlib.sha256(core_raw).hexdigest())
                )
                conditional_object(str(telemetry_path / f"diagnostics-step-{int(state.step)}.json")).write(
                    TypeAdapter(list[PendingAcquisition]).dump_json(pending) + b"\n", expected_version=None
                )
            levanter.tracker.log(
                {
                    **metrics,
                    "sft/step_duration": duration,
                    "sft/optimizer_seconds": optimizer_seconds,
                    "sft/processed_tokens_total": int(state.processed_tokens),
                    "sft/supervised_tokens_total": int(state.supervised_tokens),
                    "sft/nonpadding_tokens_per_second": batch_tokens / duration,
                    "sft/executed_tokens_per_second": batch.tokens.size / duration,
                    **({"sft/peak_hbm_bytes": memory_bytes} if memory_bytes is not None else {}),
                },
                step=int(state.step),
            )
            if int(state.step) % config.save_every_steps == 0:
                checkpoint_start = time.perf_counter()
                save_sft_checkpoint(
                    f"{config.output_path}/adapters/step-{int(state.step)}", state, config.reference, config.sft
                )
                last_saved_step = int(state.step)
                last_saved_checkpoint = f"{config.output_path}/adapters/step-{int(state.step)}"
                if jax.process_index() == 0:
                    checkpoint_record = _checkpoint_timing(last_saved_checkpoint, state, trainer.id, checkpoint_start)
                    (telemetry_path / f"checkpoint-{int(state.step):06d}.json").write_text(
                        json.dumps(checkpoint_record, indent=2) + "\n"
                    )
            stop_reason = stop(int(state.step), int(state.processed_tokens), 0)
            if stop_reason is not None:
                break
        if stop_reason is None:
            raise RuntimeError("Training data exhausted before a declared inspection or total budget boundary")
        if int(state.step) == 0:
            raise ValueError("Token cap permits no updates; choose a smaller batch or revise the budget before launch")
        if last_saved_step != int(state.step):
            if jax.process_index() == 0:
                conditional_object(str(telemetry_path / f"diagnostics-step-{int(state.step)}.json")).write(
                    TypeAdapter(list[PendingAcquisition]).dump_json(pending) + b"\n", expected_version=None
                )
            last_saved_checkpoint = f"{config.output_path}/adapters/step-{int(state.step)}"
            checkpoint_start = time.perf_counter()
            save_sft_checkpoint(last_saved_checkpoint, state, config.reference, config.sft)
            if jax.process_index() == 0:
                (telemetry_path / f"checkpoint-{int(state.step):06d}.json").write_text(
                    json.dumps(
                        _checkpoint_timing(last_saved_checkpoint, state, trainer.id, checkpoint_start),
                        indent=2,
                    )
                    + "\n"
                )
        # The leader published this queue before the completed checkpoint; every
        # rank uses those same bytes for the collective acquisition executions.
        pending = TypeAdapter(list[PendingAcquisition]).validate_json(
            (telemetry_path / f"diagnostics-step-{int(state.step)}.json").read_text()
        )
        complete_acquisition(config, frozen, state, mesh, pending)
        native_path = f"{config.output_path}/export/step-{int(state.step)}"
        export_sft(native_path, frozen, state, config.reference, config.sft)
        assert last_saved_checkpoint is not None
        handoff = write_sft_handoff(config.output_path, native_path, last_saved_checkpoint, state, config.reference)
        if config.pilot_inspection is not None and jax.process_index() == 0:
            compact = json.loads((StoragePath(last_saved_checkpoint) / "metadata.json").read_text())
            inspection = HeroSftInspection(
                handoff=handoff,
                compact_metadata_digest=digest(compact),
                sft=config.sft,
                measurement_identity_digest=measurement_identity,
                inspection=config.pilot_inspection,
                max_processed_tokens=config.max_processed_tokens,
                stop_reason=stop_reason,
            )
            save_pilot_inspection(config.output_path, inspection)
    levanter.tracker.current_tracker().finish()


def run_hero_sft(config: HeroSftRunConfig) -> None:
    """Dispatch a bounded adapter run only when explicitly invoked by a launch."""
    if config.max_processed_tokens < 1 or config.save_every_steps < 1:
        raise ValueError("Declare a positive nonpadding token cap and checkpoint cadence before launch")
    if config.diagnostic is None:
        raise ValueError("Pin a fixed acquisition diagnostic callback before launching a smoke or pilot")
    if not config.maximum_metrics:
        raise ValueError("Pin baseline-derived loss/routing maximums before launching a smoke or pilot")
    if any(not math.isfinite(maximum) for _, maximum in config.maximum_metrics):
        raise ValueError("Diagnostic maximums must be finite")
    if not config.data_manifest_digest or not config.acquisition_panel_digest:
        raise ValueError("Pin training data and the acquisition panel before collecting smoke measurements")
    if not config.output_path or config.trainer.id is None:
        raise ValueError("Pin a durable artifact path and run ID before loading model arrays")
    _apply_hero_ep_runtime_defaults(
        inline_watch_enabled=False,
        processes_per_task=1,
        moe_implementation=config.model.moe_implementation,
        remat_mode=config.model.remat_mode,
    )
    dispatch_grug_training_run(
        run_id=config.trainer.id,
        config=config,
        local_entrypoint=_run_sft_local,
        resources=config.resources,
        processes_per_task=1,
        max_retries_failure=0,
        max_task_failures=0,
    )


@dataclass(frozen=True)
class HeroSftModel:
    """Current Hero ModelSource; shared SFT rendering/packing stays in SFTSpec."""

    reference: BaseReference
    model: GrugModelConfig
    sft: SftConfig
    expert_axis_size: int
    context_axis_size: int
    replica_axis_size: int
    max_processed_tokens: int
    save_every_steps: int
    maximum_metrics: tuple[tuple[str, float], ...]
    diagnostic: Callable
    data_manifest_digest: str
    acquisition_panel_digest: str
    resume_checkpoint: str | None = None
    resume_sft_config: SftConfig | None = None
    pilot_inspection: PilotInspection | None = None

    def tokenizer_cache_key(self) -> str:
        return f"{self.reference.tokenizer}@{self.reference.tokenizer_revision}"

    def resolve_tokenizer(self, ctx: StepContext) -> str:
        return self.tokenizer_cache_key()

    @property
    def run(self) -> Callable[..., None]:
        return run_hero_sft

    def init_deps(self) -> tuple[ArtifactStep, ...]:
        return ()

    def build_train_config(
        self, ctx: StepContext, spec: SFTSpec, data_config: LmDataConfig, resources: ResourceConfig, num_train_steps: int
    ) -> HeroSftRunConfig:
        if spec.seq_len != self.model.max_seq_len:
            raise ValueError("SFT must preserve the pinned current-Hero forward context length")
        if num_train_steps != self.sft.num_steps:
            raise ValueError("Pin the SFT optimizer schedule to the resolved number of updates")
        trainer = _trainer(spec, num_train_steps=num_train_steps, gpu_allocator=True, use_explicit_mesh_axes=True)
        trainer = dataclasses.replace(trainer, id=spec.name.split("/")[-1], load_checkpoint=False)
        # _trainer's Levanter checkpointer is not used: compact checkpoints live in this artifact.
        return HeroSftRunConfig(
            self.reference,
            self.model,
            self.sft,
            data_config,
            trainer,
            resources,
            ctx.output_path,
            self.expert_axis_size,
            self.context_axis_size,
            self.replica_axis_size,
            self.max_processed_tokens,
            self.save_every_steps,
            self.maximum_metrics,
            self.diagnostic,
            self.data_manifest_digest,
            self.acquisition_panel_digest,
            self.resume_checkpoint,
            self.resume_sft_config,
            self.pilot_inspection,
        )
