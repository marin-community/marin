# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind a fresh #310 identity SFT run to an explicit 16- or 64-update horizon, with no retries.

``sft_worker`` runs on every replica. It builds the source-derived recipe with the measured router policy, replaces
the factory's data with the frozen mixture (``mixture-seed310-v1``, data config checked by hash) and calls
``run_identity_sft``. That function relies on a passing LR0 run for the precision gate and re-runs only the cheap
binds:

- source bind: the six pinned metadata hashes, every tensor header, and the tensor names and shapes of the LR0
  source receipt (whose 502 tensor hashes the full LR0 source audit wrote) -> ``audits/source-bind.json``
- derived recipe == the reviewed runtime receipt; loaded model config == the receipt
- full master-step-0 audit against the LR0 source receipt -> ``audits/master-step-0.json``
- router-dot witness on the compute view: f32 operands and output
- router biases equal the loaded source before and after training (they are frozen)

Then the dataset audit reads the same iterator the loop consumes (64 updates, the frozen 512 sequences) before any
update; each update checks its consumed batch against the frozen order immediately before its
``train_step``. The loss uses query segment IDs for dispatch validity and reports unique dropped assignments from
that same forward. Missing metrics or nonzero drops stop the run before the next update. Counts are gathered as
global per-layer totals, without summing replicated copies. At the configured horizon the native checkpoint is
committed first, then
the BF16 + FP32-router-bias HF export and its save audit. Process 0 writes ``sft-result.json`` last. The milestone-16
spending gate is evaluated afterwards on the export, not here.

``main`` prints the job plan; with ``--run`` it submits. ``--run`` needs the passing LR0 run root (its ``gate.json``
with no failures, four ``updated`` rank receipts and ``lr0-result.json``), the capacity record and the spending gate
file; it writes ``--launch-receipt`` before the submit and again with the Iris job id after it.
"""

import argparse
import datetime
import hashlib
import json
import subprocess
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path

import draccus
import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
from iris.cluster.types import Entrypoint
from levanter.compat.hf_export import run_on_export_writer, write_export_json
from levanter.compute_dtype import cast_model_for_compute
from levanter.data.text.datasets import LmDataConfig
from levanter.main.train_lm import TrainLmConfig, preserve_hf_runtime_settings
from levanter.models.lm_model import LmExample
from levanter.models.snowball import GrugMoeHfConfig, SnowballConfig, snowball_loss_with_dispatch_stats
from levanter.tracker import NoopConfig, NoopTracker, current_tracker
from levanter.trainer import Trainer, initialize
from levanter.trainer_state import TrainerState
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.interp.snowball_identity.levanter_positions import router_dot_dtypes, router_dot_is_fp32
from experiments.interp.snowball_identity.multihost import gather_global, gather_host_rows, gather_host_scalar
from experiments.interp.snowball_identity.precision_gate import (
    device_memory_report,
    dispatch_failures,
    local_device_memory,
)
from experiments.post_training.identity_context_distill.dataset_audit import (
    ACCOUNTING_MODULE_SHA256,
    FREEZE_SHA256,
    INVENTORY_SHA256,
    audit_training_iterator,
    checked_update,
    load_dataset_audit_inputs,
)
from experiments.post_training.identity_context_distill.lr0_probe import (
    MAX_TIMEOUT_SECONDS,
    PRIORITY_BANDS,
    REPLICAS,
    RESOURCES,
    SEED,
    SOURCE_FILE_HASHES,
    T310,
    approved_receipt,
    check_pod_storage,
    mesh_layout,
    submit_entrypoint,
)
from experiments.post_training.identity_context_distill.precision_audit import audit_initialized_masters
from experiments.post_training.identity_context_distill.precision_probe import router_values
from experiments.post_training.identity_context_distill.precision_source_audit import (
    INDEX,
    SOURCE_FILES,
    check_source_metadata,
    checkpoint_records,
)
from experiments.post_training.identity_context_distill.runtime_recipe import (
    check_precision_runtime_receipt,
    check_xla_precision,
    configure_precision_runtime,
    require_xla_precision,
    xla_precision_env,
    xla_precision_in_process,
)
from experiments.post_training.identity_context_distill.save_audit import audit_trained_export
from experiments.post_training.identity_context_distill.sft import (
    BASE_MODEL,
    check_export_metadata,
    check_restored_progress,
    check_runtime_recipe,
    identity_sft_config,
    source_export_overrides,
)

MIXTURE_ROOT = f"{T310}/mixture-seed310-v1"
DATA_CONFIG_SHA256 = "91a6423d0f7b708cf0c9846bf04d90fed46b6242705b6089038594ba6aa044df"
CACHE_CATALOG_SHA256 = "cfe8af5fbd4324db5b465dcd02be9416eb4fe61cd5d5419cc3227a682eb580a4"
INVENTORY_PATH = f"{T310}/mixture-seed310-v1-supplement-v1/native-token-inventory.jsonl"
"""The native token inventory of the 767 frozen rows that the dataset audit counts against."""
ACCOUNTING_TEST = "tests/training/test_chat_token_accounting.py"
SPENDING_GATE_SHA256 = "d854803cee2f53f77ffa45b7850cb060ad1c6cfd1fb6c128a1674c3c37b518ad"
"""milestone16-spending-gate-v1.json: evaluated on the step-16 export after the job, not inside it."""
TRAINING_HORIZONS = (16, 64)
FRESH64_DECISION_SHA256 = "a36790d35432902ea3bf34961c4c8d997514120938a330d0cf49b0302da61ac4"
FRESH64_SUPPLEMENT_SHA256 = "47878cd192adc37d186994d370ecbf6f6c10fb20323e7e115013297f0423dda2"
R3_RESULT_SHA256 = "7133c366bb04818378a2d38f6636c3f3aa9e2cb06c74d03932cf90b4cd20c293"
R3_RESULT_PATH = f"{T310}/sft-m16-r3/sft-result.json"
AUDITED_UPDATES = 64
"""The dataset audit pre-reads the frozen 512 sequences: 64 updates of 8."""
SOURCE_SHARDS = 39
ROUTER_BIAS_DTYPES = {f"model.layers.{layer}.mlp.router.bias": "float32" for layer in range(26)}
RUNTIME_RECEIPT_SHA256S = frozenset(
    {
        "565cf0b2e09b0e1d58b647b698364a04d41b80615b47f8e6e1dbc7167f0d8e9e",
    }
)
DISPATCH_METRICS = ("moe_dropped", "moe_sender_dropped", "moe_receiver_dropped")


def sft_runtime_receipt(receipt_bytes: bytes, receipt_sha256: str) -> dict:
    """Require the reviewed schema-6 arm-(a) forward."""
    if receipt_sha256 not in RUNTIME_RECEIPT_SHA256S:
        raise ValueError("SFT requires a reviewed schema-6 runtime receipt")
    receipt = approved_receipt(receipt_bytes, receipt_sha256)
    model = receipt["runtime_model_config"]
    if (
        receipt["schema_version"] != 6
        or model["moe_capacity_factor"] != 4.0
        or model["router_combine_fp32"] is not True
        or model["fp32_activations"] is not True
    ):
        raise ValueError("SFT requires dropless dispatch, FP32 combine and FP32 activations")
    if (model["attention_implementation"], receipt["xla_precision"]) != ("torch_order", "default"):
        raise ValueError("SFT attention and XLA precision must match arm (a)")
    return receipt


class DispatchMetrics(NoopTracker):
    """Read the actual train_step loss metrics through the public tracker boundary."""

    def __init__(self):
        self.values = {}

    def log(self, metrics, *, step, commit=None):
        for name in DISPATCH_METRICS:
            key = f"train/{name}"
            if key in metrics:
                self.values[name] = metrics[key]

    def checked_counts(self) -> dict:
        missing = set(self.values) != set(DISPATCH_METRICS)
        if np.any(gather_host_scalar(int(missing))):
            raise ValueError("Training forward did not report all dispatch metrics")
        counts = {name: gather_global(self.values[name]) for name in DISPATCH_METRICS}
        if any(
            value.shape != (26,)
            or not np.all(np.isfinite(value))
            or np.any(value < 0)
            or np.any(value != np.floor(value))
            for value in counts.values()
        ):
            raise ValueError("Invalid per-layer dispatch metrics")
        if not np.array_equal(counts["moe_dropped"], counts["moe_sender_dropped"] + counts["moe_receiver_dropped"]):
            raise ValueError("Inconsistent unique-assignment dispatch counts")
        failures = dispatch_failures({name.removeprefix("moe_"): value for name, value in counts.items()})
        if failures:
            raise ValueError(f"Training forward dropped valid expert assignments: {failures}")
        return {name: value.astype(np.int64).tolist() for name, value in counts.items()}


@dataclass(frozen=True)
class SftAudits:
    """The trainer-side audits, each called on every rank and raising on every rank on failure."""

    audit_iterator: Callable[[Iterator[LmExample], TrainLmConfig], tuple[Iterator[LmExample], dict]]
    """Pre-reads the first 64 batches of the loop's own iterator, checks them against the frozen mixture, and
    returns an iterator over the same batch objects followed by the rest, plus its report."""
    check_batch: Callable[[LmExample, int], dict]
    """Checks one consumed batch against the frozen order for that update; returns its per-update receipt."""
    save_audit: Callable[[TrainerState, str], None]
    """Checks the written HF export against the in-memory masters and the frozen source router biases."""


@dataclass(frozen=True)
class SftRun:
    """Explicit fresh-run horizon and the reviewed evidence required beyond the milestone."""

    stop_after_updates: int
    reference_result_bytes: bytes | None
    decision_bytes: bytes | None
    supplement_bytes: bytes | None


def checked_run_reference(run: SftRun) -> dict | None:
    """Validate authorization bytes without touching JAX or loading model/data payloads."""
    if run.stop_after_updates not in TRAINING_HORIZONS:
        raise ValueError("The fresh SFT horizon must be 16 or 64")
    if run.stop_after_updates == 16:
        if any(value is not None for value in (run.reference_result_bytes, run.decision_bytes, run.supplement_bytes)):
            raise ValueError("The 16-update run does not consume 64-update authorization")
        return None
    if run.reference_result_bytes is None or run.decision_bytes is None or run.supplement_bytes is None:
        raise ValueError("64 updates requires the r3 result, owner decision and supplement 2")
    if sha256(run.decision_bytes) != FRESH64_DECISION_SHA256:
        raise ValueError("64-update owner decision differs from the reviewed record")
    if sha256(run.supplement_bytes) != FRESH64_SUPPLEMENT_SHA256:
        raise ValueError("64-update decision supplement differs from the reviewed record")
    if sha256(run.reference_result_bytes) != R3_RESULT_SHA256:
        raise ValueError("64-update reference differs from the reviewed r3 result")
    return json.loads(run.reference_result_bytes)


def run_provenance(run: SftRun) -> dict:
    return {
        "initialization": "fresh_hf",
        "initial_completed_updates": 0,
        "stop_after_updates": run.stop_after_updates,
        "reference_sft_result": (
            {"path": R3_RESULT_PATH, "sha256": R3_RESULT_SHA256} if run.reference_result_bytes is not None else None
        ),
        "owner_decision_sha256": sha256(run.decision_bytes) if run.decision_bytes is not None else None,
        "owner_decision_supplement_sha256": sha256(run.supplement_bytes) if run.supplement_bytes is not None else None,
    }


def check_reference_batch(receipt: dict, update: int, reference: dict) -> None:
    expected = reference["dataset_audit"]["updates"][update]
    if any(receipt[name] != expected[name] for name in ("ids", "supervised_tokens")):
        raise ValueError(f"Update {update} differs from r3's frozen ids or supervised tokens")
    if update < 16:
        consumed = reference["updates"][update]
        if any(receipt[name] != consumed[name] for name in ("ids", "supervised_tokens")):
            raise ValueError(f"Update {update} differs from r3's consumed ids or supervised tokens")


def frozen_data_config(data_config_bytes: bytes) -> LmDataConfig:
    if hashlib.sha256(data_config_bytes).hexdigest() != DATA_CONFIG_SHA256:
        raise ValueError("Data config differs from the frozen mixture's data-config.json")
    data = draccus.decode(LmDataConfig, json.loads(data_config_bytes))
    for name, component in data.components.items():
        if component.pack is not False or component.format.pack is not False:
            raise ValueError(f"Frozen component {name} must be unpacked")
        if component.format.slice_strategy != "raise" or not component.format.mask_user_turns:
            raise ValueError(f"Frozen component {name} must raise on overflow and mask user turns")
    return data


def sft_config(
    run_root: str,
    run_id: str,
    config_bytes: bytes,
    generation_bytes: bytes,
    data_config_bytes: bytes,
    runtime_receipt: dict,
    run: SftRun,
) -> TrainLmConfig:
    """Bind a fresh factory recipe to its explicit horizon and frozen mixture."""
    checked_run_reference(run)
    data = frozen_data_config(data_config_bytes)
    identity = data.components["identity"]
    base = identity_sft_config(
        dataset_uri=identity.source.train_urls[0],
        cache_uri=data.cache_dir,
        output_uri=run_root,
        run_id=run_id,
        seed=SEED,
        source_config=json.loads(config_bytes),
        source_generation_config=json.loads(generation_bytes),
        hosts=REPLICAS,
        moe_capacity_factor=4.0,
        router_combine_fp32=True,
        fp32_activations=True,
        attention_implementation=runtime_receipt["runtime_model_config"]["attention_implementation"],
        stop_after_updates=run.stop_after_updates,
    )
    return replace(base, data=data)


def source_bind(lr0_source_receipt: dict, source_objects: dict) -> dict:
    """Writer only: every model object (the six metadata files and every index shard) has the size and ETag listed
    around the LR0 audit; the pinned metadata and headers hold; the tensor names and shapes are the ones the LR0
    audit hashed. Other files under the source root (README, evals) are not model inputs and are not compared."""
    if source_objects["source_index_sha256"] != SOURCE_FILE_HASHES[INDEX]:
        raise ValueError("Source object list was taken against a different index")
    check_source_metadata(BASE_MODEL, SOURCE_FILE_HASHES)
    records = checkpoint_records(BASE_MODEL)
    expected = {name: tuple(entry["shape"]) for name, entry in lr0_source_receipt["tensors"].items()}
    if {name: tuple(record.shape) for name, record in records.items()} != expected:
        raise ValueError("Source tensor names or shapes differ from the LR0 source receipt")
    shards = {record.file_path.rsplit("/", 1)[-1] for record in records.values()}
    if len(shards) != SOURCE_SHARDS:
        raise ValueError(f"Source index names {len(shards)} shards, not {SOURCE_SHARDS}")
    model_objects = set(SOURCE_FILES) | shards
    if set(source_objects["objects"]) != model_objects:
        raise ValueError("Source object list does not cover exactly the metadata files and index shards")
    fs, root = url_to_fs(BASE_MODEL)
    changed = []
    for name, listed in sorted(source_objects["objects"].items()):
        info = fs.info(f"{root.rstrip('/')}/{name}")
        if (info["size"], info.get("ETag")) != (listed["bytes"], listed["etag"]):
            changed.append(name)
    if changed:
        raise ValueError(f"Source objects differ from the LR0 object list: {changed}")
    return {"metadata_sha256": SOURCE_FILE_HASHES, "objects": len(model_objects), "tensor_count": len(records)}


def train_checked_updates(
    trainer: Trainer,
    state: TrainerState,
    batches: Iterator[LmExample],
    check_batch: Callable[[LmExample, int], dict],
    count: int,
    *,
    reference: dict | None,
) -> tuple[TrainerState, list[dict]]:
    """Run ``count`` updates, checking each consumed batch immediately before its update."""
    first = int(state.step)
    updates = []
    for update in range(first, first + count):
        batch = next(batches)
        receipt = check_batch(batch, update)
        if reference is not None:
            check_reference_batch(receipt, update, reference)
        dispatch = DispatchMetrics()
        with current_tracker(dispatch):
            info = trainer.train_step(state, batch)
        counts = dispatch.checked_counts()
        state = info.state
        if int(state.step) != update + 1 or not np.isfinite(info.loss):
            raise ValueError(f"Update {update} did not complete with a finite loss")
        comparison = {}
        if reference is not None and update < 16:
            previous_loss = reference["updates"][update]["loss"]
            comparison = {"r3_loss": previous_loss, "loss_delta_vs_r3": float(info.loss) - previous_loss}
        updates.append({**receipt, "update": update, "loss": float(info.loss), "dispatch": counts, **comparison})
    return state, updates


def run_identity_sft(
    config: TrainLmConfig,
    *,
    runtime_receipt: dict,
    receipt_sha256: str,
    lr0_source: dict,
    source_objects: dict,
    run_root: str,
    load_audits: Callable[[], SftAudits],
    run: SftRun,
) -> dict:
    """Load the pinned source, run the checked fresh horizon, checkpoint, export and audit.

    Call on every JAX rank. Relies on a passing LR0 run of the same pin for the precision gate.
    """
    reference = checked_run_reference(run)
    horizon = run.stop_after_updates
    if config.trainer.num_train_steps != horizon or config.trainer.load_checkpoint_path is not None:
        raise ValueError("The SFT config must match its explicit fresh-run horizon")
    if reference is not None and reference["runtime_receipt_sha256"] != receipt_sha256:
        raise ValueError("The fresh run and r3 must use the same runtime receipt")
    if config.z_loss_weight != 0.0:
        raise ValueError("The checked SFT loss requires the frozen zero auxiliary-loss recipe")
    export_path = f"{config.hf_save_path}/step-{horizon - 1}"
    if (StoragePath(export_path) / "config.json").exists():
        raise FileExistsError("SFT export path already holds a model; refusing to reuse it")
    config_bytes = (StoragePath(BASE_MODEL) / "config.json").read_bytes()
    generation_bytes = (StoragePath(BASE_MODEL) / "generation_config.json").read_bytes()
    rounding_path = runtime_receipt["routing_reference"]["path"]
    check_precision_runtime_receipt(
        config,
        runtime_receipt,
        config_bytes=config_bytes,
        generation_bytes=generation_bytes,
        rounding_bytes=StoragePath(rounding_path).read_bytes(),
    )
    source = json.loads(config_bytes)
    generation = json.loads(generation_bytes)
    lr0_source_bytes = StoragePath(lr0_source["path"]).read_bytes()
    if sha256(lr0_source_bytes) != lr0_source["sha256"]:
        raise ValueError("LR0 source receipt differs from the one the launch cited")
    lr0_source_receipt = json.loads(lr0_source_bytes)
    object_list_bytes = StoragePath(source_objects["path"]).read_bytes()
    if sha256(object_list_bytes) != source_objects["sha256"]:
        raise ValueError("Source object list differs from the one the launch cited")
    object_list = json.loads(object_list_bytes)
    initialize(config)
    if jax.process_count() != REPLICAS:
        raise RuntimeError(f"SFT requires {REPLICAS} JAX processes, found {jax.process_count()}")
    audits = load_audits()
    started = time.monotonic()

    optimizer = config.optimizer.build(config.trainer.num_train_steps)
    with Trainer(config.trainer, optimizer, snowball_loss_with_dispatch_stats, add_default_hooks=False) as trainer:
        check_runtime_recipe(config, source_config=source, source_generation_config=generation)
        run_on_export_writer(
            lambda: write_export_json(
                StoragePath(run_root) / "audits/source-bind.json", source_bind(lr0_source_receipt, object_list)
            )
        )
        tokenizer = config.data.the_tokenizer
        converter = config.model.hf_checkpoint_converter().replaced(reference_checkpoint=BASE_MODEL, tokenizer=tokenizer)
        # The HF metadata is architecture-authoritative; runtime precision is restored AFTER reload.
        reloaded = converter.config_from_hf_config(GrugMoeHfConfig(**source))
        reloaded = preserve_hf_runtime_settings(reloaded, config.model, config.trainer.compute_dtype_overrides)
        config = replace(config, model=reloaded)
        check_runtime_recipe(config, source_config=source, source_generation_config=generation)
        model = converter.load_pretrained(
            config.model.model_type, config=reloaded, axis_mapping=trainer.parameter_axis_mapping, dtype=None
        )
        loaded_model_config = asdict(model.config)
        if loaded_model_config != runtime_receipt["runtime_model_config"]:
            raise ValueError("HF loader discarded or changed the approved runtime model config")
        source_biases = router_values(model)
        state = trainer.initial_state(
            jrandom.PRNGKey(config.trainer.seed), model=model, is_trainable=config.adapter.trainable_filter(model)
        )
        del model
        check_restored_progress(config, state, completed_updates=0)
        audit_initialized_masters(
            state.model.to_state_dict(), lr0_source["path"], f"{run_root}/audits/master-step-0.json"
        )
        for name, value in router_values(state.model).items():
            np.testing.assert_array_equal(value, source_biases[name])
        overrides = config.trainer.compute_dtype_overrides
        with hax.partitioning.set_mesh(trainer.device_mesh), hax.axis_mapping(trainer.compute_axis_mapping):
            view = cast_model_for_compute(state.model, trainer.mp, overrides)
            dot_dtypes = router_dot_dtypes(view, trainer.mp.compute_dtype)
        del view
        if not router_dot_is_fp32(dot_dtypes):
            raise ValueError(f"Router dot is not FP32 on the compute view: {dot_dtypes}")

        pos = config.model.max_Pos.resize(config.train_seq_len)
        train_dataset = config.data.train_set(pos, config.trainer.batch_schedule, key=jrandom.PRNGKey(config.data_seed))
        batches, dataset_report = audits.audit_iterator(trainer.data_loader(train_dataset).iter_from_step(0), config)
        if reference is not None and dataset_report["updates"] != reference["dataset_audit"]["updates"]:
            raise ValueError("The full 64-update dataset plan differs from r3")
        train_started = time.monotonic()
        state, updates = train_checked_updates(trainer, state, batches, audits.check_batch, horizon, reference=reference)
        memory = device_memory_report(gather_host_rows(local_device_memory()))
        train_seconds = time.monotonic() - train_started
        check_restored_progress(config, state, completed_updates=horizon)
        for name, value in router_values(state.model).items():
            np.testing.assert_array_equal(value, source_biases[name])

        label = horizon - 1
        checkpointer = config.trainer.checkpointer.create(trainer.run_id)
        checkpointer.on_step(tree=state.saveable_state, step=label, force=True)
        checkpointer.wait_until_finished()
        checkpoint_path = f"{config.trainer.checkpointer.expanded_path(trainer.run_id)}/step-{label}"

        save_started = time.monotonic()
        converter.with_config_overrides(source_export_overrides(source)).save_pretrained(
            state.model,
            export_path,
            dtype=jnp.bfloat16,
            dtype_overrides=ROUTER_BIAS_DTYPES,
            generation_config=generation,
            upload_to_hf=False,
        )
        save_seconds = time.monotonic() - save_started
        run_on_export_writer(
            lambda: check_export_metadata(source, json.loads((StoragePath(export_path) / "config.json").read_bytes()))
        )
        audits.save_audit(state, export_path)
        return {
            **run_provenance(run),
            "completed_updates": int(state.step),
            "next_data_offset": config.trainer.batch_schedule.global_data_offset_by_step(int(state.step)),
            "updates": updates,
            "dataset_audit": dataset_report,
            "native_checkpoint": checkpoint_path,
            "hf_path": export_path,
            "router_dot_dtypes": dot_dtypes,
            "compute_dtype_overrides": overrides,
            "learning_rate": config.optimizer.learning_rate,
            "train_seconds": train_seconds,
            "save_seconds": save_seconds,
            "worker_seconds": time.monotonic() - started,
            "runtime_recipe_receipt": runtime_receipt,
            "runtime_receipt_sha256": receipt_sha256,
            "xla_precision": runtime_receipt["xla_precision"],
            "device_memory": {"after_update": memory},
            "lr0_source_receipt": lr0_source,
            "source_object_list": source_objects,
        }


def sft_worker(
    run_root: str,
    run_id: str,
    receipt_bytes: bytes,
    receipt_sha256: str,
    data_config_bytes: bytes,
    lr0_source: dict,
    source_objects: dict,
    load_audits: Callable[[], SftAudits],
    run: SftRun,
) -> None:
    """Run on every replica; the byte arguments are the reviewed receipt's and the frozen data config's bytes."""
    runtime_receipt = sft_runtime_receipt(receipt_bytes, receipt_sha256)
    check_xla_precision(runtime_receipt["xla_precision"])
    check_pod_storage()
    reference = runtime_receipt["routing_reference"]
    config_bytes = (StoragePath(BASE_MODEL) / "config.json").read_bytes()
    generation_bytes = (StoragePath(BASE_MODEL) / "generation_config.json").read_bytes()
    base = sft_config(run_root, run_id, config_bytes, generation_bytes, data_config_bytes, runtime_receipt, run)
    config, observed = configure_precision_runtime(
        base,
        config_bytes=config_bytes,
        generation_bytes=generation_bytes,
        rounding_bytes=StoragePath(reference["path"]).read_bytes(),
        rounding_path=reference["path"],
        rounding_sha256=reference["sha256"],
        xla_precision=xla_precision_in_process(),
    )
    if observed != runtime_receipt:
        raise ValueError("Derived runtime recipe differs from the reviewed receipt")
    if config.data != base.data:
        raise ValueError("The precision runtime changed the frozen data config")
    config = replace(config, trainer=replace(config.trainer, tracker=NoopConfig()))
    result = run_identity_sft(
        config,
        runtime_receipt=runtime_receipt,
        receipt_sha256=receipt_sha256,
        lr0_source=lr0_source,
        source_objects=source_objects,
        run_root=run_root,
        load_audits=load_audits,
        run=run,
    )
    # Collective: every rank calls it; process 0 writes.
    run_on_export_writer(lambda: write_export_json(StoragePath(run_root) / "sft-result.json", result))


def load_sft_audits(run_root: str, lr0_source: dict) -> SftAudits:
    """Load collective audit inputs after the trainer initializes distributed JAX."""
    frozen = load_dataset_audit_inputs(MIXTURE_ROOT, INVENTORY_PATH)
    return SftAudits(
        audit_iterator=lambda batches, config: audit_training_iterator(
            batches,
            config=config,
            completed_updates=0,
            frozen=frozen,
            receipt_path=f"{run_root}/audits/dataset-step-0.json",
        ),
        check_batch=lambda batch, update: checked_update(batch, update, frozen),
        save_audit=lambda state, export_path: audit_trained_export(
            state,
            export_path,
            source_root=BASE_MODEL,
            source_receipt_path=lr0_source["path"],
            source_receipt_sha256=lr0_source["sha256"],
            expected_hashes=SOURCE_FILE_HASHES,
        ),
    )


def bound_sft_worker(
    run_root: str,
    run_id: str,
    receipt_bytes: bytes,
    receipt_sha256: str,
    data_config_bytes: bytes,
    lr0_source: dict,
    source_objects: dict,
    run: SftRun,
) -> None:
    """Check flags before backend startup; defer collective inputs to trainer initialization."""
    runtime_receipt = sft_runtime_receipt(receipt_bytes, receipt_sha256)
    require_xla_precision(runtime_receipt["xla_precision"])
    checked_run_reference(run)
    check_pod_storage()
    sft_worker(
        run_root,
        run_id,
        receipt_bytes,
        receipt_sha256,
        data_config_bytes,
        lr0_source,
        source_objects,
        partial(load_sft_audits, run_root, lr0_source),
        run,
    )


def read_remote(path: str) -> bytes:
    fs, local = filesystem_for(path)
    return fs.cat_file(local)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def lr0_pass(lr0_root: str, runtime_receipt: dict) -> dict:
    """The passing LR0 run this SFT relies on; raises unless every reading-rule receipt says pass."""
    gate_path = prefix_join(lr0_root, "precision-gate/gate.json")
    gate_bytes = read_remote(gate_path)
    gate = json.loads(gate_bytes)
    if gate.get("failures") != []:
        raise ValueError(f"LR0 gate did not pass: {gate.get('failures')}")
    for rank in range(REPLICAS):
        receipt = json.loads(read_remote(prefix_join(lr0_root, f"precision-gate/ranks/{rank}.json")))
        if receipt.get("stage") != "updated" or receipt.get("process") != rank:
            raise ValueError(f"LR0 rank {rank} did not complete the probe update: {receipt}")
    result_bytes = read_remote(prefix_join(lr0_root, "lr0-result.json"))
    result = json.loads(result_bytes)
    if result.get("completed_updates") != 1:
        raise ValueError("LR0 result does not record one completed update")
    if result.get("runtime_recipe_receipt") != runtime_receipt:
        raise ValueError("LR0 used a different runtime recipe from the SFT forward")
    source_path = prefix_join(lr0_root, "audits/source-tensor-hashes.json")
    source_bytes = read_remote(source_path)
    if len(json.loads(source_bytes)["tensors"]) != 502:
        raise ValueError("LR0 source receipt does not hold 502 tensors")
    return {
        "run_root": lr0_root,
        "gate": {"path": gate_path, "sha256": sha256(gate_bytes)},
        "result": {"path": prefix_join(lr0_root, "lr0-result.json"), "sha256": sha256(result_bytes)},
        "source_receipt": {"path": source_path, "sha256": sha256(source_bytes)},
    }


def source_object_list(path: str) -> dict:
    data = read_remote(path)
    value = json.loads(data)
    if value["source_root"].rstrip("/") != BASE_MODEL.rstrip("/"):
        raise ValueError("Source object list is for a different source root")
    if value["source_index_sha256"] != SOURCE_FILE_HASHES[INDEX]:
        raise ValueError("Source object list was taken against a different index")
    return {"path": path, "sha256": sha256(data), "objects": len(value["objects"])}


def frozen_mixture() -> tuple[bytes, dict]:
    """The frozen data config bytes and the hashes of the three mixture files the launch cites."""
    files = {
        "freeze.json": FREEZE_SHA256,
        "data-config.json": DATA_CONFIG_SHA256,
        "cache-catalog.json": CACHE_CATALOG_SHA256,
    }
    contents = {name: read_remote(f"{MIXTURE_ROOT}/{name}") for name in files}
    for name, expected in files.items():
        if sha256(contents[name]) != expected:
            raise ValueError(f"Mixture {name} differs from the frozen hash")
    return contents["data-config.json"], {"root": MIXTURE_ROOT, "sha256": files}


def plan(
    job_name: str,
    run_root: str,
    timeout_seconds: int,
    priority: str,
    config: TrainLmConfig,
    receipt_sha256: str,
    xla_precision: str,
    run: SftRun,
) -> dict:
    assert isinstance(config.model, SnowballConfig)
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
    return {
        "job": job_name,
        "commit": head.stdout.strip(),
        "clean": not dirty.stdout,
        "run_root": run_root,
        **run_provenance(run),
        "optimizer": {
            "learning_rate": config.optimizer.learning_rate,
            "lr_schedule": config.optimizer.lr_schedule,
            "warmup": config.optimizer.warmup,
            "weight_decay": config.optimizer.weight_decay,
            "max_grad_norm": config.optimizer.max_grad_norm,
        },
        "batch": {
            "train_batch_size": config.trainer.train_batch_size,
            "per_device_parallelism": config.trainer.per_device_parallelism,
            "seq_len": config.train_seq_len,
        },
        "mesh": mesh_layout(config.trainer.mesh),
        "forward": {
            "moe_capacity_factor": config.model.moe_capacity_factor,
            "router_combine_fp32": config.model.router_combine_fp32,
            "fp32_activations": config.model.fp32_activations,
            "attention_implementation": config.model.attention_implementation,
            "token_valid": "query_segment_ids >= 0",
        },
        "runtime_receipt_sha256": receipt_sha256,
        "xla_precision": xla_precision,
        "env_vars": xla_precision_env(xla_precision),
        "region": RESOURCES.target_cluster,
        "replicas": REPLICAS,
        "gpus": 8 * REPLICAS,
        "resources": {"cpu": RESOURCES.cpu, "ram": RESOURCES.ram, "disk": RESOURCES.disk},
        "priority": priority,
        "ports": ["jax"],
        "secret_behavior": "isolate: CW_KEY_ID/SECRET, also as AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY",
        "retries_failure": 0,
        "retries_preemption": 0,
        "timeout_seconds": timeout_seconds,
        "h100_hour_ceiling": 8 * REPLICAS * timeout_seconds / 3600,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--run-root", required=True, help="fresh s3://marin-rno2a/ prefix; must not exist")
    parser.add_argument("--runtime-receipt", required=True, help="local reviewed schema-6 runtime receipt")
    parser.add_argument("--runtime-receipt-sha256", required=True)
    parser.add_argument("--stop-after-updates", type=int, choices=TRAINING_HORIZONS, required=True)
    parser.add_argument("--reference-sft-result", help="reviewed r3 result; required for 64 updates")
    parser.add_argument("--decision-record", help="local owner authorization record; required for 64 updates")
    parser.add_argument("--decision-supplement", help="local owner supplement 2; required for 64 updates")
    parser.add_argument("--timeout-seconds", type=int, required=True)
    parser.add_argument("--allow-long-timeout", action="store_true", help=f"permit more than {MAX_TIMEOUT_SECONDS} s")
    parser.add_argument("--expect-commit", help="the reviewed worker pin; required with --run, must equal HEAD")
    parser.add_argument("--priority", choices=sorted(PRIORITY_BANDS), default="interactive")
    parser.add_argument("--lr0-run-root", help="the passing LR0 run root; required with --run")
    parser.add_argument("--source-object-list", help="S3 source object list (names, sizes, ETags); required with --run")
    parser.add_argument("--spending-gate", help="local milestone16-spending-gate-v1.json; required with --run")
    parser.add_argument("--capacity-record", help="the launch owner's capacity guard JSON; required with --run")
    parser.add_argument("--launch-receipt", help="local JSON receipt to write; required with --run")
    parser.add_argument("--run", action="store_true", help="submit; without it only print the plan")
    args = parser.parse_args()
    run_root = args.run_root.rstrip("/")
    if not run_root.startswith("s3://marin-rno2a/"):
        raise SystemExit("run root must be on s3://marin-rno2a/")
    if args.timeout_seconds <= 0:
        raise SystemExit("timeout must be a positive number of seconds")
    if args.timeout_seconds > MAX_TIMEOUT_SECONDS and not args.allow_long_timeout:
        raise SystemExit(f"timeout above {MAX_TIMEOUT_SECONDS} s needs --allow-long-timeout")
    receipt_bytes = Path(args.runtime_receipt).read_bytes()
    runtime_receipt = sft_runtime_receipt(receipt_bytes, args.runtime_receipt_sha256)
    run = SftRun(
        args.stop_after_updates,
        read_remote(args.reference_sft_result) if args.reference_sft_result else None,
        Path(args.decision_record).read_bytes() if args.decision_record else None,
        Path(args.decision_supplement).read_bytes() if args.decision_supplement else None,
    )
    checked_run_reference(run)
    config_bytes = read_remote(f"{BASE_MODEL}config.json")
    generation_bytes = read_remote(f"{BASE_MODEL}generation_config.json")
    if sha256(config_bytes) != SOURCE_FILE_HASHES["config.json"]:
        raise SystemExit("source config.json differs from the pinned hash")
    if sha256(generation_bytes) != SOURCE_FILE_HASHES["generation_config.json"]:
        raise SystemExit("source generation_config.json differs from the pinned hash")
    data_config_bytes, mixture = frozen_mixture()
    config = sft_config(run_root, args.job_name, config_bytes, generation_bytes, data_config_bytes, runtime_receipt, run)
    lr0 = lr0_pass(args.lr0_run_root.rstrip("/"), runtime_receipt) if args.lr0_run_root else None
    source_objects = source_object_list(args.source_object_list) if args.source_object_list else None
    job_plan = {
        **plan(
            args.job_name,
            run_root,
            args.timeout_seconds,
            args.priority,
            config,
            args.runtime_receipt_sha256,
            runtime_receipt["xla_precision"],
            run,
        ),
        "mixture": mixture,
        "spending_gate_sha256": SPENDING_GATE_SHA256,
        "dataset_audit": {
            "inventory": {"path": INVENTORY_PATH, "sha256": INVENTORY_SHA256},
            "accounting_module_sha256": ACCOUNTING_MODULE_SHA256,
            "accounting_test_sha256": sha256(Path(ACCOUNTING_TEST).read_bytes()),
        },
        "lr0_pass": lr0,
        "source_object_list": source_objects,
    }
    print(json.dumps(job_plan, indent=2))
    if not args.run:
        return
    if not job_plan["clean"]:
        raise SystemExit("worktree is not clean")
    if source_objects is None:
        raise SystemExit("--run needs --source-object-list: the cheap source bind compares against it")
    if lr0 is None:
        raise SystemExit("--run needs --lr0-run-root: the SFT relies on a passing LR0 run of the same pin")
    if not (args.capacity_record and args.launch_receipt and args.expect_commit and args.spending_gate):
        raise SystemExit("--run needs --capacity-record, --launch-receipt, --expect-commit and --spending-gate")
    if job_plan["commit"] != args.expect_commit:
        raise SystemExit(f"HEAD {job_plan['commit']} is not the reviewed pin {args.expect_commit}")
    if sha256(Path(args.spending_gate).read_bytes()) != SPENDING_GATE_SHA256:
        raise SystemExit("spending gate file differs from the pre-registered hash")
    fs, root = filesystem_for(run_root)
    if fs.exists(root):
        raise SystemExit(f"run root already exists: {run_root}")
    capacity = Path(args.capacity_record).read_bytes()
    launch = {
        **job_plan,
        "state": "pre-submit",
        "submit_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "spending_gate": {"path": args.spending_gate, "sha256": SPENDING_GATE_SHA256},
        "capacity_record": {"path": args.capacity_record, "sha256": sha256(capacity), "content": json.loads(capacity)},
    }
    Path(args.launch_receipt).write_text(json.dumps(launch, indent=2) + "\n")
    entrypoint = Entrypoint.from_callable(
        bound_sft_worker,
        run_root,
        args.job_name,
        receipt_bytes,
        args.runtime_receipt_sha256,
        data_config_bytes,
        lr0["source_receipt"],
        source_objects,
        run,
    )
    job_id = submit_entrypoint(
        args.job_name, entrypoint, args.timeout_seconds, args.priority, runtime_receipt["xla_precision"]
    )
    launch.update(state="submitted", iris_job_id=job_id)
    Path(args.launch_receipt).write_text(json.dumps(launch, indent=2) + "\n")
    print(json.dumps({"job": job_id, "status": "submitted"}))


if __name__ == "__main__":
    main()
