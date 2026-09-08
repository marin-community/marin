# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay the exact 280-row Delphi fit swarm at total-parameter TPP 40.

The model architecture and 80/20 WSD schedule are the same as the Delphi 3e18
fit swarm. Only the training horizon changes. The token-aware AdamH schedule is
recomputed for the longer horizon, while source coordinates and random seeds
remain matched to the original 280-row panel. A distinct output namespace
deliberately retrains all rows and permanently retains the state immediately
after phase 0 for shared-prefix continuation experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

import fsspec
from fray.cluster import ResourceConfig
from levanter.data.text.datasets import DatasetComponent
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.processing.tokenize.data_configs import ExistingTokenizedCacheConfig
from rigging.filesystem import marin_prefix_for_region

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    build_top_level_domains,
    executor_status_succeeded,
)
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

TOTAL_TOKENS_PER_PARAMETER = 40.0
EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_augmented_swarm_tpp40_phase0_checkpoint_20260815"
LOCAL_ARTIFACT_DIR = (
    base.REFERENCE_OUTPUT_DIR / "delphi_augmented_swarm_tpp40_phase0_checkpoint_20260815" / "launch_dry_run"
)
EXPECTED_SOURCE_PANEL_SHA256 = "4f283bacb4ef269c396277cbd518ef74212a51741c909a1e1e9ace040751d507"
EXPECTED_SOURCE_COORDINATE_HASH = "4db8e7f70dda72f2bc8a04fa4b8271f1a5959aa27e7119ad4f62f951cd1b2864"
EXPECTED_FIXED_IDENTITY_HASH = "f626888135ba77cd55fea69fedc6ebaaa24066d225be7c33aba218f2f7816171"
DEFAULT_MAX_CONCURRENT = 56
STEPS_PER_EVAL = 5000
EXPECTED_PHASE0_CHECKPOINT_STEP = 21855
EXPECTED_FINAL_CHECKPOINT_STEP = 27335
ASSIGNMENT_REGION_TO_TPU_REGION = {"east5": "us-east5", "europe": "europe-west4"}
DEFAULT_TRAINING_WANDB_GROUP = "delphi_tpp40_augmented_swarm"
DEFAULT_TABLE9_WANDB_GROUP = "olmo_base_eval_table9_delphi_tpp40_augmented_swarm"
EXPECTED_TPU_DEVICE_COUNTS = {"v5p-8": 4, "v6e-8": 8}
TPU_ZONES_BY_TYPE = {
    "v5p-8": frozenset({"us-central1-a", "us-east5-a"}),
    "v6e-8": frozenset({"europe-west4-a", "us-east1-d", "us-east5-b"}),
}
HORIZON_FIELDS = frozenset(
    {
        "target_flops",
        "train_steps",
        "realized_train_tokens",
        "expected_checkpoint_step",
    }
)
DEPLOYMENT_FIELDS = frozenset({"tpu_type", "tpu_region", "tpu_zone", "tensor_parallel_size"})


def _require_tpu_placement(*, tpu_type: str, region: str, zone: str) -> None:
    zone_region = zone.rsplit("-", maxsplit=1)[0]
    if zone_region != region:
        raise ValueError(f"TPU zone {zone!r} does not belong to region {region!r}")
    try:
        allowed_zones = TPU_ZONES_BY_TYPE[tpu_type]
    except KeyError as error:
        raise ValueError(f"Unsupported TPP40 TPU type {tpu_type!r}") from error
    if zone not in allowed_zones:
        raise ValueError(
            f"TPU type {tpu_type!r} is not configured in zone {zone!r}; expected one of {sorted(allowed_zones)}"
        )


def _tensor_parallel_size(hidden_dim: int, tpu_type: str) -> int:
    try:
        device_count = EXPECTED_TPU_DEVICE_COUNTS[tpu_type]
    except KeyError as error:
        raise ValueError(f"Unsupported TPP40 TPU type {tpu_type!r}") from error
    tensor_parallel_size = 1
    while hidden_dim % (device_count // tensor_parallel_size) != 0:
        tensor_parallel_size *= 2
        if tensor_parallel_size > device_count:
            raise ValueError(f"Could not find tensor parallel size for hidden_dim={hidden_dim}, {tpu_type=}")
    return tensor_parallel_size


def _reject_assignment_option_typos(remaining: list[str]) -> None:
    suspicious = [value for value in remaining if value.startswith(("--assignment", "--expect-assignment"))]
    if suspicious:
        raise ValueError(f"Unknown assignment options: {suspicious}")


def _require_assignment_contract(*, experiment_name: str, tpu_region: str, assignment_file: str | None) -> None:
    if experiment_name == EXPERIMENT_NAME and tpu_region == "europe-west4" and assignment_file is None:
        raise ValueError("Production Europe TPP40 launch requires --assignment-file")


def _phase_0_checkpoint_step(train_steps: int) -> tuple[int, int]:
    """Return the last phase-0 update and the first phase-1 update."""
    late_phase_start_step = base.PHASE_SCHEDULE.phases[1].get_start_step_aligned(
        train_steps,
        base.TARGET_BATCH_SIZE,
        base.MIXTURE_BLOCK_SIZE,
    )
    checkpoint_step = late_phase_start_step - 1
    if checkpoint_step <= 0:
        raise ValueError(f"Invalid phase-0 checkpoint step {checkpoint_step} for {train_steps} train steps")
    if 2 * checkpoint_step <= train_steps - 1:
        raise ValueError("Phase-0 checkpoint interval would create an unintended second intermediate checkpoint")
    return checkpoint_step, late_phase_start_step


def _panel_hash(run_specs: list[base.DelphiSwarmRunSpec]) -> str:
    payload = [
        {
            "source_run_name": spec.source_run_name,
            "panel_source": spec.panel_source,
            "phase_weights": spec.phase_weights,
        }
        for spec in run_specs
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _fixed_identity_hash(run_specs: list[base.DelphiSwarmRunSpec]) -> str:
    payload = []
    for spec in run_specs:
        row = asdict(spec)
        for field in HORIZON_FIELDS:
            row.pop(field)
        payload.append(row)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _scientific_identity_hash(run_specs: list[base.DelphiSwarmRunSpec]) -> str:
    payload = []
    for spec in run_specs:
        row = asdict(spec)
        for field in HORIZON_FIELDS | DEPLOYMENT_FIELDS:
            row.pop(field)
        payload.append(row)
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _regional_input_path(canonical_path: str, *, region: str) -> str:
    canonical_prefix = marin_prefix_for_region(base.DEFAULT_TPU_REGION).rstrip("/")
    if not canonical_path.startswith(f"{canonical_prefix}/"):
        raise ValueError(f"Canonical input is outside {canonical_prefix}: {canonical_path}")
    relative_path = canonical_path.removeprefix(canonical_prefix).lstrip("/")
    return f"{marin_prefix_for_region(region).rstrip('/')}/{relative_path}"


def _require_regional_input_path(path: str, *, region: str, label: str) -> str:
    required_prefix = marin_prefix_for_region(region).rstrip("/") + "/"
    if not path.startswith(required_prefix):
        raise ValueError(f"{label} must be under {required_prefix}, got {path!r}")
    return path


def build_run_specs(
    *,
    source_panel: str,
    analysis_output_path: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
) -> tuple[list[base.DelphiSwarmRunSpec], dict[str, object]]:
    """Resolve the fixed architecture and replace only its training horizon."""
    source_specs = base.load_source_panel(
        source_panel=source_panel,
        analysis_output_path=analysis_output_path,
        tpu_region=base.DEFAULT_TPU_REGION,
        tpu_zone=base.DEFAULT_TPU_ZONE,
    )
    source_coordinate_hash = _panel_hash(source_specs)
    if base.SOURCE_PANEL_SHA256 != EXPECTED_SOURCE_PANEL_SHA256:
        raise ValueError(f"Unexpected source panel hash: {base.SOURCE_PANEL_SHA256}")
    if source_coordinate_hash != EXPECTED_SOURCE_COORDINATE_HASH:
        raise ValueError(f"Unexpected source coordinate hash: {source_coordinate_hash}")
    total_params = {spec.total_trainable_params for spec in source_specs}
    if len(total_params) != 1:
        raise ValueError(f"Source panel has inconsistent parameter counts: {sorted(total_params)}")
    total_trainable_params = total_params.pop()
    requested_train_tokens = round(total_trainable_params * TOTAL_TOKENS_PER_PARAMETER)
    tokens_per_step = base.TARGET_BATCH_SIZE * base.SEQ_LEN_DELPHI
    train_steps = round(requested_train_tokens / tokens_per_step)
    realized_train_tokens = train_steps * tokens_per_step
    if realized_train_tokens > base.SIMULATED_EPOCH_TARGET_BUDGET:
        raise ValueError(
            f"TPP-40 realizes {realized_train_tokens} tokens, exceeding simulated-epoch target "
            f"budget {base.SIMULATED_EPOCH_TARGET_BUDGET}"
        )
    realized_tpp = realized_train_tokens / total_trainable_params
    actual_approximate_flops = 6 * total_trainable_params * realized_train_tokens

    _require_tpu_placement(tpu_type=tpu_type, region=tpu_region, zone=tpu_zone)
    tensor_parallel_size = _tensor_parallel_size(source_specs[0].model_hidden_dim, tpu_type)
    run_specs = [
        replace(
            spec,
            target_flops=actual_approximate_flops,
            train_steps=train_steps,
            realized_train_tokens=realized_train_tokens,
            expected_checkpoint_step=train_steps - 1,
            tpu_type=tpu_type,
            tpu_region=tpu_region,
            tpu_zone=tpu_zone,
            tensor_parallel_size=tensor_parallel_size,
        )
        for spec in source_specs
    ]
    if _panel_hash(run_specs) != source_coordinate_hash:
        raise ValueError("TPP-40 materialization changed a source coordinate")
    source_identity_hash = _fixed_identity_hash(source_specs)
    if source_identity_hash != EXPECTED_FIXED_IDENTITY_HASH:
        raise ValueError(f"Unexpected fixed source identity hash: {source_identity_hash}")
    scientific_identity_hash = _scientific_identity_hash(source_specs)
    if _scientific_identity_hash(run_specs) != scientific_identity_hash:
        raise ValueError("TPP-40 materialization changed a scientific source identity field")
    data_seeds_matched = [spec.data_seed for spec in run_specs] == [spec.data_seed for spec in source_specs]
    if not data_seeds_matched:
        raise ValueError("TPP-40 materialization changed source data seeds")
    phase_0_checkpoint_step, late_phase_start_step = _phase_0_checkpoint_step(train_steps)
    audit: dict[str, object] = {
        "experiment_name": EXPERIMENT_NAME,
        "source_panel": source_panel,
        "source_panel_sha256": base.SOURCE_PANEL_SHA256,
        "run_count": len(run_specs),
        "source_coordinate_hash": source_coordinate_hash,
        "fixed_identity_hash": source_identity_hash,
        "scientific_identity_hash": scientific_identity_hash,
        "deployment_tpu_type": tpu_type,
        "deployment_tpu_region": tpu_region,
        "deployment_tpu_zone": tpu_zone,
        "deployment_tensor_parallel_size": tensor_parallel_size,
        "deployment_tpu_device_count": EXPECTED_TPU_DEVICE_COUNTS[tpu_type],
        "architecture_target_flops": base.TARGET_FLOPS,
        "actual_approximate_flops_per_run": actual_approximate_flops,
        "total_trainable_params": total_trainable_params,
        "non_embedding_params": run_specs[0].non_embedding_params,
        "requested_train_tokens": requested_train_tokens,
        "realized_train_tokens": realized_train_tokens,
        "train_steps": train_steps,
        "expected_checkpoint_step": train_steps - 1,
        "target_total_tokens_per_parameter": TOTAL_TOKENS_PER_PARAMETER,
        "realized_total_tokens_per_parameter": realized_tpp,
        "realized_non_embedding_tokens_per_parameter": realized_train_tokens / run_specs[0].non_embedding_params,
        "phase_fractions": base.PHASE_FRACTIONS,
        "realized_late_phase_start_step": late_phase_start_step,
        "realized_late_phase_start_fraction": late_phase_start_step / train_steps,
        "data_seeds_matched_to_original_swarm": data_seeds_matched,
        "steps_per_eval": STEPS_PER_EVAL,
        "temporary_checkpoint_interval_minutes": 10,
        "phase_0_checkpoint_step": phase_0_checkpoint_step,
        "phase_0_checkpoint_state_next_step": late_phase_start_step,
        "permanent_checkpoint_interval": phase_0_checkpoint_step,
        "native_table9_scheduled": True,
    }
    return run_specs, audit


def _parse_run_orders(value: str, *, expected_runs: int = base.EXPECTED_RUNS) -> tuple[int, ...]:
    if value == "all":
        return tuple(range(expected_runs))

    orders: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            raise ValueError("--run-orders contains an empty item")
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending run-order range {item!r}")
            orders.extend(range(start, end + 1))
        else:
            orders.append(int(item))

    if len(set(orders)) != len(orders):
        raise ValueError("--run-orders contains duplicates")
    invalid = [order for order in orders if not 0 <= order < expected_runs]
    if invalid:
        raise ValueError(f"Run orders out of range [0, {expected_runs}): {invalid}")
    if not orders:
        raise ValueError("--run-orders selected no rows")
    return tuple(orders)


def _assignment_orders(
    assignment_file: str,
    assignment_region: str,
    *,
    tpu_region: str,
    experiment_name: str,
    expected_assignment_sha256: str,
    expected_runs: int = base.EXPECTED_RUNS,
) -> tuple[tuple[int, ...], dict[str, object]]:
    if assignment_region not in {"east5", "europe"}:
        raise ValueError("--assignment-region must be 'east5' or 'europe'")
    required_tpu_region = ASSIGNMENT_REGION_TO_TPU_REGION[assignment_region]
    if tpu_region != required_tpu_region:
        raise ValueError(
            f"Assignment region {assignment_region!r} requires TPU region {required_tpu_region!r}, got {tpu_region!r}"
        )
    with fsspec.open(assignment_file, "rt") as handle:
        payload = json.load(handle)
    if payload.get("expected_runs") != expected_runs:
        raise ValueError(f"Assignment expected_runs changed: {payload.get('expected_runs')} != {expected_runs}")

    expected_hash = payload.get("assignment_sha256")
    unhashed_payload = {key: value for key, value in payload.items() if key != "assignment_sha256"}
    observed_hash = hashlib.sha256(
        json.dumps(unhashed_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if observed_hash != expected_hash:
        raise ValueError(f"Assignment SHA-256 mismatch: {observed_hash} != {expected_hash}")
    if expected_hash != expected_assignment_sha256:
        raise ValueError(f"Unexpected assignment SHA-256: {expected_hash} != {expected_assignment_sha256}")

    root_key = f"{assignment_region}_root"
    expected_root = f"{marin_prefix_for_region(tpu_region).rstrip('/')}/{experiment_name}"
    if payload.get(root_key) != expected_root:
        raise ValueError(f"Assignment {root_key} changed: {payload.get(root_key)!r} != {expected_root!r}")

    assignments = payload.get("assignments")
    if not isinstance(assignments, dict):
        raise ValueError("Assignment file lacks an assignments object")
    required_groups = {"completed", "east5", "europe", "resumable_east5"}
    if not required_groups <= assignments.keys():
        raise ValueError(f"Assignment file lacks groups: {sorted(required_groups - assignments.keys())}")
    parsed: dict[str, tuple[int, ...]] = {}
    for group in required_groups:
        values = assignments[group]
        if not isinstance(values, list) or any(type(order) is not int for order in values):
            raise ValueError(f"Assignment group {group!r} must be a list of integer run orders")
        orders = tuple(values)
        if len(set(orders)) != len(orders):
            raise ValueError(f"Assignment group {group!r} contains duplicate run orders")
        invalid = [order for order in orders if not 0 <= order < expected_runs]
        if invalid:
            raise ValueError(f"Assignment group {group!r} has out-of-range run orders: {invalid}")
        if orders != tuple(sorted(orders)):
            raise ValueError(f"Assignment group {group!r} must be sorted")
        parsed[group] = orders
    all_orders = set(range(expected_runs))
    completed, east5, europe = map(set, (parsed["completed"], parsed["east5"], parsed["europe"]))
    if completed & east5 or completed & europe or east5 & europe:
        raise ValueError("Assignment groups overlap")
    if completed | east5 | europe != all_orders:
        raise ValueError("Assignment groups are not exhaustive")
    if not set(parsed["resumable_east5"]) <= east5:
        raise ValueError("Every resumable East5 row must remain assigned to East5")
    # Completed training rows remain in the East5 graph so downstream Table-9
    # obligations are not lost. Their training steps skip idempotently.
    selected = tuple(sorted(completed | east5)) if assignment_region == "east5" else parsed[assignment_region]
    if not selected:
        raise ValueError(f"Assignment selected no rows for {assignment_region}")
    return selected, {
        "assignment_file": assignment_file,
        "assignment_region": assignment_region,
        "assignment_sha256": expected_hash,
        "assignment_completed_count": len(completed),
        "assignment_east5_count": len(east5),
        "assignment_europe_count": len(europe),
        "assignment_resumable_east5_count": len(parsed["resumable_east5"]),
        "assignment_selected_count": len(selected),
        "assignment_completed_replayed_for_eval_count": len(completed) if assignment_region == "east5" else 0,
        "assignment_strata": payload.get("strata"),
    }


def _runtime_cache_paths(region: str) -> tuple[str, ...]:
    required_prefix = marin_prefix_for_region(region).rstrip("/") + "/"
    paths: list[str] = []
    for domain in build_top_level_domains(runtime_cache_region=region, require_prebuilt_complete=True):
        for component in domain.components:
            config = component.step_fn()
            if not isinstance(config, ExistingTokenizedCacheConfig):
                config_type = type(config).__name__
                raise ValueError(
                    f"Runtime domain {domain.name}/{component.name} would launch data preparation: {config_type}"
                )
            if not config.cache_path.startswith(required_prefix):
                raise ValueError(
                    f"Runtime cache for {domain.name}/{component.name} is not region-local: {config.cache_path}"
                )
            paths.append(config.cache_path)
    if len(paths) != 140:
        raise ValueError(f"Expected 140 frozen runtime cache bindings, got {len(paths)}")
    if len(set(paths)) != len(paths):
        raise ValueError("Frozen runtime cache bindings contain duplicate paths")
    return tuple(paths)


def _runtime_paths_sha256(paths: tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(sorted(paths), separators=(",", ":")).encode()).hexdigest()


def _require_validation_caches(validation_configs: dict[str, DatasetComponent], *, region: str) -> tuple[str, ...]:
    required_prefix = marin_prefix_for_region(region).rstrip("/") + "/"
    paths: list[str] = []
    fs = fsspec.filesystem("gcs")
    for name, component in sorted(validation_configs.items()):
        cache_dir = component.cache_dir
        if not cache_dir.startswith(required_prefix):
            raise ValueError(f"Validation cache {name!r} is not region-local: {cache_dir}")
        status_path = f"{cache_dir.rstrip('/')}/.executor_status"
        try:
            with fs.open(status_path, "rt") as handle:
                status = handle.read()
        except FileNotFoundError as exc:
            raise ValueError(f"Validation cache {name!r} lacks executor status: {cache_dir}") from exc
        if not executor_status_succeeded(status):
            raise ValueError(f"Validation cache {name!r} is incomplete: {cache_dir}")
        if not fs.exists(f"{cache_dir.rstrip('/')}/validation/.stats.json"):
            raise ValueError(f"Validation cache {name!r} lacks validation/.stats.json: {cache_dir}")
        paths.append(cache_dir)
    if len(paths) != 23:
        raise ValueError(f"Expected 23 validation cache bindings, got {len(paths)}")
    return tuple(paths)


def _write_local_dry_run(
    *,
    output_dir: str,
    source_panel: str,
    analysis_output_path: str,
    run_specs: list[base.DelphiSwarmRunSpec],
    audit: dict[str, object],
    experiment_name: str,
) -> None:
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    base.save_delphi_swarm_manifest(
        base.SaveDelphiSwarmManifestConfig(
            output_path=str(artifact_dir),
            experiment_name=experiment_name,
            source_panel=source_panel,
            source_panel_sha256=base.SOURCE_PANEL_SHA256,
            analysis_output_path=analysis_output_path,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs], sort_keys=True),
            architecture_target_flops=base.TARGET_FLOPS,
        )
    )
    (artifact_dir / "launch_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel")
    parser.add_argument("--analysis-output-path")
    parser.add_argument("--tpu-type", default=base.TARGET_TPU_TYPE)
    parser.add_argument("--tpu-region", default=base.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=base.DEFAULT_TPU_ZONE)
    parser.add_argument("--table9-tpu-type")
    parser.add_argument("--table9-tpu-zone")
    parser.add_argument("--run-orders", help="Comma-separated indices/ranges, or 'all'.")
    parser.add_argument("--assignment-file")
    parser.add_argument("--assignment-region", choices=("east5", "europe"))
    parser.add_argument("--expect-assignment-sha256")
    parser.add_argument("--experiment-name", default=EXPERIMENT_NAME)
    parser.add_argument("--training-wandb-group", default=DEFAULT_TRAINING_WANDB_GROUP)
    parser.add_argument("--table9-wandb-group", default=DEFAULT_TABLE9_WANDB_GROUP)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-output", default=str(LOCAL_ARTIFACT_DIR))
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = _parse_args()
    _reject_assignment_option_typos(remaining)
    sys.argv = [sys.argv[0], *remaining]
    if not 1 <= args.max_concurrent <= DEFAULT_MAX_CONCURRENT:
        raise ValueError(f"--max-concurrent must be in [1, {DEFAULT_MAX_CONCURRENT}]")
    if args.experiment_name != EXPERIMENT_NAME:
        if args.training_wandb_group == DEFAULT_TRAINING_WANDB_GROUP:
            raise ValueError("A non-production experiment name requires an explicit training W&B group")
        if args.table9_wandb_group == DEFAULT_TABLE9_WANDB_GROUP:
            raise ValueError("A non-production experiment name requires an explicit Table-9 W&B group")
    _require_assignment_contract(
        experiment_name=args.experiment_name,
        tpu_region=args.tpu_region,
        assignment_file=args.assignment_file,
    )
    _require_tpu_placement(tpu_type=args.tpu_type, region=args.tpu_region, zone=args.tpu_zone)

    expected_prefix = marin_prefix_for_region(args.tpu_region)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != expected_prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match required prefix {expected_prefix!r}")
    os.environ["MARIN_PREFIX"] = expected_prefix

    source_panel = args.source_panel or _regional_input_path(base.DEFAULT_SOURCE_PANEL, region=args.tpu_region)
    analysis_output_path = args.analysis_output_path or _regional_input_path(
        base.DEFAULT_ANALYSIS_OUTPUT_PATH,
        region=args.tpu_region,
    )
    source_panel = _require_regional_input_path(
        source_panel,
        region=args.tpu_region,
        label="source panel",
    )
    analysis_output_path = _require_regional_input_path(
        analysis_output_path,
        region=args.tpu_region,
        label="scaling analysis input",
    )

    run_specs, audit = build_run_specs(
        source_panel=source_panel,
        analysis_output_path=analysis_output_path,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
    )
    table9_tpu_type = args.table9_tpu_type or args.tpu_type
    table9_tpu_zone = args.table9_tpu_zone or args.tpu_zone
    _require_tpu_placement(tpu_type=table9_tpu_type, region=args.tpu_region, zone=table9_tpu_zone)
    _tensor_parallel_size(run_specs[0].model_hidden_dim, table9_tpu_type)
    if args.assignment_file is not None:
        if args.run_orders is not None:
            raise ValueError("Use either --assignment-file or --run-orders, not both")
        if args.assignment_region is None:
            raise ValueError("--assignment-region is required with --assignment-file")
        if args.expect_assignment_sha256 is None:
            raise ValueError("--expect-assignment-sha256 is required with --assignment-file")
        selected_orders, assignment_audit = _assignment_orders(
            args.assignment_file,
            args.assignment_region,
            tpu_region=args.tpu_region,
            experiment_name=args.experiment_name,
            expected_assignment_sha256=args.expect_assignment_sha256,
        )
    else:
        if args.assignment_region is not None:
            raise ValueError("--assignment-region requires --assignment-file")
        if args.expect_assignment_sha256 is not None:
            raise ValueError("--expect-assignment-sha256 requires --assignment-file")
        selected_orders = _parse_run_orders(args.run_orders or "all")
        assignment_audit = {}
    run_specs = [run_specs[order] for order in selected_orders]
    audit.update(
        {
            "experiment_name": args.experiment_name,
            "table9_tpu_type": table9_tpu_type,
            "table9_tpu_zone": table9_tpu_zone,
            "selected_run_count": len(run_specs),
            "selected_run_orders": list(selected_orders),
            "selected_run_orders_sha256": (
                hashlib.sha256(json.dumps(selected_orders, separators=(",", ":")).encode()).hexdigest()
            ),
            **assignment_audit,
        }
    )
    phase_0_checkpoint_step, _ = _phase_0_checkpoint_step(run_specs[0].train_steps)
    if phase_0_checkpoint_step != EXPECTED_PHASE0_CHECKPOINT_STEP:
        raise ValueError(f"Phase-0 checkpoint changed: {phase_0_checkpoint_step} != {EXPECTED_PHASE0_CHECKPOINT_STEP}")
    if run_specs[0].expected_checkpoint_step != EXPECTED_FINAL_CHECKPOINT_STEP:
        raise ValueError(
            f"Final checkpoint changed: {run_specs[0].expected_checkpoint_step} != {EXPECTED_FINAL_CHECKPOINT_STEP}"
        )
    runtime_cache_paths = _runtime_cache_paths(args.tpu_region)
    audit["runtime_cache_count"] = len(runtime_cache_paths)
    audit["runtime_cache_paths_sha256"] = _runtime_paths_sha256(runtime_cache_paths)
    validation_steps = base._default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }
    validation_cache_paths = _require_validation_caches(validation_configs, region=args.tpu_region)
    audit["validation_cache_count"] = len(validation_cache_paths)
    audit["validation_cache_paths_sha256"] = hashlib.sha256(
        json.dumps(validation_cache_paths, separators=(",", ":")).encode()
    ).hexdigest()
    if args.dry_run:
        _write_local_dry_run(
            output_dir=args.dry_run_output,
            source_panel=source_panel,
            analysis_output_path=analysis_output_path,
            run_specs=run_specs,
            audit=audit,
            experiment_name=args.experiment_name,
        )
        logger.info("Wrote %d TPP-40 run specs under %s", len(run_specs), args.dry_run_output)
        return

    with executor_context():
        table9_resources = ResourceConfig.with_tpu(
            table9_tpu_type,
            regions=[args.tpu_region],
            zone=table9_tpu_zone,
            disk="80g",
        )
        artifacts = base.build_launch_artifacts(
            run_specs=run_specs,
            analysis_output_path=analysis_output_path,
            source_panel=source_panel,
            validation_configs=validation_configs,
            experiment_name=args.experiment_name,
            architecture_target_flops=base.TARGET_FLOPS,
            wandb_tags=(
                "delphi-tpp40-augmented-swarm",
                "architecture=3e18-selected",
                "total-tpp=40",
                "fit-panel",
                "two-phase",
                f"deployment={args.tpu_region}-{args.tpu_type}",
            ),
            training_wandb_group=args.training_wandb_group,
            table9_wandb_group=args.table9_wandb_group,
            provenance_panel="delphi_tpp40_augmented_fit_swarm",
            provenance_scale="fixed_n_total_tpp40",
            steps_per_eval=STEPS_PER_EVAL,
            permanent_checkpoint_interval=phase_0_checkpoint_step,
            table9_eval_resources=table9_resources,
        )
    if os.getenv("CI") is not None:
        logger.info(
            "Built TPP-40 graph with %d training and %d Table-9 steps; skipping launch in CI",
            len(artifacts.training_steps),
            len(artifacts.eval_steps),
        )
        return

    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=artifacts.steps,
        description=(
            f"{args.experiment_name}: {len(run_specs)} rows from the exact 280-row augmented fit panel on the "
            "3e18-selected architecture at total TPP 40 with native Table-9 evaluation"
        ),
    )


if __name__ == "__main__":
    main()
