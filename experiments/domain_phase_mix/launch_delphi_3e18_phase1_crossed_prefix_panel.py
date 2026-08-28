# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the frozen Delphi crossed-prefix continuation panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import cast

import fsspec
import pandas as pd
from fray.cluster import ResourceConfig
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main, get_git_commit
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, VersionedValue, this_output_path, versioned
from marin.processing.tokenize import step_to_lm_mixture_component
from rigging.filesystem import marin_prefix_for_region

from experiments.domain_phase_mix import launch_delphi_3e18_phase0_harsh_cap_candidates as harsh
from experiments.domain_phase_mix import launch_delphi_3e18_phase0_prefix_replay as replay
from experiments.domain_phase_mix import launch_delphi_3e18_phase1_harsh_cap_branches as runtime
from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_phase1_crossed_prefix_panel_20260827 as design,
)
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_3e18_phase1_crossed_prefix_panel_20260827"
EXPECTED_CONTRACT_VERSION = "delphi_phase1_crossed_prefix_panel_20260827_v3"
LOCAL_SCREEN_EXPERIMENT_NAME = (
    "pinlin_calvin_xu/data_mixture/delphi_3e18_phase1_local_crossed_prefix_fit079_screen10_20260828"
)
LOCAL_SCREEN_CONTRACT_VERSION = "delphi_phase1_local_crossed_prefix_fit079_20260828_v5"
LOCAL_SCREEN_PANEL_SOURCE = "delphi_phase1_local_crossed_prefix_fit079_screen10"
REFERENCE_OUTPUTS = Path(__file__).resolve().parent / "exploratory" / "two_phase_many" / "reference_outputs"
DEFAULT_DESIGN_DIR = REFERENCE_OUTPUTS / "delphi_phase1_crossed_prefix_panel_v3_20260827"
DEFAULT_MANIFEST = DEFAULT_DESIGN_DIR / "manifest.json"
DEFAULT_PREFIX_REGISTRY = DEFAULT_DESIGN_DIR / "prefix_registry.json"
DEFAULT_PANEL_ROWS = DEFAULT_DESIGN_DIR / "panel_rows.csv"
DEFAULT_PANEL_WEIGHTS = DEFAULT_DESIGN_DIR / "panel_weights.csv"
CONTINUATION_HARDWARE = runtime.TPU_HARDWARE
EXPECTED_PREFIX_COUNT = 9
RUN_NAME_MAX_LENGTH = 180


@dataclass(frozen=True)
class PanelContract:
    contract_version: str
    panel_source: str
    experiment_name: str
    fit_branches_per_prefix: int
    controls_per_prefix: int
    rank_key: str
    tangent_rank: int
    residual_degrees_of_freedom: int
    manifest_binds_names: bool


PANEL_CONTRACTS = {
    EXPECTED_CONTRACT_VERSION: PanelContract(
        contract_version=EXPECTED_CONTRACT_VERSION,
        panel_source="delphi_phase1_crossed_prefix_panel",
        experiment_name=EXPERIMENT_NAME,
        fit_branches_per_prefix=50,
        controls_per_prefix=3,
        rank_key="centered_tangent_rank",
        tangent_rank=38,
        residual_degrees_of_freedom=11,
        manifest_binds_names=False,
    ),
    LOCAL_SCREEN_CONTRACT_VERSION: PanelContract(
        contract_version=LOCAL_SCREEN_CONTRACT_VERSION,
        panel_source=LOCAL_SCREEN_PANEL_SOURCE,
        experiment_name=LOCAL_SCREEN_EXPERIMENT_NAME,
        fit_branches_per_prefix=10,
        controls_per_prefix=3,
        rank_key="anchor_tangent_rank",
        tangent_rank=6,
        residual_degrees_of_freedom=3,
        manifest_binds_names=True,
    ),
}


@dataclass(frozen=True)
class FrozenPrefix:
    state_id: str
    candidate_id: str
    repeat_seed: int
    checkpoint_uri: str
    provenance_sha256: str
    source_family: str
    source_weights_sha256: str
    source_aliases_sha256: str | None
    prefix_replay_code_commit: str
    checkpoint_ready_at_design_time: bool
    run_spec: base.DelphiSwarmRunSpec


@dataclass(frozen=True)
class CrossedBranchTrainingConfig:
    state_id: str
    branch_config: runtime.HarshBranchTrainingConfig
    expected_prefix_fields_json: str


@dataclass(frozen=True)
class SaveResolvedPanelConfig:
    output_path: str
    experiment_name: str
    code_commit: str
    design_manifest_sha256: str | VersionedValue[str]
    prefix_registry_sha256: str
    panel_rows_sha256: str
    panel_weights_sha256: str
    full_rows_json: str
    launched_run_orders: tuple[int, ...] | VersionedValue[tuple[int, ...]]
    bridge_output_root: str


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def panel_contract(contract_version: str) -> PanelContract:
    try:
        return PANEL_CONTRACTS[contract_version]
    except KeyError as error:
        raise ValueError(f"Unknown crossed-panel contract version: {contract_version}") from error


def read_uri_bytes(uri: str) -> bytes:
    fs, path = fsspec.core.url_to_fs(uri)
    with fs.open(path, "rb") as handle:
        return handle.read()


def resolve_prefix(prefix: FrozenPrefix, code_commit: str) -> FrozenPrefix:
    if prefix.state_id != design.BRIDGE_STATE_ID or prefix.checkpoint_ready_at_design_time:
        return prefix
    return replace(
        prefix,
        checkpoint_uri=prefix.checkpoint_uri.format(code_commit=code_commit),
        provenance_sha256=design.RUNTIME_CODE_COMMIT,
        prefix_replay_code_commit=code_commit,
    )


def load_artifacts(
    manifest_path: Path,
    expected_manifest_sha256: str,
    expected_contract_version: str,
    registry_path: Path,
    expected_registry_sha256: str,
    rows_path: Path,
    expected_rows_sha256: str,
    weights_path: Path,
    expected_weights_sha256: str,
) -> tuple[dict[str, object], list[FrozenPrefix], pd.DataFrame, pd.DataFrame]:
    contract = panel_contract(expected_contract_version)
    for path, expected in (
        (manifest_path, expected_manifest_sha256),
        (registry_path, expected_registry_sha256),
        (rows_path, expected_rows_sha256),
        (weights_path, expected_weights_sha256),
    ):
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"Frozen artifact changed: {path} has {actual}, expected {expected}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("contract_version") != expected_contract_version:
        raise ValueError("Frozen crossed-panel contract version changed")
    if contract.manifest_binds_names and manifest.get("panel_source") != contract.panel_source:
        raise ValueError("Frozen crossed-panel source identity changed")
    if contract.manifest_binds_names and manifest.get("experiment_name") != contract.experiment_name:
        raise ValueError("Frozen crossed-panel experiment identity changed")
    if manifest.get("prefix_registry_sha256") != expected_registry_sha256:
        raise ValueError("Manifest references a different prefix registry")
    if manifest.get("panel_rows_sha256") != expected_rows_sha256:
        raise ValueError("Manifest references different panel rows")
    if manifest.get("panel_weights_sha256") != expected_weights_sha256:
        raise ValueError("Manifest references different panel weights")
    prefix_count = int(manifest.get("prefix_count", -1))
    fit_branches_per_prefix = int(manifest.get("fit_branches_per_prefix", -1))
    controls_per_prefix = int(manifest.get("controls_per_prefix", -1))
    fit_rows = int(manifest.get("fit_rows", -1))
    reused_fit_rows = int(manifest.get("reused_fit_rows", -1))
    new_fit_rows = int(manifest.get("new_fit_rows", -1))
    control_rows = int(manifest.get("new_control_rows", -1))
    total_rows = int(manifest.get("total_rows", -1))
    if prefix_count != EXPECTED_PREFIX_COUNT:
        raise ValueError("Frozen crossed-panel prefix count changed")
    if fit_branches_per_prefix != contract.fit_branches_per_prefix:
        raise ValueError("Frozen crossed-panel fit branches per prefix changed")
    if controls_per_prefix != contract.controls_per_prefix:
        raise ValueError("Frozen crossed-panel controls per prefix changed")
    if fit_rows != reused_fit_rows + new_fit_rows:
        raise ValueError("Frozen crossed-panel fit allocation is inconsistent")
    if new_fit_rows != prefix_count * fit_branches_per_prefix:
        raise ValueError("Frozen crossed-panel fit rows do not form a complete cross")
    if control_rows != prefix_count * controls_per_prefix:
        raise ValueError("Frozen crossed-panel controls do not form a complete cross")
    if total_rows != new_fit_rows + control_rows:
        raise ValueError("Frozen crossed-panel total allocation is inconsistent")
    rank_audit = manifest.get("rank_audit")
    if not isinstance(rank_audit, dict):
        raise ValueError("Frozen crossed-panel rank audit is missing")
    tangent_rank = int(rank_audit.get(contract.rank_key, -1))
    residual_degrees = int(rank_audit.get("residual_degrees_of_freedom_per_state", -1))
    if tangent_rank != contract.tangent_rank:
        raise ValueError("Frozen crossed-panel tangent rank changed")
    if residual_degrees != contract.residual_degrees_of_freedom:
        raise ValueError("Frozen crossed-panel residual degrees of freedom changed")

    registry = json.loads(registry_path.read_text())
    prefixes = [
        FrozenPrefix(
            state_id=str(row["state_id"]),
            candidate_id=str(row["candidate_id"]),
            repeat_seed=int(row["repeat_seed"]),
            checkpoint_uri=str(row["checkpoint_uri"]),
            provenance_sha256=str(row["provenance_sha256"]),
            source_family=str(row["source_family"]),
            source_weights_sha256=str(row["source_weights_sha256"]),
            source_aliases_sha256=cast(str | None, row["source_aliases_sha256"]),
            prefix_replay_code_commit=str(row["prefix_replay_code_commit"]),
            checkpoint_ready_at_design_time=bool(row["checkpoint_ready_at_design_time"]),
            run_spec=base.DelphiSwarmRunSpec(**row["run_spec"]),
        )
        for row in registry["prefixes"]
    ]
    if tuple(row.state_id for row in prefixes) != design.STATE_IDS:
        raise ValueError("Prefix registry state identities changed")

    rows = pd.read_csv(rows_path)
    weights = pd.read_csv(weights_path)
    if len(rows) != total_rows or rows.row_id.nunique() != total_rows:
        raise ValueError("Panel row identities changed")
    if len(weights) != total_rows * 39:
        raise ValueError("Panel weight rows changed")
    if not rows.run_order.equals(pd.Series(range(total_rows))):
        raise ValueError("Panel run orders must be contiguous")
    if rows.run_id.nunique() != total_rows or weights.groupby("row_id").size().ne(39).any():
        raise ValueError("Panel runtime identities are not one-to-one")
    if int(rows.fit_budget.sum()) != new_fit_rows:
        raise ValueError("Fit-budget row count changed")
    if weights.groupby("row_id").phase_1_count.sum().ne(replay.MIXTURE_BLOCK_SIZE).any():
        raise ValueError("A continuation does not sum to the runtime mixture block")
    if not (weights.phase_1_weight == weights.phase_1_count / replay.MIXTURE_BLOCK_SIZE).all():
        raise ValueError("Continuation weights are not runtime-exact")
    state_counts = rows.groupby("prefix_state_id").size()
    rows_per_prefix = fit_branches_per_prefix + controls_per_prefix
    if set(state_counts.index) != set(design.STATE_IDS) or state_counts.ne(rows_per_prefix).any():
        raise ValueError("Each prefix state must have the frozen number of fit rows and controls")
    return manifest, prefixes, rows, weights


def expected_prefix_core(prefix: FrozenPrefix) -> dict[str, object]:
    spec = prefix.run_spec
    return {
        "experiment_name": spec.source_experiment,
        "candidate_id": prefix.candidate_id,
        "candidate_weights_sha256": prefix.source_weights_sha256,
        "candidate_aliases_sha256": prefix.source_aliases_sha256,
        "checkpoint_uri": prefix.checkpoint_uri,
        "checkpoint_step": replay.EXPECTED_PREFIX_HF_STEP,
        "trainer_state_step": replay.EXPECTED_PREFIX_TRAIN_STEPS,
        "trainer_seed": prefix.repeat_seed,
        "data_seed": spec.data_seed,
        "replay_code_commit": prefix.prefix_replay_code_commit,
        "run_name": spec.run_name,
        "phase_weights_sha256": runtime.phase_weights_sha256(spec.phase_weights),
    }


def validate_prefix(prefix: FrozenPrefix) -> None:
    spec = prefix.run_spec
    if (
        spec.train_steps != replay.EXPECTED_FULL_TRAIN_STEPS
        or spec.expected_checkpoint_step != replay.EXPECTED_PREFIX_HF_STEP
    ):
        raise ValueError(f"Prefix horizon changed: {prefix.state_id}")
    if not prefix.checkpoint_uri.startswith("gs://marin-us-east5/"):
        raise ValueError(f"Prefix checkpoint is not east5-local: {prefix.checkpoint_uri}")
    if not prefix.checkpoint_uri.endswith(f"/checkpoints/step-{replay.EXPECTED_PREFIX_HF_STEP}"):
        raise ValueError(f"Prefix is not the exact phase boundary: {prefix.checkpoint_uri}")
    expected_hardware = runtime.TpuHardware(tpu_type=spec.tpu_type, region=spec.tpu_region, zone=spec.tpu_zone)
    if prefix.source_family == "cap10_v5p" and expected_hardware != runtime.TpuHardware(
        tpu_type="v5p-8", region="us-east5", zone="us-east5-a"
    ):
        raise ValueError(f"Cap-10 prefix hardware changed: {prefix.state_id}")
    if prefix.source_family != "cap10_v5p" and expected_hardware != CONTINUATION_HARDWARE:
        raise ValueError(f"v6e prefix hardware changed: {prefix.state_id}")
    if not prefix.checkpoint_ready_at_design_time:
        if prefix.state_id != design.BRIDGE_STATE_ID:
            raise ValueError(f"Unexpected pending prefix state: {prefix.state_id}")
        return
    fs, checkpoint_path = fsspec.core.url_to_fs(prefix.checkpoint_uri)
    with fs.open(os.path.join(checkpoint_path, "metadata.json")) as handle:
        metadata = json.load(handle)
    if metadata.get("step") != replay.EXPECTED_PREFIX_HF_STEP or metadata.get("is_temporary") is not False:
        raise ValueError(f"Prefix checkpoint is not permanent: {prefix.checkpoint_uri}")
    output_root = prefix.checkpoint_uri.rsplit("/checkpoints/", maxsplit=1)[0]
    provenance_bytes = read_uri_bytes(f"{output_root}/prefix_provenance.json")
    if hashlib.sha256(provenance_bytes).hexdigest() != prefix.provenance_sha256:
        raise ValueError(f"Prefix provenance changed: {prefix.state_id}")
    provenance = json.loads(provenance_bytes)
    expected_core = expected_prefix_core(prefix)
    observed_core = {key: provenance.get(key) for key in expected_core}
    if observed_core != expected_core:
        raise ValueError(f"Prefix provenance does not match its source spec: {prefix.state_id}")


def row_weights(weights: pd.DataFrame, row_id: str) -> dict[str, float]:
    group = weights.loc[weights.row_id.eq(row_id)]
    return dict(zip(group.bucket.astype(str), group.phase_1_weight.astype(float), strict=True))


def branch_run_spec(
    prefix: FrozenPrefix,
    row: pd.Series,
    weights: pd.DataFrame,
    experiment_name: str,
    panel_source: str,
) -> base.DelphiSwarmRunSpec:
    phase_weights = {
        "phase_0": prefix.run_spec.phase_weights["phase_0"],
        "phase_1": row_weights(weights, str(row.row_id)),
    }
    max_epoch, q95_epoch, phase_tv = base._weight_diagnostics(phase_weights)
    run_name = f"cross_{row.row_id}"
    if len(run_name) > RUN_NAME_MAX_LENGTH:
        raise ValueError(f"Run name is too long: {run_name}")
    return replace(
        prefix.run_spec,
        run_order=int(row.run_order),
        run_id=int(row.run_id),
        run_name=run_name,
        source_run_name=run_name,
        source_experiment=experiment_name,
        panel_source=panel_source,
        data_seed=int(row.data_seed),
        trainer_seed=int(row.trainer_seed),
        tpu_type=CONTINUATION_HARDWARE.tpu_type,
        tpu_region=CONTINUATION_HARDWARE.region,
        tpu_zone=CONTINUATION_HARDWARE.zone,
        tensor_parallel_size=base._tensor_parallel_size(
            prefix.run_spec.model_hidden_dim,
            CONTINUATION_HARDWARE.tpu_type,
        ),
        max_simulated_epoch=max_epoch,
        q95_simulated_epoch=q95_epoch,
        mean_phase_tv_to_proportional=phase_tv,
        phase_weights=phase_weights,
    )


def run_crossed_branch(config: CrossedBranchTrainingConfig) -> None:
    """Validate the exact prefix provenance before running one continuation."""
    checkpoint = config.branch_config.prefix_checkpoint
    output_root = checkpoint.checkpoint_uri.rsplit("/checkpoints/", maxsplit=1)[0]
    provenance_bytes = read_uri_bytes(f"{output_root}/prefix_provenance.json")
    provenance = json.loads(provenance_bytes)
    expected_fields = json.loads(config.expected_prefix_fields_json)
    observed_fields = {key: provenance.get(key) for key in expected_fields}
    if observed_fields != expected_fields:
        raise ValueError(f"Prefix provenance fields changed for {config.state_id}")
    provenance_sha256 = hashlib.sha256(provenance_bytes).hexdigest()
    if checkpoint.provenance_sha256 not in (design.RUNTIME_CODE_COMMIT, provenance_sha256):
        raise ValueError(f"Prefix provenance hash changed for {config.state_id}")
    runtime.run_phase_1_branch(
        replace(
            config.branch_config,
            prefix_checkpoint=replace(checkpoint, provenance_sha256=provenance_sha256),
        )
    )


def bridge_prefix_step(
    prefix: FrozenPrefix,
    code_commit: str,
    analysis_output_path: str,
    validation_configs: dict,
) -> ExecutorStep:
    if prefix.state_id != design.BRIDGE_STATE_ID or prefix.checkpoint_ready_at_design_time:
        raise ValueError("Bridge step requested for a non-bridge prefix")
    resources = ResourceConfig.with_tpu(
        CONTINUATION_HARDWARE.tpu_type,
        regions=[CONTINUATION_HARDWARE.region],
        zone=CONTINUATION_HARDWARE.zone,
    )
    output_root = prefix.checkpoint_uri.rsplit("/checkpoints/", maxsplit=1)[0]
    step = ExecutorStep(
        name=f"{prefix.run_spec.source_experiment}/{prefix.run_spec.run_name}",
        fn=remote(
            harsh.run_harsh_candidate_prefix,
            resources=resources,
            env_vars={base.HF_HUB_DISABLE_XET_ENV_VAR: "1"},
        ),
        resources=resources,
        config=harsh.HarshCandidatePrefixTrainingConfig(
            prefix_config=replay.PrefixTrainingConfig(
                analysis_output_path=analysis_output_path,
                output_path=this_output_path(),
                run_spec=prefix.run_spec,
                validation_configs=validation_configs,
                prefix_train_steps=replay.EXPECTED_PREFIX_TRAIN_STEPS,
                optimizer_schedule_num_train_steps=replay.EXPECTED_FULL_TRAIN_STEPS,
                replay_code_commit=code_commit,
                tracker_tags=(
                    "issue-6611",
                    "delphi-crossed-prefix-hardware-bridge",
                    f"prefix_candidate={prefix.candidate_id}",
                    f"replay_code_commit={code_commit}",
                    "prefix_tpu=v6e-8",
                    "prefix_zone=us-east5-b",
                ),
            ),
            experiment_name=prefix.run_spec.source_experiment,
            candidate_id=prefix.candidate_id,
            candidate_weights_sha256=prefix.source_weights_sha256,
            candidate_aliases_sha256=prefix.source_aliases_sha256,
        ),
    )
    return step.with_output_path(output_root)


def save_resolved_panel(config: SaveResolvedPanelConfig) -> None:
    fs, path = fsspec.core.url_to_fs(config.output_path)
    fs.makedirs(path, exist_ok=True)
    payload = {
        "experiment_name": config.experiment_name,
        "code_commit": config.code_commit,
        "design_manifest_sha256": cast(str, config.design_manifest_sha256),
        "prefix_registry_sha256": config.prefix_registry_sha256,
        "panel_rows_sha256": config.panel_rows_sha256,
        "panel_weights_sha256": config.panel_weights_sha256,
        "full_rows": json.loads(config.full_rows_json),
        "launched_run_orders": list(cast(tuple[int, ...], config.launched_run_orders)),
        "bridge_output_root": config.bridge_output_root,
        "continuation_hardware": asdict(CONTINUATION_HARDWARE),
        "minimum_initial_step": replay.EXPECTED_PREFIX_TRAIN_STEPS,
        "terminal_checkpoint_step": replay.EXPECTED_FULL_TRAIN_STEPS - 1,
    }
    output_path = os.path.join(path, "resolved_panel_manifest.json")
    output_bytes = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if fs.exists(output_path):
        with fs.open(output_path, "rb") as handle:
            if handle.read() != output_bytes:
                raise ValueError(f"Refusing to replace a different resolved panel manifest: {output_path}")
        return
    with fs.open(output_path, "wb") as handle:
        handle.write(output_bytes)


def default_dry_run_output_dir(
    manifest_parent: Path,
    manifest_sha256: str,
    code_commit: str,
    launched_orders: tuple[int, ...],
) -> Path:
    identity = json.dumps(
        {
            "manifest_sha256": manifest_sha256,
            "code_commit": code_commit,
            "launched_orders": launched_orders,
        },
        sort_keys=True,
    ).encode()
    return manifest_parent / "launch_dry_run" / hashlib.sha256(identity).hexdigest()[:20]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-contract-version", required=True)
    parser.add_argument("--expected-panel-source", required=True)
    parser.add_argument("--prefix-registry", type=Path, default=DEFAULT_PREFIX_REGISTRY)
    parser.add_argument("--expected-prefix-registry-sha256", required=True)
    parser.add_argument("--panel-rows", type=Path, default=DEFAULT_PANEL_ROWS)
    parser.add_argument("--expected-panel-rows-sha256", required=True)
    parser.add_argument("--panel-weights", type=Path, default=DEFAULT_PANEL_WEIGHTS)
    parser.add_argument("--expected-panel-weights-sha256", required=True)
    parser.add_argument("--analysis-output-path", default=base.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--max-concurrent", type=int)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--run-order", action="append", type=int, dest="run_orders")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-output-dir", type=Path)
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = parse_args()
    sys.argv = [sys.argv[0], *remaining]
    expected_prefix = marin_prefix_for_region(CONTINUATION_HARDWARE.region)
    if os.environ.get("MARIN_PREFIX", expected_prefix) != expected_prefix:
        raise ValueError(f"MARIN_PREFIX must be {expected_prefix}")
    os.environ["MARIN_PREFIX"] = expected_prefix
    code_commit = replay.validate_replay_code_commit(args.code_commit, get_git_commit())
    manifest, frozen_prefixes, rows, weights = load_artifacts(
        args.manifest,
        args.expected_manifest_sha256,
        args.expected_contract_version,
        args.prefix_registry,
        args.expected_prefix_registry_sha256,
        args.panel_rows,
        args.expected_panel_rows_sha256,
        args.panel_weights,
        args.expected_panel_weights_sha256,
    )
    contract = panel_contract(args.expected_contract_version)
    if args.expected_panel_source != contract.panel_source:
        raise ValueError("Command uses the wrong crossed-panel source identity")
    if args.experiment_name != contract.experiment_name:
        raise ValueError("Command uses the wrong crossed-panel experiment identity")
    max_concurrent = len(rows) + 1 if args.max_concurrent is None else args.max_concurrent
    if not 1 <= max_concurrent <= len(rows) + 1:
        raise ValueError(f"--max-concurrent must be in [1, {len(rows) + 1}]")
    prefixes = [resolve_prefix(prefix, code_commit) for prefix in frozen_prefixes]
    prefixes_by_state = {prefix.state_id: prefix for prefix in prefixes}
    for prefix in prefixes:
        validate_prefix(prefix)

    resolved_specs = {
        int(row.run_order): branch_run_spec(
            prefixes_by_state[str(row.prefix_state_id)],
            row,
            weights,
            args.experiment_name,
            args.expected_panel_source,
        )
        for _, row in rows.iterrows()
    }
    launch_rows = rows
    if args.run_orders is not None:
        requested = tuple(dict.fromkeys(args.run_orders))
        unknown = sorted(set(requested) - set(launch_rows.run_order.astype(int)))
        if unknown:
            raise ValueError(f"Requested run orders are absent: {unknown}")
        launch_rows = launch_rows.loc[launch_rows.run_order.isin(requested)]
    launched_orders = tuple(launch_rows.run_order.astype(int))
    bridge_prefix = prefixes_by_state[design.BRIDGE_STATE_ID]
    bridge_output_root = bridge_prefix.checkpoint_uri.rsplit("/checkpoints/", maxsplit=1)[0]
    serializable_rows = json.loads(rows.to_json(orient="records"))
    dry_run_output_dir = args.dry_run_output_dir or default_dry_run_output_dir(
        args.manifest.parent,
        args.expected_manifest_sha256,
        code_commit,
        launched_orders,
    )
    save_config = SaveResolvedPanelConfig(
        output_path=str(dry_run_output_dir),
        experiment_name=args.experiment_name,
        code_commit=code_commit,
        design_manifest_sha256=args.expected_manifest_sha256,
        prefix_registry_sha256=args.expected_prefix_registry_sha256,
        panel_rows_sha256=args.expected_panel_rows_sha256,
        panel_weights_sha256=args.expected_panel_weights_sha256,
        full_rows_json=json.dumps(serializable_rows, sort_keys=True),
        launched_run_orders=launched_orders,
        bridge_output_root=bridge_output_root,
    )
    if args.dry_run:
        save_resolved_panel(save_config)
        logger.info("Validated %d rows; %d selected for launch", len(rows), len(launch_rows))
        return

    validation_steps = base._default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }
    resources = ResourceConfig.with_tpu(
        CONTINUATION_HARDWARE.tpu_type,
        regions=[CONTINUATION_HARDWARE.region],
        zone=CONTINUATION_HARDWARE.zone,
    )
    panel_identity = versioned(args.expected_manifest_sha256)
    with executor_context():
        bridge_rows_selected = launch_rows.prefix_state_id.eq(design.BRIDGE_STATE_ID).any()
        bridge_step = (
            bridge_prefix_step(bridge_prefix, code_commit, args.analysis_output_path, validation_configs)
            if bridge_rows_selected and not bridge_prefix.checkpoint_ready_at_design_time
            else None
        )
        steps = [] if bridge_step is None else [bridge_step]
        for _, row in launch_rows.iterrows():
            prefix = prefixes_by_state[str(row.prefix_state_id)]
            run_spec = resolved_specs[int(row.run_order)]
            checkpoint_uri = prefix.checkpoint_uri
            if prefix.state_id == design.BRIDGE_STATE_ID and not prefix.checkpoint_ready_at_design_time:
                if bridge_step is None:
                    raise ValueError("Bridge continuation selected without its prefix dependency")
                checkpoint_uri = cast(str, bridge_step / f"checkpoints/step-{replay.EXPECTED_PREFIX_HF_STEP}")
            prefix_checkpoint = runtime.PrefixCheckpoint(
                candidate_id=prefix.candidate_id,
                repeat_seed=prefix.repeat_seed,
                checkpoint_uri=checkpoint_uri,
                provenance_sha256=prefix.provenance_sha256,
            )
            expected_fields = expected_prefix_core(prefix)
            steps.append(
                ExecutorStep(
                    name=f"{args.experiment_name}/{run_spec.run_name}",
                    fn=remote(
                        run_crossed_branch,
                        resources=resources,
                        env_vars={base.HF_HUB_DISABLE_XET_ENV_VAR: "1"},
                    ),
                    resources=resources,
                    config=CrossedBranchTrainingConfig(
                        state_id=prefix.state_id,
                        branch_config=runtime.HarshBranchTrainingConfig(
                            experiment_name=args.experiment_name,
                            analysis_output_path=args.analysis_output_path,
                            output_path=this_output_path(),
                            run_spec=run_spec,
                            validation_configs=validation_configs,
                            prefix_checkpoint=prefix_checkpoint,
                            prefix_replay_code_commit=prefix.prefix_replay_code_commit,
                            candidate_weights_sha256=prefix.source_weights_sha256,
                            candidate_aliases_sha256=prefix.source_aliases_sha256,
                            continuation_weights_sha256=args.expected_panel_weights_sha256,
                            design_manifest_sha256=args.expected_manifest_sha256,
                            continuation_id=str(row.continuation_id),
                            code_commit=code_commit,
                            prefix_hardware=runtime.TpuHardware(
                                tpu_type=prefix.run_spec.tpu_type,
                                region=prefix.run_spec.tpu_region,
                                zone=prefix.run_spec.tpu_zone,
                            ),
                            panel_identity=panel_identity,
                        ),
                        expected_prefix_fields_json=json.dumps(expected_fields, sort_keys=True),
                    ),
                )
            )
        steps.append(
            ExecutorStep(
                name=f"{args.experiment_name}/resolved_manifest",
                fn=save_resolved_panel,
                config=replace(
                    save_config,
                    output_path=this_output_path(),
                    design_manifest_sha256=versioned(args.expected_manifest_sha256),
                    launched_run_orders=versioned(launched_orders),
                ),
            )
        )
    if os.getenv("CI") is not None:
        logger.info("Built %d crossed-panel steps; skipping launch in CI", len(steps))
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=max_concurrent),
        steps=steps,
        description=(
            f"{args.experiment_name}: nine prefix states crossed with "
            f"{manifest['fit_branches_per_prefix']} common phase-1 actions; "
            f"{manifest['reused_fit_rows']} old cells reused"
        ),
    )


if __name__ == "__main__":
    main()
