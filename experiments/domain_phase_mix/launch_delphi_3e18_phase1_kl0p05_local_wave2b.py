# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Continue the frozen KL0.05 prefix over the local 96-row Wave-2B panel."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from fray.cluster import ResourceConfig
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main, get_git_commit
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, this_output_path, versioned
from marin.processing.tokenize import step_to_lm_mixture_component

from experiments.domain_phase_mix import launch_delphi_3e18_phase0_prefix_replay as replay
from experiments.domain_phase_mix import launch_delphi_3e18_phase1_common_branches as common
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_3e18_phase1_kl0p05_local_wave2b_v6e8_20260825"
TARGET_PREFIX = "shared_bounded_ensemble_kl0p05"
EXPECTED_CONTINUATIONS = 96
EXPECTED_FIT_CONTINUATIONS = 80
EXPECTED_REFEREE_HOLDOUTS = 8
EXPECTED_TIED_CONTROLS = 8
EXPECTED_PREFIX_SEEDS = (0, 1)
SELECTION_MODES = ("post_wave1_local_redesign",)
BRANCH_RUN_ID_BASE = 954_000
DEFAULT_MAX_CONCURRENT = 96
LOCAL_ARTIFACT_ROOT = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "delphi_phase1_kl0p05_local_wave2b_launch_20260825"
)
DEFAULT_SELECTION_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "delphi_phase1_kl0p05_local_wave2b_20260825"
)
DEFAULT_CONTINUATION_WEIGHTS = DEFAULT_SELECTION_DIR / "continuation_weights.csv"
DEFAULT_SELECTION_MANIFEST = DEFAULT_SELECTION_DIR / "manifest.json"
DEFAULT_SELECTION_CONTRACT = DEFAULT_SELECTION_DIR / "contract.json"


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"Expected a boolean value, found {value!r}")


def verify_selection_artifacts(
    continuation_weights: Path,
    expected_continuation_sha256: str,
    selection_manifest: Path,
    expected_selection_manifest_sha256: str,
    selection_contract: Path,
    expected_selection_contract_sha256: str,
    expected_selection_mode: str,
) -> None:
    if common.file_sha256(continuation_weights) != expected_continuation_sha256:
        raise ValueError("Local Wave-2B continuation weights changed")
    if common.file_sha256(selection_manifest) != expected_selection_manifest_sha256:
        raise ValueError("Local Wave-2B selection manifest changed")
    if common.file_sha256(selection_contract) != expected_selection_contract_sha256:
        raise ValueError("Local Wave-2B selection contract changed")
    manifest = json.loads(selection_manifest.read_text())
    contract = json.loads(selection_contract.read_text())
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or artifacts.get("continuation_weights.csv") != expected_continuation_sha256:
        raise ValueError("Selection manifest references different continuation weights")
    if manifest.get("contract_sha256") != expected_selection_contract_sha256:
        raise ValueError("Selection manifest references a different frozen contract")
    expected_rows = {
        "fit": EXPECTED_FIT_CONTINUATIONS,
        "referee": EXPECTED_REFEREE_HOLDOUTS,
        "tied_control": EXPECTED_TIED_CONTROLS,
        "total": EXPECTED_CONTINUATIONS,
    }
    if manifest.get("target_prefix") != TARGET_PREFIX or manifest.get("rows") != expected_rows:
        raise ValueError("Selection manifest target or row count changed")
    if manifest.get("selection_mode") != expected_selection_mode:
        raise ValueError(f"Selection mode changed: {manifest.get('selection_mode')!r} != {expected_selection_mode!r}")
    if contract.get("target_prefix") != TARGET_PREFIX or contract.get("target_metric") != "uncheatable_bpb":
        raise ValueError("Selection contract target prefix changed")
    if contract.get("rows") != expected_rows:
        raise ValueError("Selection contract Wave-2 budget changed")
    contract_artifacts = contract.get("artifacts")
    if (
        not isinstance(contract_artifacts, dict)
        or contract_artifacts.get("continuation_weights.csv") != expected_continuation_sha256
    ):
        raise ValueError("Selection contract references different continuation weights")


def load_wave2_continuations(
    path: Path,
    expected_sha256: str,
    candidate_weights_path: Path,
    expected_candidate_weights_sha256: str,
) -> tuple[tuple[str, ...], list[dict[str, object]]]:
    if common.file_sha256(path) != expected_sha256:
        raise ValueError("Local Wave-2B continuation weights changed")
    frame = pd.read_csv(path)
    required = {
        "continuation_id",
        "role",
        "selection_tranche",
        "fit_budget",
        "referee_holdout",
        "prefix_repeat_seed",
        "data_seed",
        "bucket",
        "phase_1_count",
        "phase_1_weight",
        "phase_1_materialized_epochs",
        "phase_1_support_cap",
        "total_support_cap",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Wave-2 continuation weights are missing columns: {sorted(required - set(frame.columns))}")
    if common.file_sha256(candidate_weights_path) != expected_candidate_weights_sha256:
        raise ValueError("Candidate prefix weights changed")
    candidate_frame = pd.read_csv(candidate_weights_path)
    candidate_required = {
        "candidate_id",
        "bucket",
        "phase_0_weight",
        "phase_0_materialized_epochs",
    }
    if not candidate_required.issubset(candidate_frame.columns):
        raise ValueError("Candidate prefix weights are missing exposure columns")
    target_candidate = candidate_frame[candidate_frame.candidate_id.eq(TARGET_PREFIX)]
    if target_candidate.empty:
        raise ValueError(f"Candidate weights do not contain {TARGET_PREFIX}")
    phase_0_scales = common.recover_epoch_scales(target_candidate, "phase_0_materialized_epochs", "phase_0_weight")
    phase_1_scales = common.recover_epoch_scales(frame, "phase_1_materialized_epochs", "phase_1_weight")
    continuation_ids = tuple(frame.continuation_id.drop_duplicates())
    if len(continuation_ids) != EXPECTED_CONTINUATIONS:
        raise ValueError(f"Expected {EXPECTED_CONTINUATIONS} Wave-2 continuations; found {len(continuation_ids)}")
    buckets = tuple(str(bucket) for bucket in frame.loc[frame.continuation_id.eq(continuation_ids[0]), "bucket"])
    rows: list[dict[str, object]] = []
    fit_count = 0
    referee_count = 0
    tied_control_count = 0
    for continuation_id in continuation_ids:
        group = frame[frame.continuation_id.eq(continuation_id)]
        if tuple(group.bucket) != buckets:
            raise ValueError(f"Bucket order changed for {continuation_id}")
        counts = group.phase_1_count.to_numpy(dtype=int)
        weights = group.phase_1_weight.to_numpy(dtype=float)
        if counts.sum() != replay.MIXTURE_BLOCK_SIZE or np.any(counts < 0):
            raise ValueError(f"Invalid runtime counts for {continuation_id}")
        if not np.array_equal(weights, counts / replay.MIXTURE_BLOCK_SIZE):
            raise ValueError(f"Wave-2 weights are not runtime exact for {continuation_id}")
        expected_exposure = weights * np.asarray([phase_1_scales[str(bucket)] for bucket in buckets])
        if not np.allclose(group.phase_1_materialized_epochs.to_numpy(dtype=float), expected_exposure, atol=1e-9):
            raise ValueError(f"Stored phase-1 exposure changed for {continuation_id}")
        phase_1_caps = group.phase_1_support_cap.to_numpy(dtype=float)
        total_caps = group.total_support_cap.to_numpy(dtype=float)
        if np.any(expected_exposure > phase_1_caps + 1e-12):
            raise ValueError(f"Per-bucket phase-1 support exceeded by {continuation_id}")
        if tuple(target_candidate.bucket) != buckets:
            raise ValueError("Prefix and local Wave-2B bucket orders disagree")
        phase_0_exposure = target_candidate.phase_0_weight.to_numpy(dtype=float) * np.asarray(
            [phase_0_scales[str(bucket)] for bucket in buckets]
        )
        if np.any(phase_0_exposure + expected_exposure > total_caps + 1e-12):
            raise ValueError(f"Per-bucket total-exposure support exceeded by {continuation_id}")
        fit_values = {parse_bool(value) for value in group.fit_budget}
        referee_values = {parse_bool(value) for value in group.referee_holdout}
        roles = set(group.role.astype(str))
        tranches = set(group.selection_tranche.astype(str))
        prefix_seeds = set(group.prefix_repeat_seed.astype(int))
        data_seeds = set(group.data_seed.astype(int))
        if any(len(values) != 1 for values in (fit_values, referee_values, roles, tranches, prefix_seeds, data_seeds)):
            raise ValueError(f"Local Wave-2B metadata changes within {continuation_id}")
        tranche = next(iter(tranches))
        fit_budget = next(iter(fit_values))
        referee = next(iter(referee_values))
        prefix_seed = next(iter(prefix_seeds))
        data_seed = next(iter(data_seeds))
        if prefix_seed not in EXPECTED_PREFIX_SEEDS:
            raise ValueError(f"Unexpected prefix seed {prefix_seed} for {continuation_id}")
        if fit_budget and referee:
            raise ValueError(f"Referee row {continuation_id} cannot enter the fit budget")
        if fit_budget:
            fit_count += 1
        referee_count += referee
        if tranche == "tied_control":
            tied_control_count += 1
            tied_counts = target_candidate.phase_0_weight.to_numpy(dtype=float)
            tied_counts = common.replay.MIXTURE_BLOCK_SIZE * tied_counts
            if not np.array_equal(counts, np.rint(tied_counts).astype(int)):
                raise ValueError(f"Tied control {continuation_id} no longer matches the runtime-exact prefix")
        elif prefix_seed != 0 or data_seed != 930_000:
            raise ValueError(f"Fit/referee row {continuation_id} changed its prefix or data seed")
        rows.append(
            {
                "continuation_id": str(continuation_id),
                "role": next(iter(roles)),
                "selection_tranche": tranche,
                "fit_budget": fit_budget,
                "referee_holdout": referee,
                "prefix_repeat_seed": prefix_seed,
                "data_seed": data_seed,
                "weights": dict(zip(buckets, weights, strict=True)),
            }
        )
    if fit_count != EXPECTED_FIT_CONTINUATIONS:
        raise ValueError(f"Expected {EXPECTED_FIT_CONTINUATIONS} fit rows; found {fit_count}")
    if referee_count != EXPECTED_REFEREE_HOLDOUTS:
        raise ValueError(f"Local Wave-2B referee count changed: {referee_count}")
    if tied_control_count != EXPECTED_TIED_CONTROLS:
        raise ValueError(f"Local Wave-2B tied-control count changed: {tied_control_count}")
    return buckets, rows


def target_prefix_checkpoints(
    prefixes: list[common.PrefixCheckpoint],
) -> dict[int, common.PrefixCheckpoint]:
    matches = [
        prefix
        for prefix in prefixes
        if prefix.candidate_id == TARGET_PREFIX and prefix.repeat_seed in EXPECTED_PREFIX_SEEDS
    ]
    by_seed = {prefix.repeat_seed: prefix for prefix in matches}
    if tuple(sorted(by_seed)) != EXPECTED_PREFIX_SEEDS or len(matches) != len(by_seed):
        raise ValueError(f"Expected one exact {TARGET_PREFIX} checkpoint for seeds {EXPECTED_PREFIX_SEEDS}")
    return by_seed


def wave2_rows(
    prefixes: dict[int, common.PrefixCheckpoint],
    prefix_specs: dict[tuple[str, int], common.base.DelphiSwarmRunSpec],
    continuations: list[dict[str, object]],
    run_id_base: int,
) -> list[dict[str, object]]:
    rows = []
    for run_order, continuation in enumerate(continuations):
        prefix_seed = int(continuation["prefix_repeat_seed"])
        prefix = prefixes[prefix_seed]
        prefix_spec = prefix_specs[(prefix.candidate_id, prefix.repeat_seed)]
        rows.append(
            {
                "run_order": run_order,
                "fit_budget": continuation["fit_budget"],
                "branch_role": continuation["role"],
                "prefix": prefix,
                "continuation_id": continuation["continuation_id"],
                "continuation_role": continuation["role"],
                "selection_tranche": continuation["selection_tranche"],
                "referee_holdout": continuation["referee_holdout"],
                "data_seed": continuation["data_seed"],
                "phase_weights": {
                    "phase_0": prefix_spec.phase_weights["phase_0"],
                    "phase_1": continuation["weights"],
                },
            }
        )
    enriched = common.enrich_branch_rows(rows, prefix_specs, run_id_base)
    if len(enriched) != EXPECTED_CONTINUATIONS:
        raise ValueError("Local Wave-2B row count changed")
    if sum(bool(row["fit_budget"]) for row in enriched) != EXPECTED_FIT_CONTINUATIONS:
        raise ValueError("Local Wave-2B fit budget changed")
    return enriched


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-weights", type=Path, default=common.DEFAULT_CANDIDATE_WEIGHTS)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--continuation-weights", type=Path, default=DEFAULT_CONTINUATION_WEIGHTS)
    parser.add_argument("--expected-continuation-sha256", required=True)
    parser.add_argument("--selection-manifest", type=Path, default=DEFAULT_SELECTION_MANIFEST)
    parser.add_argument("--expected-selection-manifest-sha256", required=True)
    parser.add_argument("--selection-contract", type=Path, default=DEFAULT_SELECTION_CONTRACT)
    parser.add_argument("--expected-selection-contract-sha256", required=True)
    parser.add_argument("--expected-selection-mode", choices=SELECTION_MODES, required=True)
    parser.add_argument("--selected-prefixes", required=True)
    parser.add_argument("--expected-selected-prefixes-sha256", required=True)
    parser.add_argument("--prefix-replay-code-commit", required=True)
    parser.add_argument("--analysis-output-path", default=common.base.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--branch-tpu-type", required=True)
    parser.add_argument("--branch-tpu-region", required=True)
    parser.add_argument("--branch-tpu-zone", required=True)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--branch-run-id-base", type=int, default=BRANCH_RUN_ID_BASE)
    parser.add_argument("--run-order", action="append", type=int, dest="run_orders")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-output-dir", type=Path)
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = parse_args()
    sys.argv = [sys.argv[0], *remaining]
    deployment = common.resolve_branch_deployment(
        args.branch_tpu_type,
        args.branch_tpu_region,
        args.branch_tpu_zone,
    )
    if deployment != common.V6E_DEPLOYMENT:
        raise ValueError(f"Local Wave 2B is frozen to {common.V6E_DEPLOYMENT.hardware}")
    if not 1 <= args.max_concurrent <= DEFAULT_MAX_CONCURRENT:
        raise ValueError(f"--max-concurrent must be in [1, {DEFAULT_MAX_CONCURRENT}]")
    if args.branch_run_id_base != BRANCH_RUN_ID_BASE:
        raise ValueError(f"Local Wave-2B run-ID base must remain {BRANCH_RUN_ID_BASE}")
    expected_prefix = common.marin_prefix_for_region(deployment.hardware.region)
    if os.environ.get("MARIN_PREFIX", expected_prefix) != expected_prefix:
        raise ValueError(f"MARIN_PREFIX must be {expected_prefix}")
    os.environ["MARIN_PREFIX"] = expected_prefix
    code_commit = replay.validate_replay_code_commit(args.code_commit, get_git_commit())
    verify_selection_artifacts(
        args.continuation_weights,
        args.expected_continuation_sha256,
        args.selection_manifest,
        args.expected_selection_manifest_sha256,
        args.selection_contract,
        args.expected_selection_contract_sha256,
        args.expected_selection_mode,
    )
    logger.info("Launching the explicitly acknowledged local Wave-2B mode %s", args.expected_selection_mode)

    buckets, continuations = load_wave2_continuations(
        args.continuation_weights,
        args.expected_continuation_sha256,
        args.candidate_weights,
        args.expected_candidate_sha256,
    )
    prefix_specs = common.source_prefix_specs(
        candidate_weights_path=args.candidate_weights,
        candidate_weights_sha256=args.expected_candidate_sha256,
        analysis_output_path=args.analysis_output_path,
        tpu_region=common.PREFIX_HARDWARE.region,
        tpu_zone=common.PREFIX_HARDWARE.zone,
    )
    expected_phase_hashes = {
        identity: common.phase_weights_sha256(spec.phase_weights) for identity, spec in prefix_specs.items()
    }
    selected_prefixes = common.load_selected_prefixes(
        args.selected_prefixes,
        args.expected_selected_prefixes_sha256,
        args.expected_candidate_sha256,
        args.prefix_replay_code_commit,
        expected_phase_hashes,
    )
    prefixes = target_prefix_checkpoints(selected_prefixes)
    runtime_buckets = tuple(prefix_specs[(TARGET_PREFIX, 0)].phase_weights["phase_0"])
    if set(runtime_buckets) != set(buckets):
        raise ValueError("Prefix and local Wave-2B bucket sets disagree")
    for continuation in continuations:
        weights = cast(dict[str, float], continuation["weights"])
        continuation["weights"] = {bucket: weights[bucket] for bucket in runtime_buckets}
    all_rows = wave2_rows(prefixes, prefix_specs, continuations, args.branch_run_id_base)
    rows = all_rows
    if args.run_orders is not None:
        selected_orders = tuple(dict.fromkeys(args.run_orders))
        unknown = sorted(set(selected_orders) - set(range(EXPECTED_CONTINUATIONS)))
        if unknown:
            raise ValueError(f"Unknown --run-order values: {unknown}")
        rows = [row for row in rows if int(row["run_order"]) in selected_orders]
    serializable_rows = [{**row, "prefix": asdict(row["prefix"])} for row in rows]

    manifest_config = common.SaveBranchManifestConfig(
        experiment_name=EXPERIMENT_NAME,
        output_path=str(args.dry_run_output_dir or LOCAL_ARTIFACT_ROOT),
        selected_prefixes_json=json.dumps([asdict(prefixes[seed]) for seed in EXPECTED_PREFIX_SEEDS], sort_keys=True),
        selected_prefixes_sha256=args.expected_selected_prefixes_sha256,
        candidate_weights_sha256=args.expected_candidate_sha256,
        continuation_weights_sha256=args.expected_continuation_sha256,
        prefix_replay_code_commit=args.prefix_replay_code_commit,
        code_commit=code_commit,
        branch_run_id_base=args.branch_run_id_base,
        branch_noise_design_sha256=None,
        expected_full_design_rows=EXPECTED_CONTINUATIONS,
        continuation_weights_version=versioned(args.expected_continuation_sha256),
        branch_run_id_base_version=versioned(args.branch_run_id_base),
        branch_rows_json=json.dumps(serializable_rows, sort_keys=True),
        selected_run_orders=versioned(tuple(int(row["run_order"]) for row in serializable_rows)),
        prefix_hardware=common.PREFIX_HARDWARE,
        continuation_hardware=deployment.hardware,
        continuation_hardware_version=versioned(common.hardware_identity(deployment.hardware)),
        selection_manifest_sha256=args.expected_selection_manifest_sha256,
        selection_contract_sha256=args.expected_selection_contract_sha256,
    )
    if args.dry_run:
        common.save_branch_manifest(manifest_config)
        logger.info("Wrote %d Wave-2 branch specs under %s", len(rows), manifest_config.output_path)
        return

    validation_steps = common.base._default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }
    steps = []
    with executor_context():
        for row in rows:
            run_order = int(row["run_order"])
            run_name = str(row["run_name"])
            prefix = cast(common.PrefixCheckpoint, row["prefix"])
            prefix_spec = prefix_specs[(prefix.candidate_id, prefix.repeat_seed)]
            run_spec = common.move_run_spec_to_branch_hardware(
                replace(
                    prefix_spec,
                    run_order=run_order,
                    run_id=int(row["run_id"]),
                    run_name=run_name,
                    source_run_name=run_name,
                    source_experiment=EXPERIMENT_NAME,
                    panel_source="sequential_phase1_kl0p05_local_wave2b",
                    data_seed=int(row["data_seed"]),
                    trainer_seed=int(row["trainer_seed"]),
                    max_simulated_epoch=float(row["max_simulated_epoch"]),
                    q95_simulated_epoch=float(row["q95_simulated_epoch"]),
                    mean_phase_tv_to_proportional=float(row["mean_phase_tv_to_proportional"]),
                    phase_weights=row["phase_weights"],
                ),
                deployment,
            )
            resources = ResourceConfig.with_tpu(
                run_spec.tpu_type,
                regions=[run_spec.tpu_region],
                zone=run_spec.tpu_zone,
            )
            steps.append(
                ExecutorStep(
                    name=f"{EXPERIMENT_NAME}/{run_name}",
                    fn=remote(
                        common.run_phase_1_branch,
                        resources=resources,
                        env_vars={common.base.HF_HUB_DISABLE_XET_ENV_VAR: "1"},
                    ),
                    resources=resources,
                    config=common.BranchTrainingConfig(
                        experiment_name=EXPERIMENT_NAME,
                        analysis_output_path=args.analysis_output_path,
                        output_path=this_output_path(),
                        run_spec=run_spec,
                        validation_configs=validation_configs,
                        prefix_checkpoint=prefix,
                        prefix_replay_code_commit=args.prefix_replay_code_commit,
                        candidate_weights_sha256=args.expected_candidate_sha256,
                        continuation_weights_sha256=args.expected_continuation_sha256,
                        continuation_id=str(row["continuation_id"]),
                        code_commit=code_commit,
                        prefix_hardware=common.PREFIX_HARDWARE,
                        continuation_hardware=deployment.hardware,
                        continuation_hardware_version=versioned(common.hardware_identity(deployment.hardware)),
                        selection_manifest_sha256=args.expected_selection_manifest_sha256,
                        selection_contract_sha256=args.expected_selection_contract_sha256,
                    ),
                )
            )
        steps.append(
            ExecutorStep(
                name=f"{EXPERIMENT_NAME}/manifest",
                fn=common.save_branch_manifest,
                config=replace(manifest_config, output_path=this_output_path()),
            )
        )
    if os.getenv("CI") is not None:
        logger.info("Built %d Wave-2 steps; skipping launch in CI", len(steps))
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=steps,
        description=f"{EXPERIMENT_NAME}: local antithetic fixed-prefix phase-1 continuation panel",
    )


if __name__ == "__main__":
    main()
