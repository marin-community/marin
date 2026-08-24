# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Continue selected Delphi prefixes over a frozen fully crossed phase-1 panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import fsspec
import pandas as pd
from fray.cluster import ResourceConfig
from levanter.data.text.datasets import DatasetComponent
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main, get_git_commit
from marin.execution.remote import remote
from marin.execution.types import ExecutorStep, this_output_path
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.rl.placement import marin_prefix_for_region
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm

from experiments.domain_phase_mix import launch_delphi_3e18_phase0_prefix_candidates as candidates
from experiments.domain_phase_mix import launch_delphi_3e18_phase0_prefix_replay as replay
from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_3e18_phase1_common_branches_20260824"
DEFAULT_CANDIDATE_WEIGHTS = candidates.DEFAULT_CANDIDATE_WEIGHTS
DEFAULT_CONTINUATION_WEIGHTS = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "delphi_phase1_common_branches_20260824"
    / "continuation_weights.csv"
)
LOCAL_ARTIFACT_DIR = (
    Path(__file__).resolve().parent
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "delphi_phase1_common_branches_20260824"
    / "launch_dry_run"
)
SELECTED_PREFIX_COUNT = 4
COMMON_FIT_CONTINUATION_COUNT = 50
COMMON_CONTROL_CONTINUATION_COUNT = 3
COMMON_CONTINUATION_COUNT = COMMON_FIT_CONTINUATION_COUNT + COMMON_CONTROL_CONTINUATION_COUNT
PRIMARY_BRANCH_SEED = 0
STABILITY_BRANCH_SEED = 1
STABILITY_CONTINUATION_COUNT = 3
STABILITY_CONTROL_IDS = {"control_proportional", "control_incumbent_planned"}
BRANCH_NOISE_CONTINUATION_ID = "control_proportional"
BRANCH_NOISE_REPEAT_COUNT = 2
BRANCH_NOISE_DATA_SEED_BASE = 970_000
REQUIRED_SELECTED_CANDIDATES = {"observed_cap10_best"}
HISTORICAL_PHASE_1_EPOCH_CAP = 62.281654
BRANCH_RUN_ID_BASE = 950_000
TOTAL_BRANCH_ROWS = SELECTED_PREFIX_COUNT * (
    COMMON_CONTINUATION_COUNT + 1 + STABILITY_CONTINUATION_COUNT + BRANCH_NOISE_REPEAT_COUNT
)
DEFAULT_MAX_CONCURRENT = 64


@dataclass(frozen=True)
class PrefixCheckpoint:
    candidate_id: str
    repeat_seed: int
    checkpoint_uri: str


@dataclass(frozen=True)
class BranchTrainingConfig:
    analysis_output_path: str
    output_path: str
    run_spec: base.DelphiSwarmRunSpec
    validation_configs: dict[str, DatasetComponent] | None
    prefix_checkpoint: PrefixCheckpoint
    code_commit: str


@dataclass(frozen=True)
class SaveBranchManifestConfig:
    output_path: str
    selected_prefixes_json: str
    selected_prefixes_sha256: str
    candidate_weights_sha256: str
    continuation_weights_sha256: str
    code_commit: str
    branch_rows_json: str


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_uri_bytes(uri: str) -> bytes:
    fs, path = fsspec.core.url_to_fs(uri)
    with fs.open(path, "rb") as handle:
        return handle.read()


def load_selected_prefixes(
    path: str,
    expected_sha256: str,
    expected_candidate_weights_sha256: str,
    expected_code_commit: str,
) -> list[PrefixCheckpoint]:
    payload_bytes = read_uri_bytes(path)
    actual = hashlib.sha256(payload_bytes).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"Selected-prefix manifest changed: {actual} != {expected_sha256}")
    payload = json.loads(payload_bytes)
    if payload.get("candidate_weights_sha256") != expected_candidate_weights_sha256:
        raise ValueError("Selected-prefix manifest references different candidate weights")
    if payload.get("prefix_replay_code_commit") != expected_code_commit:
        raise ValueError("Selected-prefix checkpoint code differs from the continuation code")
    rows = [PrefixCheckpoint(**item) for item in payload["prefixes"]]
    candidate_ids = {row.candidate_id for row in rows}
    if len(candidate_ids) != SELECTED_PREFIX_COUNT:
        raise ValueError(f"Expected {SELECTED_PREFIX_COUNT} distinct selected prefixes")
    if not REQUIRED_SELECTED_CANDIDATES.issubset(candidate_ids):
        raise ValueError(f"Selected prefixes must include {sorted(REQUIRED_SELECTED_CANDIDATES)}")
    expected_rows = SELECTED_PREFIX_COUNT * 2
    if len(rows) != expected_rows:
        raise ValueError(f"Expected primary and stability checkpoints for each prefix ({expected_rows} rows)")
    for candidate_id in candidate_ids:
        seeds = {row.repeat_seed for row in rows if row.candidate_id == candidate_id}
        if seeds != {PRIMARY_BRANCH_SEED, STABILITY_BRANCH_SEED}:
            raise ValueError(f"Prefix {candidate_id} has checkpoint seeds {sorted(seeds)}")
    for row in rows:
        if row.candidate_id not in candidates.CANDIDATE_IDS:
            raise ValueError(f"Unknown candidate prefix: {row.candidate_id}")
        if not row.checkpoint_uri.startswith("gs://marin-us-east5/"):
            raise ValueError(f"Prefix checkpoint is not east5-local: {row.checkpoint_uri}")
        if not row.checkpoint_uri.endswith(f"/checkpoints/step-{replay.EXPECTED_PREFIX_HF_STEP}"):
            raise ValueError(f"Prefix checkpoint is not the exact post-update-2400 state: {row.checkpoint_uri}")
        expected_run_fragment = f"prefix_{row.candidate_id}_seed{row.repeat_seed}"
        if expected_run_fragment not in row.checkpoint_uri:
            raise ValueError(f"Prefix checkpoint URI does not match its candidate identity: {row.checkpoint_uri}")
        fs, checkpoint_path = fsspec.core.url_to_fs(row.checkpoint_uri)
        if not fs.exists(checkpoint_path):
            raise FileNotFoundError(f"Prefix checkpoint does not exist: {row.checkpoint_uri}")
        metadata_path = os.path.join(checkpoint_path, "metadata.json")
        if not fs.exists(metadata_path):
            raise FileNotFoundError(f"Prefix checkpoint metadata does not exist: {row.checkpoint_uri}")
        with fs.open(metadata_path) as handle:
            metadata = json.load(handle)
        if metadata.get("step") != replay.EXPECTED_PREFIX_HF_STEP or metadata.get("is_temporary") is not False:
            raise ValueError(f"Prefix checkpoint metadata is not the permanent boundary state: {metadata}")
    return rows


def load_continuations(path: Path, expected_sha256: str) -> tuple[tuple[str, ...], list[dict[str, object]]]:
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"Continuation weights changed: {actual} != {expected_sha256}")
    frame = pd.read_csv(path)
    required = {
        "continuation_id",
        "role",
        "fit_budget",
        "bucket",
        "phase_1_count",
        "phase_1_weight",
        "phase_1_materialized_epochs",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Continuation weights are missing columns: {sorted(required - set(frame.columns))}")
    continuation_ids = tuple(frame.continuation_id.drop_duplicates())
    if len(continuation_ids) != COMMON_CONTINUATION_COUNT:
        raise ValueError(f"Expected {COMMON_CONTINUATION_COUNT} common continuations")
    buckets = tuple(frame.loc[frame.continuation_id.eq(continuation_ids[0]), "bucket"])
    rows = []
    for continuation_id in continuation_ids:
        group = frame[frame.continuation_id.eq(continuation_id)]
        if tuple(group.bucket) != buckets:
            raise ValueError(f"Bucket order changed for {continuation_id}")
        counts = group.phase_1_count.to_numpy(dtype=int)
        weights = group.phase_1_weight.to_numpy(dtype=float)
        if counts.sum() != replay.MIXTURE_BLOCK_SIZE or not (counts >= 0).all():
            raise ValueError(f"Invalid runtime counts for {continuation_id}")
        if not (weights == counts / replay.MIXTURE_BLOCK_SIZE).all():
            raise ValueError(f"Weights are not runtime-exact for {continuation_id}")
        if float(group.phase_1_materialized_epochs.max()) > HISTORICAL_PHASE_1_EPOCH_CAP + 1e-12:
            raise ValueError(f"Historical phase-1 support exceeded by {continuation_id}")
        fit_budget_values = set(group.fit_budget)
        if len(fit_budget_values) != 1:
            raise ValueError(f"Fit-budget flag changes within {continuation_id}")
        rows.append(
            {
                "continuation_id": continuation_id,
                "role": str(group.role.iloc[0]),
                "fit_budget": bool(next(iter(fit_budget_values))),
                "max_phase_1_materialized_epoch": float(group.phase_1_materialized_epochs.max()),
                "weights": dict(zip(buckets, weights, strict=True)),
            }
        )
    if sum(bool(row["fit_budget"]) for row in rows) != COMMON_FIT_CONTINUATION_COUNT:
        raise ValueError(f"Expected {COMMON_FIT_CONTINUATION_COUNT} fit-budget continuations")
    if sum(not bool(row["fit_budget"]) for row in rows) != COMMON_CONTROL_CONTINUATION_COUNT:
        raise ValueError(f"Expected {COMMON_CONTROL_CONTINUATION_COUNT} common controls")
    return buckets, rows


def source_prefix_specs(
    *,
    candidate_weights_path: Path,
    candidate_weights_sha256: str,
    analysis_output_path: str,
    tpu_region: str,
    tpu_zone: str,
) -> dict[tuple[str, int], base.DelphiSwarmRunSpec]:
    specs, _ = candidates.candidate_specs(
        candidate_weights_path=candidate_weights_path,
        expected_sha256=candidate_weights_sha256,
        analysis_output_path=analysis_output_path,
        tpu_region=tpu_region,
        tpu_zone=tpu_zone,
    )
    by_identity = {}
    for spec in specs:
        suffix = f"_seed{spec.trainer_seed}"
        candidate_id = spec.run_name.removeprefix("prefix_").removesuffix(suffix)
        by_identity[(candidate_id, spec.trainer_seed)] = spec
    expected = {
        (candidate_id, seed)
        for candidate_id in candidates.CANDIDATE_IDS
        for seed in (PRIMARY_BRANCH_SEED, STABILITY_BRANCH_SEED)
    }
    if not expected.issubset(by_identity):
        raise ValueError("Primary or stability candidate prefix specs are incomplete")
    return by_identity


def branch_rows(
    *,
    prefixes: list[PrefixCheckpoint],
    prefix_specs: dict[tuple[str, int], base.DelphiSwarmRunSpec],
    continuations: list[dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    run_order = 0
    primary_prefixes = [row for row in prefixes if row.repeat_seed == PRIMARY_BRANCH_SEED]
    stability_prefixes = {row.candidate_id: row for row in prefixes if row.repeat_seed == STABILITY_BRANCH_SEED}
    stability_continuations = [row for row in continuations if row["continuation_id"] in STABILITY_CONTROL_IDS]
    if {row["continuation_id"] for row in stability_continuations} != STABILITY_CONTROL_IDS:
        raise ValueError("Stability controls are incomplete")
    highest_exposure = max(
        (row for row in continuations if bool(row["fit_budget"])),
        key=lambda row: float(row["max_phase_1_materialized_epoch"]),
    )
    stability_continuations.append(highest_exposure)
    if len(stability_continuations) != STABILITY_CONTINUATION_COUNT:
        raise ValueError("Stability-sentinel continuation count changed")
    branch_noise_continuation = next(
        (row for row in continuations if row["continuation_id"] == BRANCH_NOISE_CONTINUATION_ID),
        None,
    )
    if branch_noise_continuation is None:
        raise ValueError(f"Missing branch-noise continuation {BRANCH_NOISE_CONTINUATION_ID}")

    for prefix in primary_prefixes:
        source = prefix_specs[(prefix.candidate_id, prefix.repeat_seed)]
        phase_0_weights = source.phase_weights["phase_0"]
        for continuation in continuations:
            rows.append(
                {
                    "run_order": run_order,
                    "fit_budget": bool(continuation["fit_budget"]),
                    "branch_role": "primary_cross",
                    "prefix": prefix,
                    "continuation_id": continuation["continuation_id"],
                    "continuation_role": continuation["role"],
                    "phase_weights": {
                        "phase_0": phase_0_weights,
                        "phase_1": continuation["weights"],
                    },
                }
            )
            run_order += 1
        rows.append(
            {
                "run_order": run_order,
                "fit_budget": False,
                "branch_role": "prefix_tied_control",
                "prefix": prefix,
                "continuation_id": "tied_control",
                "continuation_role": "tied_control",
                "phase_weights": {"phase_0": phase_0_weights, "phase_1": phase_0_weights},
            }
        )
        run_order += 1

        for repeat in range(BRANCH_NOISE_REPEAT_COUNT):
            rows.append(
                {
                    "run_order": run_order,
                    "fit_budget": False,
                    "branch_role": "same_prefix_branch_noise_repeat",
                    "prefix": prefix,
                    "continuation_id": f"{BRANCH_NOISE_CONTINUATION_ID}_noise{repeat + 1}",
                    "continuation_role": "branch_noise_control",
                    "branch_data_seed": BRANCH_NOISE_DATA_SEED_BASE + repeat,
                    "phase_weights": {
                        "phase_0": phase_0_weights,
                        "phase_1": branch_noise_continuation["weights"],
                    },
                }
            )
            run_order += 1

        stability_prefix = stability_prefixes[prefix.candidate_id]
        stability_source = prefix_specs[(stability_prefix.candidate_id, stability_prefix.repeat_seed)]
        stability_phase_0 = stability_source.phase_weights["phase_0"]
        for continuation in stability_continuations:
            rows.append(
                {
                    "run_order": run_order,
                    "fit_budget": False,
                    "branch_role": "prefix_seed_stability_sentinel",
                    "prefix": stability_prefix,
                    "continuation_id": continuation["continuation_id"],
                    "continuation_role": continuation["role"],
                    "phase_weights": {
                        "phase_0": stability_phase_0,
                        "phase_1": continuation["weights"],
                    },
                }
            )
            run_order += 1
    if sum(bool(row["fit_budget"]) for row in rows) != SELECTED_PREFIX_COUNT * COMMON_FIT_CONTINUATION_COUNT:
        raise ValueError("Round-1 fit budget changed")
    if len(rows) != TOTAL_BRANCH_ROWS:
        raise ValueError("Round-1 branch count changed")
    return rows


def run_phase_1_branch(config: BranchTrainingConfig) -> None:
    """Restore one exact prefix trainer state and continue through update 3007."""
    run_spec = config.run_spec
    expected_prefix_steps, expected_prefix_hf_step = replay.phase_0_boundary(run_spec.train_steps, run_spec.batch_size)
    if (expected_prefix_steps, expected_prefix_hf_step) != (
        replay.EXPECTED_PREFIX_TRAIN_STEPS,
        replay.EXPECTED_PREFIX_HF_STEP,
    ):
        raise ValueError(
            "Phase boundary changed: "
            f"{(expected_prefix_steps, expected_prefix_hf_step)} != "
            f"{(replay.EXPECTED_PREFIX_TRAIN_STEPS, replay.EXPECTED_PREFIX_HF_STEP)}"
        )
    if run_spec.train_steps != replay.EXPECTED_FULL_TRAIN_STEPS:
        raise ValueError(f"Full training horizon changed: {run_spec.train_steps}")
    scaling_fits = base._read_scaling_fits(config.analysis_output_path)
    candidate = base._candidate_for_run_spec(scaling_fits=scaling_fits, run_spec=run_spec)
    if candidate.train_steps != run_spec.train_steps:
        raise ValueError(f"Resolved training horizon changed: {candidate.train_steps} != {run_spec.train_steps}")
    params = candidate.model_config.total_trainable_params(base.completed_adamh_heuristic.vocab_size)
    if int(params) != run_spec.total_trainable_params:
        raise ValueError(f"Resolved parameter count changed: {int(params)} != {run_spec.total_trainable_params}")
    inner = replay._prefix_train_config(
        run_spec=run_spec,
        candidate=candidate,
        validation_configs=config.validation_configs,
        replay_code_commit=config.code_commit,
    )
    tracker = replace(
        inner.trainer.tracker,
        tags=[
            *(inner.trainer.tracker.tags or []),
            "issue-6611",
            "delphi-3e18-phase1-common-branches",
            f"prefix_candidate={config.prefix_checkpoint.candidate_id}",
            f"prefix_repeat_seed={config.prefix_checkpoint.repeat_seed}",
            f"replay_code_commit={config.code_commit}",
            f"data_seed={run_spec.data_seed}",
            f"trainer_seed={run_spec.trainer_seed}",
        ],
    )
    trainer = replace(
        inner.trainer,
        tracker=tracker,
        num_train_steps=replay.EXPECTED_FULL_TRAIN_STEPS,
        # None first resumes a partial checkpoint from this branch output. If no
        # local checkpoint exists, Levanter initializes from the exact prefix.
        load_checkpoint=None,
        load_checkpoint_path=None,
        initialize_from=config.prefix_checkpoint.checkpoint_uri,
    )
    if trainer.num_train_steps != replay.EXPECTED_FULL_TRAIN_STEPS:
        raise ValueError(f"Branch trainer horizon changed: {trainer.num_train_steps}")
    inner = replace(
        inner,
        trainer=trainer,
        optimizer_schedule_num_train_steps=replay.EXPECTED_FULL_TRAIN_STEPS,
    )
    if inner.optimizer_schedule_num_train_steps != run_spec.train_steps:
        raise ValueError("Branch optimizer schedule no longer matches the full training horizon")
    resources = ResourceConfig.with_tpu(
        run_spec.tpu_type,
        regions=[run_spec.tpu_region],
        zone=run_spec.tpu_zone,
    )
    run_levanter_train_lm(
        TrainLmOnPodConfig(
            train_config=inner,
            resources=resources,
            output_path=config.output_path,
            env_vars={
                "GIT_COMMIT": config.code_commit,
                "MARIN_PREFIX": marin_prefix_for_region(run_spec.tpu_region),
                base.SKIP_EVAL_HARNESS_ENV_VAR: "1",
            },
        )
    )


def save_branch_manifest(config: SaveBranchManifestConfig) -> None:
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    payload = {
        "experiment_name": EXPERIMENT_NAME,
        "selected_prefixes": json.loads(config.selected_prefixes_json),
        "selected_prefixes_sha256": config.selected_prefixes_sha256,
        "candidate_weights_sha256": config.candidate_weights_sha256,
        "continuation_weights_sha256": config.continuation_weights_sha256,
        "code_commit": config.code_commit,
        "prefix_completed_updates": replay.EXPECTED_PREFIX_TRAIN_STEPS,
        "prefix_checkpoint_step": replay.EXPECTED_PREFIX_HF_STEP,
        "terminal_completed_updates": replay.EXPECTED_FULL_TRAIN_STEPS,
        "terminal_checkpoint_step": replay.EXPECTED_FULL_TRAIN_STEPS - 1,
        "optimizer_schedule_num_train_steps": replay.EXPECTED_FULL_TRAIN_STEPS,
        "fit_budget_rows": SELECTED_PREFIX_COUNT * COMMON_FIT_CONTINUATION_COUNT,
        "control_rows": TOTAL_BRANCH_ROWS - SELECTED_PREFIX_COUNT * COMMON_FIT_CONTINUATION_COUNT,
        "same_prefix_branch_noise_rows": SELECTED_PREFIX_COUNT * BRANCH_NOISE_REPEAT_COUNT,
        "branch_rows": json.loads(config.branch_rows_json),
    }
    with fs.open(os.path.join(config.output_path, "manifest.json"), "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-weights", type=Path, default=DEFAULT_CANDIDATE_WEIGHTS)
    parser.add_argument("--expected-candidate-sha256", required=True)
    parser.add_argument("--continuation-weights", type=Path, default=DEFAULT_CONTINUATION_WEIGHTS)
    parser.add_argument("--expected-continuation-sha256", required=True)
    parser.add_argument("--selected-prefixes", required=True)
    parser.add_argument("--expected-selected-prefixes-sha256", required=True)
    parser.add_argument("--analysis-output-path", default=base.DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--tpu-region", default=base.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=base.DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = parse_args()
    sys.argv = [sys.argv[0], *remaining]
    if args.tpu_region != base.DEFAULT_TPU_REGION or args.tpu_zone != base.DEFAULT_TPU_ZONE:
        raise ValueError(f"This launcher is pinned to {base.DEFAULT_TPU_REGION}/{base.DEFAULT_TPU_ZONE}")
    if not 1 <= args.max_concurrent <= TOTAL_BRANCH_ROWS:
        raise ValueError(f"--max-concurrent must be in [1, {TOTAL_BRANCH_ROWS}]")
    expected_prefix = marin_prefix_for_region(args.tpu_region)
    if os.environ.get("MARIN_PREFIX", expected_prefix) != expected_prefix:
        raise ValueError(f"MARIN_PREFIX must be {expected_prefix}")
    os.environ["MARIN_PREFIX"] = expected_prefix
    code_commit = replay.validate_replay_code_commit(args.code_commit, get_git_commit())

    prefixes = load_selected_prefixes(
        args.selected_prefixes,
        args.expected_selected_prefixes_sha256,
        args.expected_candidate_sha256,
        code_commit,
    )
    buckets, continuations = load_continuations(
        args.continuation_weights,
        args.expected_continuation_sha256,
    )
    prefix_specs = source_prefix_specs(
        candidate_weights_path=args.candidate_weights,
        candidate_weights_sha256=args.expected_candidate_sha256,
        analysis_output_path=args.analysis_output_path,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
    )
    runtime_buckets = tuple(next(iter(prefix_specs.values())).phase_weights["phase_0"])
    if set(runtime_buckets) != set(buckets):
        raise ValueError("Prefix and continuation bucket sets disagree")
    for continuation in continuations:
        weights = continuation["weights"]
        continuation["weights"] = {bucket: weights[bucket] for bucket in runtime_buckets}
    rows = branch_rows(prefixes=prefixes, prefix_specs=prefix_specs, continuations=continuations)
    serializable_rows = []
    for row in rows:
        prefix = row["prefix"]
        serializable_rows.append({**row, "prefix": asdict(prefix)})

    if args.dry_run:
        save_branch_manifest(
            SaveBranchManifestConfig(
                output_path=str(LOCAL_ARTIFACT_DIR),
                selected_prefixes_json=json.dumps([asdict(row) for row in prefixes], sort_keys=True),
                selected_prefixes_sha256=args.expected_selected_prefixes_sha256,
                candidate_weights_sha256=args.expected_candidate_sha256,
                continuation_weights_sha256=args.expected_continuation_sha256,
                code_commit=code_commit,
                branch_rows_json=json.dumps(serializable_rows, sort_keys=True),
            )
        )
        logger.info("Wrote %d phase-1 branch specs under %s", len(rows), LOCAL_ARTIFACT_DIR)
        return

    validation_steps = base._default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }
    steps = []
    with executor_context():
        for row in rows:
            prefix = row["prefix"]
            source = prefix_specs[(prefix.candidate_id, prefix.repeat_seed)]
            max_epoch, q95_epoch, phase_tv = base._weight_diagnostics(row["phase_weights"])
            run_order = int(row["run_order"])
            continuation_id = str(row["continuation_id"])
            run_name = f"branch_{prefix.candidate_id}_seed{prefix.repeat_seed}_{continuation_id}"
            run_spec = replace(
                source,
                run_order=run_order,
                run_id=BRANCH_RUN_ID_BASE + run_order,
                run_name=run_name,
                source_run_name=run_name,
                source_experiment=EXPERIMENT_NAME,
                panel_source="sequential_phase1_common_branch",
                data_seed=int(row.get("branch_data_seed", source.data_seed)),
                max_simulated_epoch=max_epoch,
                q95_simulated_epoch=q95_epoch,
                mean_phase_tv_to_proportional=phase_tv,
                phase_weights=row["phase_weights"],
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
                        run_phase_1_branch,
                        resources=resources,
                        env_vars={base.HF_HUB_DISABLE_XET_ENV_VAR: "1"},
                    ),
                    resources=resources,
                    config=BranchTrainingConfig(
                        analysis_output_path=args.analysis_output_path,
                        output_path=this_output_path(),
                        run_spec=run_spec,
                        validation_configs=validation_configs,
                        prefix_checkpoint=prefix,
                        code_commit=code_commit,
                    ),
                )
            )
        steps.append(
            ExecutorStep(
                name=f"{EXPERIMENT_NAME}/manifest",
                fn=save_branch_manifest,
                config=SaveBranchManifestConfig(
                    output_path=this_output_path(),
                    selected_prefixes_json=json.dumps([asdict(row) for row in prefixes], sort_keys=True),
                    selected_prefixes_sha256=args.expected_selected_prefixes_sha256,
                    candidate_weights_sha256=args.expected_candidate_sha256,
                    continuation_weights_sha256=args.expected_continuation_sha256,
                    code_commit=code_commit,
                    branch_rows_json=json.dumps(serializable_rows, sort_keys=True),
                ),
            )
        )
    if os.getenv("CI") is not None:
        logger.info("Built %d branch steps; skipping launch in CI", len(steps))
        return
    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=steps,
        description=f"{EXPERIMENT_NAME}: fully crossed state-conditioned phase-1 continuation panel",
    )


if __name__ == "__main__":
    main()
