# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
#   "wandb",
# ]
# ///

"""Collect Stage 2 and execute its frozen 8B paired-confirmation analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import wandb
from scipy import stats

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_starcoder_wsd80_scale_bayesian_refinement_stage2_20260801 as frozen_designer,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    starcoder_wsd80_training_identity as stream_identity,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
PANEL_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_scale_bayesian_refinement_20260731"
DEFAULT_OUTPUT_DIR = PANEL_DIR / "stage2_results_20260801"
FROZEN_DESIGN_PATH = SCRIPT_DIR.parents[1] / "starcoder_wsd80_scale_bayesian_refinement_stage2_design_20260801.json"
STAGE1_OBSERVATIONS_PATH = PANEL_DIR / "results_20260801" / "stage1_observations.csv"

TRAIN_PROJECT = "marin-community/marin"
TRAIN_TAG = "starcoder_wsd80_scale_bo_stage2"
OBJECTIVE_METRIC = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
EXPECTED_RUNS = 26
REFERENCE_SEED = 20_260_711
CONFIRMATION_SEEDS = (
    20_260_712,
    20_260_713,
    20_260_714,
    20_260_715,
    20_260_801,
    20_260_802,
    20_260_803,
    20_260_804,
)
CONFIRMATION_BUDGET = 8_000_000_000
CANDIDATE = (0.07, 0.87)
INCUMBENT = (0.02, 0.82)
TOKENS_PER_STEP = 128 * 2048
PRIMARY_TEST = (
    "two-sided paired t-test at alpha 0.05; the directional confirmation rule has operative one-sided alpha 0.025"
)
PRIMARY_INTERVAL = "two-sided 95% Student-t confidence interval for the mean paired difference"
DECISION_RULE = "confirm only if mean candidate-minus-incumbent < 0 and CI upper bound < 0"


@dataclass(frozen=True)
class PersistedMetric:
    """One final checkpoint metric recovered independently of W&B state."""

    value: float
    step: int
    uri: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_final_step(token_budget: int) -> int:
    """Return the last zero-indexed training step for a materialized token budget."""
    return token_budget // TOKENS_PER_STEP - 1


def verify_source_hashes(design: dict[str, Any]) -> None:
    """Fail if any preregistration input changed after Stage-2 design freeze."""
    hashes = design.get("data_use", {}).get("source_sha256", {})
    if not hashes:
        raise ValueError("Frozen design has no source hashes")
    for relative_path, expected in hashes.items():
        path = REPO_ROOT / relative_path
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"Frozen source changed: {relative_path}; expected {expected}, found {actual}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=FROZEN_DESIGN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wandb-timeout", type=int, default=240)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def persisted_final_metric(run: Any) -> PersistedMetric:
    """Read the final objective from a run's durable checkpoint output."""
    checkpoint_root = str(run.config["trainer"]["checkpointer"]["base_path"])
    uri = f"{checkpoint_root}/eval_metrics.jsonl"
    result = subprocess.run(
        ["gcloud", "storage", "cat", uri],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    finite = [
        row for row in rows if row.get(OBJECTIVE_METRIC) is not None and math.isfinite(float(row[OBJECTIVE_METRIC]))
    ]
    if not finite:
        raise ValueError(f"{run.name}: no finite {OBJECTIVE_METRIC} in {uri}")
    final = max(finite, key=lambda row: int(row["step"]))
    return PersistedMetric(float(final[OBJECTIVE_METRIC]), int(final["step"]), uri)


def verify_historical_pairing_audit(design: dict[str, Any], api: Any) -> dict[int, str]:
    """Recheck the frozen historical configs and return their stream digests by seed."""
    audit = design.get("confirmation", {}).get("historical_pairing_audit", {})
    rows = audit.get("rows", [])
    if len(rows) != len(frozen_designer.EXISTING_CONFIRMATION_SEEDS):
        raise ValueError("Frozen historical-pairing audit has an unexpected row count")
    result = {}
    for row in rows:
        seed = int(row["seed"])
        run = api.run(f"{TRAIN_PROJECT}/{row['historical_wandb_id']}")
        config = dict(run.config)
        config_digest = stream_identity.canonical_sha256(config)
        if config_digest != row["historical_wandb_config_sha256"]:
            raise ValueError(f"Historical W&B config changed after preregistration for seed {seed}")
        identity_digest = stream_identity.canonical_sha256(stream_identity.wandb_stream_identity(config))
        if identity_digest != row["historical_stream_identity_sha256"]:
            raise ValueError(f"Historical stream identity changed after preregistration for seed {seed}")
        result[seed] = identity_digest
    if set(result) != set(frozen_designer.EXISTING_CONFIRMATION_SEEDS):
        raise ValueError("Historical-pairing audit does not cover the expected seeds")
    return result


def verify_observed_training_streams(
    manifest: pd.DataFrame,
    ordered_runs: list[Any],
    historical_digests: dict[int, str],
) -> None:
    """Assert every policy and paired stream identity from the emitted W&B configs."""
    digests_by_seed: dict[int, list[str]] = {}
    for row, run in zip(manifest.to_dict("records"), ordered_runs, strict=True):
        observed_policy = stream_identity.policy_coordinates(run.config)
        expected_policy = [
            {"boundary_step": 0, "starcoder_weight": row["phase_0_starcoder"]},
            {"boundary_step": int(row["boundary_step"]), "starcoder_weight": row["phase_1_starcoder"]},
        ]
        differences = stream_identity.identity_differences(observed_policy, expected_policy)
        if differences:
            raise ValueError(f"Observed policy does not match frozen coordinate for {row['run_name']}: {differences}")
        if row["run_kind"] == "acquisition":
            continue
        digest = stream_identity.canonical_sha256(stream_identity.wandb_stream_identity(run.config))
        seed = int(row["pair_seed"])
        digests_by_seed.setdefault(seed, []).append(digest)
        if seed in historical_digests and digest != historical_digests[seed]:
            raise ValueError(f"Observed candidate stream does not match historical incumbent seed {seed}")
    for seed in frozen_designer.EXISTING_CONFIRMATION_SEEDS + frozen_designer.NEW_CONFIRMATION_SEEDS:
        digests = digests_by_seed.get(seed, [])
        if len(digests) != 2 or len(set(digests)) != 1:
            raise ValueError(f"Observed pair does not share one training stream for seed {seed}")


def collect_stage2(design_path: Path, timeout: int, workers: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join the immutable design to one durable final observation per run."""
    design = json.loads(design_path.read_text(encoding="utf-8"))
    if design.get("design_version") != "2026-08-01" or design.get("objective_metric") != OBJECTIVE_METRIC:
        raise ValueError("Stage-2 design version or objective metric is not the frozen preregistration")
    verify_source_hashes(design)
    frozen_rows = design["runs"]
    regenerated_rows, _, _, _ = frozen_designer.build_rows()
    frozen_manifest = frozen_designer.launch_manifest(frozen_rows)
    regenerated_manifest = frozen_designer.launch_manifest(regenerated_rows)
    if frozen_manifest != regenerated_manifest:
        raise ValueError("Current design code does not reproduce the frozen Stage-2 launch manifest")
    expected_manifest_hash = design.get("design", {}).get("launch_manifest_sha256")
    if stream_identity.canonical_sha256(frozen_manifest) != expected_manifest_hash:
        raise ValueError("Frozen Stage-2 launch-manifest hash is invalid")
    manifest = pd.DataFrame(frozen_rows)
    if len(manifest) != EXPECTED_RUNS or manifest["run_name"].duplicated().any():
        raise ValueError(f"Expected {EXPECTED_RUNS} unique design rows")
    analysis_plan = design.get("confirmation", {}).get("analysis_plan", {})
    if analysis_plan.get("selection_seed_excluded") != REFERENCE_SEED:
        raise ValueError("Frozen analysis plan does not exclude the selecting seed")
    if tuple(analysis_plan.get("paired_seeds", ())) != CONFIRMATION_SEEDS:
        raise ValueError("Frozen analysis plan has unexpected confirmation seeds")
    if analysis_plan.get("primary_test") != PRIMARY_TEST:
        raise ValueError("Frozen analysis plan has an unexpected primary test")
    if analysis_plan.get("primary_interval") != PRIMARY_INTERVAL:
        raise ValueError("Frozen analysis plan has an unexpected primary interval")
    if analysis_plan.get("decision_rule") != DECISION_RULE:
        raise ValueError("Frozen analysis plan has an unexpected decision rule")
    acquisitions = manifest.loc[manifest["run_kind"].eq("acquisition")]
    candidate_rows = manifest.loc[manifest["run_kind"].eq("candidate_confirmation")]
    incumbents = manifest.loc[manifest["run_kind"].eq("incumbent_confirmation")]
    if len(acquisitions) != 10 or len(candidate_rows) != 8 or len(incumbents) != 8:
        raise ValueError("Frozen design has unexpected acquisition or confirmation arm counts")
    if not (
        candidate_rows["token_budget_requested"].eq(CONFIRMATION_BUDGET)
        & np.isclose(candidate_rows["phase_0_starcoder"], CANDIDATE[0])
        & np.isclose(candidate_rows["phase_1_starcoder"], CANDIDATE[1])
    ).all():
        raise ValueError("Candidate confirmation rows do not match the frozen coordinate")
    if not (
        incumbents["token_budget_requested"].eq(CONFIRMATION_BUDGET)
        & np.isclose(incumbents["phase_0_starcoder"], INCUMBENT[0])
        & np.isclose(incumbents["phase_1_starcoder"], INCUMBENT[1])
    ).all():
        raise ValueError("Incumbent confirmation rows do not match the frozen coordinate")
    if REFERENCE_SEED in candidate_rows["pair_seed"].astype(int).tolist():
        raise ValueError("The selecting seed appears in the confirmation panel")

    api = wandb.Api(timeout=timeout)
    historical_digests = verify_historical_pairing_audit(design, api)
    runs = list(api.runs(TRAIN_PROJECT, filters={"tags": TRAIN_TAG}, per_page=100))
    by_name: dict[str, list[Any]] = {}
    for run in runs:
        by_name.setdefault(str(run.name), []).append(run)

    ordered_runs = []
    for run_name in manifest["run_name"]:
        matching_runs = by_name.get(str(run_name), [])
        if len(matching_runs) != 1:
            raise ValueError(f"{run_name}: expected exactly one W&B run, found {len(matching_runs)}")
        ordered_runs.append(matching_runs[0])
    verify_observed_training_streams(manifest, ordered_runs, historical_digests)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        metrics = list(executor.map(persisted_final_metric, ordered_runs))

    observations = manifest.copy()
    observations["starcoder_bpb"] = [metric.value for metric in metrics]
    observations["final_metric_step"] = [metric.step for metric in metrics]
    observations["expected_final_metric_step"] = observations["token_budget_requested"].map(
        lambda value: expected_final_step(int(value))
    )
    observations["metric_uri"] = [metric.uri for metric in metrics]
    observations["metric_source"] = "persisted eval_metrics.jsonl"
    observations["wandb_id"] = [str(run.id) for run in ordered_runs]
    observations["wandb_state"] = [str(run.state) for run in ordered_runs]
    observations["wandb_url"] = [str(run.url) for run in ordered_runs]
    if not np.isfinite(observations["starcoder_bpb"].to_numpy(dtype=float)).all():
        raise ValueError("Stage-2 observations contain non-finite BPB")
    incomplete = observations.loc[observations["final_metric_step"].ne(observations["expected_final_metric_step"])]
    if not incomplete.empty:
        details = incomplete[["run_name", "final_metric_step", "expected_final_metric_step"]].to_dict("records")
        raise ValueError(f"Stage-2 contains partial checkpoints: {details}")
    return observations, design


def paired_confirmation(
    stage2: pd.DataFrame,
    design: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Execute the preregistered eight-seed candidate-versus-incumbent comparison."""
    candidate = stage2.loc[
        stage2["run_kind"].eq("candidate_confirmation"),
        ["pair_seed", "comparison_source", "starcoder_bpb", "final_metric_step", "wandb_id", "wandb_url"],
    ].rename(
        columns={
            "starcoder_bpb": "candidate_bpb",
            "final_metric_step": "candidate_final_metric_step",
            "wandb_id": "candidate_wandb_id",
            "wandb_url": "candidate_wandb_url",
        }
    )
    incumbent = stage2.loc[
        stage2["run_kind"].eq("incumbent_confirmation"),
        ["pair_seed", "starcoder_bpb", "final_metric_step", "wandb_id", "wandb_url"],
    ].rename(
        columns={
            "starcoder_bpb": "incumbent_bpb",
            "final_metric_step": "incumbent_final_metric_step",
            "wandb_id": "incumbent_wandb_id",
            "wandb_url": "incumbent_wandb_url",
        }
    )

    pairs = candidate.merge(incumbent, on="pair_seed", how="inner", validate="one_to_one").sort_values("pair_seed")
    if tuple(pairs["pair_seed"].astype(int)) != CONFIRMATION_SEEDS:
        raise ValueError(f"Expected confirmation seeds {CONFIRMATION_SEEDS}, found {tuple(pairs['pair_seed'])}")
    if not pairs["candidate_final_metric_step"].eq(pairs["incumbent_final_metric_step"]).all():
        raise ValueError("A confirmation pair compares different final training steps")
    expected_step = expected_final_step(CONFIRMATION_BUDGET)
    if not pairs["candidate_final_metric_step"].eq(expected_step).all():
        raise ValueError(f"A confirmation pair did not reach expected final step {expected_step}")
    pairs["candidate_minus_incumbent_bpb"] = pairs["candidate_bpb"] - pairs["incumbent_bpb"]

    differences = pairs["candidate_minus_incumbent_bpb"].to_numpy(dtype=float)
    mean = float(differences.mean())
    sample_sd = float(differences.std(ddof=1))
    standard_error = sample_sd / np.sqrt(len(differences))
    half_width = float(stats.t.ppf(0.975, len(differences) - 1) * standard_error)
    test = stats.ttest_1samp(differences, popmean=0.0)
    historical_seed_block = pairs.loc[
        pairs["comparison_source"].eq("historical_seed_block"), "candidate_minus_incumbent_bpb"
    ].to_numpy(dtype=float)
    fresh_seed_block = pairs.loc[
        pairs["comparison_source"].eq("fresh_seed_block"), "candidate_minus_incumbent_bpb"
    ].to_numpy(dtype=float)
    if len(historical_seed_block) != 4 or len(fresh_seed_block) != 4:
        raise ValueError("Expected two four-seed contemporaneous confirmation blocks")
    block_test = stats.ttest_ind(historical_seed_block, fresh_seed_block, equal_var=False)

    audit = pd.DataFrame(design["confirmation"]["historical_pairing_audit"]["rows"])[
        ["seed", "historical_wandb_id"]
    ].rename(columns={"seed": "pair_seed", "historical_wandb_id": "audited_historical_wandb_id"})
    stage1 = pd.read_csv(STAGE1_OBSERVATIONS_PATH)
    historical_incumbent = stage1.loc[
        stage1["run_kind"].eq("incumbent_repeat")
        & stage1["token_budget_requested"].eq(CONFIRMATION_BUDGET)
        & np.isclose(stage1["phase_0_starcoder"], INCUMBENT[0])
        & np.isclose(stage1["phase_1_starcoder"], INCUMBENT[1]),
        ["trainer_data_seed", "starcoder_bpb", "final_metric_step", "wandb_id", "wandb_url"],
    ].rename(
        columns={
            "trainer_data_seed": "pair_seed",
            "starcoder_bpb": "historical_incumbent_bpb",
            "final_metric_step": "historical_final_metric_step",
            "wandb_id": "historical_wandb_id",
            "wandb_url": "historical_wandb_url",
        }
    )
    current_incumbent = incumbent.rename(
        columns={
            "incumbent_bpb": "current_incumbent_bpb",
            "incumbent_final_metric_step": "current_final_metric_step",
            "incumbent_wandb_id": "current_wandb_id",
            "incumbent_wandb_url": "current_wandb_url",
        }
    )
    drift = (
        audit.merge(historical_incumbent, on="pair_seed", how="inner", validate="one_to_one")
        .merge(current_incumbent, on="pair_seed", how="inner", validate="one_to_one")
        .sort_values("pair_seed")
    )
    if tuple(drift["pair_seed"].astype(int)) != frozen_designer.EXISTING_CONFIRMATION_SEEDS:
        raise ValueError("Historical-incumbent drift diagnostic has unexpected seeds")
    if not drift["historical_wandb_id"].eq(drift["audited_historical_wandb_id"]).all():
        raise ValueError("Historical incumbent CSV rows do not match the frozen W&B config audit")
    if not drift["historical_final_metric_step"].eq(drift["current_final_metric_step"]).all():
        raise ValueError("Historical-incumbent drift diagnostic compares different final steps")
    drift["current_minus_historical_incumbent_bpb"] = drift["current_incumbent_bpb"] - drift["historical_incumbent_bpb"]
    drift_values = drift["current_minus_historical_incumbent_bpb"].to_numpy(dtype=float)
    drift_test = stats.ttest_1samp(drift_values, popmean=0.0)
    summary: dict[str, object] = {
        "candidate_phase_0": CANDIDATE[0],
        "candidate_phase_1": CANDIDATE[1],
        "incumbent_phase_0": INCUMBENT[0],
        "incumbent_phase_1": INCUMBENT[1],
        "paired_seeds": len(differences),
        "mean_candidate_minus_incumbent_bpb": mean,
        "sample_sd_bpb": sample_sd,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "paired_t_two_sided_p": float(test.pvalue),
        "candidate_better_count": int(np.sum(differences < 0.0)),
        "confirmed": bool(mean < 0.0 and mean + half_width < 0.0),
        "selection_seed_excluded": REFERENCE_SEED,
        "historical_seed_block_mean_difference_bpb": float(historical_seed_block.mean()),
        "fresh_seed_block_mean_difference_bpb": float(fresh_seed_block.mean()),
        "seed_block_welch_p": float(block_test.pvalue),
        "historical_incumbent_drift_pair_count": len(drift_values),
        "current_minus_historical_incumbent_mean_bpb": float(drift_values.mean()),
        "historical_incumbent_drift_two_sided_p": float(drift_test.pvalue),
    }
    return pairs.reset_index(drop=True), summary, drift.reset_index(drop=True)


def write_outputs() -> None:
    """Persist complete observations and the frozen confirmation result."""
    args = parse_args()
    observations, design = collect_stage2(args.design, args.wandb_timeout, args.workers)
    pairs, confirmation, drift = paired_confirmation(observations, design)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations.to_csv(args.output_dir / "stage2_observations.csv", index=False)
    pairs.to_csv(args.output_dir / "confirmation_pairs.csv", index=False)
    seed_block_summary = (
        pairs.groupby("comparison_source", as_index=False)["candidate_minus_incumbent_bpb"]
        .agg(pair_count="count", mean_difference_bpb="mean", sd_difference_bpb="std")
        .sort_values("comparison_source")
    )
    seed_block_summary.to_csv(args.output_dir / "confirmation_seed_block_summary.csv", index=False)
    drift.to_csv(args.output_dir / "historical_incumbent_drift.csv", index=False)
    (args.output_dir / "confirmation_summary.json").write_text(
        json.dumps(confirmation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    acquisitions = observations.loc[observations["run_kind"].eq("acquisition")]
    best_acquisitions = (
        acquisitions.sort_values("starcoder_bpb")
        .groupby("token_budget_requested", as_index=False)
        .first()[["token_budget_requested", "phase_0_starcoder", "phase_1_starcoder", "starcoder_bpb", "wandb_url"]]
    )
    decision = "confirmed" if confirmation["confirmed"] else "not confirmed; result is inconclusive, not equivalent"
    report = [
        "# StarCoder WSD80 scale Bayesian refinement: Stage-2 results",
        "",
        f"- Durable coverage: {len(observations)}/{EXPECTED_RUNS} runs.",
        f"- Preregistered 8B comparison: {decision}.",
        f"- Candidate minus incumbent: {confirmation['mean_candidate_minus_incumbent_bpb']:+.6f} BPB "
        f"(95% CI [{confirmation['ci95_low']:+.6f}, {confirmation['ci95_high']:+.6f}], "
        f"two-sided paired t p={confirmation['paired_t_two_sided_p']:.6g}, "
        f"sign count {confirmation['candidate_better_count']}/{confirmation['paired_seeds']}).",
        "- All eight candidate/incumbent comparisons are contemporaneous same-seed pairs; the selecting reference seed "
        "is excluded from confirmation inference.",
        f"- Historical-incumbent drift: current minus historical "
        f"{confirmation['current_minus_historical_incumbent_mean_bpb']:+.6f} BPB over "
        f"{confirmation['historical_incumbent_drift_pair_count']} same-policy, same-seed pairs "
        f"(two-sided paired t p={confirmation['historical_incumbent_drift_two_sided_p']:.6g}); "
        "this diagnostic is not a gate.",
        "",
        "## Paired confirmation",
        "",
        pairs[
            [
                "pair_seed",
                "comparison_source",
                "candidate_bpb",
                "incumbent_bpb",
                "candidate_minus_incumbent_bpb",
            ]
        ].to_markdown(index=False, floatfmt=".7f"),
        "",
        "## Seed-block diagnostic",
        "",
        seed_block_summary.to_markdown(index=False, floatfmt=".7f"),
        "",
        f"Exploratory Welch p={confirmation['seed_block_welch_p']:.6g}. This checks whether the two four-seed blocks "
        "behave differently; it does not alter the preregistered pooled confirmation gate.",
        "",
        "## Historical-incumbent drift diagnostic",
        "",
        drift[
            [
                "pair_seed",
                "historical_incumbent_bpb",
                "current_incumbent_bpb",
                "current_minus_historical_incumbent_bpb",
            ]
        ].to_markdown(index=False, floatfmt=".7f"),
        "",
        "## Best Stage-2 spatial acquisition by rung",
        "",
        best_acquisitions.to_markdown(index=False, floatfmt=".7f"),
        "",
        "## Interpretation boundary",
        "",
        str(design["interpretation_boundary"]),
        "",
        "W&B state is recorded for provenance but is not a completion gate. Final values and steps come from durable "
        "checkpoint eval_metrics.jsonl files.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    write_outputs()
