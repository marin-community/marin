# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scipy",
#   "tabulate",
#   "wandb",
# ]
# ///

"""Execute the frozen paired analysis for matched-N,D confirmation runs."""

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
    design_starcoder_wsd80_matched_nd_confirmation_20260801 as frozen_designer,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    starcoder_wsd80_training_identity as stream_identity,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
PANEL_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_matched_nd_stage1_20260731"
DEFAULT_OUTPUT_DIR = PANEL_DIR / "confirmation_results_20260801"
FROZEN_DESIGN_PATH = SCRIPT_DIR.parents[1] / "starcoder_wsd80_matched_nd_confirmation_design_20260801.json"

TRAIN_PROJECT = "marin-community/marin"
TRAIN_TAG = "starcoder_wsd80_matched_nd_confirmation"
OBJECTIVE_METRIC = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
PAIR_COUNT_PER_CELL = 8
MINIMUM_WIN_COUNT = 7
ALPHA = 0.05


@dataclass(frozen=True)
class PersistedMetric:
    """One final objective recovered from durable checkpoint metrics."""

    value: float
    step: int
    uri: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _verify_design(design: dict[str, Any]) -> pd.DataFrame:
    if design.get("design_version") != "2026-08-01":
        raise ValueError("Unexpected confirmation design version")
    if design.get("objective_metric") != OBJECTIVE_METRIC:
        raise ValueError("Frozen confirmation objective does not match the durable-metric reader")
    if design.get("pair_count_per_cell") != PAIR_COUNT_PER_CELL:
        raise ValueError("Frozen confirmation does not contain eight pairs per cell")
    rows = design.get("runs")
    expected_count = int(design.get("expected_run_count", 0))
    if not isinstance(rows, list) or not rows or len(rows) != expected_count:
        raise ValueError("Frozen confirmation has an invalid run count")
    manifest = pd.DataFrame(rows)
    if manifest["run_name"].duplicated().any():
        raise ValueError("Frozen confirmation run names are not unique")
    expected_hash = design.get("design", {}).get("launch_manifest_sha256")
    if stream_identity.canonical_sha256(frozen_designer.launch_manifest(rows)) != expected_hash:
        raise ValueError("Frozen confirmation launch-manifest hash is invalid")
    source_hashes = design.get("data_use", {}).get("source_sha256", {})
    if not source_hashes:
        raise ValueError("Frozen confirmation has no source hashes")
    for relative_path, expected in source_hashes.items():
        path = REPO_ROOT / relative_path
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"Frozen confirmation source changed: {relative_path}; {actual} != {expected}")
    return manifest


def _ordered_runs(manifest: pd.DataFrame, timeout: int) -> list[Any]:
    api = wandb.Api(timeout=timeout)
    runs = list(api.runs(TRAIN_PROJECT, filters={"tags": TRAIN_TAG}, per_page=250))
    by_name: dict[str, list[Any]] = {}
    for run in runs:
        by_name.setdefault(str(run.name), []).append(run)
    ordered = []
    for run_name in manifest["run_name"]:
        candidates = by_name.get(str(run_name), [])
        if len(candidates) != 1:
            raise ValueError(f"{run_name}: expected exactly one W&B run, found {len(candidates)}")
        ordered.append(candidates[0])
    return ordered


def _verify_observed_streams(manifest: pd.DataFrame, runs: list[Any]) -> list[str]:
    digests = []
    digests_by_pair: dict[tuple[str, int], set[str]] = {}
    for row, run in zip(manifest.to_dict("records"), runs, strict=True):
        expected_policy = [
            {"boundary_step": 0, "starcoder_weight": float(row["phase_0_starcoder"])},
            {"boundary_step": int(row["boundary_step"]), "starcoder_weight": float(row["phase_1_starcoder"])},
        ]
        differences = stream_identity.identity_differences(
            stream_identity.policy_coordinates(run.config), expected_policy
        )
        if differences:
            raise ValueError(f"{row['run_name']}: observed policy differs from the frozen design: {differences}")
        digest = stream_identity.canonical_sha256(stream_identity.wandb_stream_identity(run.config))
        digests.append(digest)
        digests_by_pair.setdefault((str(row["cell_id"]), int(row["pair_seed"])), set()).add(digest)
    inconsistent = {pair: values for pair, values in digests_by_pair.items() if len(values) != 1}
    if inconsistent:
        raise ValueError(f"Observed candidate/comparator pairs do not share one training stream: {inconsistent}")
    return digests


def collect_confirmation(design_path: Path, timeout: int, workers: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join the frozen paired manifest to durable final checkpoint metrics."""
    design = json.loads(design_path.read_text(encoding="utf-8"))
    manifest = _verify_design(design)
    roles_by_pair = manifest.groupby(["cell_id", "pair_seed"])["role"].agg(lambda values: set(values))
    if not roles_by_pair.map(lambda roles: roles == set(frozen_designer.ROLES)).all():
        raise ValueError("Frozen confirmation contains an incomplete candidate/comparator pair")
    pair_counts = manifest.groupby("cell_id")["pair_seed"].nunique()
    if not pair_counts.eq(PAIR_COUNT_PER_CELL).all():
        raise ValueError(f"Unexpected confirmation pair counts: {pair_counts.to_dict()}")

    runs = _ordered_runs(manifest, timeout)
    stream_digests = _verify_observed_streams(manifest, runs)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        metrics = list(executor.map(persisted_final_metric, runs))
    observations = manifest.copy()
    observations["starcoder_bpb"] = [metric.value for metric in metrics]
    observations["final_metric_step"] = [metric.step for metric in metrics]
    observations["expected_final_metric_step"] = observations["total_steps"].astype(int) - 1
    observations["metric_uri"] = [metric.uri for metric in metrics]
    observations["metric_source"] = "persisted eval_metrics.jsonl"
    observations["wandb_id"] = [str(run.id) for run in runs]
    observations["wandb_state"] = [str(run.state) for run in runs]
    observations["wandb_url"] = [str(run.url) for run in runs]
    observations["observed_stream_identity_sha256"] = stream_digests
    if observations["metric_uri"].nunique() != len(observations):
        raise ValueError("Confirmation rows do not resolve to distinct durable metric files")
    misplaced = observations.loc[~observations.apply(lambda row: str(row["run_name"]) in str(row["metric_uri"]), axis=1)]
    if not misplaced.empty:
        raise ValueError(
            f"A confirmation metric path is misplaced: {misplaced[['run_name', 'metric_uri']].to_dict('records')}"
        )
    incomplete = observations.loc[observations["final_metric_step"].ne(observations["expected_final_metric_step"])]
    if not incomplete.empty:
        raise ValueError(
            f"Confirmation contains partial checkpoints: "
            f"{incomplete[['run_name', 'final_metric_step', 'expected_final_metric_step']].to_dict('records')}"
        )
    return observations, design


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * float(p_values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def paired_results(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the frozen paired rule independently in each promoted cell."""
    candidate = observations.loc[
        observations["role"].eq("untied_candidate"),
        ["cell_id", "pair_seed", "starcoder_bpb", "wandb_url"],
    ].rename(columns={"starcoder_bpb": "candidate_bpb", "wandb_url": "candidate_wandb_url"})
    comparator = observations.loc[
        observations["role"].eq("tied_comparator"),
        ["cell_id", "pair_seed", "starcoder_bpb", "wandb_url"],
    ].rename(columns={"starcoder_bpb": "comparator_bpb", "wandb_url": "comparator_wandb_url"})
    pairs = candidate.merge(comparator, on=["cell_id", "pair_seed"], validate="one_to_one")
    pairs["gain_tied_minus_untied_bpb"] = pairs["comparator_bpb"] - pairs["candidate_bpb"]

    rows = []
    for cell_id, group in pairs.groupby("cell_id", sort=True):
        values = group["gain_tied_minus_untied_bpb"].to_numpy(dtype=float)
        if len(values) != PAIR_COUNT_PER_CELL:
            raise ValueError(f"{cell_id}: expected eight paired differences")
        mean = float(values.mean())
        sample_sd = float(values.std(ddof=1))
        standard_error = sample_sd / np.sqrt(len(values))
        half_width = float(stats.t.ppf(0.975, len(values) - 1) * standard_error)
        test = stats.ttest_1samp(values, popmean=0.0, alternative="greater")
        rows.append(
            {
                "cell_id": cell_id,
                "pair_count": len(values),
                "candidate_win_count": int(np.sum(values > 0.0)),
                "mean_gain_bpb": mean,
                "sample_sd_bpb": sample_sd,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
                "paired_t_one_sided_p": float(test.pvalue),
            }
        )
    summary = pd.DataFrame(rows)
    summary["paired_t_holm_p"] = _holm_adjust(summary["paired_t_one_sided_p"].to_numpy(dtype=float))
    summary["confirmed"] = (
        summary["candidate_win_count"].ge(MINIMUM_WIN_COUNT)
        & summary["ci95_low"].gt(0.0)
        & summary["paired_t_holm_p"].lt(ALPHA)
    )
    return pairs.sort_values(["cell_id", "pair_seed"]).reset_index(drop=True), summary


def write_outputs() -> None:
    """Persist all paired observations and the frozen per-cell decisions."""
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    observations, design = collect_confirmation(args.design, args.wandb_timeout, args.workers)
    pairs, summary = paired_results(observations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observations.to_csv(args.output_dir / "confirmation_observations.csv", index=False)
    pairs.to_csv(args.output_dir / "confirmation_pairs.csv", index=False)
    summary.to_csv(args.output_dir / "cell_confirmation_summary.csv", index=False)
    decision_rows = summary[
        [
            "cell_id",
            "candidate_win_count",
            "mean_gain_bpb",
            "sample_sd_bpb",
            "ci95_low",
            "ci95_high",
            "paired_t_one_sided_p",
            "paired_t_holm_p",
            "confirmed",
        ]
    ]
    report = [
        "# StarCoder WSD80 matched-N,D fresh-seed confirmation",
        "",
        f"- Complete paired observations: {len(pairs)} across {summary['cell_id'].nunique()} promoted cells.",
        f"- Confirmed cells under the frozen rule: {int(summary['confirmed'].sum())}/{len(summary)}.",
        "- A cell passes only with at least 7/8 paired wins, a positive 95% paired-t lower bound, and "
        "Holm-adjusted one-sided paired-t p<0.05.",
        "",
        "## Decisions",
        "",
        decision_rows.to_markdown(index=False, floatfmt=".7f"),
        "",
        "## Claim boundary",
        "",
        str(design["analysis_plan"]["claim_limit"]),
        "",
        str(design["analysis_plan"]["non_pass_interpretation"]),
        "",
        str(design["analysis_plan"]["provenance_disclosure"]),
        "",
        str(design["analysis_plan"]["estimand"]),
        "",
        "Passing confirms only a selected discrete untied policy against its selected tied comparator. It does not "
        "identify a continuous optimum or establish a universal phase-order law.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    write_outputs()
