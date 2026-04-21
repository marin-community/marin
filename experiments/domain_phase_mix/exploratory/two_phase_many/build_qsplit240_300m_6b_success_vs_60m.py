# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "numpy", "pandas", "scipy"]
# ///
"""Build the strict terminal-success 300M/6B vs 60M comparison table."""

from __future__ import annotations

import json
from pathlib import Path
import re

import fsspec
import pandas as pd
from scipy import stats

from experiments.domain_phase_mix.two_phase_many_observed_runs import (
    CORE_BASELINE_RUN_NAMES,
    load_original_qsplit240_with_core_baselines,
)

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_CSV = SCRIPT_DIR / "two_phase_many.csv"
OUTPUT_CSV = SCRIPT_DIR / "qsplit240_300m_6b_completed_vs_60m.csv"
OUTPUT_SUMMARY_JSON = SCRIPT_DIR / "qsplit240_300m_6b_completed_vs_60m_summary.json"
OBJECTIVE_METRIC = "eval/uncheatable_eval/bpb"
STRATIFIED_RUN_NAME = "baseline_stratified"
QSPLIT240_300M_CHECKPOINT_PREFIX = "marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_300m_6b"
STRATIFIED_60M_CHECKPOINT_PREFIX = (
    "marin-us-central1/checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_60m_1p2b"
)
STRATIFIED_300M_CHECKPOINT_PREFIX = (
    "marin-us-central1/checkpoints/pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_300m_6b"
)
CHECKPOINT_PREFIXES_300M_6B = (
    QSPLIT240_300M_CHECKPOINT_PREFIX,
    STRATIFIED_300M_CHECKPOINT_PREFIX,
)
RUN_ROOT_PATTERN = re.compile(r"([^/]+-[0-9a-f]+)/checkpoints/eval_metrics\.jsonl$")


def _load_success_metric_by_run_name(
    fs: fsspec.AbstractFileSystem,
    checkpoint_prefix: str,
    *,
    allowed_run_names: set[str] | None = None,
) -> dict[str, dict[str, object]]:
    metrics_by_run_name: dict[str, dict[str, object]] = {}
    for metrics_path in sorted(fs.glob(f"{checkpoint_prefix}/*/checkpoints/eval_metrics.jsonl")):
        root_name_match = RUN_ROOT_PATTERN.search(metrics_path)
        if root_name_match is None:
            continue
        root_name = root_name_match.group(1)
        run_name = root_name.rsplit("-", 1)[0]
        if allowed_run_names is not None and run_name not in allowed_run_names:
            continue

        status_path = "/".join(metrics_path.split("/")[:-2]) + "/.executor_status"
        if not fs.exists(status_path):
            continue
        with fs.open(status_path, "r") as f:
            status = f.read().strip()
        if status != "SUCCESS":
            continue
        if run_name in metrics_by_run_name:
            raise ValueError(f"Found multiple SUCCESS roots for {run_name!r}")

        with fs.open(metrics_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        if not lines:
            continue
        payload = json.loads(lines[-1])
        if OBJECTIVE_METRIC not in payload:
            continue

        metrics_by_run_name[run_name] = {
            "checkpoint_root": f"gs://{'/'.join(metrics_path.split('/')[:-2])}",
            "status": status,
            "metric": float(payload[OBJECTIVE_METRIC]),
        }
    return metrics_by_run_name


def _base_60m_rows(fs: fsspec.AbstractFileSystem) -> pd.DataFrame:
    base = pd.read_csv(BASE_CSV)
    base = base[["run_name", "source_experiment", OBJECTIVE_METRIC]].dropna(subset=[OBJECTIVE_METRIC]).copy()
    base = base.rename(columns={OBJECTIVE_METRIC: "bpb_60m"})

    observed_run_names = {run.run_name for run in load_original_qsplit240_with_core_baselines()}
    base = base[base["run_name"].isin(observed_run_names)].copy()

    stratified_metrics = _load_success_metric_by_run_name(
        fs,
        STRATIFIED_60M_CHECKPOINT_PREFIX,
        allowed_run_names={STRATIFIED_RUN_NAME},
    )
    stratified_row = stratified_metrics.get(STRATIFIED_RUN_NAME)
    if stratified_row is not None:
        base = pd.concat(
            [
                base,
                pd.DataFrame(
                    [
                        {
                            "run_name": STRATIFIED_RUN_NAME,
                            "source_experiment": "pinlin_calvin_xu/data_mixture/ngd3dm2_stratified_60m_1p2b",
                            "bpb_60m": float(stratified_row["metric"]),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    return base


def _strict_success_rows() -> pd.DataFrame:
    fs = fsspec.filesystem("gs")
    base = _base_60m_rows(fs)
    observed_run_names = set(base["run_name"])

    rows: list[dict[str, object]] = []
    seen_run_names: set[str] = set()
    for checkpoint_prefix in CHECKPOINT_PREFIXES_300M_6B:
        metrics_by_run_name = _load_success_metric_by_run_name(
            fs,
            checkpoint_prefix,
            allowed_run_names=observed_run_names,
        )
        for run_name, payload in metrics_by_run_name.items():
            if run_name in seen_run_names:
                raise ValueError(f"Found multiple SUCCESS roots for {run_name!r}")
            seen_run_names.add(run_name)
            rows.append(
                {
                    "run_name": run_name,
                    "checkpoint_root": payload["checkpoint_root"],
                    "status_300m_6b": payload["status"],
                    "bpb_300m_6b": float(payload["metric"]),
                }
            )

    strict = pd.DataFrame(rows)
    strict = strict.merge(base[["run_name", "bpb_60m"]], on="run_name", how="inner")
    strict = strict.sort_values("bpb_300m_6b", ascending=True).reset_index(drop=True)
    strict["rank_300m_6b"] = strict.index + 1
    strict["rank_60m_within_completed"] = strict["bpb_60m"].rank(method="min", ascending=True).astype(int)
    strict["rank_shift"] = strict["rank_60m_within_completed"] - strict["rank_300m_6b"]
    return strict[
        [
            "run_name",
            "checkpoint_root",
            "status_300m_6b",
            "bpb_300m_6b",
            "bpb_60m",
            "rank_300m_6b",
            "rank_60m_within_completed",
            "rank_shift",
        ]
    ]


def _summary(frame: pd.DataFrame) -> dict[str, object]:
    pearson_r, pearson_pvalue = stats.pearsonr(frame["bpb_60m"], frame["bpb_300m_6b"])
    spearman_rho, spearman_pvalue = stats.spearmanr(frame["bpb_60m"], frame["bpb_300m_6b"])
    kendall_tau, kendall_pvalue = stats.kendalltau(frame["bpb_60m"], frame["bpb_300m_6b"])

    best_300m = frame.loc[frame["bpb_300m_6b"].idxmin()]
    best_60m = frame.loc[frame["bpb_60m"].idxmin()]
    reference_run_names = {run.run_name for run in load_original_qsplit240_with_core_baselines()}
    reference_run_names.add(STRATIFIED_RUN_NAME)
    pending_run_names = sorted(reference_run_names - set(frame["run_name"]))

    return {
        "objective_metric": OBJECTIVE_METRIC,
        "n_completed": len(frame),
        "n_sampled_runs_completed": int(frame["run_name"].str.startswith("run_").sum()),
        "n_core_baselines_completed": int(frame["run_name"].isin(CORE_BASELINE_RUN_NAMES).sum()),
        "n_stratified_baselines_completed": int(frame["run_name"].eq(STRATIFIED_RUN_NAME).sum()),
        "best_300m_6b_run_name": str(best_300m["run_name"]),
        "best_300m_6b_bpb": float(best_300m["bpb_300m_6b"]),
        "best_300m_6b_60m_bpb": float(best_300m["bpb_60m"]),
        "best_300m_6b_60m_rank_within_completed": int(best_300m["rank_60m_within_completed"]),
        "best_60m_within_completed_run_name": str(best_60m["run_name"]),
        "best_60m_within_completed_bpb": float(best_60m["bpb_60m"]),
        "best_60m_within_completed_300m_6b_bpb": float(best_60m["bpb_300m_6b"]),
        "best_60m_within_completed_300m_6b_rank": int(best_60m["rank_300m_6b"]),
        "pearson_r": float(pearson_r),
        "pearson_pvalue": float(pearson_pvalue),
        "spearman_rho": float(spearman_rho),
        "spearman_pvalue": float(spearman_pvalue),
        "kendall_tau": float(kendall_tau),
        "kendall_pvalue": float(kendall_pvalue),
        "pending_reference_runs": pending_run_names,
    }


def main() -> None:
    frame = _strict_success_rows()
    frame.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_SUMMARY_JSON.write_text(json.dumps(_summary(frame), indent=2, sort_keys=True))
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
