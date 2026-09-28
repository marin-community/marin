# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas"]
# ///
"""Build metric comparison tables for power_family_penalty vs 60M baselines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.summarize_eval_signal_to_noise import (
    QSPLIT240_OVERLAP_RESULTS_GLOB,
    QSPLIT240_SL_VERB_RESULTS_URI,
    _list_gcs_paths,
    _read_gcs_csv,
)

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_RANKED_CSV = SCRIPT_DIR / "eval_signal_to_noise_ranked.csv"
INPUT_SWARM_CSV = SCRIPT_DIR / "two_phase_many_all_60m_1p2b.csv"
OUTPUT_WIDE_CSV = SCRIPT_DIR / "power_family_penalty_vs_baselines_ranked_metrics_wide.csv"
OUTPUT_COMMON_WIDE_CSV = SCRIPT_DIR / "power_family_penalty_vs_baselines_ranked_metrics_common_wide.csv"
OUTPUT_LONG_CSV = SCRIPT_DIR / "power_family_penalty_vs_baselines_ranked_metrics_long.csv"
OUTPUT_SUMMARY_JSON = SCRIPT_DIR / "power_family_penalty_vs_baselines_ranked_metrics_summary.json"

PRIMARY_METRIC = "eval/uncheatable_eval/bpb"
SL_VERB_METRIC = "lm_eval/mmlu_sl_verb_5shot/bpb"


@dataclass(frozen=True)
class BaselineSpec:
    """One compared run."""

    run_name: str
    label: str


RUN_SPECS = (
    BaselineSpec("baseline_genericfamily_power_family_penalty_raw_optimum", "power_family_penalty"),
    BaselineSpec("baseline_proportional", "proportional"),
    BaselineSpec("baseline_unimax", "unimax"),
    BaselineSpec("baseline_stratified", "uniform_stratified"),
    BaselineSpec("baseline_olmix_loglinear_uncheatable_bpb", "olmix"),
)
OVERLAP_BACKFILL_RUN_NAMES = {"baseline_proportional", "baseline_unimax"}
SELECTED_BASELINES_OVERLAP_RESULTS_GLOB = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_selected_baselines_olmo_base_easy_overlap_rerun/**/collect_results*/results.csv"
)
SELECTED_BASELINES_OVERLAP_RUN_NAMES = {
    "baseline_genericfamily_power_family_penalty_raw_optimum",
    "baseline_stratified",
    "baseline_olmix_loglinear_uncheatable_bpb",
}


def _requested_metrics() -> tuple[list[str], pd.DataFrame]:
    ranked = pd.read_csv(INPUT_RANKED_CSV)
    deduped_metrics = list(dict.fromkeys(ranked["metric"].tolist()))
    metrics = [PRIMARY_METRIC] + [metric for metric in deduped_metrics if metric != PRIMARY_METRIC]
    metadata = ranked.drop_duplicates(subset=["metric"]).set_index("metric").reindex(metrics)
    return metrics, metadata


def _base_rows() -> pd.DataFrame:
    frame = pd.read_csv(INPUT_SWARM_CSV)
    selected = frame.loc[frame["run_name"].isin([spec.run_name for spec in RUN_SPECS])].copy()
    counts = selected["run_name"].value_counts()
    missing = [spec.run_name for spec in RUN_SPECS if spec.run_name not in counts.index]
    if missing:
        raise ValueError(f"Missing runs in {INPUT_SWARM_CSV}: {missing}")
    duplicated = counts[counts != 1]
    if not duplicated.empty:
        raise ValueError(f"Expected one row per run in {INPUT_SWARM_CSV}, got duplicates: {duplicated.to_dict()}")
    return selected.set_index("run_name")


def _load_overlap_backfill() -> pd.DataFrame:
    shard_uris = sorted(_list_gcs_paths(QSPLIT240_OVERLAP_RESULTS_GLOB))
    if len(shard_uris) != 8:
        raise ValueError(f"Expected 8 overlap shard results, found {len(shard_uris)}")
    return pd.concat([_read_gcs_csv(uri) for uri in shard_uris], ignore_index=True).set_index("run_name")


def _load_selected_baselines_overlap_backfill() -> pd.DataFrame:
    shard_uris = sorted(_list_gcs_paths(SELECTED_BASELINES_OVERLAP_RESULTS_GLOB))
    if len(shard_uris) != 3:
        raise ValueError(f"Expected 3 selected-baselines overlap shard results, found {len(shard_uris)}")
    return pd.concat([_read_gcs_csv(uri) for uri in shard_uris], ignore_index=True).set_index("run_name")


def _load_sl_verb_backfill() -> pd.DataFrame:
    return _read_gcs_csv(QSPLIT240_SL_VERB_RESULTS_URI).set_index("run_name")


def _merged_metric_frame() -> tuple[pd.DataFrame, list[str], list[str]]:
    metrics, _ = _requested_metrics()
    base = _base_rows()
    overlap = _load_overlap_backfill()
    selected_overlap = _load_selected_baselines_overlap_backfill()
    sl_verb = _load_sl_verb_backfill()
    backfilled_metrics: list[str] = []
    backfilled_run_names = set(OVERLAP_BACKFILL_RUN_NAMES)

    for run_name in OVERLAP_BACKFILL_RUN_NAMES:
        if run_name not in overlap.index:
            raise ValueError(f"Missing overlap rerun row for {run_name}")
        for metric in metrics:
            if metric in overlap.columns and pd.notna(overlap.loc[run_name, metric]):
                base.loc[run_name, metric] = overlap.loc[run_name, metric]
                backfilled_metrics.append(metric)
        if run_name not in sl_verb.index:
            raise ValueError(f"Missing mmlu_sl_verb rerun row for {run_name}")
        base.loc[run_name, SL_VERB_METRIC] = sl_verb.loc[run_name, SL_VERB_METRIC]
        backfilled_metrics.append(SL_VERB_METRIC)

    for run_name in SELECTED_BASELINES_OVERLAP_RUN_NAMES:
        if run_name not in selected_overlap.index:
            raise ValueError(f"Missing selected-baselines overlap rerun row for {run_name}")
        for metric in metrics:
            if metric in selected_overlap.columns and pd.notna(selected_overlap.loc[run_name, metric]):
                base.loc[run_name, metric] = selected_overlap.loc[run_name, metric]
                backfilled_metrics.append(metric)
        backfilled_run_names.add(run_name)

    return base, sorted(set(backfilled_metrics)), sorted(backfilled_run_names)


def _wide_frame(metric_frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in RUN_SPECS:
        row = {
            "label": spec.label,
            "run_name": spec.run_name,
        }
        for metric in metrics:
            row[metric] = metric_frame.loc[spec.run_name, metric] if metric in metric_frame.columns else pd.NA
        rows.append(row)
    return pd.DataFrame(rows)


def _long_frame(metric_frame: pd.DataFrame, metrics: list[str], metadata: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in metrics:
        meta_row = metadata.loc[metric] if metric in metadata.index else None
        row: dict[str, object] = {
            "metric": metric,
            "eval_name": None if meta_row is None else meta_row["eval_name"],
            "source": None if meta_row is None else meta_row["source"],
            "primary_metric_kind": None if meta_row is None else meta_row["primary_metric_kind"],
            "signal_n": None if meta_row is None else meta_row["signal_n"],
            "noise_n": None if meta_row is None else meta_row["noise_n"],
            "signal_scale": None if meta_row is None else meta_row["signal_scale"],
            "noise_scale": None if meta_row is None else meta_row["noise_scale"],
            "signal_range": None if meta_row is None else meta_row["signal_range"],
            "signal_to_noise": None if meta_row is None else meta_row["signal_to_noise"],
        }
        for spec in RUN_SPECS:
            row[spec.label] = metric_frame.loc[spec.run_name, metric] if metric in metric_frame.columns else pd.NA
        rows.append(row)
    return pd.DataFrame(rows)


def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    metrics, metadata = _requested_metrics()
    metric_frame, backfilled_metrics, backfilled_run_names = _merged_metric_frame()

    missing_columns = [metric for metric in metrics if metric not in metric_frame.columns]
    missing_values = {
        spec.run_name: [
            metric
            for metric in metrics
            if metric in metric_frame.columns and pd.isna(metric_frame.loc[spec.run_name, metric])
        ]
        for spec in RUN_SPECS
    }
    missing_values = {run_name: values for run_name, values in missing_values.items() if values}

    wide = _wide_frame(metric_frame, metrics)
    common_metrics = [
        metric
        for metric in metrics
        if metric in metric_frame.columns
        and all(not pd.isna(metric_frame.loc[spec.run_name, metric]) for spec in RUN_SPECS)
    ]
    common_wide = _wide_frame(metric_frame, common_metrics)
    long = _long_frame(metric_frame, metrics, metadata)
    summary = {
        "requested_metric_count": len(metrics),
        "common_metric_count": len(common_metrics),
        "run_count": len(RUN_SPECS),
        "primary_metric": PRIMARY_METRIC,
        "backfilled_run_names": backfilled_run_names,
        "backfilled_metric_count": len(backfilled_metrics),
        "backfilled_metrics": backfilled_metrics,
        "missing_columns": missing_columns,
        "missing_values": missing_values,
        "wide_csv": str(OUTPUT_WIDE_CSV),
        "common_wide_csv": str(OUTPUT_COMMON_WIDE_CSV),
        "long_csv": str(OUTPUT_LONG_CSV),
    }
    return wide, common_wide, long, summary


def main() -> None:
    wide, common_wide, long, summary = build_tables()
    wide.to_csv(OUTPUT_WIDE_CSV, index=False)
    common_wide.to_csv(OUTPUT_COMMON_WIDE_CSV, index=False)
    long.to_csv(OUTPUT_LONG_CSV, index=False)
    OUTPUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {OUTPUT_WIDE_CSV}")
    print(f"Wrote {OUTPUT_COMMON_WIDE_CSV}")
    print(f"Wrote {OUTPUT_LONG_CSV}")
    print(f"Wrote {OUTPUT_SUMMARY_JSON}")


if __name__ == "__main__":
    main()
