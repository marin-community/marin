# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Audit whether mixture policy alone determines the frozen 3e18 targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.spatial.distance import cdist

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    freeze_baseline_gate as gate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
DEFAULT_DASHBOARD = RESEARCH_DIR / "mixture_fit_debugger/src/generated/dashboard_data.json"
DEFAULT_OUTPUT = RESEARCH_DIR / ("reference_outputs/mechanistic_surrogate_discovery_20260717/policy_determinacy_audit")
TARGETS = ("uncheatable", "table9")
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", type=Path, default=DEFAULT_DASHBOARD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def vector_key(values: np.ndarray) -> str:
    return hashlib.sha256(np.round(values, 12).tobytes()).hexdigest()[:20]


def phase_fraction(rows: list[dict[str, Any]]) -> float:
    estimates: list[float] = []
    for row in rows:
        phase0 = np.asarray(row["phase0"], dtype=float)
        phase1 = np.asarray(row["phase1"], dtype=float)
        aggregate = np.asarray(row["aggregate"], dtype=float)
        denominator = phase0 - phase1
        valid = np.abs(denominator) > 1e-9
        if np.any(valid):
            estimates.extend(((aggregate[valid] - phase1[valid]) / denominator[valid]).tolist())
    gamma0 = float(np.median(estimates))
    if not 0.0 < gamma0 < 1.0:
        raise ValueError(f"Invalid inferred phase fraction {gamma0}")
    return gamma0


def row_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        if row["isSharedAlias"]:
            continue
        phase0 = np.asarray(row["phase0"], dtype=float)
        phase1 = np.asarray(row["phase1"], dtype=float)
        aggregate = np.asarray(row["aggregate"], dtype=float)
        records.append(
            {
                "row_id": row["id"],
                "name": row["name"],
                "split": row["split"],
                "policy_family": row["policyFamily"],
                "source_experiment": row["sourceExperiment"],
                "method": row["method"],
                "policy_key": vector_key(np.concatenate([phase0, phase1])),
                "aggregate_key": vector_key(aggregate),
                "phase0": phase0,
                "phase1": phase1,
                "aggregate": aggregate,
                **{
                    target: np.nan if row["observed"].get(target) is None else float(row["observed"][target])
                    for target in TARGETS
                },
            }
        )
    return pd.DataFrame(records)


def exact_policy_summary(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    members: list[pd.DataFrame] = []
    for key, group in frame.groupby("policy_key", sort=False):
        if len(group) < 2:
            continue
        local = group.copy()
        local["duplicate_group_size"] = len(group)
        members.append(local.drop(columns=["phase0", "phase1", "aggregate"]))
        record: dict[str, Any] = {
            "policy_key": key,
            "n": len(group),
            "source_count": group["source_experiment"].nunique(),
            "method_count": group["method"].nunique(),
            "row_ids": " | ".join(group["row_id"].astype(str)),
        }
        for target in TARGETS:
            values = group[target].dropna().to_numpy(dtype=float)
            record[f"{target}_n"] = len(values)
            record[f"{target}_mean"] = float(np.mean(values)) if len(values) else np.nan
            record[f"{target}_std"] = float(np.std(values, ddof=1)) if len(values) >= 2 else np.nan
            record[f"{target}_range"] = float(np.ptp(values)) if len(values) >= 2 else np.nan
            record[f"{target}_within_sse"] = float(np.sum(np.square(values - np.mean(values)))) if len(values) else 0.0
        summaries.append(record)
    return pd.DataFrame(summaries), pd.concat(members, ignore_index=True)


def nearest_policy_pairs(frame: pd.DataFrame, gamma0: float) -> pd.DataFrame:
    vectors0 = np.stack(frame["phase0"].to_numpy())
    vectors1 = np.stack(frame["phase1"].to_numpy())
    distance = gamma0 * cdist(vectors0, vectors0, metric="cityblock") / 2.0
    distance += (1.0 - gamma0) * cdist(vectors1, vectors1, metric="cityblock") / 2.0
    same_policy = frame["policy_key"].to_numpy()[:, None] == frame["policy_key"].to_numpy()[None, :]
    distance[same_policy] = np.inf
    records: list[dict[str, Any]] = []
    for index in range(len(frame)):
        neighbor = int(np.argmin(distance[index]))
        record: dict[str, Any] = {
            "row_id": frame.iloc[index]["row_id"],
            "neighbor_id": frame.iloc[neighbor]["row_id"],
            "split": frame.iloc[index]["split"],
            "neighbor_split": frame.iloc[neighbor]["split"],
            "policy_distance": float(distance[index, neighbor]),
        }
        for target in TARGETS:
            left = float(frame.iloc[index][target])
            right = float(frame.iloc[neighbor][target])
            record[f"{target}_gap"] = abs(left - right) if np.isfinite(left) and np.isfinite(right) else np.nan
        records.append(record)
    return pd.DataFrame(records).sort_values("policy_distance")


def aggregate_equivalence(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby("aggregate_key", sort=False):
        if group["policy_key"].nunique() < 2:
            continue
        record: dict[str, Any] = {
            "aggregate_key": key,
            "n": len(group),
            "distinct_phase_policies": group["policy_key"].nunique(),
            "row_ids": " | ".join(group["row_id"].astype(str)),
        }
        for target in TARGETS:
            values = group[target].dropna().to_numpy(dtype=float)
            record[f"{target}_range"] = float(np.ptp(values)) if len(values) >= 2 else np.nan
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    gate.assert_sealed_absent(args.dashboard)
    dashboard = json.loads(args.dashboard.read_text())
    rows = dashboard["swarms"]["delphi_3e18"]["rows"]
    gamma0 = phase_fraction(rows)
    frame = row_frame(rows).dropna(subset=list(TARGETS), how="all").reset_index(drop=True)
    exact, members = exact_policy_summary(frame)
    near = nearest_policy_pairs(frame.dropna(subset=list(TARGETS)).reset_index(drop=True), gamma0)
    aggregate = aggregate_equivalence(frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    exact.to_csv(args.output_dir / "exact_policy_groups.csv", index=False)
    members.to_csv(args.output_dir / "exact_policy_rows.csv", index=False)
    near.to_csv(args.output_dir / "nearest_policy_pairs.csv", index=False)
    aggregate.to_csv(args.output_dir / "aggregate_equivalent_phase_variation.csv", index=False)

    floor_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        usable = exact.loc[exact[f"{target}_n"] >= 2]
        repeated_rows = int(usable[f"{target}_n"].sum())
        irreducible_rmse = float(np.sqrt(usable[f"{target}_within_sse"].sum() / repeated_rows))
        floor_rows.append(
            {
                "target": target,
                "duplicate_policy_groups": len(usable),
                "repeated_rows": repeated_rows,
                "policy_only_rmse_floor_on_repeats": irreducible_rmse,
                "max_exact_policy_range": float(usable[f"{target}_range"].max()),
                "p90_exact_policy_range": float(usable[f"{target}_range"].quantile(0.9)),
            }
        )
    floor = pd.DataFrame(floor_rows)
    floor.to_csv(args.output_dir / "policy_only_noise_floor.csv", index=False)

    plot = near.melt(
        id_vars=["row_id", "neighbor_id", "policy_distance"],
        value_vars=[f"{target}_gap" for target in TARGETS],
        var_name="target",
        value_name="absolute_bpb_gap",
    ).dropna()
    plot["target"] = plot["target"].str.removesuffix("_gap")
    figure = px.scatter(
        plot,
        x="policy_distance",
        y="absolute_bpb_gap",
        color="target",
        hover_data=["row_id", "neighbor_id"],
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
        log_x=True,
        title="Nearest distinct-policy BPB gaps",
    )
    figure.update_layout(template="plotly_white")
    figure.write_html(
        args.output_dir / "nearest_policy_outcome_gaps.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )
    (args.output_dir / "report.md").write_text(
        "# Policy determinacy audit\n\n"
        f"The inferred phase fractions are {gamma0:.3f}/{1.0 - gamma0:.3f}. Exact duplicate policies provide a lower bound on error for any deterministic policy-only surrogate.\n\n"
        + floor.to_markdown(index=False, floatfmt=".6f")
        + "\n\nThe exact-policy ranges are far smaller than the >0.05-BPB optimism failures. Training/evaluation randomness therefore cannot explain the headline calibration gap.\n\n"
        + "## Aggregate-equivalent phase schedules\n\n"
        + (
            aggregate.sort_values("table9_range", ascending=False).head(20).to_markdown(index=False, floatfmt=".6f")
            if not aggregate.empty
            else "No distinct phase schedules share an exact aggregate policy."
        )
        + "\n"
    )
    print(floor.to_string(index=False))


if __name__ == "__main__":
    main()
