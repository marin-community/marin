# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Compare repaired RPL to the frozen 300M baseline by paired resampling."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_expanded_300m_pareto_baseline_20260731 as baseline,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_repaired_rpl_300m_20260731 as candidate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    bootstrap_expanded_300m_pareto_baseline_20260731 as bootstrap,
)

DEFAULT_BASELINE_DIR = SCRIPT_DIR / "reference_outputs" / "expanded_300m_pareto_baseline_20260731"
DEFAULT_CANDIDATE_DIR = SCRIPT_DIR / "reference_outputs" / "repaired_rpl_300m_20260731"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "repaired_rpl_300m_bootstrap_20260731"
PROTOCOL_VERSION = "repaired-rpl-300m-bootstrap-v2"
PRIMARY_METRICS = (
    "all_rmse",
    "asymmetric_rmse",
    "pair_delta_rmse",
    "asymmetric_regret_at_1",
    "asymmetric_low_tail_rmse",
    "asymmetric_calibration_slope",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--draws", type=int, default=bootstrap.DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=bootstrap.DEFAULT_SEED)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol_payload(
    baseline_protocol: dict[str, Any],
    candidate_protocol: dict[str, Any],
    draws: int,
    seed: int,
) -> dict[str, Any]:
    payload = {
        "version": PROTOCOL_VERSION,
        "baseline_protocol_hash": baseline_protocol["protocol_hash"],
        "candidate_protocol_hash": candidate_protocol["protocol_hash"],
        "candidate": candidate.MODEL_ID,
        "targets": list(baseline.TARGETS),
        "draws": draws,
        "seed": seed,
        "smooth_metric_bootstrap_unit": "phase_correspondence_key",
        "smooth_metric_bootstrap_strata": "outer_fold",
        "regret_bootstrap_unit": "outer_fold",
        "regret_candidate_population": "fixed full outer-fold test set",
        "metric_directions": bootstrap.METRIC_DIRECTIONS,
        "source_hashes": {
            str(path.relative_to(REPO_ROOT)): file_hash(path) for path in (Path(__file__), Path(bootstrap.__file__))
        },
    }
    encoded = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "protocol_hash": hashlib.sha256(encoded).hexdigest()}


def extended_target(
    baseline_dir: Path,
    candidate_dir: Path,
    baseline_protocol_hash: str,
    candidate_protocol_hash: str,
    target: str,
) -> bootstrap.TargetData:
    data = bootstrap.load_target(baseline_dir, baseline_protocol_hash, target)
    path = candidate_dir / "cells" / target / candidate.MODEL_ID
    marker = json.loads((path / "complete.json").read_text())
    if marker.get("protocol_hash") != candidate_protocol_hash:
        raise ValueError(f"stale candidate cell: {path}")
    frame = pd.read_csv(path / "predictions.csv").sort_values("row_index").reset_index(drop=True)
    bootstrap._aligned_frame(data.frame, frame, candidate.MODEL_ID)
    prediction = frame["predicted"].to_numpy(dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError(f"non-finite candidate predictions for {target}")
    return replace(
        data,
        model_ids=(*data.model_ids, candidate.MODEL_ID),
        predictions=np.concatenate([data.predictions, prediction[None, :]], axis=0),
    )


def orient_candidate_pairwise(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[frame["candidate"].eq(candidate.MODEL_ID) | frame["comparator"].eq(candidate.MODEL_ID)].copy()
    reverse = selected["comparator"].eq(candidate.MODEL_ID)
    original_candidate = selected["candidate"].copy()
    selected.loc[reverse, "candidate"] = candidate.MODEL_ID
    selected.loc[reverse, "comparator"] = original_candidate[reverse]
    for column in ("point_loss_difference", "bootstrap_mean_loss_difference"):
        selected.loc[reverse, column] = -selected.loc[reverse, column]
    old_lower = selected.loc[reverse, "ci_lower"].copy()
    selected.loc[reverse, "ci_lower"] = -selected.loc[reverse, "ci_upper"]
    selected.loc[reverse, "ci_upper"] = -old_lower
    old_better = selected.loc[reverse, "probability_candidate_better"].copy()
    selected.loc[reverse, "probability_candidate_better"] = selected.loc[reverse, "probability_candidate_worse"]
    selected.loc[reverse, "probability_candidate_worse"] = old_better
    probability_total = selected[
        [
            "probability_candidate_better",
            "probability_candidate_tied",
            "probability_candidate_worse",
        ]
    ].sum(axis=1)
    if not np.allclose(probability_total, 1.0, atol=1e-12, rtol=0.0):
        raise RuntimeError("pairwise comparison probabilities do not sum to one")
    return selected.sort_values(["target", "metric", "comparator"]).reset_index(drop=True)


def write_report(
    output_dir: Path,
    protocol: dict[str, Any],
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
) -> None:
    candidate_summary = summary.loc[summary["model"].eq(candidate.MODEL_ID) & summary["metric"].isin(PRIMARY_METRICS)]
    headline = comparisons.loc[
        comparisons["metric"].isin(PRIMARY_METRICS)
        & comparisons["comparator"].isin(("hierarchical_phase_replay", "retained_power_law"))
    ]
    lines = [
        "# Repaired RPL: Correspondence-Cluster Bootstrap",
        "",
        f"- Protocol: `{protocol['protocol_hash']}`",
        f"- Baseline protocol: `{protocol['baseline_protocol_hash']}`",
        f"- Candidate protocol: `{protocol['candidate_protocol_hash']}`",
        f"- Draws: {protocol['draws']}.",
        "- Smooth-error unit: `phase_correspondence_key`, resampled within outer fold.",
        "- Regret unit: outer fold, with the full candidate population fixed.",
        "- Win probabilities are strict; ties and losses are reported separately.",
        "",
        "## Candidate Intervals",
        "",
        candidate_summary[["target", "metric", "point_estimate", "ci_lower", "ci_upper"]].to_markdown(
            index=False, floatfmt=".6f"
        ),
        "",
        "## Candidate Minus Comparator Loss",
        "",
        "Negative differences favor repaired RPL.",
        "",
        headline[
            [
                "target",
                "comparator",
                "metric",
                "point_loss_difference",
                "ci_lower",
                "ci_upper",
                "probability_candidate_better",
                "probability_candidate_tied",
                "probability_candidate_worse",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.draws < 100:
        raise ValueError("at least 100 bootstrap draws are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_protocol = json.loads((args.baseline_dir / "protocol.json").read_text())
    candidate_protocol = json.loads((args.candidate_dir / "protocol.json").read_text())
    protocol = protocol_payload(baseline_protocol, candidate_protocol, args.draws, args.seed)
    write_json(args.output_dir / "protocol.json", protocol)
    marker_path = args.output_dir / "complete.json"
    if not args.force and marker_path.exists():
        marker = json.loads(marker_path.read_text())
        if marker.get("protocol_hash") == protocol["protocol_hash"]:
            print(f"skip complete bootstrap protocol {protocol['protocol_hash']}", flush=True)
            return

    summaries = []
    pairwise = []
    for target_index, target in enumerate(baseline.TARGETS):
        print(f"bootstrap {target}", flush=True)
        data = extended_target(
            args.baseline_dir,
            args.candidate_dir,
            str(baseline_protocol["protocol_hash"]),
            str(candidate_protocol["protocol_hash"]),
            target,
        )
        target_summary, target_pairwise = bootstrap.bootstrap_target(
            data,
            args.draws,
            args.seed + target_index,
        )
        summaries.append(target_summary)
        pairwise.append(target_pairwise)

    summary = pd.concat(summaries, ignore_index=True)
    comparisons = orient_candidate_pairwise(pd.concat(pairwise, ignore_index=True))
    summary.to_csv(args.output_dir / "bootstrap_metric_intervals.csv", index=False)
    comparisons.to_csv(args.output_dir / "candidate_pairwise_differences.csv", index=False)
    write_report(args.output_dir, protocol, summary, comparisons)
    write_json(
        marker_path,
        {
            "protocol_hash": protocol["protocol_hash"],
            "baseline_protocol_hash": protocol["baseline_protocol_hash"],
            "candidate_protocol_hash": protocol["candidate_protocol_hash"],
            "draws": args.draws,
        },
    )
    print(f"completed bootstrap protocol {protocol['protocol_hash']}", flush=True)


if __name__ == "__main__":
    main()
