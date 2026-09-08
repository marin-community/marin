# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Test whether RPL phase predictions are driven by unconserved retained mass.

This is the preregistered no-fit diagnostic following WSD80-SUR-046. It uses
the already selected outer-fold shapes and OOF predictions. No response
parameter, feature, fold, or threshold is selected from the correlations
reported here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_wsd80_cross_metric_rpl_20260730 as wsd_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_aggregate_conditioned_replay_control_20260730 as expanded,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_expanded_300m_pareto_baseline_20260731 as baseline,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_repaired_rpl_300m_20260731 as repaired_300m,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_repaired_rpl_wsd80_controls_20260731 as wsd_controls,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_model_20260728 as rpl,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    starcoder_wsd80_panel_20260728 as wsd80,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "retained_mass_artifact_diagnostic_20260731"
RPL_300M_DIR = SCRIPT_DIR / "reference_outputs" / "repaired_rpl_300m_20260731"
RPL_WSD_DIR = SCRIPT_DIR / "reference_outputs" / "repaired_rpl_wsd80_controls_20260731"
PROTOCOL_VERSION = "retained-mass-artifact-diagnostic-v2-full-shape"
OBSERVED_CORRELATION_LIMIT = 0.2
PREDICTED_CORRELATION_FLOOR = 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def source_inputs() -> tuple[Path, ...]:
    files = [
        Path(__file__),
        Path(rpl.__file__),
        Path(repaired_300m.__file__),
        Path(wsd_controls.__file__),
    ]
    for target in baseline.TARGETS:
        cell = RPL_300M_DIR / "cells" / target / repaired_300m.MODEL_ID
        files.extend(
            (
                cell / "predictions.csv",
                cell / "full_fit.json",
            )
        )
    for metric in wsd_controls.TARGETS:
        cell = wsd_controls.cell_dir(RPL_WSD_DIR, "random", metric)
        files.extend(
            (
                cell / "predictions.csv",
                cell / "full_fit.json",
            )
        )
    return tuple(files)


def protocol_payload() -> dict[str, Any]:
    inputs = source_inputs()
    missing = [str(path) for path in inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen diagnostic inputs: {missing}")
    payload: dict[str, Any] = {
        "version": PROTOCOL_VERSION,
        "candidate": "WSD80-SUR-046 diagnostic only",
        "statistic": "Spearman correlation of retained-mass ratio with exact-pair BPB deltas",
        "state_shape": "target-specific frozen full-fit shape",
        "prediction": "OOF prediction",
        "v2_reason": (
            "WSD80 random folds do not keep aggregate counterparts together, so "
            "fold-specific shapes cannot define one shared pair state"
        ),
        "observed_correlation_limit": OBSERVED_CORRELATION_LIMIT,
        "predicted_correlation_floor": PREDICTED_CORRELATION_FLOOR,
        "interpretation": {
            "supports_conservation": (
                "absolute observed correlation below 0.2 and absolute predicted " "correlation above 0.6"
            ),
            "contradicts_conservation": "absolute observed correlation at least 0.2",
            "otherwise": "inconclusive",
        },
        "source_hashes": {str(path.relative_to(REPO_ROOT)): baseline.file_hash(path) for path in inputs},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "protocol_hash": hashlib.sha256(encoded).hexdigest()}


def full_shape(path: Path) -> rpl.Shape:
    row = json.loads(path.read_text())
    return rpl.Shape(**row["shape"])


def fold_assignment(
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    rows: int,
) -> np.ndarray:
    assignment = np.full(rows, -1, dtype=int)
    for fold_id, (_train, test) in enumerate(folds):
        assignment[test] = fold_id
    if np.any(assignment < 0):
        raise ValueError("fold assignment does not cover every row")
    return assignment


def retained_mass(weights: np.ndarray, geometry: rpl.Geometry, shape: rpl.Shape) -> np.ndarray:
    return rpl.retained_share(
        weights,
        geometry,
        shape.retention,
        shape.late_multiplier,
    ).sum(axis=1)


def paired_rows(
    panel: str,
    target: str,
    weights: np.ndarray,
    observed: np.ndarray,
    predicted: np.ndarray,
    tied_rows: np.ndarray,
    asymmetric_rows: np.ndarray,
    folds: np.ndarray,
    shape: rpl.Shape,
    geometry: rpl.Geometry,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for tied, asymmetric in zip(tied_rows, asymmetric_rows, strict=True):
        mass = retained_mass(weights[[tied, asymmetric]], geometry, shape)
        if mass[0] <= 0.0:
            raise ValueError("tied retained mass must be positive")
        rows.append(
            {
                "panel": panel,
                "target": target,
                "tied_fold": int(folds[tied]),
                "asymmetric_fold": int(folds[asymmetric]),
                "tied_row": int(tied),
                "asymmetric_row": int(asymmetric),
                "retained_mass_tied": float(mass[0]),
                "retained_mass_asymmetric": float(mass[1]),
                "retained_mass_ratio": float(mass[1] / mass[0]),
                "log_retained_mass_ratio": float(np.log(mass[1] / mass[0])),
                "observed_delta": float(observed[asymmetric] - observed[tied]),
                "predicted_delta": float(predicted[asymmetric] - predicted[tied]),
            }
        )
    return pd.DataFrame(rows)


def load_300m_pairs(target: str) -> pd.DataFrame:
    dataset = expanded.load_300m(target)
    pooled = baseline.as_pooled(dataset)
    context = repaired_300m.selection_context(pooled)
    cell = RPL_300M_DIR / "cells" / target / repaired_300m.MODEL_ID
    predictions = pd.read_csv(cell / "predictions.csv").sort_values("row_index")
    expected_rows = np.arange(dataset.n)
    if not np.array_equal(predictions["row_index"].to_numpy(dtype=int), expected_rows):
        raise ValueError(f"stored 300M predictions are not row-aligned for {target}")
    folds = predictions["outer_fold"].to_numpy(dtype=int)
    return paired_rows(
        "300m",
        target,
        dataset.weights,
        dataset.y,
        predictions["predicted"].to_numpy(dtype=float),
        context.pair_tied,
        context.pair_asymmetric,
        folds,
        full_shape(cell / "full_fit.json"),
        baseline.retained_geometry(pooled, dataset.family_index),
    )


def load_wsd_pairs(
    metric: str,
    panel: wsd80.Panel,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    context = wsd_controls.selection_context(panel.weights)
    cell = wsd_controls.cell_dir(RPL_WSD_DIR, "random", metric)
    predictions = pd.read_csv(cell / "predictions.csv").sort_values("row_index")
    expected_rows = np.arange(len(panel.y))
    if not np.array_equal(predictions["row_index"].to_numpy(dtype=int), expected_rows):
        raise ValueError(f"stored WSD80 predictions are not row-aligned for {metric}")
    outer = wsd_controls.fold_builder(
        "random",
        panel.weights,
        expected_rows,
        wsd_controls.OUTER_SPLITS,
        wsd_controls.OUTER_SEED,
    )
    geometry = rpl.Geometry(
        c0=panel.c0,
        c1=panel.c1,
        phase_0_fraction=wsd80.REALIZED_PHASE_0_FRACTION,
    )
    return paired_rows(
        "starcoder_wsd80",
        metric,
        panel.weights,
        frame[metric].to_numpy(dtype=float),
        predictions["predicted"].to_numpy(dtype=float),
        context.pair_tied,
        context.pair_asymmetric,
        fold_assignment(outer, len(panel.y)),
        full_shape(cell / "full_fit.json"),
        geometry,
    )


def correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    result = spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def summarize(block: pd.DataFrame) -> dict[str, Any]:
    mass = block["log_retained_mass_ratio"].to_numpy(dtype=float)
    observed = block["observed_delta"].to_numpy(dtype=float)
    predicted = block["predicted_delta"].to_numpy(dtype=float)
    observed_rho, observed_p = correlation(mass, observed)
    predicted_rho, predicted_p = correlation(mass, predicted)
    delta_rho, delta_p = correlation(observed, predicted)

    if abs(observed_rho) < OBSERVED_CORRELATION_LIMIT and abs(predicted_rho) > PREDICTED_CORRELATION_FLOOR:
        decision = "supports_conservation"
    elif abs(observed_rho) >= OBSERVED_CORRELATION_LIMIT:
        decision = "contradicts_conservation"
    else:
        decision = "inconclusive"
    return {
        "panel": str(block["panel"].iloc[0]),
        "target": str(block["target"].iloc[0]),
        "pairs": len(block),
        "retained_mass_ratio_min": float(block["retained_mass_ratio"].min()),
        "retained_mass_ratio_median": float(block["retained_mass_ratio"].median()),
        "retained_mass_ratio_max": float(block["retained_mass_ratio"].max()),
        "spearman_mass_observed_delta": observed_rho,
        "spearman_mass_observed_delta_p": observed_p,
        "spearman_mass_predicted_delta": predicted_rho,
        "spearman_mass_predicted_delta_p": predicted_p,
        "spearman_observed_predicted_delta": delta_rho,
        "spearman_observed_predicted_delta_p": delta_p,
        "decision": decision,
    }


def write_report(output_dir: Path, summary: pd.DataFrame, protocol: dict[str, Any]) -> None:
    lines = [
        "# Retained-Mass Artifact Diagnostic",
        "",
        f"- Protocol: `{protocol['protocol_hash']}`",
        "- No model was fitted and no response parameter was changed.",
        "- Retained mass uses each target's frozen full-fit RPL shape.",
        "- Pair deltas use the already stored OOF predictions.",
        f"- Conservation support gate: `|rho_observed| < {OBSERVED_CORRELATION_LIMIT}` and "
        f"`|rho_predicted| > {PREDICTED_CORRELATION_FLOOR}`.",
        "",
        "## Results",
        "",
        summary.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation",
        "",
    ]
    counts = summary["decision"].value_counts()
    lines.extend(
        [
            f"- Supports conservation: {int(counts.get('supports_conservation', 0))} cells.",
            f"- Contradicts conservation: {int(counts.get('contradicts_conservation', 0))} cells.",
            f"- Inconclusive: {int(counts.get('inconclusive', 0))} cells.",
            "",
            "A conserved retained-share candidate is admissible only if the primary "
            "300M and WSD80 controls do not contradict conservation. Mixed evidence "
            "requires a narrower mechanism rather than averaging the cells.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = protocol_payload()
    baseline.write_json(args.output_dir / "protocol.json", protocol)
    if args.prepare_only:
        print(f"prepared protocol {protocol['protocol_hash']} in {args.output_dir}", flush=True)
        return

    blocks = [load_300m_pairs(target) for target in baseline.TARGETS]
    panel, frame, available = wsd_audit.load_metric_panel()
    missing = sorted(set(wsd_controls.TARGETS) - set(available))
    if missing:
        raise ValueError(f"WSD80 controls are incomplete: {missing}")
    blocks.extend(load_wsd_pairs(metric, panel, frame) for metric in wsd_controls.TARGETS)
    pairs = pd.concat(blocks, ignore_index=True)
    summary = pd.DataFrame(summarize(block) for _keys, block in pairs.groupby(["panel", "target"], sort=True))
    pairs.to_csv(args.output_dir / "pair_diagnostics.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    write_report(args.output_dir, summary, protocol)
    baseline.write_json(
        args.output_dir / "complete.json",
        {
            "protocol_hash": protocol["protocol_hash"],
            "cells": len(summary),
            "pairs": len(pairs),
        },
    )
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
