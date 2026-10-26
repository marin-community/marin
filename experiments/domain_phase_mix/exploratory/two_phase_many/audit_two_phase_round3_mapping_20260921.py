# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Check exact tied MARINER mapping for every final round-three fitted head."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import audit_nextgen_two_phase_20260921 as geometry
import nextgen_two_phase_round3_20260921 as candidate
import numpy as np
from threadpoolctl import threadpool_limits

LIMIT = 1e-8


def load_fit(record: dict[str, Any]) -> candidate.Fit:
    head = candidate.log_head.Head(
        record["intercept"],
        np.asarray(record["aggregate_coefficients"]),
        np.asarray(record["phase_coefficients"]),
        record["floor"],
    )
    return candidate.Fit(
        record["model"],
        record["shape"],
        record["forgetting"],
        record["ridge"],
        record["gamma"],
        record["flat_profile"],
        record["inner_rmse"],
        head,
    )


def native_prediction(fit: candidate.Fit, panel: candidate.benchmark.data.BenchPanel, rows: np.ndarray) -> np.ndarray:
    native = candidate.mariner.Head(fit.head.intercept, fit.head.aggregate_coefficients, fit.head.floor)
    return native.predict(candidate.mariner.design_matrix(panel.epochs[rows], fit.shape))


def check_fit(fit: candidate.Fit, panel: candidate.benchmark.data.BenchPanel, rows: np.ndarray) -> dict[str, float]:
    predicted = fit.predict(panel, rows)
    native = native_prediction(fit, panel, rows)
    individual = np.concatenate([fit.predict(panel, np.asarray([row])) for row in rows])
    tied = geometry.tied_queries(panel)
    all_rows = np.arange(len(panel.rows))
    return {
        "max_bpb_error": float(np.max(np.abs(predicted - native))),
        "actual_tied_rowwise_vs_batch_max_bpb_error": float(np.max(np.abs(predicted - individual))),
        "counterfactual_all_aggregates_max_bpb_error": float(
            np.max(np.abs(fit.predict(tied, all_rows) - native_prediction(fit, tied, all_rows)))
        ),
    }


def main() -> None:
    sources = candidate.source_hashes()
    panel = geometry.geometry_panel(geometry.PREPARED, "starcoder_wsd80_dense")
    rows = np.flatnonzero(panel.tied)
    assert len(rows) == 20
    generic: list[dict[str, Any]] = []
    learned: list[dict[str, Any]] = []
    for model in candidate.MODELS:
        for shape in (
            {"rate": 0.05, "power": 0.3, "threshold": 1.0},
            {"rate": 0.5, "power": 0.7, "threshold": 3.0},
            {"rate": 4.0, "power": 1.0, "threshold": 6.0},
        ):
            phase_width = candidate.phase_design(panel, shape, model).shape[1]
            for mu in candidate.FORGETTING if model == "NG2-008" else (0.0,):
                for floor in (-3.0, 0.7):
                    head = candidate.log_head.Head(
                        -0.2, np.asarray([0.8, 0.3, 0.1, 0.04]), np.linspace(-0.7, 0.5, phase_width), floor
                    )
                    fit = candidate.Fit(model, shape, mu, 0.0, 1.5, True, 0.0, head)
                    generic.append(
                        {"model": model, "shape": shape, "forgetting": mu, "floor": floor, **check_fit(fit, panel, rows)}
                    )
    for path in sorted((candidate.ROOT / "starcoder_screen_v1/shards").glob("*/fit.json")):
        completed = json.loads((path.parent / "complete.json").read_text())
        assert completed["identity"]["sources"] == sources
        assert candidate.benchmark.sha256(path) == completed["outputs"]["fit.json"]
        record = json.loads(path.read_text())
        fit = load_fit(record)
        learned.append(
            {
                "model": fit.model,
                "target": completed["identity"]["target"],
                "outer_fold": completed["identity"]["outer_fold"],
                "fit": candidate.benchmark.receipt(path),
                **check_fit(fit, panel, rows),
            }
        )
    assert len(learned) == 100
    model_status = {}
    for model in candidate.MODELS:
        selected = [item for item in learned if item["model"] == model]
        assert len(selected) == 25
        checks = selected + [item for item in generic if item["model"] == model]
        maximum = max(
            item[key]
            for item in checks
            for key in (
                "max_bpb_error",
                "actual_tied_rowwise_vs_batch_max_bpb_error",
                "counterfactual_all_aggregates_max_bpb_error",
            )
        )
        model_status[model] = "pass" if maximum <= LIMIT else "fail"
    assert candidate.source_hashes() == sources
    result = {
        "sources": sources,
        "audit_source": candidate.benchmark.receipt(Path(__file__)),
        "status": "pass" if all(value == "pass" for value in model_status.values()) else "fail",
        "model_status": model_status,
        "numeric_limit": LIMIT,
        "query_row_ids": panel.rows.iloc[rows].row_id.astype(str).tolist(),
        "geometry": candidate.geometry_identity(panel),
        "generic": generic,
        "learned_heads": learned,
        "scope": (
            "Actual20tiedStarCoder rows for all100 final learned heads; generic heads and tied counterfactuals separate."
        ),
        "outcomes_read": False,
        "new_fits": 0,
    }
    candidate.benchmark.write_json(candidate.ROOT / "structural_mapping.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "model_status": model_status,
                "max_learned_bpb_error": max(item["max_bpb_error"] for item in learned),
            }
        )
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
