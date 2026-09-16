# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "tabulate"]
# ///
"""Fit repetition-aware data-filtering-law adaptations on the 300M swarm.

This is the fixed-scale companion to
``fit_repetition_aware_variable_size_law_nd.py``. It tests the closest Marin
translation of Goyal et al. (arXiv:2404.07177) on all 242 signal rows in the
current 300M/6B raw metric matrix, separately for uncheatable BPB and the
issue #5416 aggregate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.two_phase_many.fit_repetition_aware_variable_size_law_nd import (
    D_REF,
    N_REF,
    SPECS,
    decoded_params,
    fit_law,
    metrics,
    parameter_count,
    predict_law,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.metric_registry.issue5416_aggregate import (
    fit_issue5416_projection,
    score_issue5416_aggregate,
    select_issue5416_task_columns,
    write_issue5416_projection,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    MANY_DOMAIN_TARGET,
    load_two_phase_many_packet,
)
from experiments.domain_phase_mix.static_batch_selection import build_dataset_spec_from_frame

SCRIPT_DIR = Path(__file__).resolve().parent
MATRIX_DIR = SCRIPT_DIR / "metric_registry" / "raw_metric_matrix_300m"
SIGNAL_CSV = MATRIX_DIR / "raw_metric_matrix_300m.csv"
VARIABLE_NOISE_CSV = MATRIX_DIR / "noise_baseline_run00097_variable_subset_300m.csv"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "repetition_aware_data_filtering_law_300m_20260521"
CV_SPLITS = 5
CV_SEED = 0


def _load_signal_targets() -> tuple[dict[str, tuple[pd.DataFrame, np.ndarray]], Any]:
    """Load full 300M signal rows and both target orientations."""

    signal = pd.read_csv(SIGNAL_CSV, low_memory=False)
    noise = pd.read_csv(VARIABLE_NOISE_CSV, low_memory=False)
    if len(signal) != 242:
        raise ValueError(f"Expected 242 signal rows, found {len(signal)}")
    if len(noise) != 10:
        raise ValueError(f"Expected 10 variable-noise rows, found {len(noise)}")
    task_columns, _task_signs = select_issue5416_task_columns(list(signal.columns))
    issue_complete_mask = signal.loc[:, list(task_columns)].notna().all(axis=1)
    issue_signal = signal.loc[issue_complete_mask].reset_index(drop=True)
    projection = fit_issue5416_projection(signal_frame=issue_signal, noise_frame=noise)
    issue5416_scores = score_issue5416_aggregate(issue_signal, projection, fail_missing=True)
    if issue5416_scores.isna().any():
        missing = int(issue5416_scores.isna().sum())
        raise ValueError(f"Issue #5416 aggregate has {missing} missing signal rows")
    uncheatable_series = pd.to_numeric(signal[MANY_DOMAIN_TARGET], errors="coerce")
    uncheatable_mask = np.isfinite(uncheatable_series.to_numpy(dtype=float))
    uncheatable_signal = signal.loc[uncheatable_mask].reset_index(drop=True)
    uncheatable = uncheatable_series.loc[uncheatable_mask].to_numpy(dtype=float)
    return {
        "uncheatable_bpb": (uncheatable_signal, uncheatable),
        "issue5416_loss": (issue_signal, -issue5416_scores.to_numpy(dtype=float)),
    }, projection


def _data_packet(signal: pd.DataFrame, y: np.ndarray, target_name: str) -> dict[str, Any]:
    """Build the data dict expected by the ND RAML fitting helpers."""

    reference = load_two_phase_many_packet(target=MANY_DOMAIN_TARGET)
    frame = signal.copy()
    frame["objective_metric"] = y
    spec = build_dataset_spec_from_frame(
        frame,
        objective_metric="objective_metric",
        name=f"raml_300m_{target_name}",
        loop=None,
    )
    weights = np.asarray(spec.weights, dtype=float)
    c0 = np.asarray(reference.c0, dtype=float)
    c1 = np.asarray(reference.c1, dtype=float)
    domain_names = np.asarray(spec.domain_names, dtype=object).astype(str)
    if list(domain_names) != list(reference.domain_names):
        raise ValueError("Domain ordering mismatch against canonical two-phase packet")
    epoch_multipliers = np.broadcast_to(np.stack([c0, c1], axis=0), weights.shape).copy()
    return {
        "frame": frame.reset_index(drop=True),
        "weights": weights,
        "y": np.asarray(y, dtype=float),
        "mixture_ids": frame["run_name"].astype(str).to_numpy(),
        "run_names": frame["run_name"].astype(str).to_numpy(),
        "scale_names": np.asarray(["300m_6b"], dtype=object),
        "scale_index": np.zeros(len(frame), dtype=np.int64),
        "domain_names": domain_names,
        "model_sizes": np.full(len(frame), N_REF, dtype=float),
        "realized_train_tokens": np.full(len(frame), D_REF, dtype=float),
        "simulated_epoch_multipliers": epoch_multipliers,
    }


def _cross_validated_summary(data: dict[str, Any], target_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every RAML spec with 5-fold OOF predictions."""

    indices = np.arange(len(data["y"]))
    splitter = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_SEED)
    summary_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    for law_spec in SPECS:
        print(f"Fitting {target_name} / {law_spec.name}", flush=True)
        oof = np.full(len(indices), np.nan, dtype=float)
        if law_spec.name == "scale_only_chinchilla_size":
            # At fixed 300M/6B, the pure N,D law has no mixture-varying feature.
            for train_idx, test_idx in splitter.split(indices):
                oof[test_idx] = float(np.mean(data["y"][indices[train_idx]]))
            optimizer_success = True
            optimizer_message = "fixed-scale constant OOF mean baseline"
            param_count = 1
            decoded: dict[str, Any] = {}
        else:
            full_model = fit_law(data, indices, law_spec)
            for train_idx, test_idx in splitter.split(indices):
                model = fit_law(data, indices[train_idx], law_spec, initial=full_model.theta)
                oof[test_idx] = predict_law(model, data, indices[test_idx])
            optimizer_success = full_model.optimizer_success
            optimizer_message = full_model.optimizer_message
            param_count = parameter_count(law_spec, len(data["domain_names"]))
            decoded = decoded_params(full_model, len(data["domain_names"]))
        if np.isnan(oof).any():
            raise ValueError(f"Missing OOF predictions for {target_name}/{law_spec.name}")
        row = metrics(data["y"], oof)
        row.update(
            {
                "target": target_name,
                "model": law_spec.name,
                "parameter_count": param_count,
                "optimizer_success": optimizer_success,
                "optimizer_message": optimizer_message,
            }
        )
        row.update(decoded)
        summary_rows.append(row)
        pred_frame = data["frame"][["run_name"]].copy()
        pred_frame["target"] = target_name
        pred_frame["model"] = law_spec.name
        pred_frame["actual"] = data["y"]
        pred_frame["predicted_oof"] = oof
        pred_frame["residual_oof"] = oof - data["y"]
        prediction_frames.append(pred_frame)
        print(
            f"  rmse={row['rmse']:.6f} spearman={row['spearman']:.6f} regret_at_1={row['regret_at_1']:.6f}",
            flush=True,
        )
    return pd.DataFrame.from_records(summary_rows), pd.concat(prediction_frames, ignore_index=True)


def _write_report(summary: pd.DataFrame) -> None:
    """Write a compact Markdown report."""

    row_counts = summary.groupby("target")["n"].max().astype(int).to_dict()
    cols = [
        "target",
        "model",
        "parameter_count",
        "rmse",
        "mae",
        "spearman",
        "pearson",
        "regret_at_1",
        "top8_overlap",
        "optimizer_success",
    ]
    lines = [
        "# Repetition-Aware Data-Filtering Law on 300M Swarm",
        "",
        "## Data",
        "",
        f"- Source signal rows: `{len(pd.read_csv(SIGNAL_CSV, usecols=['run_name']))}`",
        f"- Complete target rows: `{row_counts}`",
        f"- Source matrix: `{SIGNAL_CSV}`",
        "- Targets: `eval/uncheatable_eval/bpb` and `-issue5416_aggregate`.",
        "- Caveat: each target is fitted only on rows complete for that target.",
        "",
        "## Form",
        "",
        "This fixed-scale screen reuses the RAML adaptations from the ND script. Domain exposure is",
        "",
        r"\(h_i = 0.8w_{0i}+0.2w_{1i}\), \(r_i=w_{0i}c_{0i}+w_{1i}c_{1i}\),",
        "",
        "and repeated exposure is discounted by",
        "",
        r"\(r_{i,\mathrm{eff}}=r_i\) for \(r_i\le1\), and \(r_{i,\mathrm{eff}}=1+r_1(1-\exp(-(r_i-1)/r_1))\) for \(r_i>1\).",
        "",
        r"The scalar law uses \(D_{\mathrm{eff}}=\sum_i h_iD\,r_{i,\mathrm{eff}}/r_i\), optionally with positive per-domain value weights and a signed linear mixture head.",
        "",
        "## 5-Fold OOF Results",
        "",
        summary[cols].sort_values(["target", "rmse"]).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation",
        "",
        "- The fixed-scale form is tested on every currently target-complete row: 100 for uncheatable BPB and 93 for issue #5416 aggregate.",
        "- Domain value weights are required for useful rank fit; uniform-domain repetition is too weak as a mixture model.",
        "- The more flexible signed-head/per-domain-r1 forms should be read as high-capacity comparators because optimizer convergence is not always clean.",
        "",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targets, projection = _load_signal_targets()
    write_issue5416_projection(projection, OUTPUT_DIR / "issue5416_projection.json")
    summaries = []
    predictions = []
    for target_name, (signal, y) in targets.items():
        summary, pred = _cross_validated_summary(_data_packet(signal, y, target_name), target_name)
        summaries.append(summary)
        predictions.append(pred)
    summary_frame = pd.concat(summaries, ignore_index=True)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    summary_frame.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    prediction_frame.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary_frame.to_dict(orient="records"), indent=2))
    _write_report(summary_frame)
    print(f"Wrote {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
