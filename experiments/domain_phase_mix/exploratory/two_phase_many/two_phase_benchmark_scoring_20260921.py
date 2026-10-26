# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas"]
# ///
"""Score lower-is-better endpoints and apply frozen empirical screening gates.

Each cell is a separate held-out policy menu. No raw BPB is pooled across panels.
The protocol declares exact required cells, evidence names, and per-panel/target
gates. Every gate metric declares exactly one of ``allowed_increase`` (candidate
minus reference), ``absolute_limit`` (upper bound), or ``minimum`` (lower bound).
Tolerances must carry external or reference-calibration provenance and have been
frozen before candidate evaluation. Passing is an empirical screen, never a
confidence-interval or independent-training-seed certificate.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd

CELL_COLUMNS = ["panel", "target", "repeat", "outer_fold", "role"]
SCORE_COLUMNS = ["model", *CELL_COLUMNS]
ROW_COLUMNS = ["model", "panel", "target", "repeat", "outer_fold", "row_id"]
IDENTITY_COLUMNS = ["row_id", "policy_id", "group_id", "source", "region", "role"]
REQUIRED_COLUMNS = list(dict.fromkeys([*ROW_COLUMNS, *IDENTITY_COLUMNS, "observed", "predicted"]))
METRICS = [
    "n",
    "rmse",
    "mae",
    "predictive_r2",
    "signed_bias",
    "optimism",
    "calibration_slope",
    "calibration_intercept",
    "regret_at1",
    "regret_at3",
    "regret_at5",
    "selected_observed",
    "selected_predicted",
    "selected_optimism",
    "frontier_n",
    "frontier_rmse",
    "frontier_optimism",
]
BOUND_NAMES = ("allowed_increase", "absolute_limit", "minimum")
CLAIM = "empirical_screen_only"


def _validate_predictions(frame: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Prediction frame is missing columns: {missing}")
    if frame.empty:
        return
    if frame[REQUIRED_COLUMNS].isna().any().any():
        raise ValueError("Missing predictions, observations, or identity fields; exclude missing targets upstream")
    if frame.duplicated(ROW_COLUMNS).any():
        raise ValueError(f"Duplicate prediction identity: {ROW_COLUMNS}")
    if not frame.role.isin(["test", "tied_test"]).all():
        raise ValueError("Only held-out test and tied_test roles can be scored")
    if any(not pd.api.types.is_numeric_dtype(frame[column]) for column in ("observed", "predicted")):
        raise ValueError("Prediction and observed endpoint columns must be numeric")
    if not np.isfinite(frame[["observed", "predicted"]].to_numpy(dtype=float)).all():
        raise ValueError("Predictions and observed endpoints must be finite; exclude missing targets upstream")


def _identity_token(value: Any) -> str:
    # Type tags avoid making integer 1 and string "1" identical in paired menus.
    return f"{type(value).__name__}:{value!s}"


def _menu_fingerprint(menu: pd.DataFrame) -> str:
    rows = [
        [*(_identity_token(value) for value in row[:-1]), float(row[-1]).hex()]
        for row in menu[["repeat", "outer_fold", *IDENTITY_COLUMNS, "observed"]].itertuples(index=False, name=None)
    ]
    payload = json.dumps(sorted(rows), ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _joint_menus(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([frame, frame.assign(role="joint_test")], ignore_index=True)


def score_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Return metrics for each model and exact held-out menu.

    Calibration regresses observed on predicted. Optimism is observed minus
    predicted, so positive values indicate overly favorable predictions. Regret
    at k evaluates the best observed outcome among the k predicted choices.
    Selection uses predicted endpoint, policy ID, then row ID, never outcomes.
    The frontier is the lowest predicted ceil(20% * n) rows, with minimum one.
    Alongside input roles, joint_test compares their union within each fold so
    systematic optimism for asymmetric policies cannot evade selection checks.
    Constant observed endpoints have undefined predictive R-squared; constant
    predictions have undefined calibration slope and intercept.
    """
    _validate_predictions(frame)
    rows = []
    for key, menu in _joint_menus(frame).groupby(SCORE_COLUMNS, sort=True, dropna=False):
        ordered = menu.assign(
            _policy_tie=menu.policy_id.map(_identity_token),
            _row_tie=menu.row_id.map(_identity_token),
        ).sort_values(["predicted", "_policy_tie", "_row_tie"], kind="stable")
        observed = ordered.observed.to_numpy(dtype=float)
        predicted = ordered.predicted.to_numpy(dtype=float)
        error = predicted - observed
        observed_centered = observed - observed.mean()
        predicted_centered = predicted - predicted.mean()
        sst = float(observed_centered @ observed_centered)
        prediction_variation = float(predicted_centered @ predicted_centered)
        slope = (
            float(predicted_centered @ observed_centered / prediction_variation)
            if prediction_variation > 0
            else float("nan")
        )
        frontier_n = max(1, math.ceil(0.2 * len(menu)))
        row = dict(zip(SCORE_COLUMNS, key, strict=True))
        row.update(
            n=len(menu),
            rmse=float(np.sqrt(np.mean(error**2))),
            mae=float(np.mean(np.abs(error))),
            predictive_r2=1.0 - float(error @ error) / sst if sst > 0 else float("nan"),
            signed_bias=float(error.mean()),
            optimism=float(-error.mean()),
            calibration_slope=slope,
            calibration_intercept=float(observed.mean() - slope * predicted.mean()),
            selected_row_id=ordered.iloc[0].row_id,
            selected_policy_id=ordered.iloc[0].policy_id,
            selected_observed=float(observed[0]),
            selected_predicted=float(predicted[0]),
            selected_optimism=float(-error[0]),
            frontier_n=frontier_n,
            frontier_rmse=float(np.sqrt(np.mean(error[:frontier_n] ** 2))),
            frontier_optimism=float(-error[:frontier_n].mean()),
            menu_fingerprint=_menu_fingerprint(menu),
        )
        for k in (1, 3, 5):
            row[f"regret_at{k}"] = float(observed[:k].min() - observed.min())
        rows.append(row)
    columns = [*SCORE_COLUMNS, *METRICS, "selected_row_id", "selected_policy_id", "menu_fingerprint"]
    return pd.DataFrame(rows, columns=columns)


def summarize_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Pool fit diagnostics across folds while averaging fold-menu selection metrics.

    Each repeat remains separate, and outer_fold is the literal 'macro'. Regret
    and selected optimism weight fold menus equally. Frontier metrics pool only
    rows selected by the lowest predicted 20% rule within their original folds.
    Such summaries describe OOF endpoints, not independent training replicates.
    """
    folds = score_predictions(frame)
    rows = []
    group_columns = ["model", "panel", "target", "repeat", "role"]
    for key, menu in _joint_menus(frame).groupby(group_columns, sort=True, dropna=False):
        matching = folds
        for column, value in zip(group_columns, key, strict=True):
            matching = matching[matching[column] == value]
        observed = menu.observed.to_numpy(dtype=float)
        predicted = menu.predicted.to_numpy(dtype=float)
        error = predicted - observed
        observed_centered = observed - observed.mean()
        predicted_centered = predicted - predicted.mean()
        sst = float(observed_centered @ observed_centered)
        variation = float(predicted_centered @ predicted_centered)
        slope = float(predicted_centered @ observed_centered / variation) if variation > 0 else float("nan")
        frontier_n = int(matching.frontier_n.sum())
        row = dict(zip(group_columns, key, strict=True))
        row.update(
            outer_fold="macro",
            n=len(menu),
            rmse=float(np.sqrt(np.mean(error**2))),
            mae=float(np.mean(np.abs(error))),
            predictive_r2=1.0 - float(error @ error) / sst if sst > 0 else float("nan"),
            signed_bias=float(error.mean()),
            optimism=float(-error.mean()),
            calibration_slope=slope,
            calibration_intercept=float(observed.mean() - slope * predicted.mean()),
            frontier_n=frontier_n,
            frontier_rmse=float(np.sqrt((matching.frontier_rmse**2 * matching.frontier_n).sum() / frontier_n)),
            frontier_optimism=float((matching.frontier_optimism * matching.frontier_n).sum() / frontier_n),
            selected_row_id=None,
            selected_policy_id=None,
            menu_fingerprint=_menu_fingerprint(menu),
        )
        for metric in (
            "regret_at1",
            "regret_at3",
            "regret_at5",
            "selected_observed",
            "selected_predicted",
            "selected_optimism",
        ):
            row[metric] = float(matching[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows, columns=folds.columns)


def _reason(code: str, **context: Any) -> dict[str, Any]:
    return {"code": code, **context}


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool) and math.isfinite(value)
    )


def _protocol_reasons(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    reasons = []
    provenance = protocol.get("tolerance_provenance", {})
    if not protocol.get("protocol_id"):
        reasons.append(_reason("missing_protocol_id"))
    if not protocol.get("required_evidence"):
        reasons.append(_reason("missing_required_evidence_declaration"))
    if (
        protocol.get("tolerances_frozen") is not True
        or provenance.get("kind") not in {"external", "reference_calibration"}
        or not provenance.get("source")
        or provenance.get("frozen_before_candidate_evaluation") is not True
    ):
        reasons.append(_reason("unfrozen_or_unproven_tolerances"))
    required = protocol.get("required_cells", [])
    if not required or any(set(cell) != set(CELL_COLUMNS) for cell in required):
        reasons.append(_reason("missing_or_invalid_required_cells", columns=CELL_COLUMNS))
    elif len({tuple(cell[column] for column in CELL_COLUMNS) for cell in required}) != len(required):
        reasons.append(_reason("duplicate_required_cells"))
    gates = protocol.get("gates", [])
    if not gates:
        reasons.append(_reason("missing_gates"))
    for index, gate in enumerate(gates):
        if not all(gate.get(column) is not None for column in ("panel", "target", "role")) or not gate.get("metrics"):
            reasons.append(_reason("invalid_gate", gate=index))
        for rule in gate.get("metrics", []):
            bounds = [name for name in BOUND_NAMES if name in rule]
            if rule.get("metric") not in METRICS or len(bounds) != 1 or not _finite_number(rule[bounds[0]]):
                reasons.append(_reason("invalid_metric_bound", gate=index, metric=rule.get("metric")))
    return reasons


def _cell_key(cell: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(cell[column] for column in CELL_COLUMNS)


def _same_menu(candidate: Any, reference: Any) -> bool:
    return (
        isinstance(candidate, str)
        and len(candidate) == 64
        and all(character in "0123456789abcdef" for character in candidate)
        and candidate == reference
    )


def _candidate_gates(
    metrics: pd.DataFrame,
    candidate: str,
    reference: str,
    protocol: dict[str, Any],
    reasons: list[dict[str, Any]],
) -> dict[str, Any]:
    reasons = list(reasons)
    checks = []
    if reasons:
        return {"model": candidate, "status": "incomplete", "reasons": reasons, "checks": checks}
    candidate_rows = metrics[metrics.model == candidate].set_index(CELL_COLUMNS)
    reference_rows = metrics[metrics.model == reference].set_index(CELL_COLUMNS)
    required_cells = protocol.get("required_cells", [])
    candidate_gap = False
    benchmark_gap = False
    if candidate_rows.empty:
        reasons.append(_reason("missing_candidate"))
    if reference_rows.empty:
        reasons.append(_reason("missing_reference", reference=reference))
    for cell in required_cells:
        if set(cell) != set(CELL_COLUMNS):
            continue
        key = _cell_key(cell)
        if key not in reference_rows.index:
            benchmark_gap = True
            reasons.append(_reason("missing_required_cell", cell=cell, side="reference"))
            continue
        if key not in candidate_rows.index:
            candidate_gap = True
            reasons.append(_reason("missing_candidate_required_cell", cell=cell))
            continue
        chosen = candidate_rows.loc[key]
        baseline = reference_rows.loc[key]
        if "menu_fingerprint" not in metrics or not _same_menu(chosen.menu_fingerprint, baseline.menu_fingerprint):
            reasons.append(_reason("unpaired_policy_menu", cell=cell))
            continue
        gates = [
            gate
            for gate in protocol.get("gates", [])
            if all(gate.get(column) == cell[column] for column in ("panel", "target", "role"))
        ]
        if not gates:
            reasons.append(_reason("ungated_required_cell", cell=cell))
        for gate in gates:
            if gate.get("reference", reference) != reference:
                reasons.append(_reason("gate_reference_mismatch", cell=cell))
                continue
            for rule in gate.get("metrics", []):
                metric = rule.get("metric")
                bounds = [name for name in BOUND_NAMES if name in rule]
                if metric not in METRICS or len(bounds) != 1 or not _finite_number(rule[bounds[0]]):
                    continue
                candidate_value = chosen.get(metric)
                reference_value = baseline.get(metric)
                if not _finite_number(candidate_value) or not _finite_number(reference_value):
                    reasons.append(_reason("undefined_metric", cell=cell, metric=metric))
                    continue
                bound = bounds[0]
                limit = float(rule[bound])
                delta = float(candidate_value - reference_value)
                value = delta if bound == "allowed_increase" else float(candidate_value)
                passed = value >= limit if bound == "minimum" else value <= limit
                checks.append(
                    {
                        "cell": cell,
                        "metric": metric,
                        "bound": bound,
                        "limit": limit,
                        "candidate": float(candidate_value),
                        "reference": float(reference_value),
                        "delta": delta,
                        "passed": bool(passed),
                    }
                )
    # Protocol/reference gaps do not establish a scientific rejection. Candidate
    # failure to produce an available required cell is an estimation failure.
    failures = [check for check in checks if not check["passed"]]
    rejection_class = None
    if benchmark_gap:
        status = "incomplete"
    elif candidate_gap:
        status = "reject"
        rejection_class = "estimation"
    elif failures:
        status = "reject"
        rejection_class = "prediction"
        reasons.append(_reason("metric_gate_failed", count=len(failures)))
    elif reasons or not checks:
        status = "incomplete"
    elif protocol.get("require_any_improvement", False) and not any(
        check["bound"] == "allowed_increase" and check["delta"] < 0 for check in checks
    ):
        status = "inconclusive"
        reasons.append(_reason("no_paired_improvement"))
    else:
        status = "advance"
    return {
        "model": candidate,
        "status": status,
        "rejection_class": rejection_class,
        "reasons": reasons,
        "checks": checks,
    }


def evaluate_gates(metrics: pd.DataFrame, evidence: dict[str, bool], protocol: dict[str, Any]) -> dict[str, Any]:
    """Apply a frozen protocol to scored candidate/reference menus.

    Required cells contain every CELL_COLUMNS field, including repeat and fold.
    Required evidence must be explicitly true. The reference is protocol['reference'];
    optional protocol['candidates'] also detects candidates with no prediction rows.
    Preserve score_predictions' menu fingerprints to certify identical row IDs,
    policy metadata and observed endpoints within each required cell. Additional
    diagnostic cells do not affect pairing or the verdict.
    Inconclusive means a complete point-estimate screen did not demonstrate any
    paired improvement when require_any_improvement is true; it is not a test of
    confidence-interval width or training-seed uncertainty.
    """
    missing = sorted(set(SCORE_COLUMNS) - set(metrics.columns))
    if missing:
        raise ValueError(f"Metric frame is missing columns: {missing}")
    if metrics.duplicated(SCORE_COLUMNS).any():
        raise ValueError("Duplicate metric cell")
    reasons = _protocol_reasons(protocol)
    reference = protocol.get("reference", "")
    if not reference:
        reasons.append(_reason("missing_reference_name"))
    for name in protocol.get("required_evidence", []):
        if evidence.get(name) is not True:
            reasons.append(_reason("missing_required_evidence", evidence=name))
    candidates = protocol.get("candidates", sorted(set(metrics.model) - {reference}))
    if reference in candidates or len(set(candidates)) != len(candidates):
        reasons.append(_reason("invalid_candidate_list"))
    results = [_candidate_gates(metrics, candidate, reference, protocol, reasons) for candidate in candidates]
    statuses = {result["status"] for result in results}
    status = next(
        (value for value in ("reject", "incomplete", "inconclusive", "advance") if value in statuses), "incomplete"
    )
    return {
        "status": status,
        "claim": CLAIM,
        "protocol_id": protocol.get("protocol_id"),
        "reference": reference,
        "tolerance_provenance": protocol.get("tolerance_provenance"),
        "uncertainty": "point_estimates_only; no independent training-seed noise estimate",
        "required_evidence": protocol.get("required_evidence", []),
        "evidence": evidence,
        "reasons": reasons + ([_reason("no_candidates")] if not results else []),
        "candidates": results,
    }


def compare_models(frame: pd.DataFrame, reference: str, policy: dict[str, Any]) -> dict[str, Any]:
    """Compare paired predictions; aggregation='fold_macro' gates fold summaries.

    Unlike the strict score API, promotion records nonfinite predictions as a
    missing model cell. It discards the entire affected role and joint menu,
    never just the failed rows, preserving the required-menu coverage check.
    """
    if "reference" in policy and policy["reference"] != reference:
        raise ValueError("Policy reference does not match the requested reference")
    protocol = {**policy, "reference": reference}
    aggregation = policy.get("aggregation", "per_fold")
    if aggregation not in {"per_fold", "fold_macro"}:
        raise ValueError(f"Unknown gate aggregation: {aggregation}")
    # Validate identity and endpoint evidence independently of prediction success.
    _validate_predictions(frame.assign(predicted=0.0))
    if not frame.empty and not pd.api.types.is_numeric_dtype(frame.predicted):
        raise ValueError("Prediction columns must be numeric")
    finite = np.isfinite(frame.predicted.to_numpy(dtype=float))
    invalid = frame.loc[~finite, SCORE_COLUMNS].copy()
    if aggregation == "fold_macro":
        invalid["outer_fold"] = "macro"
    invalid = pd.concat([invalid, invalid.assign(role="joint_test")]).drop_duplicates()
    finite_frame = frame.loc[finite]
    metrics = summarize_predictions(finite_frame) if aggregation == "fold_macro" else score_predictions(finite_frame)
    invalid_keys = pd.MultiIndex.from_frame(invalid)
    metrics = metrics.loc[~pd.MultiIndex.from_frame(metrics[SCORE_COLUMNS]).isin(invalid_keys)]
    protocol.setdefault("candidates", sorted(set(frame.model) - {reference}))
    result = evaluate_gates(metrics, policy.get("evidence", {}), protocol)
    result["nonfinite_prediction_cells"] = invalid.to_dict(orient="records")
    return result
