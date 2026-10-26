# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""TP047: all selected TP046 fixed pairs, including failed query evaluations.

Only saved heads and immutable cached features; no fitting, integration, policy
search, empirical labels, candidate selection or replacement of prior gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    diagnose_tpp40_finite_menu_decisions_20261015 as menus,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import diagnose_tpp40_gaussian_pairs_20261015 as pairs

BASE, PARENT = pairs.BASE, pairs.PARENT
INPUT = PARENT / "whole_response_fold3_noise"
OUTPUT = PARENT / "whole_response_noisy_pairs"
NOISES = ("20261015", "20261016", "20261017")
DIRECT_PINS = {
    Path(pairs.__file__): "f4a56fae7501848bf143b0e0fe87928d179be61b5e6d3050c1762c56b50beaa8",
    Path(menus.__file__): "025b88dea356fcae1c3915e643581452d1cae2ec530d57029f4698ea035ab5de",
    Path(menus.risk.__file__): "4093857e51f3d1498f03f8989d6c73c5e99b04d087aaa2e7080f5e2d5afff4ac",
}
SEAL = INPUT / "result_manifest_v1.json"
AUDIT = INPUT / "audit/author_replay_report_v1.json"
CONTRACT = {
    "candidate_id": "TP40-047",
    "scope": "all_30_TP046_selected_heads_frozen_Gaussian_pairs",
    "outer_fold": 3,
    "noise_seeds": list(NOISES),
    "worlds": list(pairs.WORLDS),
    "heads": 30,
    "menus": {k: list(v) for k, v in menus.MENUS.items()},
    "menu_records": 90,
    "sigma": pairs.SIGMA,
    "class": "nonnegative absolute floor and scale; one shared shape per function, allowed to differ across hypotheses",
    "M": "heterogeneous teacher descriptive only; no within-class lower bound",
    "rows": "every original training and query row; same-aggregate gains paired by row identity",
    "nonfinite": "preserve row status and log response; no finite risk for unavailable means or distance",
    "analytic_eligibility": "common family, finite training and all query means, no evaluated exponent underflow",
    "adaptivity": "fresh hypothetical Gaussian experiment for the frozen pair; no adaptive error guarantee",
    "realized_likelihood": "0.5*||u||^2-z.u on saved observations; adaptive descriptive identity, not a calibrated test",
    "prior_failures": "retain all query failures, solver flags, original metrics and historical checks",
    "new_fits": False,
    "new_integrations": False,
    "new_queries": False,
    "empirical_outcomes": False,
    "scientific_gate": None,
    "original_stage2_remains_blocked": True,
    "scientific_promotion": False,
}
read, digest, write = pairs.read, pairs.digest, pairs.write


def inventory() -> list[tuple[str, str, str]]:
    return [(n, w, c) for n in NOISES for w in pairs.WORLDS for c in pairs.candidate_names(w)]


def inherited_sources() -> dict[str, str]:
    result: dict[str, str] = {}
    for folder in (pairs.OUTPUT, menus.OUTPUT, INPUT):
        for path, expected in read(folder / "registration.json")["source_hashes"].items():
            if path in result and result[path] != expected:
                raise ValueError("Conflicting inherited source identities")
            if digest(Path(path)) != expected:
                raise ValueError(f"Inherited source changed: {path}")
            result[path] = expected
    for path, expected in DIRECT_PINS.items():
        if digest(path) != expected:
            raise ValueError(f"Direct arithmetic helper changed: {path}")
        result[str(path)] = expected
    return result


def input_paths() -> set[Path]:
    return {
        SEAL,
        AUDIT,
        INPUT / "registration.json",
        INPUT / "execution_identity.json",
        INPUT / "complete.json",
        pairs.OUTPUT / "registration.json",
        menus.OUTPUT / "registration.json",
        pairs.ALL_FOLDS / "features.sqlite",
        pairs.ALL_FOLDS / "registration.json",
        pairs.ALL_FOLDS / "result_manifest_v1.json",
        pairs.STAGE1 / "result_manifest.json",
        *[pairs.STAGE1 / "prepared" / f for f in ("clean.npz", "matched_tied.npz", "noise.npz")],
        *[pairs.helper.DATA / f for f in ("metadata.json", "policies.npz", "rows.csv", "folds.csv")],
    }


def registration_contents() -> dict[str, Any]:
    sources = inherited_sources()
    for path in (Path(__file__), BASE / "test_tpp40_whole_response_noisy_pairs_20261015.py", OUTPUT / "proposal_v1.md"):
        sources[str(path)] = digest(path)
    return {
        "contract": CONTRACT,
        "runtime": pairs.runtime(),
        "source_hashes": sources,
        "input_hashes": {str(p): digest(p) if p.exists() else None for p in sorted(input_paths())},
    }


def verify_seal(path: Path) -> None:
    for name, record in read(path)["files"].items():
        if digest(path.parent / name) != record["sha256"]:
            raise ValueError(f"Sealed prerequisite changed: {name}")


def identity() -> dict[str, Any]:
    registration = read(OUTPUT / "registration.json")
    expected = registration_contents()
    if registration.get("status") != "registered" or any(registration.get(k) != v for k, v in expected.items()):
        raise ValueError("Exact source, scope, runtime and input registration required")
    if any(v is None for v in expected["input_hashes"].values()):
        raise ValueError("TP046 audit and seal must exist before execution")
    for path in (SEAL, pairs.ALL_FOLDS / "result_manifest_v1.json", pairs.STAGE1 / "result_manifest.json"):
        verify_seal(path)
    if read(AUDIT)["status"] != "author_saved_output_replay_passed":
        raise ValueError("Require passed TP046 saved-output replay")
    if read(pairs.STAGE1 / "result_manifest.json")["learned_stage_ready"]:
        raise ValueError("Preserve the original failed learned-stage prerequisite")
    complete = read(INPUT / "complete.json")
    if complete["completed_heads"] != 30 or complete["identity_sha256"] != digest(INPUT / "execution_identity.json"):
        raise ValueError("All thirty TP046 heads must be complete")
    for name, expected_hash in complete["files"].items():
        if digest(INPUT / name) != expected_hash:
            raise ValueError("TP046 completion receipt changed")
    return {"registration_sha256": digest(OUTPUT / "registration.json"), **expected}


def execution_identity() -> dict[str, Any]:
    current = identity()
    if current != read(OUTPUT / "execution_identity.json"):
        raise ValueError("Frozen identity changed")
    approval = read(OUTPUT / "execution_approval.json")
    expected = {
        "scope": CONTRACT["scope"],
        "registration_sha256": current["registration_sha256"],
        "execution_identity_sha256": digest(OUTPUT / "execution_identity.json"),
    }
    if any(approval.get(k) != v for k, v in expected.items()):
        raise ValueError("Require exact no-refit diagnostic approval")
    return current


def response_rows(matrix: np.ndarray, head: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    """Evaluate each saved normalized response without clipping or dropping failures."""
    if head is None:
        return np.full(len(matrix), np.nan), np.full(len(matrix), np.nan), ["missing_head"] * len(matrix), 0
    amplitudes = np.asarray(head["amplitudes"], float)
    if not np.isfinite(amplitudes).all() or np.any(amplitudes < 0):
        raise ValueError("Nonnegative finite amplitudes required")
    if (
        not all(math.isfinite(head[k]) for k in ("floor", "scale", "normalization"))
        or min(head["floor"], head["scale"]) < 0
    ):
        raise ValueError("Saved head is outside the nonnegative response class")
    if head["scale"] == 0:
        return (
            np.full(len(matrix), head["floor"]),
            np.full(len(matrix), -np.inf),
            ["zero_scale_closure"] * len(matrix),
            0,
        )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        log_response = math.log(head["scale"]) + (matrix @ amplitudes - head["normalization"])
        deficit = np.exp(log_response)
        predicted = head["floor"] + deficit
    statuses = ["finite" if np.isfinite(v) else "nonfinite_response" for v in predicted]
    return predicted, log_response, statuses, int(np.sum(deficit == 0))


def pair_summary(candidate: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            result = pairs.gaussian_pair(candidate, truth, pairs.SIGMA)
    except ValueError as error:
        return {
            "pair_status": "unavailable",
            "pair_failure": str(error),
            "distance": None,
            "bayes_error": None,
            "log_bayes_error": None,
        }
    if not all(math.isfinite(v) for v in result.values()):
        return {
            "pair_status": "unavailable",
            "pair_failure": "Nonrepresentable Gaussian summary",
            **{k: v if math.isfinite(v) else None for k, v in result.items()},
        }
    return {"pair_status": "finite", "pair_failure": None, **result}


def realized_likelihood(candidate: np.ndarray, truth: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    """Report the saved-data likelihood contrast without post-selection guarantees."""
    if candidate.shape != truth.shape or observed.shape != truth.shape or truth.ndim != 1 or not len(truth):
        raise ValueError("Likelihood inputs require the same nonempty training rows")
    result = {
        "realized_likelihood_status": "unavailable_nonfinite",
        "realized_log_likelihood_ratio_truth_over_candidate": None,
        "realized_noise_alignment": None,
        "half_squared_distance": None,
    }
    if not all(np.isfinite(v).all() for v in (candidate, truth, observed)):
        return result
    with np.errstate(over="ignore", invalid="ignore"):
        u = (candidate.astype(np.longdouble) - truth.astype(np.longdouble)) / pairs.SIGMA
        z = (observed.astype(np.longdouble) - truth.astype(np.longdouble)) / pairs.SIGMA
        half_distance, alignment = (u @ u) / 2, z @ u
        ratio = half_distance - alignment
    values = (ratio, alignment, half_distance)
    if not all(np.isfinite(v) and abs(v) <= np.finfo(float).max for v in values):
        result["realized_likelihood_status"] = "unavailable_unrepresentable"
        return result
    return {
        "realized_likelihood_status": "finite_adaptive_descriptive",
        "realized_log_likelihood_ratio_truth_over_candidate": float(ratio),
        "realized_noise_alignment": float(alignment),
        "half_squared_distance": float(half_distance),
    }


def scalar_bound(delta: float, pair: dict[str, Any], eligible: bool) -> dict[str, Any]:
    result = {
        "signed_separation": delta if math.isfinite(delta) else None,
        "absolute_separation": abs(delta) if math.isfinite(delta) else None,
        "pair_mse_lower_bound": None,
        "pair_mse_log_lower_bound": None,
        "bound_status": "unavailable",
    }
    if not math.isfinite(delta):
        result["bound_status"] = "nonfinite_functional"
    elif pair["pair_status"] != "finite":
        result["bound_status"] = "unavailable_training_pair"
    elif not eligible:
        result["bound_status"] = "not_within_class_eligible"
    elif delta == 0:
        result.update(pair_mse_lower_bound=0.0, bound_status="exact_zero_separation")
    else:
        log_bound = 2 * math.log(abs(delta)) - math.log(2) + pair["log_bayes_error"]
        value = math.exp(log_bound) if log_bound <= math.log(np.finfo(float).max) else None
        result.update(
            pair_mse_lower_bound=value,
            pair_mse_log_lower_bound=log_bound,
            bound_status=(
                "finite" if value is not None and value > 0 else "underflow" if value == 0 else "overflow_log_retained"
            ),
        )
    return result


def functional_tables(frame: pd.DataFrame, pair: dict[str, Any], eligible: bool) -> tuple[list[dict], list[dict]]:
    endpoints = []
    for row in frame.to_dict("records"):
        with np.errstate(over="ignore", invalid="ignore"):
            delta = float(row["predicted"] - row["truth"])
        endpoints.append(row | scalar_bound(delta, pair, eligible))
    first = frame.loc[frame.role == "asymmetric"].reset_index(drop=True)
    tied = frame.loc[frame.role == "matched_tied"].reset_index(drop=True)
    if first[["source_index", "row_id"]].to_dict("list") != tied[["source_index", "row_id"]].to_dict("list"):
        raise ValueError("Gain counterpart identities differ")
    gains = []
    for a, t in zip(first.to_dict("records"), tied.to_dict("records"), strict=True):
        with np.errstate(over="ignore", invalid="ignore"):
            candidate_gain = float(t["predicted"] - a["predicted"])
            true_gain = float(t["truth"] - a["truth"])
        gains.append(
            {
                "source_index": a["source_index"],
                "row_id": a["row_id"],
                "candidate_gain": candidate_gain,
                "true_gain": true_gain,
                **scalar_bound(candidate_gain - true_gain, pair, eligible),
            }
        )
    return endpoints, gains


def menu_diagnostic(head: dict, frame: pd.DataFrame, name: str) -> tuple[dict, list[dict]]:
    lineage = {
        "prerequisite_heads_path": str(OUTPUT / "heads.csv"),
        "source_fit_path": head["source_fit_path"],
        "source_fit_sha256": head["source_fit_sha256"],
    }
    subset = frame.loc[frame.role.isin(menus.MENUS[name])]
    if head["pair_status"] != "finite" or not np.isfinite(subset[["truth", "predicted"]].to_numpy()).all():
        return {
            "pair_id": head["pair_id"],
            "menu": name,
            "status": "unavailable_nonfinite",
            "within_class_menu_regret_bound": None,
            **lineage,
        }, []
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            record, intervals = menus.menu_rows(head, frame, name)
        return record | lineage, intervals
    except (ArithmeticError, ValueError) as error:
        return {
            "pair_id": head["pair_id"],
            "menu": name,
            "status": "numerical_failure",
            "message": str(error),
            **lineage,
        }, []


def json_finite(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: json_finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_finite(v) for v in value]
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return None
    return value


def diagnose_head(
    folder: Path,
    context: dict,
    geometry: dict,
    rows: pd.DataFrame,
    cache: Any,
    clean: np.ndarray,
    tied: np.ndarray,
    noise: np.ndarray,
) -> dict:
    fit = read(folder / "fit.json")
    spec, selected = fit["selected_candidate"], fit["selected"]
    if (spec["world"], spec["noise"], spec["outer_fold"]) != (context["world"], context["noise"], 3):
        raise ValueError("Selected model context mismatch")
    train, test, tied_test = (geometry["split"][k] for k in ("train", "test", "tied_test"))
    if fit["train_indices"] != train.tolist():
        raise ValueError("Saved full-training membership differs")
    if spec["shape"] not in pairs.helper.base.SHAPES or spec["mu"] not in pairs.helper.base.STRENGTHS:
        raise ValueError("Saved shape or strength differs from the frozen menu")
    descriptor = {
        "shape": spec["shape"],
        "alpha": geometry["alpha"],
        "parameters": {"mu": spec["mu"]},
        "feature_config": {"clock": spec["clock"]},
    }
    training_matrix = cache.matrix(geometry["first"][train], geometry["last"][train], descriptor)
    query_matrix = cache.matrix(geometry["query0"], geometry["query1"], descriptor)
    training, training_log, training_status, train_underflow = response_rows(training_matrix, selected)
    predicted, query_log, query_status, query_underflow = response_rows(query_matrix, selected)
    pair = pair_summary(training, clean[train])
    likelihood = realized_likelihood(training, clean[train], clean[train] + noise[train])
    if (folder / "predictions.csv").exists():
        saved = pd.read_csv(folder / "predictions.csv", float_precision="round_trip")
        if saved[["role", "row_id", "source_index"]].to_dict("list") != geometry["query"].to_dict("list"):
            raise ValueError("Saved query row order differs")
        np.testing.assert_allclose(predicted, saved.predicted, rtol=3e-13, atol=2e-12)
    query_truth = np.r_[clean[test], tied[test], clean[tied_test]]
    if (folder / "predictions.csv").exists():
        np.testing.assert_array_equal(saved.truth.to_numpy(), query_truth)
    original = read(folder / "scores.json") if (folder / "scores.json").exists() else read(folder / "query_failure.json")
    finite = np.isfinite(training).all() and np.isfinite(predicted).all()
    common = context["world"] != "heterogeneous_null"
    eligible = bool(common and finite and train_underflow + query_underflow == 0 and selected is not None)
    head = (
        context
        | pair
        | likelihood
        | {
            "source_fit_path": str(folder / "fit.json"),
            "source_fit_sha256": digest(folder / "fit.json"),
            "pair_id": "__".join(str(context[k]) for k in ("source", "outer_fold", "world", "noise", "candidate")),
            "analytic_pair_eligible": eligible,
            "common_family_member": common,
            "within_class_exclusion": (
                "heterogeneous teacher"
                if not common
                else "nonfinite or underflowed evaluation"
                if not eligible
                else None
            ),
            "selected_fit_unresolved": bool(
                fit["full_unresolved"] or any(r["unresolved"] for r in fit["selected_inner_records"])
            ),
            "original_numerical_status": json.dumps(original, sort_keys=True),
            "selected_candidate": json.dumps(spec, sort_keys=True),
            "inner_mse": fit["inner_mse"],
            "native_full_fit_reused": fit["native_full_fit_reused"],
            "train_underflow_count": train_underflow,
            "query_underflow_count": query_underflow,
            "nonfinite_query_count": int(np.sum(~np.isfinite(predicted))),
            "original_query_failure": (folder / "query_failure.json").exists(),
            "common_floor_domain": "nonnegative",
            "common_scale_domain": "nonnegative",
        }
    )
    frame = geometry["query"].assign(
        truth=query_truth, predicted=predicted, log_response=query_log, response_status=query_status
    )
    endpoints, gains = functional_tables(frame, pair, eligible)
    menu_records, intervals = [], []
    for name in menus.MENUS:
        record, cells = menu_diagnostic(head, frame, name)
        menu_records.append(record)
        intervals.extend(cells)
    training_frame = pd.DataFrame(
        {
            "source_index": train,
            "row_id": rows.iloc[train].row_id.to_numpy(),
            "truth": clean[train],
            "observed": clean[train] + noise[train],
            "predicted": training,
            "log_response": training_log,
            "response_status": training_status,
        }
    )
    return json_finite(
        {
            "head": head,
            "training": training_frame.to_dict("records"),
            "endpoints": endpoints,
            "gains": gains,
            "menus": menu_records,
            "intervals": intervals,
        }
    )


def cached_payload(path: Path, frozen: dict, context: dict) -> dict:
    saved = read(path)
    if saved["identity"] != frozen or saved["payload_sha256"] != canonical(saved["payload"]):
        raise ValueError("Existing diagnostic cell differs")
    payload = saved["payload"]
    if any(payload["head"].get(k) != v for k, v in context.items()):
        raise ValueError("A valid receipt belongs to another context")
    return payload


def run() -> None:
    frozen = execution_identity()
    complete = OUTPUT / "complete.json"
    if complete.exists():
        old = read(complete)
        if old["identity"] != frozen or any(digest(OUTPUT / p) != sha for p, sha in old["files"].items()):
            raise ValueError("Preserve prior completed diagnostic")
        return
    geometries, rows = pairs.geometry()
    geometry = geometries[3]
    cache_path = pairs.ALL_FOLDS / "features.sqlite"
    if Path(str(cache_path) + "-wal").exists() and Path(str(cache_path) + "-wal").stat().st_size:
        raise ValueError("Immutable feature cache requires an empty WAL")
    outputs: dict[str, str] = {}
    tables: dict[str, list] = {k: [] for k in ("heads", "training", "endpoints", "gains", "menus", "intervals")}
    with (
        closing(sqlite3.connect(f"file:{cache_path}?mode=ro&immutable=1", uri=True)) as connection,
        np.load(pairs.STAGE1 / "prepared/clean.npz") as clean,
        np.load(pairs.STAGE1 / "prepared/matched_tied.npz") as tied,
        np.load(pairs.STAGE1 / "prepared/noise.npz") as noise,
    ):
        cache = pairs.helper.base.Cache(
            connection, pairs.helper.base.Checks(), read(pairs.ALL_FOLDS / "registration.json"), geometry
        )
        for seed, world, candidate in inventory():
            context = {
                "source": "whole_response_fold3_noise",
                "outer_fold": 3,
                "world": world,
                "noise": seed,
                "candidate": candidate,
            }
            path = OUTPUT / "cells" / seed / world / f"{candidate}.json"
            if path.exists():
                payload = cached_payload(path, frozen, context)
            else:
                folder = INPUT / "cells/outer3" / seed / world / candidate
                payload = diagnose_head(folder, context, geometry, rows, cache, clean[world], tied[world], noise[seed])
                write(path, {"identity": frozen, "payload_sha256": canonical(payload), "payload": payload})
            outputs[str(path.relative_to(OUTPUT))] = digest(path)
            head = payload["head"]
            tables["heads"].append(head)
            for key in ("training", "endpoints", "gains", "menus", "intervals"):
                tables[key].extend({"pair_id": head["pair_id"], **row} for row in payload[key])
    if len(tables["heads"]) != 30 or len(tables["menus"]) != 90:
        raise ValueError("All-head/all-menu population changed")
    for key, records in tables.items():
        path = OUTPUT / f"{key}.csv"
        content = pd.DataFrame(records).to_csv(index=False)
        if path.exists() and path.read_text() != content:
            raise ValueError("Preserve previous table")
        path.write_text(content)
        outputs[path.name] = digest(path)
    summary = {
        "heads": 30,
        "menus": 90,
        "original_query_failures": sum(h["original_query_failure"] for h in tables["heads"]),
        "analytic_pair_eligible": sum(h["analytic_pair_eligible"] for h in tables["heads"]),
        "finite_menus": sum(r["status"] == "finite" for r in tables["menus"]),
        "scientific_gate": None,
        "scientific_promotion": False,
        "original_stage2_remains_blocked": True,
    }
    write(OUTPUT / "summary.json", summary)
    outputs["summary.json"] = digest(OUTPUT / "summary.json")
    write(complete, {"identity": frozen, "files": outputs, **summary})


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("draft", "freeze", "run"))
    command = parser.parse_args().command
    if command == "draft":
        write(OUTPUT / "registration.draft.json", {"status": "draft_not_authorized", **registration_contents()})
    elif command == "freeze":
        write(OUTPUT / "execution_identity.json", identity())
    else:
        run()


if __name__ == "__main__":
    main()
