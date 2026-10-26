# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Diagnose a cached choice between two completed, sealed search procedures.

All fold/task choices are committed from final inner-CV scores before prediction
tables are parsed. No fitter, optimizer, model prediction or family search is
called. This retrospective development diagnostic is not a combined-J2 estimator
or promotion evidence; its two constituent empirical failures remain unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import benchmark_two_phase_20260921 as benchmark

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reference_outputs/tpp40_cached_harm_selector_20261015"
FROZEN = ROOT / "reference_outputs/two_phase_benchmark_20260921/v3"
PREPARED = FROZEN / "prepared"
PANEL = "starcoder_wsd80_dense"
MODEL = "cached_harm_selector"
PHYSICAL, EFFECTIVE = "late_refresh_hazard", "late_refresh_effective_harm"
SOURCE_ROOTS = {
    PHYSICAL: ROOT / "reference_outputs/tpp40_late_refresh_starcoder_20261015",
    EFFECTIVE: ROOT / "reference_outputs/tpp40_effective_harm_starcoder_20261015",
}
SOURCE_SEALS = {
    PHYSICAL: "01b443d8cbeb78497ca10bda8432804e37e907637205eb9784c6b5ec0c6c3522",
    EFFECTIVE: "646bc77c80084526ed133db9004713ee08b751f1a35c837efeeb5e3620d13f0a",
}
TARGETS = (
    "eval/paloma/c4_en-llama3/bpb",
    "eval/paloma/dolma_100_programing_languages-llama3/bpb",
    "eval/paloma/falcon-refinedweb-llama3/bpb",
    "eval/uncheatable_eval/github_cpp-llama3/bpb",
    "eval/uncheatable_eval/github_python-llama3/bpb",
)
REFERENCES = ("effective_exposure_dsp", "aggregate_mariner_train_median")
EVIDENCE = ("candidate_registration", "asymmetric_only_fitting", "nested_split_integrity", "tied_class_mapping")
APPROVAL_SCOPE = "cached_completed_harm_search_selector_diagnostic"
SCOPE = {
    "candidate_id": "TP40-035",
    "stage": "cached_selector_diagnostic",
    "model": MODEL,
    "source_families": [PHYSICAL, EFFECTIVE],
    "panel": PANEL,
    "targets": list(TARGETS),
    "outer_folds": list(range(5)),
    "repeat": 0,
    "expected_contexts": 25,
    "expected_prediction_rows": 1730,
    "new_fits": 0,
    "selection_field": "inner_rmse",
    "selection_rule": "physical if physical.inner_rmse <= effective.inner_rmse else effective",
    "tie_tolerance": 0,
    "references": list(REFERENCES),
    "claim": "retrospective_cached_procedure_union_diagnostic_only",
}
LIMITATION = (
    "Union of two completed search histories, not a joint family/gamma search. "
    "Families were nominated after their held-out outcomes were known. Even a numeric "
    "policy pass cannot promote this selector or establish a mechanism or reliable optimum. "
    "Whole-procedure synthetic qualification and any subsequent evaluation need separate registration."
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_once(path: Path, value: Any) -> None:
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f"Preserve the existing artifact and use a separate revision: {path}")
    benchmark.write_json(path, value)


def source_paths() -> set[Path]:
    paths = {
        Path(__file__),
        ROOT / "test_tpp40_cached_harm_selector_20261015.py",
        *map(Path, benchmark.code_identity()),
        benchmark.PROTOCOL_DIR / "protocol.json",
        benchmark.PROTOCOL_DIR / "data_contract.json",
        PREPARED / "manifest.json",
        *(PREPARED / PANEL / name for name in ("rows.csv", "folds.csv", "metadata.json", "split_audit.json")),
        *(FROZEN / "screen_policies" / f"{name}.json" for name in REFERENCES),
        FROZEN / "screen_calibration/predictions.csv",
    }
    for root in SOURCE_ROOTS.values():
        paths.update(Path(name) for name in read_json(root / "execution_identity.json")["sources"])
        paths.update(
            root / name for name in ("result_manifest.json", "execution_identity.json", "audit/replay_report_v1.json")
        )
    return paths


def verify_scope(registration: dict[str, Any]) -> None:
    for key, value in {**SCOPE, "status": "registered", "environment": benchmark.environment_identity()}.items():
        if registration.get(key) != value:
            raise ValueError(f"Registered diagnostic scope differs: {key}")
    if registration.get("source_seals") != SOURCE_SEALS:
        raise ValueError("Both exact sealed source roots are required")
    expected = {str(path.resolve()) for path in source_paths()}
    if set(registration["frozen_reference_inputs"]) != expected:
        raise ValueError("Every current executable and transitive source/input requires a pin")
    benchmark.verify_hashes(registration["frozen_reference_inputs"])


def verify_sources() -> dict[str, Any]:
    adopted = {}
    for family, root in SOURCE_ROOTS.items():
        path = root / "result_manifest.json"
        if benchmark.sha256(path) != SOURCE_SEALS[family]:
            raise ValueError(f"Source seal changed: {family}")
        manifest = read_json(path)
        benchmark.verify_hashes({name: item["sha256"] for name, item in manifest["files"].items()}, root)
        identity = read_json(root / "execution_identity.json")
        benchmark.verify_hashes(identity["sources"])
        if identity["environment"] != benchmark.environment_identity():
            raise ValueError("Both completed procedures require the identical numerical runtime")
        audit = read_json(root / "audit/replay_report_v1.json")
        if not audit["passed"]:
            raise ValueError("Source prediction replay must pass; source empirical gate failures are preserved")
        adopted[family] = {
            "root": str(root.resolve()),
            "manifest": benchmark.receipt(path),
            "identity": benchmark.receipt(root / "execution_identity.json"),
            "audit": benchmark.receipt(root / "audit/replay_report_v1.json"),
        }
    return adopted


def prepare() -> None:
    """Hash and adopt both sealed runs without parsing predictions or selecting a family."""
    registration = read_json(OUTPUT / "registration.json")
    verify_scope(registration)
    adoption = {
        "sources": verify_sources(),
        "environment": benchmark.environment_identity(),
        "scope": SCOPE,
        "held_prediction_tables_parsed": False,
        "decisions_computed": False,
    }
    write_once(OUTPUT / "adoption.json", adoption)
    write_once(
        OUTPUT / "execution_identity.json",
        {
            "sources": {
                **registration["frozen_reference_inputs"],
                str(OUTPUT / "registration.json"): benchmark.sha256(OUTPUT / "registration.json"),
                str(OUTPUT / "adoption.json"): benchmark.sha256(OUTPUT / "adoption.json"),
            },
            "environment": benchmark.environment_identity(),
        },
    )


def identity() -> dict[str, Any]:
    registration = read_json(OUTPUT / "registration.json")
    verify_scope(registration)
    current = read_json(OUTPUT / "execution_identity.json")
    expected = {
        "sources": {
            **registration["frozen_reference_inputs"],
            str(OUTPUT / "registration.json"): benchmark.sha256(OUTPUT / "registration.json"),
            str(OUTPUT / "adoption.json"): benchmark.sha256(OUTPUT / "adoption.json"),
        },
        "environment": benchmark.environment_identity(),
    }
    if current != expected or read_json(OUTPUT / "adoption.json")["sources"] != verify_sources():
        raise ValueError("Diagnostic identity or sealed adoption changed")
    approval = read_json(OUTPUT / "execution_approval.json")
    for key, value in {
        "scope": APPROVAL_SCOPE,
        "registration_sha256": benchmark.sha256(OUTPUT / "registration.json"),
        "execution_identity_sha256": benchmark.sha256(OUTPUT / "execution_identity.json"),
    }.items():
        if approval.get(key) != value:
            raise ValueError(f"Separate exact selector execution approval required: {key}")
    return current


def fit_status(fit: dict[str, Any]) -> dict[str, Any]:
    search = fit["search_space"]
    selected = fit["selection_profile"][search["selected_trial_index"]]
    score = fit["inner_rmse"]
    if not math.isfinite(score) or score != selected["inner_rmse"]:
        raise ValueError("Final score must be finite and bound to its selected profile entry")
    strength = fit["parameters"]["lambda"]
    return {
        "inner_rmse": score,
        "lambda": strength,
        "null": strength == 0,
        "physical_strength_boundary": strength == 1,
        "selected_from_failed_refinement": search["selected_from_failed_refinement"],
        "refinement_results": search["refinement_results"],
        "selected_trial_index": search["selected_trial_index"],
        "selected_trial": selected,
        "shape": fit["shape"],
        "ridge": fit["ridge"],
        "gamma": fit["kappa"],
        "head": fit["head"],
    }


def choose_context(physical: dict[str, Any], effective: dict[str, Any]) -> dict[str, Any]:
    """Choose solely from two completed final inner-CV scores, with exact ties."""
    for key in ("training_rows", "inner_folds", "alpha", "selection_contract", "calibration_contract"):
        if physical[key] != effective[key]:
            raise ValueError(f"Completed procedures are not comparable: {key}")
    for key in ("shapes", "ridges", "gammas", "parameters", "complete_aggregate_source", "test_grid_override"):
        if physical["search_space"][key] != effective["search_space"][key]:
            raise ValueError(f"Completed search inputs differ: {key}")
    statuses = {PHYSICAL: fit_status(physical), EFFECTIVE: fit_status(effective)}
    p, e = physical["inner_rmse"], effective["inner_rmse"]
    family = PHYSICAL if p <= e else EFFECTIVE
    return {
        "source_family": family,
        "inner_rmse": statuses[family]["inner_rmse"],
        "effective_minus_physical_inner_rmse": e - p,
        "absolute_score_margin": abs(e - p),
        "exact_score_tie": p == e,
        "selected_status": statuses[family],
        "candidate_statuses": statuses,
        "training_rows": physical["training_rows"],
        "inner_folds": physical["inner_folds"],
    }


def cached_complete(stage: str, current: dict[str, Any], output: Path = OUTPUT) -> bool:
    path = output / f"{stage}_complete.json"
    if not path.exists():
        return False
    record = read_json(path)
    if record["identity"] != current or record["stage"] != stage:
        raise ValueError("Completed stage belongs to another identity or scope")
    benchmark.verify_hashes(record["outputs"], output)
    return True


def finish(stage: str, current: dict[str, Any], names: list[str], output: Path = OUTPUT) -> None:
    write_once(
        output / f"{stage}_complete.json",
        {"stage": stage, "identity": current, "outputs": {name: benchmark.sha256(output / name) for name in names}},
    )


def select() -> None:
    """Commit all 25 choices using fit JSON and outcome-independent fold metadata."""
    current = identity()
    if cached_complete("selection", current):
        return
    folds = pd.read_csv(PREPARED / PANEL / "folds.csv", keep_default_na=False)
    rows = pd.read_csv(PREPARED / PANEL / "rows.csv", keep_default_na=False).set_index("row_id")
    decisions = []
    for fold in range(5):
        train = folds[folds.repeat.eq(0) & folds.outer_fold.eq(fold) & folds.role.eq("train")]
        if rows.loc[train.row_id, "tied"].any() or set(train.inner_fold) != {0, 1, 2}:
            raise ValueError("Training must be asymmetric-only with the three frozen inner labels")
        for task_index, target in enumerate(TARGETS):
            fits, receipts = {}, {}
            for family, root in SOURCE_ROOTS.items():
                directory = root / "fits" / f"fold{fold}_task{task_index}" / family
                path = directory / "fit.json"
                complete = read_json(directory / "complete.json")
                if complete["context"] != {"fold": fold, "model": family, "stage": "fit", "target": target}:
                    raise ValueError("Cached fit context differs")
                if complete["identity"] != read_json(root / "execution_identity.json"):
                    raise ValueError("Cached fit identity differs")
                benchmark.verify_hashes(complete["outputs"], directory)
                fit = read_json(path)
                if fit["model"] != family or fit["training_rows"] != train.row_id.tolist():
                    raise ValueError("Cached fit family or ordered training rows differ")
                if fit["inner_folds"] != train.inner_fold.tolist() or fit["search_space"]["test_grid_override"]:
                    raise ValueError("Cached fit inner partitions or complete search differ")
                fits[family], receipts[family] = fit, benchmark.receipt(path)
            decisions.append(
                {
                    "panel": PANEL,
                    "target": target,
                    "task_index": task_index,
                    "repeat": 0,
                    "outer_fold": fold,
                    "source_fits": receipts,
                    **choose_context(fits[PHYSICAL], fits[EFFECTIVE]),
                }
            )
    write_once(OUTPUT / "decisions.json", {"identity": current, "decisions": decisions, "held_tables_parsed": False})
    finish("selection", current, ["decisions.json"])


def copy_predictions(decisions: list[dict[str, Any]], tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Copy rows from a committed context choice without consulting their labels."""
    keys = ["panel", "target", "repeat", "outer_fold", "row_id", "role"]
    aligned = {}
    for family in (PHYSICAL, EFFECTIVE):
        table = tables[family]
        table = table[table.model.eq(family)].sort_values(keys).reset_index(drop=True)
        if table.duplicated(keys).any() or not set(table.role).issubset({"test", "tied_test"}):
            raise ValueError("Source prediction menu contains duplicate or non-held rows")
        aligned[family] = table
    left, right = aligned[PHYSICAL], aligned[EFFECTIVE]
    common = [name for name in left.columns if name not in ("model", "predicted")]
    if set(left.columns) != set(right.columns) or not left[common].equals(right[common]):
        raise ValueError("Source prediction row identities, observed labels or metadata differ")
    pieces = []
    for decision in decisions:
        family = decision["source_family"]
        table = aligned[family]
        chosen = table[
            table.panel.eq(decision["panel"])
            & table.target.eq(decision["target"])
            & table.repeat.eq(decision["repeat"])
            & table.outer_fold.eq(decision["outer_fold"])
        ].copy()
        if chosen.empty:
            raise ValueError("Missing selected context")
        chosen["source_family"] = family
        chosen["model"] = MODEL
        pieces.append(chosen)
    result = pd.concat(pieces, ignore_index=True).sort_values(keys).reset_index(drop=True)
    if len(result) != len(left) or result.duplicated(keys).any():
        raise ValueError("Selections must cover each held context exactly once")
    return result


def materialize(current: dict[str, Any], output: Path, prediction_paths: dict[str, Path]) -> pd.DataFrame:
    """Require durable complete decisions before the first prediction CSV read."""
    if not cached_complete("selection", current, output):
        raise ValueError("All decisions must be committed before loading held prediction tables")
    record = read_json(output / "decisions.json")
    if record["identity"] != current or record["held_tables_parsed"]:
        raise ValueError("Decisions must originate from the fit-only stage")
    tables = {
        family: pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
        for family, path in prediction_paths.items()
    }
    return copy_predictions(record["decisions"], tables)


def score() -> None:
    current = identity()
    if cached_complete("scoring", current):
        return
    predictions = materialize(
        current, OUTPUT, {family: root / "predictions.csv" for family, root in SOURCE_ROOTS.items()}
    )
    if len(predictions) != SCOPE["expected_prediction_rows"]:
        raise ValueError("The full five-target frozen held menu is required")
    decisions = read_json(OUTPUT / "decisions.json")["decisions"]
    predictions.to_csv(OUTPUT / "predictions.csv", index=False)
    benchmark.scoring.score_predictions(predictions).to_csv(OUTPUT / "metrics.csv", index=False)
    benchmark.scoring.summarize_predictions(predictions).to_csv(OUTPUT / "summary.csv", index=False)
    pd.json_normalize(decisions).drop(columns=["training_rows", "inner_folds"]).to_csv(
        OUTPUT / "selected_parameters.csv", index=False
    )
    policy = read_json(FROZEN / "screen_policies" / f"{REFERENCES[0]}.json")
    evidence = {}
    for name in EVIDENCE:
        path = OUTPUT / "evidence" / f"{name}.json"
        write_once(
            path,
            {
                "evidence": name,
                "protocol_id": policy["benchmark_protocol_id"],
                "candidate_model": MODEL,
                "data_contract_sha256": policy["data_contract_sha256"],
                "status": "pass",
                "registration": benchmark.receipt(OUTPUT / "registration.json"),
                "adoption": benchmark.receipt(OUTPUT / "adoption.json"),
                "selection": benchmark.receipt(OUTPUT / "selection_complete.json"),
                "scope": "Structural provenance only; no whole-selector synthetic qualification or promotion.",
                "tied_mapping": "Choice is fixed per fold/task; both source heads have the exact native tied class.",
            },
        )
        evidence[name] = [benchmark.receipt(path)]
    write_once(OUTPUT / "evidence/evidence.json", evidence)
    finish(
        "scoring",
        current,
        ["predictions.csv", "metrics.csv", "summary.csv", "selected_parameters.csv", "evidence/evidence.json"]
        + [f"evidence/{name}.json" for name in EVIDENCE],
    )


def gate() -> None:
    current = identity()
    if cached_complete("gates", current):
        return
    if not cached_complete("scoring", current) or not cached_complete("selection", current):
        raise ValueError("Complete committed selection and unchanged scoring are required")
    verdicts = {}
    for reference in REFERENCES:
        policy_path = FROZEN / "screen_policies" / f"{reference}.json"
        evidence_path = OUTPUT / "evidence/evidence.json"
        policy = benchmark.verified_policy(policy_path, evidence_path, MODEL)
        paired = benchmark.promotion_inputs(
            OUTPUT / "predictions.csv", FROZEN / "screen_calibration/predictions.csv", policy, reference
        )
        policy["candidates"] = [MODEL]
        verdict = benchmark.scoring.compare_models(paired, reference, policy)
        verdict["receipts"] = {
            "policy": benchmark.receipt(policy_path),
            "predictions": benchmark.receipt(OUTPUT / "predictions.csv"),
            "reference_predictions": benchmark.receipt(FROZEN / "screen_calibration/predictions.csv"),
            "evidence": benchmark.receipt(evidence_path),
        }
        verdict["scope"] = LIMITATION
        write_once(OUTPUT / f"gate_{reference}.json", verdict)
        verdicts[reference] = benchmark.receipt(OUTPUT / f"gate_{reference}.json")
    write_once(
        OUTPUT / "diagnostic_report.json",
        {"identity": current, "scope": LIMITATION, "promotion_eligible": False, "raw_policy_verdicts": verdicts},
    )
    finish("gates", current, [f"gate_{reference}.json" for reference in REFERENCES] + ["diagnostic_report.json"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "select", "score", "gate"))
    args = parser.parse_args()
    {"prepare": prepare, "select": select, "score": score, "gate": gate}[args.stage]()


if __name__ == "__main__":
    main()
