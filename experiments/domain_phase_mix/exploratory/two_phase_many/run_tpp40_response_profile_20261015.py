# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""TP40-036: three fixed-law estimators on eighty sealed synthetic contexts.

No empirical outcomes, inner-subset fits, model selection, quadrature, or raw
policy optimization. Frozen-cache reads are immutable; writes stay in this root.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sqlite3
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_response_profile_20261015 as model

BASE = Path(__file__).resolve().parent
PARENT = BASE / "reference_outputs/tpp40_acquisition_retention_20261015"
OUTPUT = PARENT / "response_profile"
STAGE1, EXTENSION = PARENT / "stage1", PARENT / "oracle_extension"
DATA = BASE / "reference_outputs/two_phase_benchmark_20260921/v3/prepared/starcoder_wsd80_dense"
EXTENSION_SEAL = EXTENSION / "result_manifest_v1.json"
STAGE1_SEAL = STAGE1 / "result_manifest.json"
CACHE_PATH = EXTENSION / "features.sqlite"
PURE_REVIEW = OUTPUT / "audit/independent_pure_review_v3.json"
SCOPE = "tp036_fixed_law_response_profile_80_contexts"
ARMS = ("known_floor_log_nnls", "known_floor_response", "learned_nonnegative_floor_response")
WORLDS = (
    {"name": "native_null", "kind": "native", "clock": None, "mu": 0.0},
    {"name": "heterogeneous_null", "kind": "heterogeneous", "clock": None, "mu": 0.0},
    {"name": "equal_positive", "kind": "temporal", "clock": "equal_group", "mu": 3.0},
    {"name": "pool_positive", "kind": "temporal", "clock": "global_pool", "mu": 3.0},
)
NOISES = ("clean", "20261015", "20261016", "20261017")
CONTRACT = {
    "scope": SCOPE,
    "worlds": list(WORLDS),
    "arms": list(ARMS),
    "outer_folds": list(range(5)),
    "repeat": 0,
    "noise_names": list(NOISES),
    "training_noise_sd": 0.003,
    "query_noise": False,
    "fixed_shape": {"rate": 0.25, "power": 0.7, "threshold": 3.0},
    "known_floor": 0.6,
    "gammas_for_initialization_only": list(model.GAMMAS),
    "ridge": 0.0,
    "objective": "mean_squared_bpb_divided_by_0.003_squared",
    "solver_options": model.SOLVER_OPTIONS,
    "contexts": 80,
    "heads": 240,
    "new_response_heads": 160,
    "native_initialization_fits": 400,
    "nonlinear_starts": 800,
    "inner_subset_fits": False,
    "shape_or_kinetic_selection": False,
    "heterogeneous_world_remains_shared_shape_misspecified": True,
    "true_floor_initialization": False,
    "floor_domain": "nonnegative, no added upper bound",
    "amplitude_domain": "nonnegative, no added upper bound",
    "prediction_cap": False,
    "raw_optimization": False,
    "empirical_outcomes": False,
    "scientific_promotion": False,
    "prior_learned_floor_failure_preserved": True,
    "profile_prediction_representation": "floor + exp(log(scale) + (X amplitude - training_max_logit))",
    "gradient_representation": "normalized envelope gradient using X minus its maximum-logit training row",
    "per_start_evaluation_cap_includes_initial_call": True,
    "unresolved_diagnostics": [
        "selected source solve unsuccessful",
        "learned-floor exact or near-constant profile",
        "training exponential underflow",
        "projected gradient exceeds fixed tolerance",
    ],
}


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def runtime() -> dict[str, str]:
    return {"python": sys.version, "sqlite": sqlite3.sqlite_version} | {
        name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl")
    }


def source_paths() -> set[Path]:
    return {Path(path) for path in read(EXTENSION / "registration.json")["source_hashes"]} | {
        Path(__file__),
        Path(model.__file__),
        BASE / "test_tpp40_response_profile_20261015.py",
        BASE / "test_run_tpp40_response_profile_20261015.py",
        PARENT / "reviews/response_space_profile_diagnostic_proposal_v1.md",
        PARENT / "reviews/floor_identification_fable_v1/adjudication_v1.md",
    }


def input_paths() -> set[Path]:
    return {DATA / name for name in ("policies.npz", "metadata.json", "rows.csv", "folds.csv")} | {
        EXTENSION_SEAL,
        STAGE1_SEAL,
        CACHE_PATH,
        EXTENSION / "execution_identity.json",
        EXTENSION / "registration.json",
        EXTENSION / "oracle_completion/complete.json",
        EXTENSION / "oracle_summary.json",
        STAGE1 / "prepared/clean.npz",
        STAGE1 / "prepared/noise.npz",
        STAGE1 / "prepared/matched_tied.npz",
        STAGE1 / "prepared/preflight.json",
        PURE_REVIEW,
    }


def verify_hashes(hashes: dict[str, str]) -> None:
    for path, expected in hashes.items():
        if digest(Path(path)) != expected:
            raise ValueError(f"Frozen input changed: {path}")


def verify_prerequisites(registration: dict[str, Any]) -> None:
    review = read(PURE_REVIEW)
    if (
        digest(PURE_REVIEW) != registration["prerequisites"]["independent_pure_review_sha256"]
        or not review["passed"]
        or review["source_sha256"] != digest(Path(model.__file__))
        or review["tests_sha256"] != digest(BASE / "test_tpp40_response_profile_20261015.py")
    ):
        raise ValueError("The independent pure review does not cover this exact implementation")
    for folder, manifest, pin in (
        (STAGE1, STAGE1_SEAL, "stage1_sha256"),
        (EXTENSION, EXTENSION_SEAL, "extension_sha256"),
    ):
        if digest(manifest) != registration["prerequisites"][pin]:
            raise ValueError("The registered prerequisite seal differs")
        sealed = read(manifest)
        verify_hashes({str(folder / name): item["sha256"] for name, item in sealed["files"].items()})
    extension = read(EXTENSION_SEAL)
    if extension["status"] != "independent_numerical_replay_pass_diagnostic_only" or extension["scientific_promotion"]:
        raise ValueError("The extension is an audited diagnostic, not a promoted estimator")
    prior = read(STAGE1 / "prepared/preflight.json")
    if prior["learned_stage_ready"] or sum(not item["gamma_feasible"] for item in prior["floor_checks"]) != 137:
        raise ValueError("Preserve all inherited floor-domain failures")
    for suffix in ("-wal", "-shm"):
        if Path(str(CACHE_PATH) + suffix).exists():
            raise ValueError("The sealed feature cache must be checkpointed before immutable access")


def registration() -> dict[str, Any]:
    value = read(OUTPUT / "registration.json")
    if value.get("status") != "registered" or value.get("contract") != CONTRACT or value.get("environment") != runtime():
        raise ValueError("Register the exact diagnostic and runtime before execution")
    for name, paths in (("source_hashes", source_paths()), ("input_hashes", input_paths())):
        if set(value[name]) != {str(path.resolve()) for path in paths}:
            raise ValueError("The registered source/input whitelist differs")
        verify_hashes(value[name])
    verify_prerequisites(value)
    return value


def identity() -> dict[str, Any]:
    value = registration()
    return {name: value[name] for name in ("source_hashes", "input_hashes", "environment", "prerequisites")} | {
        "registration_sha256": digest(OUTPUT / "registration.json")
    }


def freeze() -> None:
    value = identity()
    path = OUTPUT / "execution_identity.json"
    if path.exists() and read(path) != value:
        raise ValueError("Preserve the earlier identity and register a separate revision")
    write(path, value)


def execution_identity() -> dict[str, Any]:
    value = identity()
    if read(OUTPUT / "execution_identity.json") != value:
        raise ValueError("The execution identity changed")
    approval = read(OUTPUT / "execution_approval.json")
    expected = {
        "scope": SCOPE,
        "registration_sha256": value["registration_sha256"],
        "execution_identity_sha256": digest(OUTPUT / "execution_identity.json"),
    }
    if any(approval.get(name) != expected_value for name, expected_value in expected.items()):
        raise ValueError("The exact frozen diagnostic requires its execution approval")
    return value


def receipt(folder: Path, spec: dict[str, Any], value: dict[str, Any]) -> bool:
    path = folder / "complete.json"
    if not path.exists():
        return False
    saved = read(path)
    if saved["spec"] != spec or saved["identity"] != value:
        raise ValueError("A completed diagnostic cell belongs to another execution")
    verify_hashes({str(folder / name): sha for name, sha in saved["files"].items()})
    verify_hashes(saved["dependencies"])
    return True


def finish(
    folder: Path, spec: dict[str, Any], value: dict[str, Any], names: tuple[str, ...], dependencies: tuple[Path, ...]
) -> None:
    write(
        folder / "complete.json",
        {
            "spec": spec,
            "identity": value,
            "files": {name: digest(folder / name) for name in names},
            "dependencies": {str(path): digest(path) for path in dependencies},
        },
    )


def training_values(clean: np.ndarray, noise: np.ndarray, train: np.ndarray) -> np.ndarray:
    return clean[train] + noise[train]


def geometry(outer: int) -> dict[str, Any]:
    metadata = read(DATA / "metadata.json")
    with np.load(DATA / "policies.npz", allow_pickle=False) as archive:
        if set(archive.files) != {"phase0", "phase1", "phase0_epochs", "phase1_epochs", "c0", "c1"}:
            raise ValueError("Policy-only archive whitelist changed")
        first, last = archive["phase0_epochs"], archive["phase1_epochs"]
        scales = archive["c0"] + archive["c1"]
    rows = pd.read_csv(DATA / "rows.csv", usecols=["row_id", "policy_id", "group_id", "source", "region", "tied"])
    folds = pd.read_csv(DATA / "folds.csv", usecols=["panel", "repeat", "outer_fold", "row_id", "role", "inner_fold"])
    part = folds.loc[(folds.repeat == 0) & (folds.outer_fold == outer)]
    lookup = {name: index for index, name in enumerate(rows.row_id)}
    split = {
        role: np.asarray([lookup[name] for name in part.loc[part.role == role, "row_id"]], int)
        for role in ("train", "test", "tied_test")
    }
    if rows.iloc[split["train"]].tied.any() or set(rows.iloc[split["train"]].group_id) & set(
        rows.iloc[split["test"]].group_id
    ):
        raise ValueError("The frozen asymmetric grouped split changed")
    alpha = float(metadata["alpha"])
    held, tied = split["test"], split["tied_test"]
    total = first[held] + last[held]
    q0, q1 = (
        np.vstack((first[held], alpha * total, first[tied])),
        np.vstack((last[held], (1 - alpha) * total, last[tied])),
    )
    indices = np.r_[held, held, tied]
    query = pd.DataFrame(
        {
            "role": ["asymmetric"] * len(held) + ["matched_tied"] * len(held) + ["actual_tied"] * len(tied),
            "source_index": indices,
            "row_id": rows.iloc[indices].row_id.to_numpy(),
        }
    )
    return {
        "first": first,
        "last": last,
        "query0": q0,
        "query1": q1,
        "query": query,
        "alpha": alpha,
        "scales": scales,
        "split": split,
        "rows": rows,
    }


class FrozenFeatures:
    """Read exact feature rows from the sealed immutable SQLite cache."""

    def __init__(self, connection: sqlite3.Connection, data: dict[str, Any], frozen: dict[str, Any]):
        self.connection = connection
        self.contexts = {}
        for key, body in connection.execute("SELECT key,identity FROM contexts"):
            record = json.loads(body)
            if canonical_hash(record) != key:
                raise ValueError("A cache context identity is corrupt")
            if record["rate"] != 0.25 or record["power"] != 0.7 or record["alpha"] != data["alpha"]:
                raise ValueError("The oracle cache has an unexpected fixed shape")
            for path, sha in record["sources"].items():
                if frozen["source_hashes"][path] != sha:
                    raise ValueError("The cached law source differs")
            expected_runtime = {
                name: frozen["environment"][name] for name in ("python", "numpy", "scipy", "threadpoolctl", "sqlite")
            }
            if record["runtime"] != expected_runtime:
                raise ValueError("Cached feature runtime differs")
            config = record["config"]
            pools = 1 / data["scales"]
            masses = np.full(2, 0.5) if config["clock"] == "equal_group" else pools / pools.sum()
            expected = {
                "clock": config["clock"],
                "scales": data["scales"].tolist(),
                "masses": masses.tolist(),
                "groups": ["first", "second"],
                "relative_tolerance": 1e-11,
                "max_panels": 512,
            }
            if config != expected or config["clock"] not in ("equal_group", "global_pool"):
                raise ValueError("Cached clock configuration differs")
            self.contexts[(config["clock"], record["mu"])] = key

    def matrix(self, first: np.ndarray, last: np.ndarray, world: dict[str, Any]) -> np.ndarray:
        total = first + last
        if world["mu"] == 0:
            benefit = -np.expm1(-((0.25 * total) ** 0.7))
        else:
            context = self.contexts[(world["clock"], world["mu"])]
            values = []
            for early, late in zip(first, last, strict=True):
                request = {"early": early.tolist(), "late": late.tolist()}
                row = self.connection.execute(
                    "SELECT request,response,response_sha256 FROM features WHERE context=? AND policy=?",
                    (context, canonical_hash(request)),
                ).fetchone()
                if row is None:
                    raise ValueError("A required saved feature is missing; no new integration is allowed")
                encoded, response, sha = row
                result = json.loads(response)
                if json.loads(encoded) != request or hashlib.sha256(response.encode()).hexdigest() != sha:
                    raise ValueError("A frozen cache row is corrupt")
                if result["status"] != "complete" or result["relative_error_estimate"] > 1e-11:
                    raise ArithmeticError("Preserve unconverged cached features")
                values.append(result["benefit"])
            benefit = np.asarray(values)
        harm = np.logaddexp(0.0, np.log1p(total) - 3.0) ** 2
        matrix = np.column_stack((-benefit, harm))
        if not np.isfinite(matrix).all():
            raise ArithmeticError("Nonfinite frozen design")
        return matrix


def fixed_profile(fit: dict[str, Any], mu: float) -> dict[str, Any]:
    """Use the generating kinetic profile, independent of the saved CV winner."""
    profile = next(item for item in fit["profiles"] if item["mu"] == mu)
    if profile["head"]["floor"] != 0.6:
        raise ValueError("The baseline floor is not the supplied generating floor")
    return profile


def baseline_profile(world: dict[str, Any], outer: int, noise: str) -> tuple[dict[str, Any], Path]:
    candidate = world["clock"] or "aggregate_mariner"
    folder = EXTENSION / "cells" / f"outer{outer}" / noise / world["name"] / candidate
    saved = read(folder / "complete.json")
    if saved["identity"] != read(EXTENSION / "execution_identity.json"):
        raise ValueError("The baseline execution identity differs")
    verify_hashes({str(folder / name): sha for name, sha in saved["files"].items()})
    verify_hashes(saved["dependencies"])
    fit = read(folder / "fit.json")
    return fixed_profile(fit, world["mu"]), folder / "complete.json"


def log_prediction(matrix: np.ndarray, head: dict[str, Any]) -> np.ndarray:
    return head["floor"] + np.exp(np.clip(head["intercept"] + matrix @ np.asarray(head["coefficients"]), -30, 30))


def score(predicted: np.ndarray, truth: np.ndarray, query: pd.DataFrame) -> dict[str, Any]:
    held = int(np.sum(query.role == "asymmetric"))
    gain, true_gain = predicted[held : 2 * held] - predicted[:held], truth[held : 2 * held] - truth[:held]
    error = predicted - truth
    result: dict[str, Any] = {
        "query_rmse": float(np.sqrt(np.mean(error**2))),
        "query_max_abs_error": float(np.max(abs(error))),
        "gain_rmse": float(np.sqrt(np.mean((gain - true_gain) ** 2))),
        "true_gain_rms": float(np.sqrt(np.mean(true_gain**2))),
        "maximum_absolute_predicted_gain": float(np.max(abs(gain))),
    }
    for role in ("asymmetric", "matched_tied", "actual_tied"):
        residual = error[query.role.to_numpy() == role]
        result[f"{role}_count"] = len(residual)
        result[f"{role}_rmse"] = float(np.sqrt(np.mean(residual**2))) if len(residual) else None
        result[f"{role}_bias"] = float(np.mean(residual)) if len(residual) else None
        result[f"{role}_max_abs_error"] = float(np.max(abs(residual))) if len(residual) else None
    return result


def fit_diagnostics(fit: dict[str, Any]) -> dict[str, Any]:
    """Expose every start's failure status alongside the selected head."""
    selected = fit.get("selected")
    solves = fit.get("solves", [])
    selected_start = None if selected is None else selected["start_index"]
    return {
        "starts_attempted": len(solves),
        "failed_start_count": sum(not item["success"] for item in solves),
        "nonwinning_failed_start_count": sum(
            not item["success"] and item["start_index"] != selected_start for item in solves
        ),
        "ftol_termination_count": sum(item["termination"] == "ftol" for item in solves),
        "gradient_termination_count": sum(item["termination"] == "gradient" for item in solves),
        "evaluation_cap_count": sum(item["termination"] == "evaluation_cap" for item in solves),
        "numerical_failure_evaluation_count": sum(item["status"] != "finite" for item in fit.get("trials", [])),
        "selected_start_index": selected_start,
        "selected_from_successful_solve": fit.get("selected_from_successful_solve"),
        "selected_near_constant_phi": None if selected is None else selected["near_constant_phi"],
        "selected_nonregular_learned_floor": None if selected is None else selected["nonregular_learned_floor"],
        "selected_training_underflow_count": (
            None if selected is None else selected["training_exponential_underflow_count"]
        ),
        "selected_first_order_tolerance_met": None if selected is None else selected["first_order_tolerance_met"],
        "native_exponential_clipped_count": fit.get("native_exponential_clipped_count", 0),
    }


def run_context(spec: dict[str, Any]) -> dict[str, Any]:
    frozen = execution_identity()
    world, outer, noise = spec["world"], spec["outer_fold"], spec["noise"]
    folder = OUTPUT / "cells" / f"outer{outer}" / noise / world["name"]
    data = geometry(outer)
    train, held, tied = (data["split"][name] for name in ("train", "test", "tied_test"))
    with (
        np.load(STAGE1 / "prepared/clean.npz", allow_pickle=False) as clean,
        np.load(STAGE1 / "prepared/noise.npz", allow_pickle=False) as noises,
        np.load(STAGE1 / "prepared/matched_tied.npz", allow_pickle=False) as ties,
    ):
        observed = training_values(clean[world["name"]], noises[noise], train)
        truth = np.r_[clean[world["name"]][held], ties[world["name"]][held], clean[world["name"]][tied]]
    with closing(sqlite3.connect(f"file:{CACHE_PATH}?mode=ro&immutable=1", uri=True)) as connection:
        cache = FrozenFeatures(connection, data, frozen)
        matrix = cache.matrix(data["first"][train], data["last"][train], world)
        query_matrix = cache.matrix(data["query0"], data["query1"], world)
    baseline, baseline_receipt = baseline_profile(world, outer, noise)
    if not np.allclose(
        log_prediction(query_matrix, baseline["head"]), baseline["query_predictions"], rtol=0, atol=2e-12
    ):
        raise ArithmeticError("The immutable design does not replay the saved baseline query")
    start_folder = folder / "initialization"
    start_spec = {**spec, "stage": "five_gamma_starts"}
    if receipt(start_folder, start_spec, frozen):
        starts = read(start_folder / "starts.json")["starts"]
    else:
        with threadpool_limits(limits=1):
            starts = model.gamma_starts(matrix, observed)
        write(
            start_folder / "starts.json",
            {
                "starts": starts,
                "train_indices": train.tolist(),
                "observed": observed.tolist(),
                "matrix": matrix.tolist(),
            },
        )
        finish(start_folder, start_spec, frozen, ("starts.json",), (EXTENSION_SEAL, STAGE1_SEAL, baseline_receipt))
    results = []
    for arm in ARMS:
        frozen = execution_identity()
        target = folder / arm
        arm_spec = {**spec, "arm": arm}
        if receipt(target, arm_spec, frozen):
            results.append(read(target / "scores.json"))
            continue
        if (target / "execution_failure.json").exists():
            raise ValueError("Preserve the failed cell and review a separate revision before retrying")
        fit: dict[str, Any] | None = None
        try:
            with threadpool_limits(limits=1):
                if arm == ARMS[0]:
                    fit = {
                        "status": "reused_fixed_true_kinetic_profile",
                        "profile": baseline,
                        "baseline_receipt_sha256": digest(baseline_receipt),
                        "scientific_promotion": False,
                    }
                    predicted = log_prediction(query_matrix, baseline["head"])
                    training = log_prediction(matrix, baseline["head"])
                    unresolved = False
                    logits = baseline["head"]["intercept"] + np.vstack((matrix, query_matrix)) @ np.asarray(
                        baseline["head"]["coefficients"]
                    )
                    fit["native_exponential_clipped_count"] = int(np.sum((logits < -30) | (logits > 30)))
                else:
                    fit = model.fit_response(matrix, observed, 0.6 if arm == ARMS[1] else None, starts)
                    if fit["selected"] is None:
                        raise ArithmeticError("No finite response-profile candidate")
                    predicted, training = (
                        model.predict(query_matrix, fit["selected"]),
                        model.predict(matrix, fit["selected"]),
                    )
                    unresolved = fit["unresolved"]
                report = {
                    **score(predicted, truth, data["query"]),
                    **fit_diagnostics(fit),
                    "raw_training_mse": float(np.mean((training - observed) ** 2)),
                    "unresolved": unresolved,
                    "scientific_promotion": False,
                    "learned_stage_remains_blocked": True,
                    "clean_correct_model_reference_applicable": noise == "clean" and world["kind"] != "heterogeneous",
                }
                report["clean_query_recovery_reference_met"] = (
                    report["query_max_abs_error"] <= 1e-8 if report["clean_correct_model_reference_applicable"] else None
                )
            write(target / "fit.json", fit)
            write(target / "scores.json", report)
            data["query"].assign(truth=truth, predicted=predicted).to_csv(target / "predictions.csv", index=False)
            pd.DataFrame(
                {
                    "source_index": train,
                    "row_id": data["rows"].iloc[train].row_id.to_numpy(),
                    "observed": observed,
                    "predicted": training,
                }
            ).to_csv(target / "training.csv", index=False)
            finish(
                target,
                arm_spec,
                frozen,
                ("fit.json", "scores.json", "predictions.csv", "training.csv"),
                (start_folder / "complete.json", baseline_receipt, EXTENSION_SEAL, STAGE1_SEAL),
            )
        except (ArithmeticError, ValueError, RuntimeError) as error:
            if fit is not None:
                write(target / "failed_fit.json", fit)
            write(
                target / "execution_failure.json",
                {"spec": arm_spec, "identity": frozen, "exception": type(error).__name__, "message": str(error)},
            )
            raise
        results.append(report)
    return {**spec, "arms_completed": len(results)}


def contexts() -> list[dict[str, Any]]:
    return [
        {"world": world, "outer_fold": outer, "noise": noise}
        for outer in range(5)
        for noise in NOISES
        for world in WORLDS
    ]


def summarize() -> None:
    frozen = execution_identity()
    records, dependencies = [], []
    for spec in contexts():
        folder = OUTPUT / "cells" / f"outer{spec['outer_fold']}" / spec["noise"] / spec["world"]["name"]
        for arm in ARMS:
            target = folder / arm
            if not receipt(target, {**spec, "arm": arm}, frozen):
                raise ValueError("The registered240-head diagnostic is incomplete")
            fit = read(target / "fit.json")
            selected = fit.get("selected")
            records.append(
                {
                    **spec,
                    "arm": arm,
                    **read(target / "scores.json"),
                    "floor": fit["profile"]["head"]["floor"] if selected is None else selected["floor"],
                    "selected_face": None if selected is None else selected["face"],
                    "selected_projected_gradient_inf": None if selected is None else selected["projected_gradient_inf"],
                }
            )
            dependencies.append(target / "complete.json")
    write(
        OUTPUT / "summary.json",
        {
            "status": "conditional_response_diagnostic_not_method",
            "records": records,
            "heads": len(records),
            "scientific_promotion": False,
            "learned_stage_remains_blocked": True,
        },
    )
    finish(OUTPUT, {"stage": "summary"}, frozen, ("summary.json",), tuple(dependencies))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "fit", "summarize"))
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze()
    elif args.command == "summarize":
        summarize()
    else:
        execution_identity()
        if args.workers < 1:
            raise ValueError("Use a positive worker count")
        if args.workers == 1:
            for spec in contexts():
                print(json.dumps(run_context(spec)), flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                for result in pool.map(run_context, contexts()):
                    print(json.dumps(result), flush=True)
        summarize()


if __name__ == "__main__":
    main()
