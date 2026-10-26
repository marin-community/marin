# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""TP051: surrogate-only raw policy selection followed by synthetic teacher scoring."""

from __future__ import annotations

import argparse
import math
import platform
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    run_tpp40_acquisition_retention_oracle_extension_20261015 as reference,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_response_profile_20261015 as prior
from experiments.domain_phase_mix.exploratory.two_phase_many import tpp40_two_bucket_optimization_20261015 as optimizer

oracle, models = reference.oracle, reference.models
BASE = Path(__file__).resolve().parent
OUTPUT = oracle.PARENT / "raw_policy_selection_diagnostic"
UPSTREAM = oracle.PARENT / "whole_response_ridge_stress"
SCOPE = "tp051_all30_frozen_tp049_raw_policy_falsifier_no_refits"
DEADLINE = "2026-10-15T18:00:00+00:00"
SEARCH = {"relative_tolerance": 1e-11, "max_panels": 512}
REPLAY = {"relative_tolerance": 1e-12, "max_panels": 2048}
CONTRACT: dict[str, Any] = {
    "candidate_id": "TP40-051",
    "scope": SCOPE,
    "heads": 30,
    "native_controls": 12,
    "temporal_heads": 18,
    "outer_fold": 3,
    "noise_seeds": list(oracle.SEEDS),
    "worlds": list(oracle.WORLDS),
    "domain": "physical early/late code weights in [0,1]^2; web is complement",
    "optimizer": "unchanged17x17/five16-interval lines/12Powell; retain every finite trial and tied incumbent",
    "aggregate_restriction": "exact native or selected mu0 uses diagonal-only search and returns tied policy",
    "search_numerics": SEARCH,
    "replay_numerics": REPLAY,
    "teacher_searches": "byte-adopt4 sealed oracle_extension results after exact identity verification",
    "selection": "surrogate prediction only; durable result receipt before teacher or native-control scoring",
    "teacher_response": "unchanged hand-defined unclipped exponential truth",
    "fitted_response": "saved floor+exp(log(scale)+Xamplitudes-saved_training_normalization); no clipping",
    "controls": "same-world/same-noise independently fitted native raw policy; no extra fit",
    "observed_baseline": (
        "12 training-only minimum-observed endpoints, ties by row_id then source index; "
        "frozen before teacher barrier; clean archived truth scoring only"
    ),
    "failed_inputs": "all30 identities retained regardless of inherited convergence/query status",
    "deadline_utc": DEADLINE,
    "minimum_free_bytes": 4 * 1024**3,
    "resource_stop": "checked before each scalar callback; keep partial incumbents and distinct stop status",
    "new_fits": False,
    "new_empirical_labels": False,
    "caps_or_kl": False,
    "global_optimum_certified": False,
    "promotion": False,
    "gate_changes": False,
}


class ResourceStop(ArithmeticError):
    """A resource boundary is not evidence that a scientific law failed."""


def resource_check(folder: Path) -> None:
    if datetime.now(UTC) >= datetime.fromisoformat(DEADLINE):
        raise ResourceStop("Registered18UTC deadline reached")
    if shutil.disk_usage(folder).free < CONTRACT["minimum_free_bytes"]:
        raise ResourceStop("Registered4GiB free-space boundary reached")


def specs() -> list[dict[str, Any]]:
    return [
        {"noise": str(seed), "world": world["name"], "candidate": candidate}
        for seed in oracle.SEEDS
        for world in oracle.WORLDS
        for candidate in oracle.candidates(world)
    ]


def cell(root: Path, spec: dict[str, Any]) -> Path:
    return root / "cells/outer3" / spec["noise"] / spec["world"] / spec["candidate"]


def source_paths() -> set[Path]:
    return reference.source_paths() | {
        Path(__file__),
        BASE / "test_tpp40_ridge_raw_policies_20261015.py",
        Path(prior.__file__),
        OUTPUT / "proposal_v1.md",
        *map(Path, prior.read(UPSTREAM / "registration.json")["source_hashes"]),
    }


def input_paths() -> set[Path]:
    paths = (
        oracle.input_paths()
        | {
            UPSTREAM / name
            for name in (
                "registration.json",
                "execution_identity.json",
                "complete.json",
                "summary.json",
                "result_manifest_v1.json",
                "audit/author_replay_report_v1.json",
            )
        }
        | {
            reference.OUTPUT / name
            for name in (
                "registration.json",
                "execution_identity.json",
                "raw_truth_summary.json",
                "result_manifest_v1.json",
                "audit/independent_raw_replay_report_v1.json",
            )
        }
    )
    for spec in specs():
        folder = cell(UPSTREAM, spec)
        paths |= {folder / "fit.json", folder / "fit_complete.json"}
        fitted = prior.read(folder / "fit.json")
        paths.add(Path(fitted["full_checkpoint"]))
        status_paths = [
            folder / name
            for name in ("scores.json", "score_complete.json", "query_failure.json")
            if (folder / name).exists()
        ]
        if not status_paths:
            raise ValueError("Each frozen head needs its original query outcome or failure")
        paths.update(status_paths)
    for world in oracle.WORLDS:
        paths |= {
            reference.OUTPUT / "raw_truth" / world["name"] / name for name in ("optimization.json", "complete.json")
        }
    paths.add(oracle.OUTPUT / "prepared/clean.npz")
    for spec in specs():
        if spec["candidate"] == "aggregate_mariner":
            folder = prior.OUTPUT / "cells/outer3" / spec["noise"] / spec["world"] / "initialization"
            paths.update((folder / "starts.json", folder / "complete.json"))
    return paths


def verify_upstream() -> None:
    if prior.read(UPSTREAM / "complete.json")["completed_heads"] != 30:
        raise ValueError("All30 TP049 outcomes must be finalized before binding or querying")
    registration = prior.read(UPSTREAM / "registration.json")
    prior.verify_hashes(registration["source_hashes"])
    for spec in specs():
        if spec["candidate"] == "aggregate_mariner":
            folder = prior.OUTPUT / "cells/outer3" / spec["noise"] / spec["world"] / "initialization"
            prior.verify_hashes(
                {
                    str(folder / name): registration["input_hashes"][str(folder / name)]
                    for name in ("starts.json", "complete.json")
                }
            )
    historical = prior.read(UPSTREAM / "execution_identity.json")
    if historical["source_hashes"] != registration["source_hashes"] or historical["registration_sha256"] != prior.digest(
        UPSTREAM / "registration.json"
    ):
        raise ValueError("TP049 source/registration identity changed")
    report = prior.read(UPSTREAM / "audit/author_replay_report_v1.json")
    expected = {
        "status": "author_saved_output_replay_passed",
        "unique_inner_conditional_records": 302400,
        "zero_ridge_exact_lineage_inner_count": 60480,
        "positive_ridge_retained_only_inner_count": 241920,
    }
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError("Require the saved-head TP049 replay")
    population = [
        {key: row[key] for key in ("noise", "world", "candidate")} for row in report["rows"] + report["query_failures"]
    ]
    if sorted(population, key=prior.canonical_hash) != sorted(specs(), key=prior.canonical_hash):
        raise ValueError("TP049 replay must account for exactly30 unique identities")
    seal = prior.read(UPSTREAM / "result_manifest_v1.json")
    frozen_identity = prior.digest(UPSTREAM / "execution_identity.json")
    for spec in specs():
        folder = cell(UPSTREAM, spec)
        receipt = prior.read(folder / "fit_complete.json")
        if receipt["identity_sha256"] != frozen_identity:
            raise ValueError("A frozen full-fit receipt has a foreign identity")
        prior.verify_hashes({str(folder / name): sha for name, sha in receipt["files"].items()})
        fit = prior.read(folder / "fit.json")
        checkpoint = Path(fit["full_checkpoint"])
        prior.verify_hashes({str(checkpoint): receipt["conditional_files"][str(checkpoint)]})
        for path in (folder / "fit.json", folder / "fit_complete.json", checkpoint):
            entry = seal["files"][str(path.relative_to(UPSTREAM))]
            prior.verify_hashes({str(path): entry["sha256"]})


def verify_truth(data: Any) -> None:
    if not prior.read(reference.OUTPUT / "audit/independent_raw_replay_report_v1.json")["passed"]:
        raise ValueError("Require the frozen raw-truth arithmetic audit")
    prior.verify_hashes(prior.read(reference.OUTPUT / "registration.json")["source_hashes"])
    seal = prior.read(reference.OUTPUT / "result_manifest_v1.json")
    expected_base = {
        "alpha": data.alpha,
        "scales": data.total_scales.tolist(),
        "search_numerics": SEARCH,
        "replay_numerics": REPLAY,
        "source_sha256": prior.digest(Path(optimizer.__file__)),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    }
    for world in oracle.WORLDS:
        folder = reference.OUTPUT / "raw_truth" / world["name"]
        for name in ("optimization.json", "complete.json"):
            path = folder / name
            prior.verify_hashes({str(path): seal["files"][str(path.relative_to(reference.OUTPUT))]["sha256"]})
        result = prior.read(folder / "optimization.json")
        for key, value in expected_base.items():
            if result["identity"][key] != value:
                raise ValueError("Teacher search geometry/helper/numerics/runtime differs: " + key)
        teacher = result["identity"]["predictor"]
        if teacher["world"] != world or teacher["shape"] != oracle.SHAPE or teacher["head"] != oracle.HEAD:
            raise ValueError("Teacher law/world/head differs")
        if teacher["response"] != "unclipped hand-defined exponential truth":
            raise ValueError("Teacher response differs")
        for field_name in ("source_hashes", "input_hashes"):
            prior.verify_hashes(teacher["identity"][field_name])
        if result["numerical_audit_unresolved"]:
            raise ValueError("Cannot silently replace an unresolved prior teacher search")


def identity() -> dict[str, Any]:
    value = prior.read(OUTPUT / "registration.json")
    if value.get("status") != "registered" or value["contract"] != CONTRACT or value["environment"] != prior.runtime():
        raise ValueError("Register the exact source/runtime/scope before new queries")
    verify_upstream()
    for field_name, paths in (("source_hashes", source_paths()), ("input_hashes", input_paths())):
        if set(value[field_name]) != {str(path.resolve()) for path in paths}:
            raise ValueError("Exact source/input whitelist differs")
        prior.verify_hashes(value[field_name])
    data, _ = oracle.inputs()
    if data.buckets != ("nemotron_full", "starcoder"):
        raise ValueError("Raw coordinates require web0/code1")
    verify_truth(data)
    return {key: value[key] for key in ("source_hashes", "input_hashes", "environment")} | {
        "registration_sha256": prior.digest(OUTPUT / "registration.json")
    }


def approved() -> dict[str, Any]:
    frozen = identity()
    if prior.read(OUTPUT / "execution_identity.json") != frozen:
        raise ValueError("Frozen identity changed")
    receipt = prior.read(OUTPUT / "execution_approval.json")
    expected = {
        "scope": SCOPE,
        "registration_sha256": frozen["registration_sha256"],
        "execution_identity_sha256": prior.digest(OUTPUT / "execution_identity.json"),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("Exact execution approval required")
    return frozen


@dataclass
class FittedPredictor:
    spec: dict[str, Any]
    head: dict[str, Any]
    alpha: float
    config: Any
    cache: Any
    resource_folder: Path
    underflows: int = 0
    stopped: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __call__(self, early: np.ndarray, late: np.ndarray) -> float:
        try:
            resource_check(self.resource_folder)
        except ResourceStop:
            self.stopped = True
            raise
        if self.head["scale"] == 0:
            value, logit, underflow = self.head["floor"], None, False
        else:
            matrix, _ = models.design_and_offset(
                early[None],
                late[None],
                self.alpha,
                self.spec["shape"],
                models.MODEL,
                {"mu": self.spec["mu"]},
                self.config,
                self.cache,
            )
            logit = float(
                math.log(self.head["scale"])
                + ((matrix @ np.asarray(self.head["amplitudes"]))[0] - self.head["normalization"])
            )
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                deficit = float(np.exp(logit))
            underflow = deficit == 0
            value = self.head["floor"] + deficit
        if not math.isfinite(value):
            raise ArithmeticError("Nonfinite raw fitted prediction; no clipping allowed")
        self.underflows += int(underflow)
        key = prior.canonical_hash({"early": early.tolist(), "late": late.tolist()})
        self.diagnostics[key] = {"logit": logit, "underflow": underflow, "zero_scale": self.head["scale"] == 0}
        return value


def optimize_cell(spec: dict[str, Any], frozen: dict[str, Any]) -> None:
    folder = cell(OUTPUT, spec)
    if prior.receipt(folder / "selection", spec | {"stage": "surrogate_selection"}, frozen):
        return
    if (folder / "selection/optimization.json").exists():
        raise ValueError("Preserve an unreceipted raw result before retry")
    folder.mkdir(parents=True, exist_ok=True)
    source = cell(UPSTREAM, spec)
    fitted = prior.read(source / "fit.json")
    inherited = {
        name: prior.read(source / name) for name in ("scores.json", "query_failure.json") if (source / name).exists()
    }
    result: dict[str, Any] = {
        "inherited": inherited,
        "fit_full_unresolved": fitted["full_unresolved"],
        "selected_candidate": fitted["selected_candidate"],
    }
    head = fitted["selected"]
    if head is None:
        result.update(status="upstream_fit_failed", optimization=None)
    else:
        data, folds = oracle.inputs()
        selected = fitted["selected_candidate"]
        cache = models.BenefitCache(folder / "features.sqlite")
        config = replace(oracle.feature_config(data.total_scales, selected["clock"]), **SEARCH)
        search = FittedPredictor(selected, head, data.alpha, config, cache, folder)
        replay = replace(search, config=replace(config, **REPLAY), diagnostics={})
        callbacks = optimizer.Predictors(
            search,
            replay,
            {
                "frozen_identity": frozen,
                "fit_sha256": prior.digest(source / "fit.json"),
                "spec": spec,
                "selected_candidate": selected,
                "head": head,
            },
            SEARCH,
            REPLAY,
        )
        train = oracle.split_indices(data.rows, folds, 3)["train"]
        weights = np.stack(
            (
                data.early[train] / (data.alpha * data.total_scales),
                data.late[train] / ((1 - data.alpha) * data.total_scales),
            ),
            axis=1,
        )
        optimized = optimizer.optimize_two_bucket(
            callbacks,
            alpha=data.alpha,
            scales=data.total_scales,
            restriction=optimizer.Restriction.AGGREGATE if selected["mu"] == 0 else optimizer.Restriction.TEMPORAL,
            observed_policies=weights,
        )
        result.update(
            status="resource_stopped" if search.stopped or replay.stopped else "raw_complete",
            optimization=optimized,
            prediction_numerics={
                "search_underflow_calls": search.underflows,
                "replay_underflow_calls": replay.underflows,
                "search": search.diagnostics,
                "replay": replay.diagnostics,
            },
        )
    target = folder / "selection"
    prior.write(target / "optimization.json", result)
    prior.finish(
        target,
        spec | {"stage": "surrogate_selection"},
        frozen,
        ("optimization.json",),
        (source / "fit.json", source / "fit_complete.json"),
    )


def teacher_values(result: dict, truth: Any, alpha: float, scales: np.ndarray, folder: Path) -> dict[str, float]:
    points = {
        "selected_free": (result["best"]["phase0"], result["best"]["phase1"]),
        "selected_static": (result["best_static"]["phase0"], result["best_static"]["phase1"]),
        "selected_aggregate_tied": (result["aggregate"], result["aggregate"]),
    }
    values = {}
    for name, (first, last) in points.items():
        resource_check(folder)
        values[name] = truth(alpha * scales * np.asarray(first), (1 - alpha) * scales * np.asarray(last))
    return values


def observed_choice(values: np.ndarray, indices: np.ndarray, row_ids: list[str]) -> int:
    """Choose only from observed training values; labels cannot break exact ties."""
    return min(range(len(indices)), key=lambda index: (values[index], row_ids[index], int(indices[index])))


def select_observed_baselines(frozen: dict[str, Any]) -> None:
    data, folds = oracle.inputs()
    train = oracle.split_indices(data.rows, folds, 3)["train"]
    for spec in specs():
        if spec["candidate"] != "aggregate_mariner":
            continue
        folder = OUTPUT / "observed_baseline" / spec["noise"] / spec["world"] / "selection"
        stage = spec | {"stage": "training_observed_selection"}
        if prior.receipt(folder, stage, frozen):
            continue
        source = prior.OUTPUT / "cells/outer3" / spec["noise"] / spec["world"] / "initialization/starts.json"
        saved = prior.read(source)
        values = np.asarray(saved["observed"], float)
        if saved["train_indices"] != train.tolist() or values.shape != train.shape or not np.isfinite(values).all():
            raise ValueError("Observed baseline training identity differs")
        local = observed_choice(values, train, data.rows.iloc[train].row_id.tolist())
        index = int(train[local])
        record = {
            "spec": spec,
            "source_index": index,
            "row_id": str(data.rows.iloc[index].row_id),
            "observed": float(values[local]),
            "train_count": len(train),
            "tie_rule": "observed,row_id,source_index",
            "phase0": (data.early[index] / (data.alpha * data.total_scales)).tolist(),
            "phase1": (data.late[index] / ((1 - data.alpha) * data.total_scales)).tolist(),
        }
        prior.write(folder / "selection.json", record)
        prior.finish(folder, stage, frozen, ("selection.json",), (source, source.parent / "complete.json"))


def score_observed_baselines(frozen: dict[str, Any]) -> None:
    resource_check(OUTPUT)
    source = oracle.OUTPUT / "prepared/clean.npz"
    with np.load(source, allow_pickle=False) as truth:
        for spec in specs():
            if spec["candidate"] != "aggregate_mariner":
                continue
            root = OUTPUT / "observed_baseline" / spec["noise"] / spec["world"]
            stage = spec | {"stage": "training_observed_teacher_score"}
            if prior.receipt(root / "evaluation", stage, frozen):
                continue
            resource_check(root)
            selected = prior.read(root / "selection/selection.json")
            value = float(truth[spec["world"]][selected["source_index"]])
            prior.write(
                root / "evaluation/scores.json",
                selected | {"teacher_loss": value, "optimism": value - selected["observed"], "new_integration": False},
            )
            prior.finish(
                root / "evaluation", stage, frozen, ("scores.json",), (source, root / "selection/complete.json")
            )


def accounting(result: dict, values: dict, reference_values: dict, native_value: float | None) -> dict:
    free, static, aggregate = (values[key] for key in ("selected_free", "selected_static", "selected_aggregate_tied"))
    gain = aggregate - free
    return {
        "teacher_selected_free": free,
        "teacher_selected_static": static,
        "teacher_selected_aggregate_tied": aggregate,
        "true_gain_same_aggregate": gain,
        "true_gain_vs_selected_static": static - free,
        "predicted_gain_same_aggregate": result["predicted_gain_same_aggregate"],
        "predicted_gain_vs_static": result["predicted_gain_vs_static"],
        "phase_gain_overprediction": result["predicted_gain_same_aggregate"] - gain,
        "selected_optimism": free - result["best"]["predicted"],
        "signed_regret_vs_teacher_2p_incumbent": free - reference_values["selected_free"],
        "signed_regret_vs_teacher_static_incumbent": free - reference_values["selected_static"],
        "teacher_advantage_over_native_selected": None if native_value is None else native_value - free,
    }


def adopt_truth(frozen: dict[str, Any], data: Any) -> None:
    """Replay fixed teacher coordinates after the complete surrogate-selection barrier."""
    for world in oracle.WORLDS:
        folder = OUTPUT / "adopted_truth" / world["name"]
        stage = {"stage": "teacher_incumbent_adoption", "world": world["name"]}
        if prior.receipt(folder, stage, frozen):
            continue
        source = reference.OUTPUT / "raw_truth" / world["name"] / "optimization.json"
        folder.mkdir(parents=True, exist_ok=True)
        adopted = folder / "optimization.json"
        if adopted.exists() and prior.digest(adopted) != prior.digest(source):
            raise ValueError("Preserve a changed teacher adoption")
        if not adopted.exists():
            shutil.copyfile(source, adopted)
        result = prior.read(adopted)
        record: dict[str, Any] = {"source_sha256": prior.digest(source), "reranked": False}
        try:
            cache = models.BenefitCache(folder / "features.sqlite")
            teacher = reference.TruthPredictor(world, data.alpha, data.total_scales, SEARCH, cache)
            primary = teacher_values(result, teacher, data.alpha, data.total_scales, folder)
            strict = teacher_values(result, replace(teacher, numerics=REPLAY), data.alpha, data.total_scales, folder)
            names = {
                "selected_free": "best",
                "selected_static": "best_static",
                "selected_aggregate_tied": "aggregate_tied",
            }
            previous = result["replay"]["selected_reporting_replays"]
            discrepancy = max(
                abs(values[key] - previous[name][field_name])
                for key, name in names.items()
                for values, field_name in ((primary, "search_value"), (strict, "replay_value"))
            )
            record.update(
                status="finite",
                primary=primary,
                strict=strict,
                maximum_adoption_discrepancy=discrepancy,
                numerical_unresolved=discrepancy > optimizer.REPLAY_CHANGE_LIMIT,
            )
        except (ArithmeticError, ValueError) as error:
            record.update(
                status="resource_stopped" if isinstance(error, ResourceStop) else "teacher_replay_failed",
                message=str(error),
                exception=type(error).__name__,
            )
        prior.write(folder / "replay.json", record)
        prior.finish(
            folder, stage, frozen, ("optimization.json", "replay.json"), (source, source.parent / "complete.json")
        )


def score_all(frozen: dict[str, Any]) -> None:
    if not all(
        prior.receipt(cell(OUTPUT, spec) / "selection", spec | {"stage": "surrogate_selection"}, frozen)
        for spec in specs()
    ):
        raise ValueError("Complete all30 durable surrogate selections before teacher scoring")
    if not all(
        prior.receipt(
            OUTPUT / "observed_baseline" / spec["noise"] / spec["world"] / "selection",
            spec | {"stage": "training_observed_selection"},
            frozen,
        )
        for spec in specs()
        if spec["candidate"] == "aggregate_mariner"
    ):
        raise ValueError("Freeze all12 observed-training choices before teacher scoring")
    data, _ = oracle.inputs()
    score_observed_baselines(frozen)
    adopt_truth(frozen, data)
    for spec in specs():
        folder = cell(OUTPUT, spec)
        if prior.receipt(folder / "evaluation", spec | {"stage": "teacher_evaluation"}, frozen):
            continue
        selected = prior.read(folder / "selection/optimization.json")
        result = selected["optimization"]
        record = {
            "spec": spec,
            "source_status": selected["status"],
            "fit_full_unresolved": selected["fit_full_unresolved"],
            "status": "unscored",
        }
        dependencies = [folder / "selection/complete.json"]
        if (
            selected["status"] == "resource_stopped"
            or result is None
            or result["best"] is None
            or result.get("predicted_aggregate_tied") is None
        ):
            record.update(status="no_complete_policy", metrics=None)
        else:
            world = next(item for item in oracle.WORLDS if item["name"] == spec["world"])
            adopted_folder = OUTPUT / "adopted_truth" / world["name"]
            adopted_replay = prior.read(adopted_folder / "replay.json")
            dependencies.append(adopted_folder / "complete.json")
            try:
                resource_check(folder)
                if adopted_replay["status"] != "finite":
                    raise ArithmeticError("Adopted teacher incumbent replay unavailable")
                cache = models.BenefitCache(folder / "teacher_features.sqlite")
                primary = reference.TruthPredictor(world, data.alpha, data.total_scales, SEARCH, cache)
                strict = replace(primary, numerics=REPLAY)
                values = teacher_values(result, primary, data.alpha, data.total_scales, folder)
                strict_values = teacher_values(result, strict, data.alpha, data.total_scales, folder)
                native_path = cell(OUTPUT, spec | {"candidate": "aggregate_mariner"}) / "selection/optimization.json"
                native_record = prior.read(native_path)
                native = native_record["optimization"]
                native_value, native_primary = None, None
                if native_record["status"] == "raw_complete" and native is not None and native["best"] is not None:
                    early = data.alpha * data.total_scales * np.asarray(native["best"]["phase0"])
                    late = (1 - data.alpha) * data.total_scales * np.asarray(native["best"]["phase1"])
                    resource_check(folder)
                    native_primary = primary(early, late)
                    resource_check(folder)
                    native_value = strict(early, late)
                difference = max(abs(values[key] - strict_values[key]) for key in values)
                dependencies.append(native_path.parent / "complete.json")
                record.update(
                    status="finite",
                    metrics=accounting(result, strict_values, adopted_replay["strict"], native_value),
                    primary_metrics=accounting(result, values, adopted_replay["primary"], native_primary),
                    primary_teacher_values=values,
                    strict_teacher_values=strict_values,
                    teacher_maximum_replay_change=difference,
                    teacher_replay_unresolved=difference > optimizer.REPLAY_CHANGE_LIMIT,
                    adopted_teacher_replay_unresolved=adopted_replay["numerical_unresolved"],
                    native_control_source_status=native_record["status"],
                    native_control_numerical_unresolved=None if native is None else native["numerical_audit_unresolved"],
                    raw_numerical_unresolved=result["numerical_audit_unresolved"],
                    phase_tv=result["phase_tv"],
                    coordinates=result["best"]["coordinates"],
                    inherited_query_failure="query_failure.json" in selected["inherited"],
                    source_underflows={
                        key: selected["prediction_numerics"][key]
                        for key in ("search_underflow_calls", "replay_underflow_calls")
                    },
                )
                baseline_path = OUTPUT / "observed_baseline" / spec["noise"] / spec["world"] / "evaluation/scores.json"
                baseline = prior.read(baseline_path)
                record["teacher_advantage_over_best_observed_training"] = (
                    baseline["teacher_loss"] - strict_values["selected_free"]
                )
                record["observed_baseline_row_id"] = baseline["row_id"]
                dependencies.append(baseline_path.parent / "complete.json")
            except (ArithmeticError, ValueError) as error:
                record.update(
                    status="resource_stopped" if isinstance(error, ResourceStop) else "teacher_evaluation_failed",
                    exception=type(error).__name__,
                    message=str(error),
                    metrics=None,
                )
        target = folder / "evaluation"
        prior.write(target / "scores.json", record)
        prior.finish(target, spec | {"stage": "teacher_evaluation"}, frozen, ("scores.json",), tuple(dependencies))


def summarize(frozen: dict[str, Any]) -> None:
    rows, dependencies = [], []
    for spec in specs():
        folder = cell(OUTPUT, spec) / "evaluation"
        if not prior.receipt(folder, spec | {"stage": "teacher_evaluation"}, frozen):
            raise ValueError("All30 finite/failed evaluation receipts required")
        rows.append(prior.read(folder / "scores.json"))
        dependencies.append(folder / "complete.json")
    prior.write(
        OUTPUT / "summary.json",
        {
            "heads": len(rows),
            "records": rows,
            "finite": sum(row["status"] == "finite" for row in rows),
            "global_optimum_certified": False,
            "promotion": False,
            "gate_changes": False,
            "scope": "continuous-domain numerical incumbents; separate from the previous finite-menu problem",
        },
    )
    prior.finish(OUTPUT, {"stage": "summary"}, frozen, ("summary.json",), tuple(dependencies))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "optimize", "score", "summarize"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "freeze":
        value = identity()
        path = OUTPUT / "execution_identity.json"
        if path.exists() and prior.read(path) != value:
            raise ValueError("Preserve changed frozen identity")
        prior.write(path, value)
        return
    frozen = approved()
    if args.command == "optimize":
        select_observed_baselines(frozen)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(optimize_cell, specs(), [frozen] * len(specs())))
    elif args.command == "score":
        score_all(frozen)
        summarize(frozen)
    else:
        summarize(frozen)


if __name__ == "__main__":
    main()
