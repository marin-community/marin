# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "threadpoolctl"]
# ///
"""Profile invented prefix extensions of fixed TP038 function pairs; no empirical labels."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import log_ndtr

from experiments.domain_phase_mix.exploratory.two_phase_many.tpp40_response_profile_20261015 import linear_profile

BASE = Path(__file__).resolve().parent
PARENT = BASE / "reference_outputs/tpp40_acquisition_retention_20261015"
INPUT = PARENT / "gaussian_pair_diagnostic"
OUTPUT = PARENT / "prefix_auxiliary_pair_diagnostic"
DATA = BASE / "reference_outputs/two_phase_benchmark_20260921/v3/prepared/starcoder_wsd80_dense"
SEALS = {
    "gaussian_pair_diagnostic/result_manifest_v1.json": (
        "d9dba546a0e2a0d1a3b9c6beb390a6a90306ff0b65cc672efb92f9cafb1b10a3"
    ),
    "stage1/result_manifest.json": "a8c38029614a5df658743f2a551f38b97f44fe7fcc1590a430b3ac91762b6718",
    "known_floor_diagnostic/result_manifest_v1.json": "8d6c3fafd825b432e76dc270c1f2c3dbb63c068bd1af0d3ef942f8637ae626e0",
    "known_floor_all_folds/result_manifest_v1.json": "3cf46cfc976d35a11993fbffb084d1e4ee7a5d925994a73db76536dfad0300bc",
    "response_profile/result_manifest_v1.json": "c97acfbf0434a4a2fd818983d966781718d6b6388b828c73a6492e046783a4ed",
    "unchanged_j2_diagnostic/result_manifest_v1.json": (
        "381233df0f94dae9927a1a1983687b3f1edeefa3b93d149a9770d8ae01d74ec6"
    ),
}
SHAPE = {"rate": 0.25, "power": 0.7, "threshold": 3.0}
AMPLITUDES = [1.0, 3.0, 0.1, 0.1]
BUCKETS = ["nemotron_full", "starcoder"]
RHOS = (0.0, 0.5, 0.9)
PREFIX_NOISE_RATIOS = (1.0, 3.0, 10.0)
SIGMA = 0.003
CONTRACT = {
    "candidate_id": "TP40-045",
    "pairs": 450,
    "common_family_inventory": 327,
    "heterogeneous_inventory": 123,
    "profiles": 4050,
    "correlations": list(RHOS),
    "prefix_noise_ratios": list(PREFIX_NOISE_RATIOS),
    "sigma_final": SIGMA,
    "prefix_teacher": {"shape": SHAPE, "amplitudes": AMPLITUDES, "floor": 0.6, "intercept": 1.0},
    "heterogeneous_log_rate_span": 0.25,
    "sharing": "saved_canonical_native_amplitudes_and_shape_not_intrinsic_endpoint_function_equivalence",
    "profile": "closed_nonnegative_prefix_floor_and_scale_at_each_covariance_shifted_target",
    "covariance_target": "true_prefix + rho * (sigma_prefix / sigma_final) * delta_final",
    "distance": "sum(u**2) + sum((v-rho*u)**2)/(1-rho**2)",
    "prefix_coordinate": "saved_phase0_epochs_without_another_alpha_factor",
    "hypothetical_noise": "iid_across_runs_common_known_within_run_gaussian_covariance_under_both_hypotheses",
    "functional_bound": "delta**2/2 * Phi(-joint_distance/2); latent_endpoint_or_gain_only",
    "scale_zero": "included_closed_family_boundary_not_finite_log_intercept",
    "duplicates": "retain_all_ancestry_and_parameter_representations_no_independence_or_collapsing",
    "new_empirical_labels": False,
    "new_endpoint_functions_or_policy_queries": False,
    "primary_benchmark_unchanged": True,
    "original_learned_stage_remains_blocked": True,
    "scientific_gate": None,
    "scope": "removal_or_survival_of_existing_fixed_pair_witnesses_under_an_unverified_shared_prefix_model",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def immutable_write(path: Path, content: bytes) -> None:
    """Resume exact completed outputs and preserve unexplained partial artifacts."""
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Existing output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("xb") as stream:
        stream.write(content)
    temporary.rename(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def inherited_sources(expected: dict[str, str]) -> dict[str, str]:
    """Adopt the historical helper code, refusing live replacements."""
    for name, digest in expected.items():
        if sha(Path(name)) != digest:
            raise ValueError(f"Inherited source changed: {name}")
    return dict(expected)


def verify_seals() -> dict[str, int]:
    counts = {}
    for name, expected in SEALS.items():
        seal = PARENT / name
        if sha(seal) != expected:
            raise ValueError(f"Changed prerequisite seal: {seal}")
        files = read_json(seal)["files"]
        for relative, record in files.items():
            path = seal.parent / relative
            if sha(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
                raise ValueError(f"Changed prerequisite file: {path}")
        counts[name] = len(files)
    return counts


def parameter_source(head: dict[str, Any]) -> tuple[Path, str, str]:
    source, fold, world, noise, candidate = (
        head[key] for key in ("source", "outer_fold", "world", "noise", "candidate")
    )
    if source == "known_floor_search":
        folder = (
            PARENT / "known_floor_diagnostic/cells" if fold == 0 else PARENT / f"known_floor_all_folds/cells/outer{fold}"
        )
        return folder / world / str(noise) / candidate / "fit.json", "head.coefficients", "shape"
    if source == "unchanged_j2":
        return PARENT / "unchanged_j2_diagnostic/cells" / world / candidate / "model.json", "head.coefficients", "shape"
    if source != "response_profile":
        raise ValueError(f"Unknown source: {source}")
    field = "profile.head.coefficients" if candidate == "known_floor_log_nnls" else "selected.amplitudes"
    return (
        PARENT / f"response_profile/cells/outer{fold}" / str(noise) / world / candidate / "fit.json",
        field,
        "stage1.SHAPE",
    )


def nested(value: dict[str, Any], field: str) -> Any:
    for key in field.split("."):
        value = value[key]
    return value


def ancestry(shape: dict[str, float], amplitudes: np.ndarray, world: str) -> dict[str, Any]:
    """Record exact canonical equality; no tolerance-based function deduplication."""
    same_shape = shape == SHAPE and world != "heterogeneous_null"
    same_amplitudes = bool(np.array_equal(amplitudes, AMPLITUDES))
    return {
        "same_native_shape_as_teacher": same_shape,
        "same_amplitudes_as_teacher": same_amplitudes,
        "kinetics_or_final_level_only_difference": same_shape and same_amplitudes,
        "amplitude_max_abs_difference_from_teacher": float(np.max(abs(amplitudes - AMPLITUDES))),
        "common_shape_parameter_differences": {key: shape[key] - SHAPE[key] for key in SHAPE},
        "heterogeneous_teacher_has_bucket_specific_rates": world == "heterogeneous_null",
    }


def parameter_inventory() -> list[dict[str, Any]]:
    """Read canonical parameter metadata only; no policy arrays or prefix features."""
    heads = pd.read_csv(INPUT / "heads.csv", keep_default_na=False, float_precision="round_trip")
    if len(heads) != 450 or heads.pair_id.duplicated().any() or int(heads.common_family_member.sum()) != 327:
        raise ValueError("Unexpected predecessor inventory")
    result = []
    for head in heads.to_dict("records"):
        path, field, shape_field = parameter_source(head)
        model = read_json(path)
        amplitudes = np.asarray(nested(model, field), dtype=float)
        shape = SHAPE if shape_field == "stage1.SHAPE" else model["shape"]
        if amplitudes.shape != (4,) or not np.isfinite(amplitudes).all() or np.any(amplitudes < 0):
            raise ValueError(f"Invalid amplitudes: {path}")
        if set(shape) != set(SHAPE) or not all(math.isfinite(value) and value > 0 for value in shape.values()):
            raise ValueError(f"Invalid shape: {path}")
        result.append(
            {
                "head": head,
                "source_path": str(path),
                "source_sha256": sha(path),
                "cell_receipt_path": str(path.parent / "complete.json"),
                "cell_receipt_sha256": sha(path.parent / "complete.json"),
                "amplitude_field": field,
                "shape_field": shape_field,
                "shape": shape,
                "amplitudes": amplitudes.tolist(),
                "ancestry": ancestry(shape, amplitudes, head["world"]),
                "final_temporal_parameters_metadata_only": model.get("parameters", {}),
            }
        )
    if heads.groupby("source").size().to_dict() != {
        "known_floor_search": 200,
        "response_profile": 240,
        "unchanged_j2": 10,
    }:
        raise ValueError("Unexpected source counts")
    return result


def registration_contents() -> dict[str, Any]:
    teacher = read_json(PARENT / "stage1/registration.json")["contract"]
    if (
        teacher["shape"] != SHAPE
        or teacher["head"] != {"coefficients": AMPLITUDES, "floor": 0.6, "intercept": 1.0}
        or teacher["heterogeneous_log_rate_span"] != 0.25
    ):
        raise ValueError("Inherited teacher specification changed")
    inventory = parameter_inventory()
    prior = read_json(INPUT / "registration.json")
    source_paths = [
        Path(__file__),
        BASE / "test_tpp40_prefix_auxiliary_pairs_20261015.py",
        BASE / "tpp40_response_profile_20261015.py",
        OUTPUT / "proposal_v2.md",
        OUTPUT / "source_plan_v2.md",
    ]
    sources = inherited_sources(prior["source_hashes"])
    sources.update({str(path): sha(path) for path in source_paths})
    inputs = [INPUT / name for name in ("heads.csv", "training_rows.csv", "endpoints.csv", "gains.csv")]
    inputs.extend(DATA / name for name in ("metadata.json", "policies.npz", "rows.csv", "folds.csv"))
    inputs.extend(PARENT / name for name in SEALS)
    inputs.append(PARENT / "stage1/registration.json")
    for item in inventory:
        inputs.extend(Path(item[key]) for key in ("source_path", "cell_receipt_path"))
    return {
        "contract": CONTRACT,
        "runtime": (
            {"python": sys.version} | {key: importlib.metadata.version(key) for key in ("numpy", "pandas", "scipy")}
        ),
        "source_hashes": sources,
        "input_hashes": {str(path): sha(path) for path in inputs},
        "parameter_inventory_sha256": hashlib.sha256(json_bytes(inventory)).hexdigest(),
    }


def identity() -> dict[str, Any]:
    value = read_json(OUTPUT / "registration.json")
    current = registration_contents()
    if value.get("status") != "registered" or any(value.get(key) != item for key, item in current.items()):
        raise ValueError("Registration, sources, input hashes, canonical parameters or runtime changed")
    verify_seals()
    return current | {"registration_sha256": sha(OUTPUT / "registration.json")}


def native_logits(
    epochs: np.ndarray, shape: dict[str, float], amplitudes: np.ndarray, rates: np.ndarray | None = None
) -> np.ndarray:
    """Native signed benefit/harm features at physical prefix exposure."""
    if epochs.ndim != 2 or epochs.shape[1] != 2 or np.any(epochs < 0) or not np.isfinite(epochs).all():
        raise ValueError("Expected finite nonnegative two-bucket epochs")
    rate = shape["rate"] if rates is None else rates
    benefit = -np.expm1(-((rate * epochs) ** shape["power"]))
    harm = np.logaddexp(0.0, np.log1p(epochs) - shape["threshold"]) ** 2
    result = np.column_stack((-benefit, harm)) @ amplitudes
    if not np.isfinite(result).all():
        raise ArithmeticError("Nonfinite native prefix logit")
    return result


def prefix_truth(epochs: np.ndarray, scales: np.ndarray, world: str) -> np.ndarray:
    rates = None
    if world == "heterogeneous_null":
        order = np.lexsort((np.asarray(BUCKETS), np.log(1 / scales)))
        coordinate = np.empty(2)
        coordinate[order] = [-1, 1]
        rates = SHAPE["rate"] * np.exp(0.25 * coordinate)
    logit = 1.0 + native_logits(epochs, SHAPE, np.asarray(AMPLITUDES), rates)
    with np.errstate(over="raise", invalid="raise"):
        deficit = np.exp(logit)
    if np.any(deficit == 0) or not np.isfinite(deficit).all():
        raise ArithmeticError("Prefix teacher response underflow/nonfinite")
    return 0.6 + deficit


def profiled_extension(
    logits: np.ndarray, truth_prefix: np.ndarray, delta_final: np.ndarray, rho: float, prefix_noise_ratio: float
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Find the closest closed shared-amplitude prefix extension for a fixed covariance."""
    if (
        not math.isfinite(prefix_noise_ratio)
        or prefix_noise_ratio <= 0
        or not 0 <= rho < 1
        or logits.ndim != 1
        or not len(logits)
        or logits.shape != truth_prefix.shape
        or logits.shape != delta_final.shape
    ):
        raise ValueError("Invalid covariance or aligned vector shapes")
    if not all(np.isfinite(value).all() for value in (logits, truth_prefix, delta_final)):
        raise ArithmeticError("Nonfinite prefix profile inputs")
    shift = float(np.max(logits))
    phi = np.exp(logits - shift)
    underflow = int(np.sum(phi == 0))
    if underflow:
        raise ArithmeticError(f"Normalized prefix exponential underflow: {underflow} rows")
    target = truth_prefix + rho * prefix_noise_ratio * delta_final
    profile = linear_profile(phi, target, None)
    u = delta_final / SIGMA
    v = (profile.prediction - truth_prefix) / (SIGMA * prefix_noise_ratio)
    conditional = (v - rho * u) / math.sqrt(1 - rho**2)
    d_final = float(np.linalg.norm(u))
    d_conditional = float(np.linalg.norm(conditional))
    distance = math.hypot(d_final, d_conditional)
    log_error = float(log_ndtr(-distance / 2))
    if (
        not all(math.isfinite(x) for x in (distance, log_error, profile.floor, profile.scale))
        or not np.isfinite(profile.prediction).all()
    ):
        raise ArithmeticError("Nonfinite profile, distance or Gaussian log tail")
    result = {
        "floor": float(profile.floor),
        "normalized_scale": float(profile.scale),
        "logit_shift": shift,
        "log_original_scale": None if profile.scale == 0 else math.log(profile.scale) - shift,
        "scale_zero_closure": profile.scale == 0,
        "face": profile.face,
        "constant": bool(profile.constant),
        "near_constant": bool(profile.near_constant),
        "singular_values": profile.singular_values.tolist(),
        "centered_variance": profile.centered_variance,
        "prefix_underflow_count": underflow,
        "distance_final": d_final,
        "distance_prefix_only": float(np.linalg.norm(v)),
        "prefix_final_cosine": (
            None if d_final == 0 or np.linalg.norm(v) == 0 else float((u / d_final) @ (v / np.linalg.norm(v)))
        ),
        "distance_conditional": d_conditional,
        "distance_joint": distance,
        "distance_common_prefix_reference": d_final / math.sqrt(1 - rho**2),
        "distance_unrestricted_prefix_nuisance_reference": d_final,
        "bayes_error_joint": math.exp(log_error),
        "log_bayes_error_joint": log_error,
        "profile_sse": float(np.sum((profile.prediction - target) ** 2)),
        "prefix_numeric_status": "finite",
        "prefix_decomposition_unidentified": bool(profile.near_constant),
    }
    return result, {
        "truth_prefix": truth_prefix,
        "candidate_prefix": profile.prediction,
        "shifted_target": target,
        "standardized_final_difference": u,
        "standardized_prefix_difference": v,
        "whitened_conditional_difference": conditional,
    }


def scalar_bound(separation: float, log_error: float) -> dict[str, Any]:
    if separation == 0:
        return {"pair_mse_lower_bound": 0.0, "pair_mse_log_lower_bound": None, "bound_underflowed": False}
    log_bound = 2 * math.log(abs(separation)) - math.log(2) + log_error
    bound = math.exp(log_bound)
    return {"pair_mse_lower_bound": bound, "pair_mse_log_lower_bound": log_bound, "bound_underflowed": bound == 0}


def cell_result(
    item: dict[str, Any],
    train: pd.DataFrame,
    epochs: np.ndarray,
    scales: np.ndarray,
    rho: float,
    prefix_noise_ratio: float,
) -> dict[str, Any]:
    head = item["head"]
    result = {
        "head": head,
        "canonical_parameters": item,
        "rho": rho,
        "prefix_noise_ratio": prefix_noise_ratio,
        "status": "failed",
        "analytic_joint_pair_eligible": False,
    }
    try:
        teacher = prefix_truth(epochs, scales, head["world"])
        logits = native_logits(epochs, item["shape"], np.asarray(item["amplitudes"]))
        delta = train.candidate_mean.to_numpy(float) - train.truth.to_numpy(float)
        profile, arrays = profiled_extension(logits, teacher, delta, rho, prefix_noise_ratio)
        if not math.isclose(profile["distance_final"], head["distance"], rel_tol=1e-12, abs_tol=1e-12):
            raise ArithmeticError("Final training distance changed")
        result.update(profile)
        result.update(
            {
                "status": "finite",
                "analytic_joint_pair_eligible": bool(head["analytic_pair_eligible"]),
                "rows": train[["source_index", "row_id"]].to_dict("records"),
                "prefix_arrays": {key: value.tolist() for key, value in arrays.items()},
            }
        )
    except (ArithmeticError, ValueError) as error:
        result["failure"] = {"type": type(error).__name__, "message": str(error)}
    return result


def verified_cell(folder: Path, execution: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any] | None:
    receipt_path = folder / "complete.json"
    if not receipt_path.exists():
        if folder.exists() and any(folder.iterdir()):
            raise ValueError(f"Unexplained incomplete cell preserved: {folder}")
        return None
    receipt = read_json(receipt_path)
    if (
        receipt["execution_identity_sha256"] != hashlib.sha256(json_bytes(execution)).hexdigest()
        or receipt["spec"] != spec
        or sha(folder / "result.json") != receipt["result_sha256"]
    ):
        raise ValueError(f"Changed cell or provenance: {folder}")
    return read_json(folder / "result.json")


def scientific_inputs() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], np.ndarray, np.ndarray]:
    metadata = read_json(DATA / "metadata.json")
    if metadata["buckets"] != BUCKETS:
        raise ValueError("Named bucket order changed")
    rows = pd.read_csv(DATA / "rows.csv", usecols=["row_id", "tied"])
    folds = pd.read_csv(DATA / "folds.csv", usecols=["repeat", "outer_fold", "row_id", "role"])
    train = pd.read_csv(INPUT / "training_rows.csv", keep_default_na=False, float_precision="round_trip")
    with np.load(DATA / "policies.npz", allow_pickle=False) as archive:
        early = archive["phase0_epochs"].copy()
        scales = archive["c0"] + archive["c1"]
    if scales.shape != (2,) or np.any(scales <= 0):
        raise ValueError("Invalid native exposure scales")
    groups = {str(pair): value.copy() for pair, value in train.groupby("pair_id", sort=False)}
    for pair, value in groups.items():
        outer = int(pair.split("__")[1])
        part = folds.loc[(folds["repeat"] == 0) & (folds.outer_fold == outer) & (folds.role == "train")]
        indices = value.source_index.to_numpy(int)
        if (
            len(indices) != len(set(indices))
            or rows.iloc[indices].tied.any()
            or value.row_id.tolist() != rows.iloc[indices].row_id.tolist()
            or set(value.row_id) != set(part.row_id)
        ):
            raise ValueError(f"Changed training alignment: {pair}")
    return rows, groups, early, scales


def run() -> None:
    execution = identity()
    if read_json(OUTPUT / "execution_identity.json") != execution:
        raise ValueError("Frozen execution identity changed")
    approval = read_json(OUTPUT / "approval.json")
    if approval != {
        "approved": True,
        "execution_identity_sha256": sha(OUTPUT / "execution_identity.json"),
        "scope": "4050_fixed_pair_prefix_profiles",
    }:
        raise ValueError("Exact execution approval is absent")
    inventory = parameter_inventory()
    immutable_write(OUTPUT / "parameter_inventory.json", json_bytes(inventory))
    _, groups, early, scales = scientific_inputs()
    if set(groups) != {item["head"]["pair_id"] for item in inventory}:
        raise ValueError("Training and parameter inventories differ")
    for item in inventory:
        pair = item["head"]["pair_id"]
        train = groups[pair]
        for rho in RHOS:
            for prefix_noise_ratio in PREFIX_NOISE_RATIOS:
                folder = OUTPUT / "cells" / pair / str(rho) / str(prefix_noise_ratio)
                spec = {
                    "pair_id": pair,
                    "rho": rho,
                    "prefix_noise_ratio": prefix_noise_ratio,
                    "parameter_sha256": hashlib.sha256(json_bytes(item)).hexdigest(),
                }
                if verified_cell(folder, execution, spec) is not None:
                    continue
                result = cell_result(
                    item, train, early[train.source_index.to_numpy(int)], scales, rho, prefix_noise_ratio
                )
                immutable_write(folder / "result.json", json_bytes(result))
                immutable_write(
                    folder / "complete.json",
                    json_bytes(
                        {
                            "execution_identity_sha256": hashlib.sha256(json_bytes(execution)).hexdigest(),
                            "spec": spec,
                            "result_sha256": sha(folder / "result.json"),
                        }
                    ),
                )


def summarize() -> None:
    execution = identity()
    rows, prefixes, functionals = [], [], []
    endpoints = pd.read_csv(INPUT / "endpoints.csv", keep_default_na=False, float_precision="round_trip")
    gains = pd.read_csv(INPUT / "gains.csv", keep_default_na=False, float_precision="round_trip")
    endpoint_groups = dict(tuple(endpoints.groupby("pair_id", sort=False)))
    gain_groups = dict(tuple(gains.groupby("pair_id", sort=False)))
    for item in parameter_inventory():
        head = item["head"]
        pair = head["pair_id"]
        for rho in RHOS:
            for prefix_noise_ratio in PREFIX_NOISE_RATIOS:
                spec = {
                    "pair_id": pair,
                    "rho": rho,
                    "prefix_noise_ratio": prefix_noise_ratio,
                    "parameter_sha256": hashlib.sha256(json_bytes(item)).hexdigest(),
                }
                result = verified_cell(OUTPUT / "cells" / pair / str(rho) / str(prefix_noise_ratio), execution, spec)
                if result is None:
                    raise ValueError(f"Missing cell: {pair}, rho={rho}")
                row = head | {
                    key: value
                    for key, value in result.items()
                    if key not in {"head", "canonical_parameters", "rows", "prefix_arrays"}
                }
                row["source_parameter_path"] = item["source_path"]
                row["source_parameter_sha256"] = item["source_sha256"]
                row.update(item["ancestry"])
                rows.append(row)
                if result["status"] != "finite":
                    continue
                for i, training in enumerate(result["rows"]):
                    prefixes.append(
                        training
                        | {"pair_id": pair, "rho": rho, "prefix_noise_ratio": prefix_noise_ratio}
                        | {key: values[i] for key, values in result["prefix_arrays"].items()}
                    )
                for kind, table in (("endpoint", endpoint_groups[pair]), ("gain", gain_groups[pair])):
                    for old in table.to_dict("records"):
                        record = {
                            "pair_id": pair,
                            "rho": rho,
                            "prefix_noise_ratio": prefix_noise_ratio,
                            "functional_kind": kind,
                            "source_index": old["source_index"],
                            "row_id": old["row_id"],
                            "role": old.get("role", "paired_gain"),
                            "signed_separation": old["signed_separation"],
                            "original_pair_eligible": head["analytic_pair_eligible"],
                            "analytic_joint_pair_eligible": result["analytic_joint_pair_eligible"],
                            "old_distance": head["distance"],
                            "joint_distance": result["distance_joint"],
                            "original_numerical_status": head["original_numerical_status"],
                        }
                        if result["analytic_joint_pair_eligible"]:
                            record.update(scalar_bound(old["signed_separation"], result["log_bayes_error_joint"]))
                        else:
                            record.update(
                                {
                                    "pair_mse_lower_bound": None,
                                    "pair_mse_log_lower_bound": None,
                                    "bound_underflowed": False,
                                }
                            )
                        functionals.append(record)
    table = pd.DataFrame(rows)
    if len(table) != 4050 or table[["pair_id", "rho", "prefix_noise_ratio"]].duplicated().any():
        raise ValueError("Incomplete profile inventory")
    summary = {
        "pairs": 450,
        "profiles": len(rows),
        "finite": int((table.status == "finite").sum()),
        "failures": int((table.status != "finite").sum()),
        "prefix_rows": len(prefixes),
        "functional_rows": len(functionals),
        "scientific_gate": None,
        "promotion": False,
        "original_learned_stage_ready": False,
        "canonical_parameter_sharing_not_function_equivalence": True,
        "duplicates_not_independent": True,
    }
    for name, values in (("profiles.csv", rows), ("prefix_rows.csv", prefixes), ("functionals.csv", functionals)):
        immutable_write(OUTPUT / name, pd.DataFrame(values).to_csv(index=False).encode())
    immutable_write(OUTPUT / "summary.json", json_bytes(summary))
    immutable_write(
        OUTPUT / "complete.json",
        json_bytes(
            {
                "identity": execution,
                "outputs": {
                    name: sha(OUTPUT / name)
                    for name in (
                        "parameter_inventory.json",
                        "profiles.csv",
                        "prefix_rows.csv",
                        "functionals.csv",
                        "summary.json",
                    )
                },
            }
        ),
    )
    print(json.dumps(summary))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["draft", "freeze", "run", "summarize"])
    args = parser.parse_args()
    if args.mode == "draft":
        verify_seals()
        immutable_write(OUTPUT / "registration.draft.json", json_bytes({"status": "draft", **registration_contents()}))
    elif args.mode == "freeze":
        immutable_write(OUTPUT / "execution_identity.json", json_bytes(identity()))
    elif args.mode == "run":
        run()
    else:
        summarize()


if __name__ == "__main__":
    main()
