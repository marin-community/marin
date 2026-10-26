# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Static misspecification and stronger-noise stress using frozen round five.

Only synthetic truth and artifact provenance change. The private runner keeps
round five's exact fitting, strength profiles, scoring, and numerical optimizer.
No real outcomes enter synthetic labels, selection, or truth construction.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_tangent_round5_20261015 as frozen

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "reference_outputs/tpp40_tangent_stress_20261015"
TRUTH_VARIANTS = ("native", "shared_offgrid_null", "heterogeneous_rate_null")


def heterogeneous_rates(
    pool_tokens: np.ndarray, bucket_names: np.ndarray, base_rate: float, log_rate_span: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return rates and zero-based ranks, resolving equal pools by bucket name."""
    if len(pool_tokens) != len(bucket_names) or len(pool_tokens) < 2 or np.any(pool_tokens <= 0):
        raise ValueError("Heterogeneous rates require aligned names and positive pool sizes")
    order = np.lexsort((bucket_names.astype(str), np.log(pool_tokens)))
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(len(order))
    coordinate = 2.0 * ranks / (len(order) - 1) - 1.0
    return base_rate * np.exp(log_rate_span * coordinate), ranks, coordinate


def make_truth(spec: dict[str, Any], geometry: frozen.Geometry, train: np.ndarray) -> dict[str, Any]:
    """Modify only the registered static response shape of the frozen truth."""
    truth = frozen.make_truth(spec, geometry, train)
    variant = spec["truth_variant"]
    truth["truth_variant"] = variant
    if variant == "native":
        return truth
    static = _BACKEND.registration()["truth_static_shapes"][variant]
    if variant == "shared_offgrid_null":
        shape = {key: float(static[key]) for key in ("rate", "power", "threshold")}
        if shape in frozen.temporal.mariner.SHAPES:
            raise ValueError("The registered off-grid null unexpectedly belongs to the native shape menu")
        truth["shape"] = shape
        truth["static_misspecification"] = "shared shape outside the native search grid; exact zero phase effect"
        return truth
    with np.load(_BACKEND.DATA / "arrays.npz") as arrays:
        names = arrays["bucket_names"].astype(str)
    rates, ranks, coordinate = heterogeneous_rates(
        geometry.pool_tokens, names, float(static["base_rate"]), float(static["log_rate_span"])
    )
    truth["shape"] = {
        "rate": rates.tolist(),
        "power": float(static["power"]),
        "threshold": float(static["threshold"]),
    }
    truth["rate_provenance"] = {
        "bucket_names": names.tolist(),
        "pool_tokens": geometry.pool_tokens.tolist(),
        "zero_based_pool_rank": ranks.tolist(),
        "rank_coordinate": coordinate.tolist(),
        "tie_break": "ascending bucket name within equal log pool size",
        "parameters": static,
    }
    truth["static_misspecification"] = "per-bucket benefit rate; fitted families share one rate; exact zero phase effect"
    return truth


def synthetic_prediction(truth: dict[str, Any], early: np.ndarray, late: np.ndarray, alpha: float) -> np.ndarray:
    """Evaluate a serialized static vector-rate truth or the frozen native law."""
    if truth["truth_variant"] != "heterogeneous_rate_null":
        return frozen.synthetic_prediction(truth, early, late, alpha)
    total, shape = early + late, truth["shape"]
    benefit = frozen.temporal.mariner.benefit(total, np.asarray(shape["rate"]), float(shape["power"]))
    harm = frozen.temporal.mariner.harm(total, float(shape["threshold"]))
    matrix = np.hstack((-benefit, harm))
    head = truth["head"]
    predicted = float(head["floor"]) + np.exp(float(head["intercept"]) + matrix @ np.asarray(head["coefficients"]))
    if not np.isfinite(predicted).all():
        raise FloatingPointError("Heterogeneous static truth produced nonfinite outcomes")
    return predicted


def isolated_backend() -> ModuleType:
    """Bind stress truth and provenance without changing the imported runner."""
    name = f"{__name__}._frozen_round5"
    spec = importlib.util.spec_from_file_location(name, Path(frozen.__file__))
    assert spec is not None and spec.loader is not None
    backend = importlib.util.module_from_spec(spec)
    sys.modules[name] = backend
    spec.loader.exec_module(backend)
    backend.OUTPUT = OUTPUT
    original_registration = backend.registration
    original_identity = backend.source_identity

    def registration() -> dict[str, Any]:
        value = original_registration()
        for item in value["contexts"]:
            if item["truth_mode"] not in ("null", "fixed_strength"):
                raise ValueError("This stress protocol only supports null and fixed-strength truths")
            variant = item["truth_variant"]
            if variant not in TRUTH_VARIANTS:
                raise ValueError(f"Unknown static truth variant: {variant}")
            if variant != "native" and (item["truth_mode"] != "null" or item["truth_family"] != "aggregate_mariner"):
                raise ValueError("Misspecified static truths must retain zero phase effect")
        return value

    def source_identity() -> dict[str, str]:
        identity = original_identity()
        identity[str(Path(__file__).resolve())] = frozen.benchmark.sha256(Path(__file__))
        return dict(sorted(identity.items()))

    backend.registration = registration
    backend.source_identity = source_identity
    backend.make_truth = make_truth
    backend.synthetic_prediction = synthetic_prediction
    return backend


_BACKEND = isolated_backend()


def run_context(spec: dict[str, Any]) -> dict[str, Any]:
    """Enter the unchanged round-five fitting procedure in a spawn-safe worker."""
    return _BACKEND.run_context(spec)


def optimize_context(spec: dict[str, Any]) -> dict[str, Any]:
    """Enter the unchanged raw-policy and known-truth audit."""
    return _BACKEND.optimize_context(spec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "summarize", "optimize", "summarize-optimization"))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--context")
    args = parser.parse_args()
    if args.command == "prepare":
        _BACKEND.prepare()
        return
    if args.command == "summarize":
        _BACKEND.summarize()
        return
    if args.command == "summarize-optimization":
        _BACKEND.summarize_optimization()
        return
    if args.workers < 1:
        raise ValueError("Worker count must be positive")
    _BACKEND.execution_identity()
    contexts = _BACKEND.registration()["contexts"]
    if args.context is not None:
        contexts = [item for item in contexts if item["name"] == args.context]
        if not contexts:
            raise ValueError(f"Unknown registered stress context: {args.context}")
    worker = optimize_context if args.command == "optimize" else run_context
    if args.command == "optimize" and args.workers != 1:
        raise ValueError("Raw optimization runs sequentially to preserve shared truth receipts")
    if args.workers == 1:
        for spec in contexts:
            print(json.dumps(worker(spec), sort_keys=True), flush=True)
        return
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(worker, spec) for spec in contexts]
        for future in as_completed(futures):
            print(json.dumps(future.result(), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
