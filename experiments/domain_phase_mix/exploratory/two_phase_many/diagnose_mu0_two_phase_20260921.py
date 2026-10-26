# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Registered outcome-informed removal of the round-two phase channel."""

import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import benchmark_two_phase_20260921 as benchmark
import nextgen_two_phase_round2_20260921 as surrogate
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

MODELS = {"aggregate_native": "NG2-005", "aggregate_profile": "NG2-006"}
ROOT = surrogate.ROOT / "mu0_diagnostic"


def sources() -> dict[str, str]:
    return {
        **surrogate.source_hashes(),
        str(Path(__file__).resolve()): benchmark.sha256(Path(__file__)),
        str(ROOT / "registration.json"): benchmark.sha256(ROOT / "registration.json"),
    }


def mu0_features(original: np.ndarray) -> np.ndarray:
    """Remove every nonzero-forgetting basis without changing model search code."""
    if original.ndim != 4 or original.shape[1] != len(surrogate.FORGETTING) or surrogate.FORGETTING[0] != 0:
        raise ValueError("Expected native feature geometry with mu0 as the first state")
    return np.repeat(original[:, :1], len(surrogate.FORGETTING), axis=1)


def fit_task(arguments: tuple[str, str, str, int, str]) -> dict:
    prepared_name, output_name, target, outer, label = arguments
    prepared, output = Path(prepared_name), Path(output_name)
    key = hashlib.sha256(json.dumps([target, outer, label]).encode()).hexdigest()[:20]
    folder = output / "shards" / key
    identity = {
        "sources": sources(),
        "target": target,
        "outer_fold": outer,
        "label": label,
        "prepared_manifest": benchmark.sha256(prepared / "manifest.json"),
        "derived_features": benchmark.sha256(output / "mu0_features.npz"),
        "environment": benchmark.environment_identity(),
    }
    if (folder / "complete.json").exists():
        saved = json.loads((folder / "complete.json").read_text())
        if saved["identity"] != identity:
            raise ValueError("Completed diagnostic identity changed")
        benchmark.verify_hashes(saved["outputs"], folder)
        return {"folder": str(folder), "cached": True}
    panel = benchmark.read_panel(prepared, "starcoder_wsd80_dense")
    folds = pd.read_csv(prepared / panel.name / "folds.csv", dtype={"row_id": str})
    folds = folds[(folds.repeat == 0) & (folds.outer_fold == outer)]
    lookup = pd.Series(np.arange(len(panel.rows)), index=panel.rows.row_id)
    folds = folds.loc[np.isfinite(panel.targets[target][lookup.loc[folds.row_id].to_numpy(int)])]
    training, query = folds[folds.role.eq("train")], folds[folds.role.ne("train")]
    train, indices = lookup.loc[training.row_id].to_numpy(int), lookup.loc[query.row_id].to_numpy(int)
    with np.load(output / "mu0_features.npz") as arrays:
        values = arrays["benefits"]
    started = time.monotonic()
    with threadpool_limits(limits=1):
        fitted = surrogate.fit_model(
            panel, panel.targets[target], train, training.inner_fold.to_numpy(int), MODELS[label], values
        )
        if fitted.forgetting != 0:
            raise AssertionError("Identical state scores did not choose the registered mu0 tie-break")
        prediction = fitted.predict(panel, indices)
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite diagnostic prediction")
    frame = panel.rows.iloc[indices][["row_id", "policy_id", "group_id", "source", "region"]].copy()
    frame["model"], frame["panel"], frame["target"] = label, panel.name, target
    frame["repeat"], frame["outer_fold"], frame["role"] = 0, outer, query.role.to_numpy()
    frame["observed"], frame["predicted"] = panel.targets[target][indices], prediction
    benchmark.write_json(
        folder / "fit.json",
        {
            **fitted.to_json(),
            "diagnostic_label": label,
            "training_rows": training.row_id.tolist(),
            "inner_folds": training.inner_fold.tolist(),
            "runtime_seconds": time.monotonic() - started,
        },
    )
    frame.to_csv(folder / "predictions.csv", index=False)
    benchmark.write_json(
        folder / "complete.json",
        {
            "identity": identity,
            "outputs": {name: benchmark.sha256(folder / name) for name in ["fit.json", "predictions.csv"]},
        },
    )
    return {
        "folder": str(folder),
        "cached": False,
        "model": label,
        "target": target,
        "fold": outer,
        "runtime_seconds": time.monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=benchmark.DEFAULT_OUTPUT / "v3/prepared")
    parser.add_argument("--output", type=Path, default=ROOT)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("Workers must be positive")
    registration = json.loads((ROOT / "registration.json").read_text())
    if registration["models"] != MODELS or registration["sources"] != surrogate.source_hashes():
        raise ValueError("Registered model provenance differs from current implementation")
    manifest = json.loads((args.prepared / "manifest.json").read_text())
    benchmark.verify_hashes(manifest["sources"])
    benchmark.verify_hashes(manifest["identity"]["code"])
    benchmark.verify_hashes(manifest["outputs"], args.prepared)
    if manifest["identity"]["code"] != benchmark.code_identity():
        raise ValueError("Prepared code identity differs")
    if manifest["identity"]["environment"] != benchmark.environment_identity():
        raise ValueError("Prepared numerical environment differs")
    if manifest["identity"]["data_contract_sha256"] != benchmark.sha256(benchmark.PROTOCOL_DIR / "data_contract.json"):
        raise ValueError("Prepared data contract differs")
    panel = benchmark.read_panel(args.prepared, "starcoder_wsd80_dense")
    original = surrogate.geometry_cache(panel, surrogate.ROOT / "starcoder_screen_v1/geometry", "causal")
    values = mu0_features(original)
    recipe = {
        "sources": sources(),
        "original_cache": benchmark.sha256(
            surrogate.ROOT / "starcoder_screen_v1/geometry/starcoder_wsd80_dense_causal.npz"
        ),
        "original_array_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
        "derived_array_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "shape": list(values.shape),
        "transformation": "np.repeat(original[:, :1], 5, axis=1)",
        "read_only_original": True,
    }
    receipt_path = args.output / "derived_features.json"
    if receipt_path.exists():
        if json.loads(receipt_path.read_text()) != recipe:
            raise ValueError("Derived feature recipe changed")
        with np.load(args.output / "mu0_features.npz") as arrays:
            if hashlib.sha256(arrays["benefits"].tobytes()).hexdigest() != recipe["derived_array_sha256"]:
                raise ValueError("Derived feature array differs from its registered recipe")
    else:
        benchmark.write_json(receipt_path, recipe)
        np.savez_compressed(args.output / "mu0_features.npz", benefits=values)
    targets = benchmark.selected_targets(panel, list(benchmark.QUICK_TASKS), "screen")
    tasks = [
        (str(args.prepared), str(args.output), target, fold, label)
        for label in MODELS
        for target in targets
        for fold in range(5)
    ]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(fit_task, task) for task in tasks]):
            record = future.result()
            records.append(record)
            print(json.dumps(record), flush=True)
    frames = [
        pd.read_csv(Path(record["folder"]) / "predictions.csv", keep_default_na=False, float_precision="round_trip")
        for record in records
    ]
    predictions = pd.concat(frames, ignore_index=True).sort_values(["model", "target", "outer_fold", "row_id"])
    predictions.to_csv(args.output / "predictions.csv", index=False)
    benchmark.scoring.score_predictions(predictions).to_csv(args.output / "metrics.csv", index=False)
    benchmark.scoring.summarize_predictions(predictions).to_csv(args.output / "summary.csv", index=False)
    benchmark.write_json(
        args.output / "run.json",
        {"sources": sources(), "fits": len(records), "models": MODELS, "claim": registration["claim"], "lm_jobs": 0},
    )


if __name__ == "__main__":
    main()
