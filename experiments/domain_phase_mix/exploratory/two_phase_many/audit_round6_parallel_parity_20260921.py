# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Check runtime-only shape parallelism against a preserved serial unit fit."""

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import nextgen_two_phase_round6_20260921 as runner
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def restored_fit(payload: dict[str, Any]) -> runner.Fit:
    head = runner.log_head.Head(
        payload["intercept"],
        np.asarray(payload["aggregate_coefficients"]),
        np.asarray(payload["phase_coefficients"]),
        payload["floor"],
    )
    return runner.Fit(
        payload["model"],
        payload["shape"],
        payload["q"],
        payload["kappa"],
        payload["ridge"],
        payload["inner_rmse"],
        head,
        payload["floor_search"],
    )


def audit(root: Path, workers: int) -> None:
    sources = runner.source_hashes() | {str(Path(__file__).resolve()): runner.benchmark.sha256(Path(__file__))}
    names = (
        "serial_reference.json",
        "serial_reference_receipt.json",
        "serial_fixture.npz",
        "serial_fixture_rows.csv",
        "serial_fixture_metadata.json",
        "parallel_clean_fit.json",
        "parallel_clean_run.json",
    )
    inputs = {name: runner.benchmark.sha256(root / name) for name in names}
    identity = {
        "sources": sources,
        "inputs": inputs,
        "workers": workers,
        "environment": runner.benchmark.environment_identity(),
    }
    completion = root / "parallel_parity.json"
    if completion.exists():
        saved = json.loads(completion.read_text())
        if saved["identity"] != identity:
            raise ValueError("Parallel parity inputs changed; retain the earlier receipt")
        runner.benchmark.verify_hashes(saved["outputs"], root)
        if saved["passed"] is not True:
            raise ValueError("The preserved parallel parity check failed")
        print(json.dumps({"cached": True, "passed": True}))
        return

    metadata = json.loads((root / "serial_fixture_metadata.json").read_text())
    with np.load(root / "serial_fixture.npz") as fixture:
        arrays = {key: fixture[key] for key in fixture.files}
    panel = runner.benchmark.data.BenchPanel(
        metadata["name"],
        tuple(metadata["buckets"]),
        arrays["phase0"],
        arrays["phase1"],
        metadata["alpha"],
        arrays["c0"],
        arrays["c1"],
        {},
        pd.read_csv(root / "serial_fixture_rows.csv", dtype={"row_id": str}),
        metadata["metadata"],
        metadata["input_hashes"],
    )
    poisoned = arrays["response"].copy()
    excluded = np.setdiff1d(np.arange(len(poisoned)), arrays["train"])
    poisoned[excluded] = np.resize([np.nan, np.inf, -np.inf, -1e12], len(excluded))
    poisoned_path = root / "parallel_poisoned_fit.json"
    checkpoint = root / "parallel_poisoned_complete.json"
    reused_fit = checkpoint.exists()
    if reused_fit:
        cached = json.loads(checkpoint.read_text())
        if cached["identity"] != identity:
            raise ValueError("Parallel poisoned-fit checkpoint differs from the current identity")
        runner.benchmark.verify_hashes(cached["outputs"], root)
        elapsed = cached["elapsed_seconds"]
        fitted = restored_fit(json.loads(poisoned_path.read_text()))
    else:
        started = time.monotonic()
        with threadpool_limits(limits=1):
            fitted = runner.fit_model(
                dataclasses.replace(panel, targets={"poison": poisoned}),
                poisoned,
                arrays["train"],
                arrays["inner"],
                "NG2-016",
                workers=workers,
            )
        elapsed = time.monotonic() - started
        runner.benchmark.write_json(poisoned_path, fitted.to_json())
        runner.benchmark.write_json(
            checkpoint,
            {
                "identity": identity,
                "elapsed_seconds": elapsed,
                "outputs": {poisoned_path.name: runner.benchmark.sha256(poisoned_path)},
            },
        )

    serial = json.loads((root / "serial_reference.json").read_text())
    clean = json.loads((root / "parallel_clean_fit.json").read_text())
    predictions = fitted.predict(panel, arrays["query"])
    clean_predictions = restored_fit(clean).predict(panel, arrays["query"])
    dictionary_equal = json.loads(json.dumps(fitted.to_json(), allow_nan=False)) == serial == clean
    canonical = [
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        for value in (serial, clean, fitted.to_json())
    ]
    json_equal = canonical[0] == canonical[1] == canonical[2]
    prediction_equal = predictions.tobytes() == arrays["query_prediction"].tobytes() == clean_predictions.tobytes()
    runner.benchmark.verify_hashes(sources)
    runner.benchmark.verify_hashes(inputs, root)
    result = {
        "identity": identity,
        "passed": dictionary_equal and json_equal and prediction_equal,
        "full_fit_dictionaries_identical": dictionary_equal,
        "canonical_json_bytes_identical": json_equal,
        "query_predictions_identical": prediction_equal,
        "max_poisoned_query_error": float(np.max(np.abs(predictions - arrays["query_prediction"]))),
        "max_clean_query_error": float(np.max(np.abs(clean_predictions - arrays["query_prediction"]))),
        "poisoned_parallel_seconds": elapsed,
        "poisoned_fit_reused": reused_fit,
        "claim": "Runtime-only parallel parity on an arbitrary unit fixture; no scientific outcome fits",
        "outputs": {name: runner.benchmark.sha256(root / name) for name in (poisoned_path.name, checkpoint.name)},
    }
    runner.benchmark.write_json(completion, result)
    print(json.dumps({key: value for key, value in result.items() if key not in ("identity", "outputs")}, indent=2))
    if not result["passed"]:
        raise AssertionError("Parallel clean/poisoned fitting differs from the exact serial reference")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("Workers must be positive")
    audit(args.root, args.workers)


if __name__ == "__main__":
    main()
