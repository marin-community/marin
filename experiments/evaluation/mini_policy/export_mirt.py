# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export training-only IRT state and verify reconstruction of held-out predictions."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from rigging.filesystem.buckets import filesystem_for

from experiments.evaluation.mini_policy.experiment import (
    CONTINUOUS,
    SEED,
    read_bytes,
    retained_benchmark_indices,
    retained_responses,
)
from experiments.evaluation.mini_policy.methods import estimate_ability, fit_mirt, predict_panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    settings = json.loads(read_bytes(args.study + "/settings.json"))
    raw_tasks = read_bytes(settings["source"] + "/matrix/tasks.json")
    raw_matrix = read_bytes(settings["source"] + "/matrix/responses.npz")
    if hashlib.sha256(raw_tasks + raw_matrix).hexdigest() != settings["data_sha256"]:
        raise ValueError("Study matrix no longer matches the frozen input hash")
    metadata = json.loads(raw_tasks)
    arrays = np.load(io.BytesIO(raw_matrix))
    task_benchmarks = np.asarray([row["benchmark"] for row in metadata["tasks"]])
    primary = [metadata["benchmarks"][j] for j in retained_benchmark_indices(metadata["benchmarks"])]
    if settings["target_benchmarks"] != primary:
        raise ValueError("Study target scope differs from the current 21-benchmark scope")
    fit_response = retained_responses(arrays["responses"], task_benchmarks)
    repetitions = {"aime24": 10, "olympiadbench": 10, "gpqa-diamond": 3, "ot-tblite-recovery": 3, "tb2-recovery": 3}
    count_audit = []
    for benchmark in metadata["benchmarks"]:
        counts = arrays["counts"][:, task_benchmarks == benchmark]
        expected = repetitions.get(benchmark, 1)
        excessive = np.argwhere(counts > expected)
        count_audit.append(
            {
                "benchmark": benchmark,
                "expected_maximum": expected,
                "observed_maximum": int(counts.max()),
                "excessive_cells": len(excessive),
            }
        )
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    payload = json.dumps(count_audit, indent=2).encode()
    (output / "repetition-audit.json").write_bytes(payload)
    fs, key = filesystem_for(args.destination + "/repetition-audit.json")
    fs.pipe_file(key, payload)
    if any(row["excessive_cells"] for row in count_audit):
        raise ValueError("Historical trial multiplicity exceeds the declared policy")
    candidates = np.asarray(
        [
            j
            for j, task in enumerate(metadata["tasks"])
            if task["benchmark"] not in metadata["predictor_exclusions"]
            and (task["benchmark"] != "mrcr" or int(task["strata"]["mrcr_bin_upper"]) <= 16384)
        ]
    )
    contexts = {
        row["model"]: int(row["serving"]["max_model_len"])
        for row in json.loads(read_bytes(settings["source"] + "/assembly_audit.json"))
        if row["benchmark"] == "mrcr"
    }
    receipts = []
    for fold, split in enumerate(settings["splits"]):
        train, test = np.asarray(split["train"]), np.asarray(split["test"])
        original = json.loads(read_bytes(args.study + f"/fold-{fold}-irt-fit.json"))
        model = fit_mirt(
            fit_response[train],
            arrays["counts"][train],
            task_benchmarks,
            np.isin(task_benchmarks, list(CONTINUOUS)),
            original["chosen_dimension"],
            SEED + fold,
            maxiter=6000,
        )
        if not model.converged:
            raise ArithmeticError("Exported IRT fit did not converge")
        pool = candidates[np.mean(np.isfinite(arrays["responses"][train][:, candidates]), axis=0) >= 0.75]
        differences = []
        for budget in (210, 840):
            archived = json.loads(read_bytes(args.study + f"/fold-{fold}-mirt_static-0-{budget}.json"))
            points = []
            for position, i in enumerate(test):
                selected = np.asarray(archived["probe_manifests"][position]["selected"])
                eligible = np.asarray(
                    [
                        task["benchmark"] != "mrcr"
                        or int(task["strata"]["mrcr_bin_upper"]) <= contexts[metadata["models"][i]] // 2
                        for task in metadata["tasks"]
                    ]
                )
                theta = estimate_ability(model, selected, arrays["responses"][i], arrays["counts"][i])
                points.append(predict_panel(model, theta, task_benchmarks, primary, eligible))
            difference = float(np.max(np.abs(np.asarray(points) - np.asarray(archived["predicted"]))))
            if difference > 1e-6:
                raise ArithmeticError(f"Export changed original held-out predictions: {difference}")
            differences.append({"budget": budget, "maximum_absolute_difference": difference})
        stream = io.BytesIO()
        np.savez_compressed(
            stream,
            ability=model.ability,
            discrimination=model.discrimination,
            intercept=model.intercept,
            continuous=model.continuous,
            expected_counts=model.expected_counts,
            pool=pool,
        )
        state = stream.getvalue()
        name = f"fold-{fold}.npz"
        fs, key = filesystem_for(args.destination + "/" + name)
        fs.pipe_file(key, state)
        receipt = {
            "fold": fold,
            "training_models": [metadata["models"][i] for i in train],
            "heldout_models": [metadata["models"][i] for i in test],
            "dimension": model.ability.shape[1],
            "state_sha256": hashlib.sha256(state).hexdigest(),
            "matrix_sha256": settings["data_sha256"],
            "validation": differences,
            "status": "Verified training-only artifact; no held-out response vectors are exported.",
        }
        receipts.append(receipt)
        print(json.dumps(receipt), flush=True)
    # Runtime identity data contains no historical model outcomes or target scores.
    runtime = {
        "tasks": metadata["tasks"],
        "target_benchmarks": primary,
        "source_matrix_sha256": settings["data_sha256"],
        "predictor_exclusions": metadata["predictor_exclusions"],
    }
    for name, document in [("task-index.json", runtime), ("summary.json", {"study": args.study, "fits": receipts})]:
        payload = json.dumps(document, indent=2, allow_nan=False).encode()
        (output / name).write_bytes(payload)
        fs, key = filesystem_for(args.destination + "/" + name)
        fs.pipe_file(key, payload)
    print("MIRT_EXPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
