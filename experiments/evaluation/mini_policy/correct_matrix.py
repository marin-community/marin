# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Version the study matrix with verified GSM8K grading corrections."""

import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from rigging.filesystem.buckets import filesystem_for

from experiments.evaluation.mini_policy.assemble import read
from experiments.evaluation.mini_policy.matrix import PREFIX

PARITY = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/gsm8k-grader-parity-20261030"
DEST = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/inventory-20261030-v5"


def bytes_at(url):
    fs, key = filesystem_for(url)
    return fs.cat(key)


def main():
    task_bytes = bytes_at(PREFIX + "/matrix/tasks.json")
    matrix_bytes = bytes_at(PREFIX + "/matrix/responses.npz")
    metadata = json.loads(task_bytes)
    source = np.load(io.BytesIO(matrix_bytes))
    arrays = {key: source[key].copy() for key in source.files}
    target_benchmark = metadata["benchmarks"].index("gsm8k-0shot")
    columns = {row["task_id"]: j for j, row in enumerate(metadata["tasks"]) if row["benchmark"] == "gsm8k-0shot"}
    summary = read(PARITY + "/summary.json")
    audit = read(PREFIX + "/assembly_audit.json")
    changes = []
    receipts = []
    for model_record in summary["results"]:
        model = model_record["model"]
        key = hashlib.sha256(model.encode()).hexdigest()[:16]
        result_url = PARITY + "/" + key + ".json"
        raw = bytes_at(result_url)
        replay = json.loads(raw)
        i = metadata["models"].index(model)
        if len(replay["trials"]) != len(columns):
            raise ValueError("Incomplete replay panel")
        model_changes = []
        for row in replay["trials"]:
            j = columns[row["task_id"]]
            if arrays["responses"][i, j] != row["expected"]:
                raise ValueError("Original frozen response differs from replay audit")
            if arrays["counts"][i, j] != 1:
                raise ValueError("GSM8K correction expects one trial per unique task")
            if row["replayed"] != row["expected"]:
                change = {"model": model, "task_id": row["task_id"], "before": row["expected"], "after": row["replayed"]}
                changes.append(change)
                model_changes.append(change)
            arrays["responses"][i, j] = row["replayed"]
        values = arrays["responses"][i, list(columns.values())]
        before = float(arrays["targets"][i, target_benchmark])
        arrays["targets"][i, target_benchmark] = values.mean()
        arrays["sems"][i, target_benchmark] = values.std(ddof=1) / np.sqrt(len(values))
        if len(model_changes) != replay["target_mismatches"]:
            raise ValueError("Correction count differs from independent replay")
        receipt = {
            "model": model,
            "changed_trials": len(model_changes),
            "before": before,
            "after": float(values.mean()),
            "source": result_url,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "grader_sha256": replay["target_sha256"],
        }
        receipts.append(receipt)
        for cell in audit:
            if cell["model"] == model and cell["benchmark"] == "gsm8k-0shot":
                cell["analysis_grading_override"] = receipt
    if len(changes) != summary["target_mismatches"]:
        raise ValueError("Global correction count mismatch")
    # All original non-GSM8K response, count, and target entries remain exact.
    unchanged = np.asarray([row["benchmark"] != "gsm8k-0shot" for row in metadata["tasks"]])
    np.testing.assert_array_equal(arrays["responses"][:, unchanged], source["responses"][:, unchanged])
    for key in arrays:
        if key not in {"responses", "targets", "sems"}:
            np.testing.assert_array_equal(arrays[key], source[key])
    primary = [j for j in range(len(metadata["benchmarks"])) if j != target_benchmark]
    for key in ["targets", "sems"]:
        np.testing.assert_array_equal(arrays[key][:, primary], source[key][:, primary])
    provenance = {
        "parent_source": PREFIX,
        "parent_input_sha256": hashlib.sha256(task_bytes + matrix_bytes).hexdigest(),
        "new_source": DEST,
        "raw_replay_source": PARITY,
        "changed_trials": changes,
        "model_scores": receipts,
        "scope": (
            "Restore one consistent GSM8K policy grader across the study cohort. No task identities, counts, "
            "missingness, other benchmark grades, or main campaign tracker cells changed."
        ),
    }
    metadata["analysis_grade_corrections"] = provenance
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    payloads = {
        "matrix/tasks.json": json.dumps(metadata, indent=2, allow_nan=False).encode(),
        "matrix/responses.npz": stream.getvalue(),
        "assembly_audit.json": json.dumps(audit, default=str).encode(),
        "grading_correction.json": json.dumps(provenance, indent=2, allow_nan=False).encode(),
    }
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    for name, payload in payloads.items():
        fs, key = filesystem_for(DEST + "/" + name)
        fs.pipe_file(key, payload)
        path = output / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_bytes(payload)
    print(
        json.dumps(
            {
                "destination": DEST,
                "changed_trials": len(changes),
                "changed_models": [row for row in receipts if row["changed_trials"]],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
