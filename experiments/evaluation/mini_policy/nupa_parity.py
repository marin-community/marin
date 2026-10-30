# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify the adopted NUPA policy scorer against every frozen cohort response."""

import concurrent.futures
import hashlib
import io
import json
import os
from importlib import import_module
from pathlib import Path

import numpy as np
from finestore.reader import ReadView
from rigging.filesystem.buckets import filesystem_for

from experiments.evaluation.mini_policy.assemble import original_record
from experiments.evaluation.mini_policy.experiment import SOURCE, read_bytes
from experiments.evaluation.mini_policy.identities import document_identity

SCORER = import_module("eval.chat_benchmarks.NUPA-Loose.scorer")
DEST = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/nupa-grader-parity-20261030"


def save(name, document):
    payload = json.dumps(document, indent=2, default=str).encode()
    fs, key = filesystem_for(DEST + "/" + name)
    fs.pipe_file(key, payload)
    return hashlib.sha256(payload).hexdigest()


def main():
    metadata = json.loads(read_bytes(SOURCE + "/matrix/tasks.json"))
    arrays = np.load(io.BytesIO(read_bytes(SOURCE + "/matrix/responses.npz")))
    audit = json.loads(read_bytes(SOURCE + "/assembly_audit.json"))
    tasks = {task["task_id"]: j for j, task in enumerate(metadata["tasks"]) if task["benchmark"] == "NUPA"}
    scorer_path = Path(SCORER.__file__)
    scorer_hash = hashlib.sha256(scorer_path.read_bytes()).hexdigest()
    cells = [row for row in audit if row["benchmark"] == "NUPA"]

    def replay(cell):
        record = original_record(cell, json.loads(read_bytes(cell["evidence_url"])))
        if record is None:
            raise ValueError("Raw NUPA record unavailable")
        root = record["results_path"]
        view = ReadView(root)
        table = view.scan("samples", columns=["doc_id", "doc", "output"])
        if table is None or table.num_rows != 5000:
            raise ValueError(f"Expected 5000 raw NUPA samples: {cell['model']}")
        model_index = metadata["models"].index(cell["model"])
        mismatches = []
        trials = []
        seen = set()
        for row in table.to_pylist(maps_as_pydicts="strict"):
            doc = json.loads(row["doc"])
            identity = document_identity("NUPA", doc, row["doc_id"])
            if identity in seen:
                raise ValueError("Duplicate raw NUPA task")
            seen.add(identity)
            column = tasks[identity]
            score = SCORER.score_prediction(row["output"], doc["answer"], doc["answer_format"])
            expected = float(arrays["responses"][model_index, column])
            trial = {
                "task_id": identity,
                "doc_id": row["doc_id"],
                "expected": expected,
                "replayed": score.exact_match,
                "extracted": score.extracted_answer,
                "reference": doc["answer"],
                "answer_format": doc["answer_format"],
                "response_sha256": hashlib.sha256((row["output"] or "").encode()).hexdigest(),
            }
            trials.append(trial)
            if expected != score.exact_match:
                mismatches.append({**trial, "response": row["output"]})
        if seen != set(tasks):
            raise ValueError("Raw NUPA task panel differs from frozen panel")
        key = hashlib.sha256(cell["model"].encode()).hexdigest()[:16]
        receipt = {
            "model": cell["model"],
            "source": root,
            "sample_metadata": view.table_metadata_path("samples"),
            "scorer_sha256": scorer_hash,
            "tasks": len(trials),
            "mismatches": mismatches,
            "trials": trials,
        }
        digest = save(key + ".json", receipt)
        return {
            "model": cell["model"],
            "tasks": len(trials),
            "mismatches": len(mismatches),
            "evidence": DEST + "/" + key + ".json",
            "evidence_sha256": digest,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(replay, cells))
    summary = {
        "scorer_sha256": scorer_hash,
        "models": len(results),
        "trials": sum(row["tasks"] for row in results),
        "mismatches": sum(row["mismatches"] for row in results),
        "results": results,
    }
    save("summary.json", summary)
    (Path(os.environ["IRIS_OUTPUT_DIR"]) / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}), flush=True)


if __name__ == "__main__":
    main()
