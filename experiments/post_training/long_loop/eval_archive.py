# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish native SkyRL evaluation rows through FineStore and EvalDash's record contract."""

import hashlib
import json
from collections import defaultdict

from finestore.eval import EvalSample, EvaluationStore, Grading, SampleKind
from marin.evaluation.records import EvalRunRecord, TaskCoverage, write_record
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.post_training.long_loop.archive import digest, write_once


def publish_evaluation(*, source: str, record: EvalRunRecord, records_prefix: str) -> str:
    """Keep every native row, including failed trials; index scores only when verified."""
    files = sorted((StoragePath(source) / "*.jsonl").glob(), key=str)
    if not files:
        raise ValueError("No native evaluation files to archive")
    signature = hashlib.sha256(
        json.dumps(
            {
                "record": record.model_dump(mode="json", by_alias=True),
                "sources": [(str(path), hashlib.sha256(path.read_bytes()).hexdigest()) for path in files],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    marker_path = StoragePath(record.results_path) / "publication.json"
    if marker_path.exists():
        marker = json.loads(marker_path.read_bytes())
        if marker["signature"] != signature:
            raise ValueError("Evaluation archive inputs changed")
        if digest(marker["record_path"])[1] != marker["record_sha256"]:
            raise ValueError("Published evaluation record changed")
        if digest(prefix_join(record.results_path, "HEAD"))[1] != marker["head_sha256"]:
            raise ValueError("Published evaluation archive HEAD changed")
        return marker["record_path"]
    counts = defaultdict(lambda: {"attempted": 0, "scored": 0, "correct": 0})
    scores = defaultdict(list)
    store = EvaluationStore.open(record.results_path, writer_id="long-loop-archive")
    try:
        for path in files:
            raw = path.read_bytes()
            filename = path.name
            store.add_source_artifact(filename, raw, content_type="application/x-ndjson")
            if filename == "aggregated_results.jsonl":
                continue
            for line_number, line in enumerate(raw.splitlines()):
                if not line.strip():
                    continue
                row = json.loads(line)
                task, uid = row["data_source"], row["uid"]
                trial_id = f"{filename}:{line_number}"
                name = hashlib.sha256(trial_id.encode()).hexdigest() + ".json"
                trajectory = store.add_artifact(name, line, metadata={"source": filename, "line": line_number})
                verification = row["verification"]
                verified = verification is not None and verification["status"] == "verified"
                score = verification["score"] if verified else None
                counts[task]["attempted"] += 1
                counts[task]["scored"] += int(verified)
                counts[task]["correct"] += int(verified and verification["passed"] is True)
                if verified:
                    scores[task].append(score)
                store.add_sample(
                    EvalSample(
                        task=task,
                        doc_id=uid,
                        kind=SampleKind.AGENTIC if row["env_class"] == "harbor_trial" else SampleKind.GENERATION,
                        trajectory_uri=trajectory,
                        grading=Grading(
                            method="native-verifier",
                            metric="reward",
                            score=score,
                            passed=verification["passed"] if verified else None,
                            detail=json.dumps(verification),
                        ),
                        metrics={"reward": score} if verified else {},
                        doc=json.dumps(
                            {
                                "uid": uid,
                                "data_source": task,
                                "native_source": filename,
                                "source_line": line_number,
                                "source_sha256": hashlib.sha256(raw).hexdigest(),
                            }
                        ),
                    ),
                    trial_id=trial_id,
                )
            store.flush()
        store.seal()
    finally:
        store.close()
    updated = record.model_copy(
        update={
            "metrics": {task: {"reward": sum(values) / len(values)} for task, values in scores.items()},
            "coverage": {
                task: TaskCoverage(n_attempted=c["attempted"], n_scored=c["scored"], n_correct=c["correct"])
                for task, c in counts.items()
            },
        }
    )
    path = write_record(updated, records_prefix)
    write_once(
        prefix_join(record.results_path, "publication.json"),
        json.dumps(
            {
                "signature": signature,
                "record_path": path,
                "record_sha256": digest(path)[1],
                "head_sha256": digest(prefix_join(record.results_path, "HEAD"))[1],
            },
            sort_keys=True,
        ).encode(),
    )
    return path
