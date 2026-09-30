# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture two bounded private TaskTrove projection samples for local Harbor replay."""

import hashlib
import json
import os
from pathlib import Path

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.taskcompendium.trial_tasktrove_projection import _archive_at, _source_row_index

MAX_WRAPPER_BYTES = 128 * 1024
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024
COHORTS = {
    "laion__nemotron-gym-knowledge-mcqa-v2": "qa-short-answer",
    "laion__nemo-prism-math-v3": "math-answer",
}


def _first_line(path: StoragePath) -> bytes:
    with path.open("rb") as opened:
        line = opened.readline(MAX_WRAPPER_BYTES + 1)
    if not line.endswith(b"\n") or len(line) > MAX_WRAPPER_BYTES:
        raise ValueError("Accepted TaskTrove wrapper line is missing or exceeds its size limit")
    return line


def main() -> None:
    configure_coreweave_s3()
    ingestion_uri = os.environ["TASKTROVE_OUTPUT_URI"]
    projection_uri = os.environ["TASKTROVE_ACCEPTED_OUTPUT_URI"]
    artifact_dir = Path(os.environ["IRIS_OUTPUT_DIR"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((StoragePath(projection_uri) / "manifest.json").read_bytes())
    ledger_path = StoragePath(ingestion_uri) / "ingestion-ledger.parquet"
    captured = []
    for source, family in COHORTS.items():
        groups = [group for group in manifest["outputs"].values() if group["uri"].endswith(f"/{source}/{family}.jsonl")]
        if len(groups) != 1 or groups[0]["rows"] < 1:
            raise ValueError(f"Expected one nonempty accepted projection group for {source}")
        group_path = StoragePath(groups[0]["uri"])
        wrapper_bytes = _first_line(group_path)
        wrapper = json.loads(wrapper_bytes)
        candidate = wrapper["task"]
        proof = wrapper["source_proof"]
        row_index = _source_row_index(ledger_path, candidate, proof)
        archive_bytes = _archive_at(proof["input_file"], row_index)
        if len(archive_bytes) > MAX_ARCHIVE_BYTES:
            raise ValueError(f"Accepted source archive for {source} exceeds the bounded capture limit")
        archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
        if archive_sha256 != proof["archive_sha256"]:
            raise ValueError(f"Accepted archive SHA256 differs from its proof for {source}")
        name = "mcqa" if family == "qa-short-answer" else "prism"
        cohort_dir = artifact_dir / name
        cohort_dir.mkdir(parents=True, exist_ok=True)
        (cohort_dir / "accepted-wrapper.jsonl").write_bytes(wrapper_bytes)
        (cohort_dir / "source-archive.tar.gz").write_bytes(archive_bytes)
        sidecar = {
            "projection_manifest_sha256": (
                hashlib.sha256((StoragePath(projection_uri) / "manifest.json").read_bytes()).hexdigest()
            ),
            "accepted_group_uri": groups[0]["uri"],
            "wrapper_sha256": hashlib.sha256(wrapper_bytes).hexdigest(),
            "wrapper_size_bytes": len(wrapper_bytes),
            "candidate_id": candidate["id"],
            "source_row": proof["source_row"],
            "input_file": proof["input_file"],
            "input_row": row_index,
            "input_object_pin": proof["input_object_pin"],
            "archive_path": proof["archive_path"],
            "archive_sha256": archive_sha256,
            "archive_size_bytes": len(archive_bytes),
        }
        (cohort_dir / "capture-sidecar.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
        captured.append(
            {
                "source": source,
                "candidate_id": candidate["id"],
                "wrapper_sha256": sidecar["wrapper_sha256"],
                "archive_sha256": archive_sha256,
                "archive_size_bytes": len(archive_bytes),
                "input_row": row_index,
            }
        )
    (artifact_dir / "capture-summary.json").write_text(json.dumps(captured, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
