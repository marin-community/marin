# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run private Harbor controls on deterministic TaskTrove math batch specimens."""

import asyncio
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.taskcompendium.trial_tasktrove_projection import _run

OPENMATH_SOURCE = "laion__nemotron-gym-math-openmathreasoning-v2"
CANDIDATE_FIELDS = frozenset(
    {
        "record_version",
        "id",
        "context",
        "environment_requirements",
        "tool_providers",
        "final_tools",
        "answer_type",
        "source",
        "submission_instruction",
        "tags",
        "source_category",
    }
)
PROOF_FIELDS = frozenset(
    {
        "candidate_id",
        "input_file",
        "input_row",
        "input_object_pin",
        "source",
        "path",
        "archive_sha256",
        "disposition",
    }
)


def _sample_positions(row_count: int) -> tuple[int, ...]:
    if row_count < 1:
        raise ValueError("OpenMath batch has no candidate rows to trial")
    return tuple(sorted({0, row_count // 2, row_count - 1}))


def _candidate_wrapper(candidate: dict[str, Any], proof: dict[str, Any]) -> dict[str, Any]:
    if set(candidate) != CANDIDATE_FIELDS or set(proof) != PROOF_FIELDS:
        raise ValueError("TaskTrove candidate/proof does not match the reviewed schema")
    source = candidate.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("row"), str):
        raise ValueError("TaskTrove candidate has no source row")
    source_subset, archive_path = source["row"].split(":", 1)
    if source_subset != OPENMATH_SOURCE:
        raise ValueError("TaskTrove candidate is outside the OpenMath trial cohort")
    if (
        candidate.get("source_category") != "math-answer"
        or proof.get("candidate_id") != candidate.get("id")
        or proof.get("source") != source_subset
        or proof.get("path") != archive_path
        or proof.get("disposition") != "imported"
    ):
        raise ValueError("TaskTrove candidate and ingestion proof coordinates disagree")
    return {
        "task": candidate,
        "source_proof": {
            "source_row": source["row"],
            "input_file": proof["input_file"],
            "input_object_pin": proof["input_object_pin"],
            "archive_path": archive_path,
            "archive_sha256": proof["archive_sha256"],
        },
    }


def _selected_wrappers(candidate_path: StoragePath, proof_path: StoragePath, row_count: int) -> list[dict[str, Any]]:
    positions = set(_sample_positions(row_count))
    selected = []
    source_index = 0
    with candidate_path.open("rb") as candidate_stream, proof_path.open("rb") as proof_stream:
        for candidate_line, proof_line in zip(candidate_stream, proof_stream, strict=True):
            candidate = json.loads(candidate_line)
            if candidate.get("source", {}).get("row", "").startswith(f"{OPENMATH_SOURCE}:"):
                if source_index in positions:
                    selected.append(_candidate_wrapper(candidate, json.loads(proof_line)))
                source_index += 1
    if source_index != row_count or len(selected) != len(positions):
        raise ValueError(
            f"Candidate source count or deterministic samples differ from manifest: {source_index} vs {row_count}"
        )
    return selected


async def main() -> None:
    configure_coreweave_s3()
    ingestion_uri = os.environ["TASKTROVE_OUTPUT_URI"]
    output_dir = Path(os.environ["IRIS_OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = StoragePath(ingestion_uri)
    manifest_bytes = (prefix / "ingestion-manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("status") != "complete":
        raise ValueError("Cannot trial an incomplete TaskTrove ingestion")
    expected_rows = manifest["public_candidate_counts"][f"source:{OPENMATH_SOURCE}"]
    wrappers = _selected_wrappers(prefix / "public-candidates.jsonl", prefix / "candidate-proof.jsonl", expected_rows)
    ledger_path = prefix / "ingestion-ledger.parquet"
    outcomes = []
    with tempfile.TemporaryDirectory(prefix="tasktrove-math-bulk-wrappers-") as temporary_directory:
        for index, wrapper in enumerate(wrappers):
            wrapper_path = Path(temporary_directory) / f"sample-{index}.jsonl"
            wrapper_path.write_text(json.dumps(wrapper, sort_keys=True) + "\n")
            workdir = output_dir / "lowered" / f"sample-{index}"
            artifact_dir = output_dir / f"sample-{index}"
            workdir.mkdir(parents=True)
            artifact_dir.mkdir()
            outcomes.append(
                await _run(
                    str(wrapper_path),
                    ledger_path,
                    workdir,
                    artifact_dir,
                )
            )
    summary = {
        "status": "complete",
        "source": OPENMATH_SOURCE,
        "ingestion_manifest_uri": manifest["manifest_uri"],
        "ingestion_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_manifest_sha256": manifest["source_manifest_sha256"],
        "candidate_jsonl_sha256": manifest["artifacts"]["public_candidates"]["sha256"],
        "proof_jsonl_sha256": manifest["artifacts"]["candidate_proof"]["sha256"],
        "candidate_count": expected_rows,
        "sample_positions": list(_sample_positions(expected_rows)),
        "specimens": outcomes,
        "public_projection_published": False,
    }
    (output_dir / "harbor-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({"source": OPENMATH_SOURCE, "candidate_count": expected_rows, "specimens": outcomes}, sort_keys=True)
    )


if __name__ == "__main__":
    asyncio.run(main())
