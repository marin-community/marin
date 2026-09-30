# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run no-model Harbor replay trials for accepted TaskTrove projections."""

import asyncio
import hashlib
import json
import os
from io import BytesIO
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

import pyarrow.parquet as pq
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.harbor.runner import ChatLaunch, run_trial
from taskcompendium.importers.tasktrove.convert import read_archive
from taskcompendium.importers.tasktrove.mathematical import import_task as import_math
from taskcompendium.importers.tasktrove.mcqa import import_task as import_mcqa
from taskcompendium.lowering import HarborEnvironmentConfig, lower_to_harbor
from taskcompendium.models import VerifierKind
from taskcompendium.submission import PlainText

from experiments.post_training.taskcompendium.audit_tasktrove_ingest import BATCH_SIZE
from experiments.post_training.taskcompendium.ingest_tasktrove import MCQA_SOURCE
from experiments.post_training.taskcompendium.records import public_task_record

SOURCE_NAMES = MappingProxyType(
    {
        MCQA_SOURCE: "mcqa",
        "laion__nemo-prism-math-v3": "prism",
        "laion__nemotron-gym-math-openmathreasoning-v2": "openmath",
    }
)
PROJECTED_SOURCES = (
    MCQA_SOURCE,
    "laion__nemo-prism-math-v3",
)
LEDGER_JOIN_COLUMNS = (
    "input_split",
    "input_file",
    "input_row",
    "source",
    "path",
    "input_object_pin",
    "archive_sha256",
    "disposition",
    "imported_id",
)


def _source_row_index(ledger_path: StoragePath, candidate: dict, proof: dict) -> int:
    source_row = candidate["source"]["row"]
    source_subset, archive_path = source_row.split(":", 1)
    if proof.get("source_row") != source_row or proof.get("archive_path") != archive_path:
        raise ValueError("Accepted public source proof has inconsistent source coordinates")
    matches = []
    with ledger_path.open("rb") as opened:
        parquet = pq.ParquetFile(opened)
        if set(LEDGER_JOIN_COLUMNS) - set(parquet.schema_arrow.names):
            raise ValueError("Ingestion ledger is missing source-row join fields")
        for batch in parquet.iter_batches(columns=list(LEDGER_JOIN_COLUMNS), batch_size=BATCH_SIZE):
            for row in batch.to_pylist():
                if (
                    row["input_split"] == "tasks"
                    and row["input_file"] == proof["input_file"]
                    and row["source"] == source_subset
                    and row["path"] == archive_path
                    and row["input_object_pin"] == proof["input_object_pin"]
                    and row["archive_sha256"] == proof["archive_sha256"]
                    and row["disposition"] == "imported"
                    and row["imported_id"] == candidate["id"]
                ):
                    matches.append(row["input_row"])
    if len(matches) != 1 or type(matches[0]) is not int:
        raise ValueError(f"Accepted source proof joins to {len(matches)} ingestion rows, expected exactly one")
    return matches[0]


def _archive_at(source_path: str, row_index: int) -> bytes:
    with StoragePath(source_path).open("rb") as opened:
        parquet = pq.ParquetFile(opened)
        row_offset = 0
        for row_group in range(parquet.metadata.num_row_groups):
            row_count = parquet.metadata.row_group(row_group).num_rows
            if row_offset <= row_index < row_offset + row_count:
                rows = parquet.read_row_group(row_group, columns=["task_binary"]).column("task_binary")
                value = rows[row_index - row_offset].as_py()
                if not isinstance(value, bytes):
                    raise ValueError("Accepted source row has no archive bytes")
                return value
            row_offset += row_count
    raise ValueError(f"Accepted source row {row_index} is outside the source Parquet file")


def _read_candidate_archive(archive_bytes: bytes, candidate: dict, source_subset: str, archive_path: str):
    source = candidate["source"]
    return read_archive(
        archive_bytes,
        source_subset,
        archive_path,
        source["dataset"],
        source["revision"],
    )


def _reply(specification, correct: bool) -> dict[str, str]:
    parameters = json.loads(specification.verifier.parameters_json)
    if specification.verifier.kind is VerifierKind.MCQ_ANSWER:
        expected = parameters["expected"]
        answer = expected if correct else ("B" if expected != "B" else "A")
    elif specification.verifier.kind is VerifierKind.MATHEMATICAL_ANSWER:
        expected = parameters["expected"]
        answer = expected if correct else "923487234982734"
        if not correct and answer == expected:
            answer = "-923487234982734"
    else:
        raise ValueError(f"Unsupported Harbor proof verifier: {specification.verifier.kind}")
    return {"role": "assistant", "content": answer}


async def _trial(task_dir: Path, reply: dict[str, str], trials_dir: Path, name: str):
    body = BytesIO(json.dumps({"choices": [{"message": reply}]}).encode())
    with patch("taskcompendium.harbor.adapter.urllib.request.urlopen", return_value=body):
        return await run_trial(
            task_dir,
            HarborEnvironmentConfig(),
            ChatLaunch(model="fixed-replay", api_base="https://replay.invalid"),
            trials_dir,
            name,
        )


async def _run(group_path: str, ledger_path: StoragePath, workdir: Path, artifact_dir: Path) -> dict:
    with StoragePath(group_path).open("rb") as opened:
        wrapper = json.loads(opened.readline())
    candidate = wrapper["task"]
    proof = wrapper["source_proof"]
    source_subset, archive_path = candidate["source"]["row"].split(":", 1)
    if source_subset not in SOURCE_NAMES:
        raise ValueError(f"Unreviewed Harbor trial source {source_subset}")
    input_row = _source_row_index(ledger_path, candidate, proof)
    archive_bytes = _archive_at(proof["input_file"], input_row)
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    if archive_sha256 != proof["archive_sha256"]:
        raise ValueError("Sampled archive SHA256 differs from exact accepted source proof")
    archive = _read_candidate_archive(archive_bytes, candidate, source_subset, archive_path)
    if source_subset == MCQA_SOURCE:
        specification = import_mcqa(archive)
        family = "qa-short-answer"
    else:
        specification = import_math(archive).specification
        family = "math-answer"
    if public_task_record(specification, family=family) != candidate:
        projected = public_task_record(specification, family=family)
        mismatched_fields = sorted(key for key in candidate if candidate[key] != projected.get(key))
        raise ValueError(f"Re-imported source task differs from accepted public fields: {mismatched_fields}")

    task_dir = lower_to_harbor(
        specification,
        PlainText(id="plain"),
        HarborEnvironmentConfig(),
        workdir / SOURCE_NAMES[source_subset],
    )
    trials_dir = artifact_dir / "trials" / SOURCE_NAMES[source_subset]
    correct_name, incorrect_name = "correct", "incorrect"
    correct_result = await _trial(task_dir, _reply(specification, True), trials_dir, correct_name)
    incorrect_result = await _trial(task_dir, _reply(specification, False), trials_dir, incorrect_name)
    outcomes = {}
    for name, result in ((correct_name, correct_result), (incorrect_name, incorrect_result)):
        if result.exception_info is not None:
            raise RuntimeError(f"Harbor {name} trial failed: {result.exception_info}")
        outcome = json.loads((trials_dir / name / "verifier" / "taskcompendium-result.json").read_text())
        expected_reward = 1.0 if name == correct_name else 0.0
        if outcome != {"status": "graded", "reward": expected_reward, "error": None}:
            raise ValueError(f"Harbor {name} replay returned unexpected grading outcome: {outcome}")
        outcomes[name] = outcome
    return {
        "source": source_subset,
        "source_row": candidate["source"]["row"],
        "input_row": input_row,
        "archive_sha256": archive_sha256,
        "candidate_id": candidate["id"],
        "harbor_outcomes": outcomes,
        "trace_directories": {name: str(trials_dir / name) for name in outcomes},
    }


async def main() -> None:
    configure_coreweave_s3()
    ingestion_uri = os.environ["TASKTROVE_OUTPUT_URI"]
    projection_uri = os.environ["TASKTROVE_ACCEPTED_OUTPUT_URI"]
    ledger_path = StoragePath(ingestion_uri) / "ingestion-ledger.parquet"
    artifact_dir = Path(os.environ["IRIS_OUTPUT_DIR"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    workdir = artifact_dir / "lowered"
    workdir.mkdir()
    manifest = json.loads((StoragePath(projection_uri) / "manifest.json").read_bytes())
    outcomes = []
    for source_subset in PROJECTED_SOURCES:
        name = SOURCE_NAMES[source_subset]
        groups = [
            group
            for group in manifest["outputs"].values()
            if group["uri"].endswith(f"/{source_subset}/{'qa-short-answer' if name == 'mcqa' else 'math-answer'}.jsonl")
        ]
        if len(groups) != 1 or groups[0]["rows"] < 1:
            raise ValueError(f"Expected one nonempty accepted projection group for {source_subset}")
        outcomes.append(await _run(groups[0]["uri"], ledger_path, workdir, artifact_dir))
    (artifact_dir / "harbor-summary.json").write_text(json.dumps(outcomes, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
