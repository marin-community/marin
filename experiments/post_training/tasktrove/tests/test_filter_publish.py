# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Filter and publish a small converted Parquet built from the fixtures."""

import json
import re
from dataclasses import asdict, replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.post_training.tasktrove.convert import ConvertedRecord, convert_one
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.converters.registry import converter_index
from experiments.post_training.tasktrove.dataset import SourceInfo, SourceVerdict
from experiments.post_training.tasktrove.publish import (
    DEFAULT_HF_REPO_ID,
    TASK_COLUMNS,
    export_task,
    publish_release,
    publish_to_huggingface,
)
from experiments.post_training.tasktrove.task_format import VERIFIER_TOML
from experiments.post_training.tasktrove.taskbinary import INSTRUCTION, TaskFiles, read_task_binary, write_task_binary
from experiments.post_training.tasktrove.verify import DedupStatus, filter_tasks

FIXTURES = Path(__file__).parents[1] / "fixtures"
# The release summary looks every source up in source_verdicts.json.
SOURCE = "laion__nemotron-gym-knowledge-mcqa-v2"
MATH_SOURCE = "laion__nemotron-gym-math-v5"


class RecordingHubApi:
    def __init__(self) -> None:
        self.created: dict | None = None
        self.uploaded: dict | None = None
        self.files: dict[str, bytes] = {}

    def create_repo(self, repo_id: str, **kwargs) -> None:
        self.created = {"repo_id": repo_id, **kwargs}

    def upload_folder(self, *, folder_path: str | Path, **kwargs) -> None:
        root = Path(folder_path)
        self.uploaded = {"folder_path": folder_path, **kwargs}
        self.files = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def _record(path: str, blob: bytes, source: str = SOURCE, family: str = "qa-short-answer") -> ConvertedRecord:
    info = SourceInfo(source, SourceVerdict.KEEP, family, "")
    return convert_one(info, path, blob, converter_index(), "ref")


def _reworded(blob: bytes, instruction: str) -> bytes:
    task = read_task_binary(blob)
    return write_task_binary(TaskFiles({**task.files, INSTRUCTION: instruction.encode()}))


def _leaking_math_record(path: str) -> ConvertedRecord:
    """A converted math task whose instruction quotes the expected answer."""
    record = _record(path, (FIXTURES / "nemotron_math.tar.gz").read_bytes(), MATH_SOURCE, "math-answer")
    task = read_task_binary(record.task_binary)
    answer = "x^2 + 2x + 1 = 0"
    spec = re.sub(r'expected = ".*"', f'expected = "{answer}"', task.text(VERIFIER_TOML))
    instruction = task.text(INSTRUCTION) + f"\n\nHint: the answer is {answer}."
    files = {**task.files, VERIFIER_TOML: spec.encode(), INSTRUCTION: instruction.encode()}
    return replace(record, task_binary=write_task_binary(TaskFiles(files)))


def _write_converted(root: Path, records: list[ConvertedRecord]) -> Path:
    converted = root / "converted"
    converted.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([asdict(r) for r in records]), converted / "part-00000.parquet")
    return root


def _rows(path: Path) -> dict[str, dict]:
    """Rows of one parquet, or of every parquet in a directory, keyed by task path."""
    files = sorted(path.glob("*.parquet")) if path.is_dir() else [path]
    return {row["path"]: row for f in files for row in pq.read_table(f).to_pylist()}


def test_filter_marks_repeated_instructions_and_capped_rows(tmp_path):
    blob = (FIXTURES / "nemotron_mcqa.tar.gz").read_bytes()
    original = read_task_binary(blob).text(INSTRUCTION)
    records = [
        _record("a.tar.gz", blob),
        _record("b.tar.gz", _reworded(blob, "  " + original.upper() + "\n\n")),
        _record("c.tar.gz", _reworded(blob, original + " Show your work.")),
        _record("d.tar.gz", _reworded(blob, original + " Be brief.")),
        replace(_record("e.tar.gz", blob), status=ConvertStatus.NULL_GRADER, task_binary=None),
    ]
    converted = _write_converted(tmp_path / "converted", records)

    filter_tasks(str(converted), str(tmp_path / "filtered"), max_tasks_per_source=2)
    rows = _rows(tmp_path / "filtered" / "graded")

    assert rows["b.tar.gz"]["status"] == DedupStatus.DUPLICATE and rows["b.tar.gz"]["task_binary"] is None
    assert rows["b.tar.gz"]["error"] == "same instruction as a.tar.gz"
    assert rows["e.tar.gz"]["status"] == ConvertStatus.NULL_GRADER
    statuses = sorted(rows[p]["status"] for p in ("a.tar.gz", "c.tar.gz", "d.tar.gz"))
    assert statuses == [DedupStatus.CAPPED, ConvertStatus.CONVERTED, ConvertStatus.CONVERTED]
    capped = next(p for p in ("a.tar.gz", "c.tar.gz", "d.tar.gz") if rows[p]["status"] == DedupStatus.CAPPED)
    assert rows[capped]["task_binary"] is None

    filter_tasks(str(converted), str(tmp_path / "again"), max_tasks_per_source=2)
    assert _rows(tmp_path / "again" / "graded") == rows


def test_publish_writes_survivors_and_rejection_ledger(tmp_path):
    blob = (FIXTURES / "nemotron_mcqa.tar.gz").read_bytes()
    good = _record("good.tar.gz", blob)
    duplicate = _record("later-dup.tar.gz", blob)
    leaking = _leaking_math_record("leak.tar.gz")
    unconverted = replace(_record("bad.tar.gz", blob), status=ConvertStatus.NULL_GRADER, error="empty", task_binary=None)
    converted = _write_converted(tmp_path / "converted", [good, duplicate, leaking, unconverted])
    filtered, release = (str(tmp_path / name) for name in ("filtered", "release"))

    filter_tasks(str(converted), filtered, max_tasks_per_source=None)
    publish_release(filtered, release, tool_ref="ref")

    assert [path.name for path in (tmp_path / "release" / "tasks").glob("*.parquet")] == ["part-00000.parquet"]
    tasks = _rows(tmp_path / "release" / "tasks")
    assert set(tasks) == {"good.tar.gz"}
    assert tasks["good.tar.gz"]["mode"] == "mcq" and tasks["good.tar.gz"]["converter"] == "nemotron_mcqa"
    assert set(tasks["good.tar.gz"]) == set(TASK_COLUMNS)
    manifest = json.loads((tmp_path / "release" / "manifest.json").read_text())
    assert manifest["clean_tasks"] == 1 and manifest["input_tasks"] == 4
    assert manifest["by_status"] == {"converted": 1, "duplicate": 1, "verified:gold_leak": 1, "null_grader": 1}
    assert manifest["by_check"] == {"gold_leak": 1}
    assert manifest["source_details"][SOURCE] == {
        "converters": {"nemotron_mcqa": 1},
        "modes": {"mcq": 1},
        "languages": {},
        "dockerfiles": {good.dockerfile_id: 1},
    }
    ledger = _rows(tmp_path / "release" / "ledger.parquet")
    assert {p: r["status"] for p, r in ledger.items()} == {
        "later-dup.tar.gz": "duplicate",
        "leak.tar.gz": "verified:gold_leak",
        "bad.tar.gz": "null_grader",
    }
    assert ledger["bad.tar.gz"]["error"] == "empty"
    assert (tmp_path / "release" / "report.md").read_text().startswith("# TaskTrove release")

    exported = export_task(str(tmp_path / "release" / "tasks"), "good.tar.gz", tmp_path / "export")
    assert (exported / "tests" / "verifier.toml").is_file() and (exported / "environment" / "Dockerfile").is_file()


def test_huggingface_publish_uploads_tasks_and_audit_metadata(tmp_path):
    blob = (FIXTURES / "nemotron_mcqa.tar.gz").read_bytes()
    converted = _write_converted(tmp_path / "converted", [_record("good.tar.gz", blob)])
    filtered = str(tmp_path / "filtered")
    release = str(tmp_path / "release")
    filter_tasks(str(converted), filtered, max_tasks_per_source=None)
    publish_release(filtered, release, tool_ref="ref")
    api = RecordingHubApi()

    publish_to_huggingface(release, private=True, api=api)

    assert api.created == {
        "repo_id": DEFAULT_HF_REPO_ID,
        "repo_type": "dataset",
        "private": True,
        "exist_ok": True,
    }
    assert api.uploaded is not None
    assert api.uploaded["repo_id"] == DEFAULT_HF_REPO_ID
    assert api.uploaded["repo_type"] == "dataset"
    assert api.uploaded["delete_patterns"] == "data/*.parquet"
    assert set(api.files) == {"README.md", "data/part-00000.parquet", "ledger.parquet", "manifest.json"}
    assert api.files["data/part-00000.parquet"] == (tmp_path / "release/tasks/part-00000.parquet").read_bytes()
    card = api.files["README.md"].decode()
    assert "path: data/*.parquet" in card
    assert "# TaskTrove release" in card


def test_huggingface_publish_validates_release_before_creating_repo(tmp_path):
    api = RecordingHubApi()

    with pytest.raises(FileNotFoundError, match="no task Parquet files"):
        publish_to_huggingface(str(tmp_path / "missing"), api=api)

    assert api.created is None
