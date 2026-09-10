# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The deduped, verified and clean steps over a small converted parquet built from the fixtures."""

import json
import re
from dataclasses import asdict, replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.post_training.tasktrove.clean import build_clean, export_task
from experiments.post_training.tasktrove.contract import VERIFIER_TOML
from experiments.post_training.tasktrove.convert import ConvertedRecord, convert_one
from experiments.post_training.tasktrove.converters.converted_task import ConvertStatus
from experiments.post_training.tasktrove.converters.registry import converter_index
from experiments.post_training.tasktrove.dedup import DedupStatus, dedup_tasks
from experiments.post_training.tasktrove.sources import SourceInfo, SourceVerdict
from experiments.post_training.tasktrove.taskbinary import INSTRUCTION, TaskFiles, read_task_binary, write_task_binary
from experiments.post_training.tasktrove.verify import verify_tasks

FIXTURES = Path(__file__).parents[1] / "fixtures"
SOURCE = "mcqa"


def _record(path: str, blob: bytes, source: str = SOURCE, family: str = "qa-short-answer") -> ConvertedRecord:
    info = SourceInfo(source, SourceVerdict.KEEP, family, "")
    return convert_one(info, path, blob, converter_index(), "ref")


def _reworded(blob: bytes, instruction: str) -> bytes:
    task = read_task_binary(blob)
    return write_task_binary(TaskFiles({**task.files, INSTRUCTION: instruction.encode()}))


def _leaking_math_record(path: str) -> ConvertedRecord:
    """A converted math task whose instruction quotes the expected answer."""
    record = _record(path, (FIXTURES / "nemotron_math.tar.gz").read_bytes(), "math", "math-answer")
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


def test_dedup_marks_repeated_instructions_and_capped_rows(tmp_path):
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

    dedup_tasks(str(converted), str(tmp_path / "deduped"), max_tasks_per_source=2)
    rows = _rows(tmp_path / "deduped" / "deduped")

    assert rows["b.tar.gz"]["status"] == DedupStatus.DUPLICATE and rows["b.tar.gz"]["task_binary"] is None
    assert rows["e.tar.gz"]["status"] == ConvertStatus.NULL_GRADER
    statuses = sorted(rows[p]["status"] for p in ("a.tar.gz", "c.tar.gz", "d.tar.gz"))
    assert statuses == [DedupStatus.CAPPED, ConvertStatus.CONVERTED, ConvertStatus.CONVERTED]
    capped = next(p for p in ("a.tar.gz", "c.tar.gz", "d.tar.gz") if rows[p]["status"] == DedupStatus.CAPPED)
    assert rows[capped]["task_binary"] is None

    dedup_tasks(str(converted), str(tmp_path / "again"), max_tasks_per_source=2)
    assert _rows(tmp_path / "again" / "deduped") == rows


def test_clean_keeps_survivors_and_ledgers_the_rest(tmp_path):
    blob = (FIXTURES / "nemotron_mcqa.tar.gz").read_bytes()
    good = _record("good.tar.gz", blob)
    duplicate = _record("later-dup.tar.gz", blob)
    leaking = _leaking_math_record("leak.tar.gz")
    unconverted = replace(_record("bad.tar.gz", blob), status=ConvertStatus.NULL_GRADER, error="empty", task_binary=None)
    converted = _write_converted(tmp_path / "converted", [good, duplicate, leaking, unconverted])
    deduped, verified, clean = (str(tmp_path / name) for name in ("deduped", "verified", "clean"))

    dedup_tasks(str(converted), deduped, max_tasks_per_source=None)
    verify_tasks(deduped, verified)
    build_clean(deduped, verified, clean, tool_ref="ref")

    tasks = _rows(tmp_path / "clean" / "tasks" / SOURCE)
    assert set(tasks) == {"good.tar.gz"}
    assert tasks["good.tar.gz"]["mode"] == "mcq" and tasks["good.tar.gz"]["converter"] == "nemotron_mcqa"
    manifest = json.loads((tmp_path / "clean" / "manifest.json").read_text())
    assert manifest["clean_tasks"] == 1 and manifest["input_tasks"] == 4
    assert manifest["by_status"] == {"converted": 1, "duplicate": 1, "verified:gold_leak": 1, "null_grader": 1}
    assert not (tmp_path / "clean" / "tasks" / "math").exists()
    ledger = _rows(tmp_path / "clean" / "ledger" / "convert.parquet")
    assert {p: r["status"] for p, r in ledger.items()} == {
        "later-dup.tar.gz": "duplicate",
        "leak.tar.gz": "verified:gold_leak",
        "bad.tar.gz": "null_grader",
    }
    assert (tmp_path / "clean" / "report.md").read_text().startswith("# TaskTrove Clean")

    exported = export_task(str(tmp_path / "clean" / "tasks" / SOURCE), "good.tar.gz", tmp_path / "export")
    assert (exported / "tests" / "verifier.toml").is_file() and (exported / "environment" / "Dockerfile").is_file()
