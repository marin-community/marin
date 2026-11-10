# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic questions and completions exercise GenomeQA's native task boundary."""

import hashlib
import json
import subprocess
import sys

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.evaluation.response import completion_text
from taskcompendium.importers import genomeqa as source
from taskcompendium.importers.evalchemy.scorers.genomeqa import grade_genomeqa
from taskcompendium.importers.evalchemy.scorers.main import score
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, Source, TextMessage
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.resources import resource_bytes


def completion(content, *, reasoning=None, finish="stop", refusal=None):
    return (
        completion_text({"content": content, "reasoning_content": reasoning, "refusal": refusal}, finish, "final_only"),
        finish,
    )


@pytest.mark.parametrize(
    "response,options,correct,failure",
    [
        (completion("### Analysis\nA is plausible.\n### Answer\nAnswer: B"), "AB", 1, None),
        (completion("Answer: B", reasoning="Answer: A"), "ABCD", 1, None),
        (completion(None, reasoning="### Answer\nAnswer: B"), "ABCD", 0, "missing_answer"),
        (completion("### Answer\nAnswer: B", finish="length"), "ABCD", 0, "truncation"),
        (completion("", finish="stop"), "ABCD", 0, "missing_answer"),
        (completion(None, refusal="No answer"), "ABCD", 0, "missing_answer"),
        ("<think>Answer: A</think>\n### Answer\n**B**", "AB", 1, None),
        ("<think>### Answer\nAnswer: B", "AB", 0, "missing_answer"),
        ("<think>first</think><think>Answer: B", "AB", 0, "missing_answer"),
        ("I considered B, but cannot tell.", "AB", 0, "missing_answer"),
        ("### Answer\nAnswer: B\nAnswer: A", "AB", 0, "conflicting_answers"),
        ("Answer: B\nAnswer: A", "AB", 0, "conflicting_answers"),
        ("### Answer\nAnswer: B or C", "ABCD", 0, "conflicting_answers"),
        ("### Answer\nAnswer: B and C", "ABCD", 0, "conflicting_answers"),
        ("Answer: B/C", "ABCD", 0, "conflicting_answers"),
        ("Answer: B, C", "ABCD", 0, "conflicting_answers"),
        ("Answer: B because C lacks the motif.", "ABCD", 1, None),
        ("### Answer\nAnswer: C", "AB", 0, "invalid_option"),
        ("### Answer\nAnswer: A\n### Answer\nAnswer: B", "AB", 1, None),
    ],
)
def test_final_answer_grading_preserves_protocol(response, options, correct, failure):
    text, stop = response if isinstance(response, tuple) else (response, "stop")
    grade = grade_genomeqa("B", options, 1, text, stop)
    assert grade.detail["metrics"]["accuracy"] == correct
    assert grade.detail["genomeqa_parsing"]["failure"] == failure
    assert grade.detail["genomeqa_parsing"]["answer"] == ("B" if correct else "")


def test_source_conversion_preserves_question_and_private_grading_inputs(tmp_path):
    (tmp_path / "run.py").write_text(
        'raise RuntimeError("must not execute")\nSIMPLE_SYSTEM_PROMPT = "Synthetic protocol prompt"\n'
    )
    context = ConversionContext(
        {"prompt": StoragePath(str(tmp_path))},
        EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS, docker_image="fixture@sha256:" + "0" * 64
        ),
    )
    row = RawRow(
        "synthetic",
        Source(dataset="synthetic", revision="fixture", row="binary/data.parquet:0", importer_revision="fixture"),
        {
            "id": 1,
            "subtask": "6_motif",
            "format": "binary",
            "question": "Synthetic DNA: AcgT",
            "options": ["alpha", "beta"],
            "answer_index": 1,
            "answer_text": "beta",
            "metadata": "private",
        },
    )
    task = source.convert_genomeqa(row, context)
    assert task.context.events == (
        TextMessage(role="system", content="Synthetic protocol prompt"),
        TextMessage(
            role="user",
            content="Question:\nSynthetic DNA: AcgT\n\nOptions:\nA. alpha\nB. beta\n\nPlease select the correct option.",
        ),
    )
    assert not task.resources.worker
    private = {resource.path: resource_bytes(resource) for resource in task.resources.verifier}
    config = json.loads(private["config.json"])
    for resource_path, content in private.items():
        target = tmp_path / resource_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from dataclasses import asdict; from source_grader.main import score; "
            "print(json.dumps(asdict(score('genomeqa', json.load(open('config.json')), '### Answer\\nB'))))",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout)["reward"] == 1
    verdict = score("genomeqa", config, "### Answer\nB")
    assert verdict.reward == 1
    assert verdict.detail["metric_weights"] == {"genomeqa_macro": 5}
    assert source.genomeqa_item(task).groups == {"subtask": "6_motif", "format": "binary", "cell": "6_motif/binary"}


def test_stop_reason_reaches_source_grader():
    verdict = score("genomeqa", {"expected": "B", "options": "AB", "cell_weight": 1}, "Answer: B", stop_reason="length")
    assert verdict.reward == 0
    assert verdict.detail["metrics"]["truncation"] == 1


def test_pinned_population_validates_both_formats_before_publication(tmp_path, monkeypatch):
    context = ConversionContext({}, None)
    total = 0
    for answer_format in source.FORMATS:
        rows = [
            {
                "id": index,
                "subtask": task,
                "question": "Synthetic DNA: AcgT",
                "options": ["alpha", "beta"] if answer_format == "binary" else ["alpha", "beta", "gamma", "delta"],
                "answer_index": 1,
                "answer_text": "beta",
            }
            for task in source.SUBTASKS
            for index in range(source.CELL_COUNTS[f"{task}/{answer_format}"])
        ]
        path = tmp_path / answer_format / "train-00000-of-00001.parquet"
        path.parent.mkdir()
        pq.write_table(pa.Table.from_pylist(rows), path)
        monkeypatch.setitem(source.FILE_HASHES, answer_format, hashlib.sha256(path.read_bytes()).hexdigest())
        loaded = list(source.read_genomeqa(StoragePath(str(path)), context))
        assert len(loaded) == len({(row["subtask"], row["id"]) for row in loaded}) == 2600
        total += len(loaded)
        path.write_bytes(path.read_bytes() + b"tampered")
        with pytest.raises(ValueError, match="checksum mismatch"):
            list(source.read_genomeqa(StoragePath(str(path)), context))
    assert total == 5200
