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
    "response,options,answer,failure",
    [
        (completion("### Analysis\nA is plausible.\n### Answer\nAnswer: B"), "AB", "B", None),
        (completion("Answer: B", reasoning="Answer: A"), "ABCD", "B", None),
        (completion(None, reasoning="### Answer\nAnswer: B"), "ABCD", "", "missing_answer"),
        (completion("### Answer\nAnswer: B", finish="length"), "ABCD", "", "truncation"),
        (completion("", finish="stop"), "ABCD", "", "missing_answer"),
        (completion(None, refusal="No answer"), "ABCD", "", "missing_answer"),
        ("<think>Answer: A</think>\n### Answer\n**B**", "AB", "B", None),
        ("<think>### Answer\nAnswer: B", "AB", "", "missing_answer"),
        ("<think>first</think><think>Answer: B", "AB", "", "missing_answer"),
        ("I considered B, but cannot tell.", "AB", "", "missing_answer"),
        ("### Answer\nAnswer: B\nAnswer: A", "AB", "", "conflicting_answers"),
        ("Answer: B\nAnswer: A", "AB", "", "conflicting_answers"),
        ("### Answer\nAnswer: B or C", "ABCD", "", "conflicting_answers"),
        ("### Answer\nAnswer: B and C", "ABCD", "", "conflicting_answers"),
        ("Answer: A", "AB", "A", None),
        ("Answer: B1", "AB", "", "missing_answer"),
        ("### Answer\nB1", "AB", "", "missing_answer"),
        ("Answer: B or (C)", "ABCD", "", "conflicting_answers"),
        ("Answer: B/C", "ABCD", "", "conflicting_answers"),
        ("Answer: B, C", "ABCD", "", "conflicting_answers"),
        ("Answer: B because C lacks the motif.", "ABCD", "B", None),
        ("### Answer\nAnswer: C", "AB", "", "invalid_option"),
        ("### Answer\nAnswer: A\n### Answer\nAnswer: B", "AB", "B", None),
    ],
)
def test_final_answer_grading_preserves_protocol(response, options, answer, failure):
    text, stop = response if isinstance(response, tuple) else (response, "stop")
    grade = grade_genomeqa("B", options, 1, text, stop)
    assert grade.detail["metrics"]["accuracy"] == (answer == "B")
    assert grade.detail["genomeqa_parsing"]["failure"] == failure
    assert grade.detail["genomeqa_parsing"]["answer"] == answer


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
    verdict = json.loads(completed.stdout)
    assert verdict["reward"] == 1
    assert verdict["detail"]["metric_weights"] == {"genomeqa_macro": 5}
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
            for index in range(100 if task == "6_motif" else 500)
        ]
        path = tmp_path / f"{answer_format}-override.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        monkeypatch.setitem(source.FORMAT_BY_SHA256, hashlib.sha256(path.read_bytes()).hexdigest(), answer_format)
        loaded = list(source.read_genomeqa(StoragePath(str(path)), context))
        assert len(loaded) == len({(row["subtask"], row["id"]) for row in loaded}) == 2600
        total += len(loaded)
        other_format = "mcq" if answer_format == "binary" else "binary"
        locator = Source(
            dataset="GenomeQA", revision="fixture", row=f"{other_format}/data.parquet:0", importer_revision="fixture"
        )
        with pytest.raises(ValueError, match=f"{other_format} input contains {answer_format} data"):
            source.convert_genomeqa(RawRow("fixture", locator, loaded[0]), context)
        rows[0]["question"] = "Changed synthetic DNA: TTTT"
        pq.write_table(pa.Table.from_pylist(rows), path)
        with pytest.raises(ValueError):
            list(source.read_genomeqa(StoragePath(str(path)), context))
    assert total == 5200
