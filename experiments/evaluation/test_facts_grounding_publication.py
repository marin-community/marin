# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FACTS Grounding v2 publication sends the official full prompt and binds the judge's grounding inputs."""

import csv
import json

from taskcompendium.runtime.resources import resource_bytes

from experiments.evaluation.catalog import evaluation_benchmarks
from experiments.evaluation.publication_cases import published_cases

# Longer than the standard library CSV reader's default 131,072-character field limit.
LONG_DOCUMENT = "The museum opens at nine. " * 6000
ROWS = [
    {
        "system_instruction": "Answer only from the context.",
        "user_request": "When does the museum open?",
        "context_document": LONG_DOCUMENT,
        "full_prompt": "Answer only from the context.\n\nWhen does the museum open?\n\n" + LONG_DOCUMENT,
        "domain": "Retail/Product",
        "type": "Fact Finding",
        "high_level_type": "Q&A",
    },
    {
        "system_instruction": "Summarize faithfully.",
        "user_request": 'Summarize the ruling, with "quotes".',
        "context_document": "Line one.\nLine two, with a comma.",
        "full_prompt": "Summarize faithfully.\nSummarize the ruling.\nLine one.\nLine two, with a comma.",
        "domain": "Legal",
        "type": "Summarization",
        "high_level_type": "Text Transformation",
    },
]


def write_rows(file):
    with file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ROWS[0]))
        writer.writeheader()
        writer.writerows(ROWS)


def test_publication_sends_full_prompt_and_binds_request_and_document(tmp_path, monkeypatch):
    [source] = evaluation_benchmarks()["facts-grounding-v2"].resolve()
    cases = sorted(
        published_cases(source, write_rows, tmp_path, monkeypatch, len(ROWS)), key=lambda case: case.item.source_ordinal
    )
    assert [case.item.groups for case in cases] == [{"domain": "Retail/Product"}, {"domain": "Legal"}]
    assert {case.item.operation.response_selection for case in cases} == {"final_only"}
    for case, row in zip(cases, ROWS, strict=True):
        [event] = case.task.context.events
        assert (event.role, event.content) == ("user", row["full_prompt"])
        judge = case.task.grader.judge
        assert (judge.question, judge.references) == (row["user_request"], (row["context_document"],))
        config = json.loads(resource_bytes(next(r for r in case.task.resources.verifier if r.path == "config.json")))
        assert config == {"full_prompt": row["full_prompt"]}
