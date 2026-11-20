# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FACTS Parametric publication sends the bare question and binds the judge's question and gold answer."""

import csv

from experiments.evaluation.catalog import evaluation_benchmarks
from experiments.evaluation.publication_cases import published_cases

ROWS = [
    {
        "url": "https://en.wikipedia.org/wiki/Yair_Yint_Aung",
        "query": "yair yint aung birthday",
        "answer": "August 18, 1993",
        "topic": "birthday",
    },
    {
        "url": "https://en.wikipedia.org/wiki/Example",
        "query": 'who wrote "the example", a novel',
        "answer": "Jane Doe",
        "topic": "other",
    },
]


def write_rows(file):
    with file.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ROWS[0]))
        writer.writeheader()
        writer.writerows(ROWS)


def test_publication_sends_question_and_binds_gold_answer(tmp_path, monkeypatch):
    [source] = evaluation_benchmarks()["facts-parametric"].resolve()
    cases = sorted(
        published_cases(source, write_rows, tmp_path, monkeypatch, len(ROWS)), key=lambda case: case.item.source_ordinal
    )
    assert [case.item.groups for case in cases] == [{"topic": "birthday"}, {"topic": "other"}]
    assert {case.item.operation.response_selection for case in cases} == {"final_only"}
    for case, row in zip(cases, ROWS, strict=True):
        [event] = case.task.context.events
        assert (event.role, event.content) == ("user", row["query"])
        judge = case.task.grader.judge
        assert (judge.question, judge.references) == (row["query"], (row["answer"],))
