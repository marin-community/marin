# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Request-set roundtrip and OLMo-Eval conversion.

Guards the request pipeline: a truncated artifact must fail loudly (the per-task
count is the parity check against the SC oracle), task-name normalization must map
OLMo-Eval ids to the registry, and a non-gold (accuracy) resolution must be
rejected rather than silently scored.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from marin.evaluation.olmo_base_eval.generate_requests import convert_olmo_requests, normalize_olmo_task_name
from marin.evaluation.olmo_base_eval.request_set import RequestInstance, load_request_set, write_request_set


def test_write_then_load_request_set_roundtrips(tmp_path):
    instances = [
        RequestInstance(task="arc_easy", doc_id=0, context="Q1\nAnswer:", continuation=" yes"),
        RequestInstance(task="arc_easy", doc_id=1, context="Q2\nAnswer:", continuation=" no"),
        RequestInstance(task="lambada", doc_id=0, context="once upon a", continuation=" time"),
    ]
    directory = str(tmp_path / "rs")
    manifest = write_request_set(directory, instances, olmo_eval_git_sha="abc123", source="unit-test")
    assert manifest.tasks == {"arc_easy": 2, "lambada": 1}

    loaded = load_request_set(directory)
    assert {task: len(items) for task, items in loaded.items()} == {"arc_easy": 2, "lambada": 1}
    assert loaded["lambada"][0].continuation == " time"


def test_load_request_set_rejects_count_mismatch(tmp_path):
    directory = str(tmp_path / "rs")
    write_request_set(
        directory,
        [RequestInstance(task="arc_easy", doc_id=0, context="c", continuation=" x")],
        olmo_eval_git_sha=None,
        source="unit-test",
    )
    # Truncate requests.jsonl so the loaded count disagrees with the manifest.
    (Path(directory) / "requests.jsonl").write_text("")
    with pytest.raises(ValueError, match="count mismatch"):
        load_request_set(directory)


@pytest.mark.parametrize(
    "olmo_task_name, expected",
    [
        ("arc_easy:olmo3base:bpb", "arc_easy"),
        ("mmlu_abstract_algebra:rc:bpb", "mmlu_abstract_algebra"),
        ("lambada:bpb:olmo3base", "lambada"),
        ("basic_skills_arithmetic:olmo3base:bpb", "basic_skills_arithmetic"),
        ("lab_bench_dbqa:bpb:olmo3base", None),  # not a Table 9 component -> skipped
        ("medqa_en:rc", None),
    ],
)
def test_normalize_olmo_task_name(olmo_task_name, expected):
    assert normalize_olmo_task_name(olmo_task_name) == expected


def _write_olmo_requests(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_convert_olmo_requests_extracts_gold_and_skips_unknown(tmp_path):
    requests_dir = tmp_path / "requests" / "mock"
    _write_olmo_requests(
        requests_dir / "arc_easy_olmo3base_bpb-requests.jsonl",
        [
            {"task_name": "arc_easy:olmo3base:bpb", "doc_id": 0,
             "request": {"context": "Q\nAnswer:", "continuation": " yes", "continuations": [" yes"]}},
        ],
    )
    _write_olmo_requests(
        requests_dir / "lab_bench_dbqa-requests.jsonl",
        [
            {"task_name": "lab_bench_dbqa:bpb", "doc_id": 0,
             "request": {"context": "x", "continuation": " a", "continuations": [" a"]}},
        ],
    )
    instances = convert_olmo_requests(str(tmp_path / "requests"))
    assert [(i.task, i.doc_id, i.continuation) for i in instances] == [("arc_easy", 0, " yes")]


def test_convert_olmo_requests_uses_singular_gold_for_multichoice(tmp_path):
    # piqa-style: continuations lists both choices; the singular continuation is
    # the gold (here the second choice). Only the gold is scored.
    requests_dir = tmp_path / "requests" / "mock"
    _write_olmo_requests(
        requests_dir / "piqa-requests.jsonl",
        [
            {"task_name": "piqa:olmo3base:bpb", "doc_id": 0,
             "request": {"context": "Goal\nAnswer:", "continuation": " gold two",
                         "continuations": [" choice one", " gold two"]}},
        ],
    )
    instances = convert_olmo_requests(str(tmp_path / "requests"))
    assert [(i.task, i.continuation) for i in instances] == [("piqa", " gold two")]


def test_convert_olmo_requests_rejects_duplicate_doc_id(tmp_path):
    requests_dir = tmp_path / "requests" / "mock"
    _write_olmo_requests(
        requests_dir / "csqa-requests.jsonl",
        [
            {"task_name": "csqa:bpb:olmo3base", "doc_id": 0,
             "request": {"context": "c", "continuation": " a"}},
            {"task_name": "csqa:bpb:olmo3base", "doc_id": 0,
             "request": {"context": "c", "continuation": " b"}},
        ],
    )
    # Two requests for the same doc means per-choice accuracy, not gold-only BPB.
    with pytest.raises(ValueError, match="duplicate doc_id"):
        convert_olmo_requests(str(tmp_path / "requests"))
