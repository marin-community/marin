# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Store-building and CLI tests: rows, metadata, decontamination, the live gate."""

import asyncio
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from experiments.post_training.task_curation.balanced_reasoning.bands import Band
from experiments.post_training.task_curation.balanced_reasoning.cli import build_parser, quota_from_arguments
from experiments.post_training.task_curation.balanced_reasoning.lane import run_selected_task
from experiments.post_training.task_curation.balanced_reasoning.selection import write_selection
from experiments.post_training.task_curation.balanced_reasoning.store import (
    BALANCED_CHAT_SCHEMA,
    BlendFilter,
    build_store,
    word_ngrams,
)
from experiments.post_training.task_curation.balanced_reasoning.test_lane import (
    FakeBackend,
    lane_policy_defaults,
    lane_resources,
    make_task,
    provider_completion,
    selected_task,
)

ANSWER_PROMPT = (
    "What is 40 + 2? Answer with the number only, after you explain each "
    "arithmetic step in order carefully and show the carry."
)
ANSWER = "42"


def _run_task(run_root: Path, entry, completions) -> None:
    policy = lane_policy_defaults()

    async def drive() -> None:
        task = make_task(ANSWER_PROMPT, ANSWER)
        await run_selected_task(
            entry,
            task,
            policy,
            lane_resources(policy, FakeBackend(completions), expected=ANSWER),
            run_root / "tasks" / entry.task.task_id,
        )

    asyncio.run(drive())


def test_blend_filter_hits_and_boilerplate() -> None:
    task_prompt = (
        "Solve the equation carefully: " + "explain each arithmetic step in order and then " * 3 + "give the result."
    )
    blend = BlendFilter.from_prompts([task_prompt + " Extra tail words here.", "An unrelated prompt about chess."])
    assert blend.hit(task_prompt) is not None
    assert blend.hit("A completely different puzzle about towers and bridges in the rain.") is None
    # 13-gram basics: identical short texts have no 13-gram when they are too short
    assert word_ngrams("one two three") == frozenset()


def test_build_store_rows_and_schema(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    entry = selected_task()
    _run_task(
        run_root,
        entry,
        tuple(provider_completion(ANSWER, completion_tokens=77, reasoning="Six times seven.") for _ in range(4)),
    )
    selection_path = run_root / "selection.json"
    write_selection(str(selection_path), [entry], {"medium": 1})
    manifest = build_store(str(run_root), selection_path=str(selection_path))
    assert manifest["rows"] == 1
    assert manifest["resolutions"] == {"solved_shortest": 1}
    table = pq.read_table(run_root / "store" / "rows.parquet")
    assert table.schema == BALANCED_CHAT_SCHEMA
    row = table.to_pylist()[0]
    assert row["source"] == "balanced-reasoning-v1"
    assert row["band"] == "medium"
    assert row["verified"] is True
    assert row["atlas_task_id"] == "t1"
    assistant = row["messages"][1]
    assert assistant["channel"] == "analysis"  # the trace's reasoning rode the analysis channel


def test_build_store_drops_blend_overlap(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    entry = selected_task()
    _run_task(run_root, entry, tuple(provider_completion(ANSWER, completion_tokens=77) for _ in range(4)))
    selection_path = run_root / "selection.json"
    write_selection(str(selection_path), [entry], {"medium": 1})
    blend = BlendFilter.from_prompts([ANSWER_PROMPT + " Extra tail words."])
    manifest = build_store(str(run_root), selection_path=str(selection_path), blend=blend)
    assert manifest["rows"] == 0
    assert manifest["dropped_by_decontamination"] == 1


def test_run_command_refuses_without_the_live_gate(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(json.dumps({"digest": "x", "census": {}, "tasks": []}))
    parser = build_parser()
    args = parser.parse_args(["run", "--selection", str(selection_path), "--run-root", str(tmp_path / "run")])
    exit_code = args.handler(args)
    assert exit_code == 2
    assert "gated" in capsys.readouterr().err


def test_plan_command_bands_from_a_local_ledger(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    ledger = tmp_path / "ledger" / "rows" / "part-0.jsonl"
    ledger.parent.mkdir(parents=True)
    task_sha = "ab" * 32
    rows = [
        {
            "source_id": "src",
            "task_sha256": task_sha,
            "solver": "hosted",
            "outcome": "scored",
            "reward": 1.0,
            "attempt_id": "a1",
            "task_id": "t1",
        },
        {
            "source_id": "src",
            "task_sha256": task_sha,
            "solver": "hosted",
            "outcome": "scored",
            "reward": 1.0,
            "attempt_id": "a2",
            "task_id": "t1",
        },
        {
            "source_id": "src",
            "task_sha256": task_sha,
            "solver": "hosted",
            "outcome": "scored",
            "reward": 0.0,
            "attempt_id": "a3",
            "task_id": "t1",
        },
    ]
    ledger.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps(
            {
                "task_id": "t1",
                "source_id": "src",
                "task_sha256": task_sha,
                "domain": "math",
                "task_path": "/tasks/t1.json",
            }
        )
        + "\n"
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "plan",
            "--tasks",
            str(tasks),
            "--ledger",
            str(tmp_path / "ledger"),
            "--output",
            str(tmp_path / "run"),
            "--easy-quota",
            "0",
            "--medium-quota",
            "1",
            "--hard-quota",
            "0",
            "--unsolved-quota",
            "0",
        ]
    )
    assert args.handler(args) == 0
    output = capsys.readouterr().out
    assert "selected=1" in output
    selection = json.loads((tmp_path / "run" / "selection.json").read_text())
    assert selection["tasks"][0]["band"] == "medium"


def test_quota_overrides_fill_defaults(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(["plan", "--tasks", "x", "--output", str(tmp_path), "--hard-quota", "10"])
    quota = quota_from_arguments(args)
    assert quota.per_band[Band.HARD] == 10
    assert quota.per_band[Band.EASY] == 4_000
