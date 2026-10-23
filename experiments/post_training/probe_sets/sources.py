# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned sources for the probe sets, each adapted to :class:`ProbeRow`.

Every loader reads one Hugging Face dataset at a 40-character commit and returns rows
sorted by a stable id, so selection downstream is reproducible from a seed alone.
"""

from __future__ import annotations

import ast
import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import load_dataset as hf_load_dataset

from experiments.post_training.task_data.math_r import r_prompt

_COMMIT = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class SourceSpec:
    """One dataset at a pinned commit."""

    repo: str
    revision: str
    split: str
    config: str | None = None

    def __post_init__(self) -> None:
        if not _COMMIT.fullmatch(self.revision):
            raise ValueError(f"{self.repo} must be pinned to a 40-character commit, got {self.revision!r}")


@dataclass(frozen=True)
class ProbeRow:
    """One probe prompt: a message list ending in a user turn, plus its provenance."""

    probe_id: str
    source: str
    messages: tuple[dict[str, str], ...]
    answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def user_text(self) -> str:
        """Every user turn, joined; the text that overlap checks compare."""
        return "\n".join(message["content"] for message in self.messages if message["role"] == "user")

    def record(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "source": self.source,
            "messages": list(self.messages),
            "answer": self.answer,
            "metadata": self.metadata,
        }


IFEVAL = SourceSpec("google/IFEval", "966cd89545d6b6acfd7638bc708b98261ca58e84", "train")
IFBENCH = SourceSpec("allenai/IFBench_test", "2e8a48de45ff3bf41242f927254ca81b59ca3ae2", "train")
ULTRACHAT = SourceSpec("HuggingFaceH4/ultrachat_200k", "8049631c405ae6576f93f445c6b8166f76f5505a", "test_sft")
AIME_2024 = SourceSpec("HuggingFaceH4/aime_2024", "2fe88a2f1091d5048c0f36abc874fb997b3dd99a", "train")
AIME_2025 = SourceSpec("MathArena/aime_2025", "c94da77eb22bbd6439e62a323bec18493a421302", "train")
HMMT_FEB_2025 = SourceSpec("MathArena/hmmt_feb_2025", "6fdc4277120810ff75aa22d2d5489b91f7a262a1", "train")
# Text-only, English, open-ended competition math.
OLYMPIADBENCH = SourceSpec(
    "Hothan/OlympiadBench", "91184b52131e7fc9455fef848035173aea8cc01a", "train", config="OE_TO_maths_en_COMP"
)
SWE_BENCH_VERIFIED = SourceSpec("princeton-nlp/SWE-bench_Verified", "c104f840cc67f8b6eec6f759ebc8b2693d585d4a", "test")
PILE = SourceSpec("monology/pile-uncopyrighted", "3be90335b66f24456a5d6659d9c8d208c0357119", "train")

Loader = Callable[..., Any]


def _rows(spec: SourceSpec, load_dataset: Loader, *, streaming: bool = False) -> Any:
    return load_dataset(spec.repo, spec.config, revision=spec.revision, split=spec.split, streaming=streaming)


def _single_turn(source: str, probe_id: str, prompt: str, **extra: Any) -> ProbeRow:
    return ProbeRow(probe_id=probe_id, source=source, messages=({"role": "user", "content": prompt},), **extra)


def constraint_prompts(load_dataset: Loader = hf_load_dataset) -> list[ProbeRow]:
    """IFEval and IFBench: single-turn prompts carrying verifiable constraints."""
    rows = []
    for name, spec in (("ifeval", IFEVAL), ("ifbench", IFBENCH)):
        for row in _rows(spec, load_dataset):
            rows.append(
                _single_turn(
                    name,
                    f"{name}-{row['key']}",
                    row["prompt"],
                    metadata={"instruction_id_list": list(row["instruction_id_list"])},
                )
            )
    return sorted(rows, key=lambda row: row.probe_id)


def multi_turn_chats(load_dataset: Loader = hf_load_dataset) -> list[ProbeRow]:
    """UltraChat test conversations with at least two user turns, cut before the last assistant reply."""
    rows = []
    for row in _rows(ULTRACHAT, load_dataset):
        messages = [{"role": message["role"], "content": message["content"]} for message in row["messages"]]
        last_user = max((index for index, message in enumerate(messages) if message["role"] == "user"), default=-1)
        prompt = messages[: last_user + 1]
        if sum(message["role"] == "user" for message in prompt) < 2:
            continue
        rows.append(ProbeRow(probe_id=f"ultrachat-{row['prompt_id']}", source="ultrachat", messages=tuple(prompt)))
    return sorted(rows, key=lambda row: row.probe_id)


def competition_math(load_dataset: Loader = hf_load_dataset) -> dict[str, list[ProbeRow]]:
    """The P_think benchmarks, each prompted exactly as the R task prompts its problems."""

    def math_row(source: str, index: str, problem: str, answer: str) -> ProbeRow:
        return _single_turn(source, f"{source}-{index}", r_prompt(problem), answer=str(answer))

    benchmarks = {
        "aime_2024": [
            math_row("aime_2024", row["id"], row["problem"], row["answer"]) for row in _rows(AIME_2024, load_dataset)
        ],
        "aime_2025": [
            math_row("aime_2025", row["problem_idx"], row["problem"], row["answer"])
            for row in _rows(AIME_2025, load_dataset)
        ],
        "hmmt_feb_2025": [
            math_row("hmmt_feb_2025", row["problem_idx"], row["problem"], row["answer"])
            for row in _rows(HMMT_FEB_2025, load_dataset)
        ],
        "olympiadbench": [
            math_row("olympiadbench", row["id"], row["question"], _first_answer(row["final_answer"]))
            for row in _rows(OLYMPIADBENCH, load_dataset)
            if row["is_multiple_answer"] in (False, "False")
        ],
    }
    return {name: sorted(rows, key=lambda row: row.probe_id) for name, rows in benchmarks.items()}


def _first_answer(final_answer: Any) -> str:
    answers = ast.literal_eval(final_answer) if isinstance(final_answer, str) else list(final_answer)
    if len(answers) != 1:
        raise ValueError(f"expected one OlympiadBench answer, got {answers!r}")
    return str(answers[0])


def swe_bench_instance_ids(load_dataset: Loader = hf_load_dataset) -> list[str]:
    return sorted(row["instance_id"] for row in _rows(SWE_BENCH_VERIFIED, load_dataset))


def pile_documents(load_dataset: Loader = hf_load_dataset) -> Iterator[tuple[str, str]]:
    """Stream ``(domain, text)`` pairs from the Pile test split in file order."""
    for row in _rows(PILE, load_dataset, streaming=True):
        yield row["meta"]["pile_set_name"], row["text"]


def task_probe_rows(path: Path, source: str) -> list[ProbeRow]:
    """The probe split a WP7 build wrote (``x_probe.jsonl`` or ``r_probe.jsonl``)."""
    rows = []
    for line in path.read_text().splitlines():
        task = json.loads(line)
        metadata = {key: task[key] for key in ("template_name", "parameter", "direction") if key in task}
        rows.append(_single_turn(source, task["task_id"], task["prompt"], answer=task["answer"], metadata=metadata))
    return sorted(rows, key=lambda row: row.probe_id)
