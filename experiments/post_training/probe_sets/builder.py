# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe-set assembly: seeded selection, overlap screening, and manifests.

Prompts are shared across the four architectures; only reference continuations are
per architecture, so the builder writes prompts once. Every set is screened against
Stage 0 and the task training data by 13-gram overlap of normalized user text and
must be pairwise disjoint from every other set. ``P_X`` is screened against ``X``
train by function identity instead: sibling functions differ only in a constant, so
text overlap would flag them all.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from experiments.post_training.probe_sets.sources import (
    AIME_2024,
    AIME_2025,
    HMMT_FEB_2025,
    IFBENCH,
    IFEVAL,
    OLYMPIADBENCH,
    PILE,
    SWE_BENCH_VERIFIED,
    ULTRACHAT,
    ProbeRow,
    SourceSpec,
)
from experiments.post_training.task_data.math_r import OVERLAP_N_GRAM, ngrams, normalize_problem

P_CHAT_SIZE = 1_500
P_THINK_OLYMPIADBENCH_SIZE = 210
P_AGENT_SIZE = 150
P_TASK_SIZE = 1_000
T_DOMAINS = (
    "Pile-CC",
    "PubMed Central",
    "ArXiv",
    "Github",
    "FreeLaw",
    "StackExchange",
    "USPTO Backgrounds",
    "PubMed Abstracts",
    "Wikipedia (en)",
    "DM Mathematics",
)
T_TOKENS_PER_DOMAIN = 200_000

# The eval-policy SWE-bench Verified setup the P_agent trajectories are recorded under.
HARBOR_SWE_BENCH_DATASET = "hf://DCAgent/swebench-verified"
HARBOR_SWE_BENCH_REF = "0e4345c345395a3371f0b79d891a59fdbf74e9a8"
HARBOR_SWE_BENCH_BASE_CONFIG = Path("experiments/evaluation/configs/harbor/swebench-full.yaml")
P_AGENT_CONTEXT_TOKENS = 131_072
P_AGENT_GENERATION_TOKENS = 16_384


def overlap_grams(text: str) -> frozenset[tuple[str, ...]]:
    return ngrams(normalize_problem(text), OVERLAP_N_GRAM)


def seeded_selection(rows: list[ProbeRow], size: int, *, seed: int, name: str) -> list[ProbeRow]:
    """Shuffle ``rows`` (already sorted by stable id) with ``seed`` and take ``size``; raise on a shortfall."""
    if len(rows) < size:
        raise ValueError(f"{name} needs {size} rows but only {len(rows)} survive screening")
    shuffled = list(rows)
    random.Random(f"{seed}:{name}").shuffle(shuffled)
    return shuffled[:size]


def colliding_ids(rows: Iterable[ProbeRow], texts: Iterable[str]) -> set[str]:
    """Ids of ``rows`` sharing a 13-gram with any of ``texts``.

    The probe grams are held in memory and ``texts`` is streamed, so a corpus the
    size of Stage 0 is screened in bounded memory.
    """
    owners: dict[tuple[str, ...], set[str]] = {}
    for row in rows:
        for gram in overlap_grams(row.user_text()):
            owners.setdefault(gram, set()).add(row.probe_id)
    hits: set[str] = set()
    for text in texts:
        for gram in overlap_grams(text):
            hits |= owners.get(gram, set())
    return hits


def message_file_user_texts(paths: Iterable[Path]) -> Iterator[str]:
    """User turns of Stage 0 protocol rows (``.jsonl`` or ``.jsonl.gz`` with a ``messages`` list)."""
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as handle:
            for line in handle:
                row = json.loads(line)
                yield "\n".join(message["content"] for message in row["messages"] if message["role"] == "user")


def task_file_prompts(paths: Iterable[Path]) -> Iterator[str]:
    """Prompts of WP7 task files (``x_train.jsonl``, ``r_train.jsonl``)."""
    for path in paths:
        for line in path.read_text().splitlines():
            yield json.loads(line)["prompt"]


@dataclass(frozen=True)
class ScreeningInputs:
    """Training data every probe set must be disjoint from."""

    stage0_files: tuple[Path, ...]
    task_train_files: tuple[Path, ...]
    x_train_file: Path

    def __post_init__(self) -> None:
        if not self.stage0_files or not self.task_train_files:
            raise ValueError("probe screening requires the Stage 0 files and the task training files")


def screen(rows: list[ProbeRow], inputs: ScreeningInputs, *, x_task_rows: bool) -> tuple[list[ProbeRow], dict[str, int]]:
    """Drop rows that overlap Stage 0 or task training data; return the kept rows and drop counts."""
    stage0 = colliding_ids(rows, message_file_user_texts(inputs.stage0_files))
    if x_task_rows:
        train_functions = {
            (task["template_name"], task["parameter"])
            for task in map(json.loads, inputs.x_train_file.read_text().splitlines())
        }
        task_train = {
            row.probe_id for row in rows if (row.metadata["template_name"], row.metadata["parameter"]) in train_functions
        }
        other_train = [path for path in inputs.task_train_files if path != inputs.x_train_file]
        task_train |= colliding_ids(rows, task_file_prompts(other_train))
    else:
        task_train = colliding_ids(rows, task_file_prompts(inputs.task_train_files))
    kept = [row for row in rows if row.probe_id not in stage0 | task_train]
    return kept, {"dropped_stage0_overlap": len(stage0), "dropped_task_train_overlap": len(task_train - stage0)}


def require_fixed_benchmark(rows: list[ProbeRow], kept: list[ProbeRow], name: str) -> None:
    """A fixed benchmark cannot be refilled, so any overlap is a contamination finding that stops the build."""
    if len(kept) != len(rows):
        dropped = sorted({row.probe_id for row in rows} - {row.probe_id for row in kept})
        raise ValueError(f"{name} overlaps training data and cannot be refilled: {dropped[:5]}")


def build_p_chat(
    constraints: list[ProbeRow], chats: list[ProbeRow], inputs: ScreeningInputs, *, seed: int
) -> tuple[list[ProbeRow], dict[str, Any]]:
    """Every screened constraint prompt, filled to 1,500 with seeded multi-turn chats."""
    kept_constraints, constraint_drops = screen(constraints, inputs, x_task_rows=False)
    kept_chats, chat_drops = screen(chats, inputs, x_task_rows=False)
    filler = seeded_selection(kept_chats, P_CHAT_SIZE - len(kept_constraints), seed=seed, name="P_chat")
    return kept_constraints + filler, {
        "constraint_prompts": len(kept_constraints),
        "multi_turn_chats": len(filler),
        "constraint_drops": constraint_drops,
        "chat_drops": chat_drops,
    }


def build_p_think(
    benchmarks: dict[str, list[ProbeRow]], inputs: ScreeningInputs, *, seed: int
) -> tuple[list[ProbeRow], dict[str, Any]]:
    """AIME 2024, AIME 2025 and HMMT February 2025 whole, plus 210 seeded OlympiadBench problems."""
    rows: list[ProbeRow] = []
    counts: dict[str, Any] = {}
    for name in ("aime_2024", "aime_2025", "hmmt_feb_2025"):
        kept, drops = screen(benchmarks[name], inputs, x_task_rows=False)
        require_fixed_benchmark(benchmarks[name], kept, name)
        rows += kept
        counts[name] = len(kept)
    olympiad, drops = screen(benchmarks["olympiadbench"], inputs, x_task_rows=False)
    rows += seeded_selection(olympiad, P_THINK_OLYMPIADBENCH_SIZE, seed=seed, name="P_think/olympiadbench")
    counts["olympiadbench"] = P_THINK_OLYMPIADBENCH_SIZE
    counts["olympiadbench_drops"] = drops
    return rows, counts


def build_task_probe(
    rows: list[ProbeRow], inputs: ScreeningInputs, *, name: str, seed: int, x_task_rows: bool
) -> tuple[list[ProbeRow], dict[str, Any]]:
    kept, drops = screen(rows, inputs, x_task_rows=x_task_rows)
    return seeded_selection(kept, P_TASK_SIZE, seed=seed, name=name), drops


def require_pairwise_disjoint(sets: dict[str, list[ProbeRow]]) -> None:
    """Raise if any two probe sets share a prompt id or a 13-gram of user text."""
    names = sorted(sets)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            shared_ids = {row.probe_id for row in sets[first]} & {row.probe_id for row in sets[second]}
            shared_text = colliding_ids(sets[first], (row.user_text() for row in sets[second]))
            if shared_ids or shared_text:
                raise ValueError(f"{first} and {second} overlap: {sorted(shared_ids | shared_text)[:5]}")


def build_t(
    documents: Iterable[tuple[str, str]], count_tokens: Callable[[str], int]
) -> tuple[list[dict[str, str]], dict[str, int]]:
    """Whole documents per domain, in stream order, until each holds ``T_TOKENS_PER_DOMAIN`` tokens."""
    tokens = dict.fromkeys(T_DOMAINS, 0)
    rows: list[dict[str, str]] = []
    for domain, text in documents:
        if domain not in tokens or tokens[domain] >= T_TOKENS_PER_DOMAIN:
            continue
        tokens[domain] += count_tokens(text)
        rows.append({"domain": domain, "text": text})
        if all(count >= T_TOKENS_PER_DOMAIN for count in tokens.values()):
            return rows, tokens
    raise ValueError(f"the T stream ended before every domain reached its budget: {tokens}")


def p_agent_instance_ids(instance_ids: list[str], *, seed: int) -> list[str]:
    """150 seeded SWE-bench Verified instances, sorted for a stable task list."""
    if len(instance_ids) < P_AGENT_SIZE:
        raise ValueError(f"P_agent needs {P_AGENT_SIZE} instances, the source has {len(instance_ids)}")
    shuffled = sorted(instance_ids)
    random.Random(f"{seed}:P_agent").shuffle(shuffled)
    return sorted(shuffled[:P_AGENT_SIZE])


def p_agent_harbor_config(instance_ids: list[str], base_config_path: Path = HARBOR_SWE_BENCH_BASE_CONFIG) -> dict:
    """The eval-policy SWE-bench Verified Harbor config, pinned and narrowed to the P_agent tasks."""
    config = yaml.safe_load(base_config_path.read_text())
    [dataset] = config["datasets"]
    if dataset["name"] != HARBOR_SWE_BENCH_DATASET:
        raise ValueError(f"{base_config_path} no longer points at {HARBOR_SWE_BENCH_DATASET}")
    config["datasets"] = [{"name": HARBOR_SWE_BENCH_DATASET, "ref": HARBOR_SWE_BENCH_REF, "task_names": instance_ids}]
    return config


def lines_sha256(records: Iterable[dict[str, Any]]) -> str:
    """The sha256 of the records as sorted-key JSON lines, newline-terminated."""
    digest = hashlib.sha256()
    for record in records:
        digest.update((json.dumps(record, sort_keys=True) + "\n").encode())
    return digest.hexdigest()


def write_rows(path: Path, records: list[dict[str, Any]]) -> str:
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    return lines_sha256(records)


def source_entry(spec: SourceSpec) -> dict[str, str | None]:
    return {"repo": spec.repo, "revision": spec.revision, "split": spec.split, "config": spec.config}


PROMPT_SOURCES = {
    "P_chat": (IFEVAL, IFBENCH, ULTRACHAT),
    "P_think": (AIME_2024, AIME_2025, HMMT_FEB_2025, OLYMPIADBENCH),
    "P_agent": (SWE_BENCH_VERIFIED,),
    "T": (PILE,),
}
