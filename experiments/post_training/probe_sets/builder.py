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
from experiments.post_training.task_data.math_r import OVERLAP_N_GRAM, R_INSTRUCTION, ngrams, normalize_problem

P_CHAT_SIZE = 1_500
P_THINK_SIZE = 300
STAGE0_CONTAMINATION_SHARE = 0.5
"""A probe row is contaminated by Stage 0 when at least this share of its 13-grams occur
there (decision log 2026-10-24): stock competition wording matches a few grams, a
near-copy matches most of them. Task training splits keep the any-13-gram rule."""
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
    """13-grams of normalized text with the shared R instruction removed.

    Every R and P_think prompt ends with :data:`R_INSTRUCTION`, itself exactly one
    13-gram, so left in it would make every math probe collide with all of R train.
    """
    return ngrams(normalize_problem(text.replace(R_INSTRUCTION, "")), OVERLAP_N_GRAM)


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


def matched_gram_shares(rows: Iterable[ProbeRow], texts: Iterable[str]) -> dict[str, float]:
    """Per row with any match, the share of its 13-grams that occur in ``texts`` (streamed)."""
    rows = list(rows)
    owners: dict[tuple[str, ...], set[str]] = {}
    sizes: dict[str, int] = {}
    for row in rows:
        grams = overlap_grams(row.user_text())
        sizes[row.probe_id] = len(grams)
        for gram in grams:
            owners.setdefault(gram, set()).add(row.probe_id)
    matched: dict[str, set[tuple[str, ...]]] = {}
    for text in texts:
        for gram in overlap_grams(text):
            for probe_id in owners.get(gram, ()):
                matched.setdefault(probe_id, set()).add(gram)
    return {probe_id: len(grams) / sizes[probe_id] for probe_id, grams in matched.items()}


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


@dataclass(frozen=True)
class ScreeningHits:
    """Which candidate rows overlap each training source, computed in one pass per source."""

    screened_ids: frozenset[str]
    stage0: frozenset[str]
    stage0_shares: dict[str, float]
    task_train: dict[Path, frozenset[str]]
    x_train_file: Path
    x_train_functions: frozenset[tuple[str, Any]]


def screening_hits(candidates: Iterable[ProbeRow], inputs: ScreeningInputs) -> ScreeningHits:
    """Overlap of every candidate row of every probe set with Stage 0 and each task training file.

    Stage 0 and the task files are streamed once each for all candidates together;
    screening set by set would stream Stage 0 once per set. Stage 0 hits are rows at or
    above :data:`STAGE0_CONTAMINATION_SHARE`; every Stage 0 match is kept in
    ``stage0_shares`` for the manifest.
    """
    rows = list(candidates)
    ids = [row.probe_id for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("probe candidates must have unique probe ids")
    functions = frozenset(
        (task["template_name"], task["parameter"])
        for task in map(json.loads, inputs.x_train_file.read_text().splitlines())
    )
    shares = matched_gram_shares(rows, message_file_user_texts(inputs.stage0_files))
    return ScreeningHits(
        screened_ids=frozenset(ids),
        stage0=frozenset(probe_id for probe_id, share in shares.items() if share >= STAGE0_CONTAMINATION_SHARE),
        stage0_shares=shares,
        task_train={path: frozenset(colliding_ids(rows, task_file_prompts([path]))) for path in inputs.task_train_files},
        x_train_file=inputs.x_train_file,
        x_train_functions=functions,
    )


def screen(rows: list[ProbeRow], hits: ScreeningHits, *, x_task_rows: bool) -> tuple[list[ProbeRow], dict[str, int]]:
    """Drop rows that overlap Stage 0 or task training data; return the kept rows and drop counts.

    ``P_X`` rows (``x_task_rows``) are screened against ``X`` train by function identity
    instead of text, and against the other task files by text.
    """
    unscreened = {row.probe_id for row in rows} - hits.screened_ids
    if unscreened:
        raise ValueError(f"rows were not among the screened candidates: {sorted(unscreened)[:5]}")
    ids = {row.probe_id for row in rows}
    stage0 = ids & hits.stage0
    if x_task_rows:
        task_train = {
            row.probe_id
            for row in rows
            if (row.metadata["template_name"], row.metadata["parameter"]) in hits.x_train_functions
        }
        for path, path_hits in hits.task_train.items():
            if path != hits.x_train_file:
                task_train |= ids & path_hits
    else:
        task_train = set().union(*(ids & path_hits for path_hits in hits.task_train.values()))
    kept = [row for row in rows if row.probe_id not in stage0 | task_train]
    return kept, {"dropped_stage0_overlap": len(stage0), "dropped_task_train_overlap": len(task_train - stage0)}


def build_p_chat(
    constraints: list[ProbeRow], chats: list[ProbeRow], hits: ScreeningHits, *, seed: int
) -> tuple[list[ProbeRow], dict[str, Any]]:
    """Every screened constraint prompt, filled to 1,500 with seeded multi-turn chats."""
    kept_constraints, constraint_drops = screen(constraints, hits, x_task_rows=False)
    kept_chats, chat_drops = screen(chats, hits, x_task_rows=False)
    filler = seeded_selection(kept_chats, P_CHAT_SIZE - len(kept_constraints), seed=seed, name="P_chat")
    return kept_constraints + filler, {
        "constraint_prompts": len(kept_constraints),
        "multi_turn_chats": len(filler),
        "constraint_drops": constraint_drops,
        "chat_drops": chat_drops,
    }


def build_p_think(
    benchmarks: dict[str, list[ProbeRow]], hits: ScreeningHits, *, seed: int
) -> tuple[list[ProbeRow], dict[str, Any]]:
    """AIME 2024, AIME 2025 and HMMT February 2025 less their contaminated problems, filled to 300 from OlympiadBench."""
    rows: list[ProbeRow] = []
    counts: dict[str, Any] = {}
    for name in ("aime_2024", "aime_2025", "hmmt_feb_2025"):
        kept, drops = screen(benchmarks[name], hits, x_task_rows=False)
        rows += kept
        counts[name] = len(kept)
        counts[f"{name}_drops"] = drops
    olympiad, drops = screen(benchmarks["olympiadbench"], hits, x_task_rows=False)
    fill = P_THINK_SIZE - len(rows)
    rows += seeded_selection(olympiad, fill, seed=seed, name="P_think/olympiadbench")
    counts["olympiadbench"] = fill
    counts["olympiadbench_drops"] = drops
    return rows, counts


def build_task_probe(
    rows: list[ProbeRow], hits: ScreeningHits, *, name: str, seed: int, x_task_rows: bool
) -> tuple[list[ProbeRow], dict[str, Any]]:
    kept, drops = screen(rows, hits, x_task_rows=x_task_rows)
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
