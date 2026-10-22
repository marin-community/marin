# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The R task family: long-thinking competition math with checkable final answers.

Problems are drawn from an open competition-math pool at a pinned revision (an
explicit build input, recorded in the build manifest). The benchmarks reserved for
``P_think`` — AIME, HMMT and OlympiadBench — are excluded by source tag and by
13-gram overlap against the reserved-benchmark file, which is a required build input
and must hold every problem of those benchmarks, not only the ``P_think`` subset:
rephrased reservations share long spans with the originals even when exact-text dedup
sees nothing. NuminaMath's ``olympiads`` tag names olympiad sources generally, not
OlympiadBench, so it is kept: it is the pool's main supply of problems hard enough to
need long thinking, and the overlap check removes its OlympiadBench collisions. Every kept answer is checked to be verifier-parsable before the task
ships. Grading goes through the math verifier MarinSkyRL already uses
(``marin-verifyit[math]``).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from datasets import load_dataset as hf_load_dataset
from verifyit.grade import InvalidTask
from verifyit.modes.extract import extract_boxed
from verifyit.modes.grade_math import grade_math_text_candidate
from verifyit.spec import MathSpec

# Source tags naming a benchmark reserved for P_think (substring match, so "amc_aime"
# collides); everything else is screened by n-gram overlap.
P_THINK_SOURCES = ("aime", "hmmt", "olympiadbench")

# Word n-gram length for the P_think overlap check.
OVERLAP_N_GRAM = 13

TRAIN_SPLIT = "train"
PROBE_SPLIT = "probe"


@dataclass(frozen=True)
class RPoolSpec:
    """One competition-math pool at a pinned revision."""

    repo: str
    revision: str
    config: str | None = None
    split: str = "train"


# The open competition-math pool R draws from, at a pinned revision (Apache-2.0;
# rows carry ``source``, ``problem`` and a CoT ``solution`` ending in a boxed answer).
R_POOL_SPEC = RPoolSpec(repo="AI-MO/NuminaMath-CoT", revision="9d8d210c9f6a36c8f3cd84045668c9b7800ef517")


@dataclass(frozen=True)
class RTask:
    """One competition-math task with its checkable final answer, prompt, and split."""

    task_id: str
    problem: str
    answer: str
    source: str
    split: str
    prompt: str


def normalize_problem(text: str) -> str:
    """Lowercase, collapse whitespace, and strip punctuation/LaTeX markup so that
    re-typeset duplicates of one problem collide."""
    stripped = re.sub(r"\\[a-zA-Z]+", " ", text)
    return " ".join("".join(c if c.isalnum() else " " for c in stripped.lower()).split())


def is_reserved_for_probe(source: str) -> bool:
    """Whether a problem's source tag names a P_think-reserved pool."""
    return any(reserved in source.lower() for reserved in P_THINK_SOURCES)


def ngrams(text: str, n: int = OVERLAP_N_GRAM) -> frozenset[tuple[str, ...]]:
    """The word n-grams of ``text``; a text shorter than ``n`` tokens contributes its
    full token tuple, so short exact duplicates still collide."""
    tokens = text.split()
    if not tokens:
        return frozenset()
    if len(tokens) < n:
        return frozenset({tuple(tokens)})
    return frozenset(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def r_prompt(problem: str) -> str:
    """The canonical user prompt for one R task."""
    return f"{problem.strip()}\n\nSolve the problem step by step, then give the final answer in \\boxed{{}}."


def answer_is_verifier_parsable(answer: str) -> bool:
    """Whether the math verifier can grade ``answer`` as a checkable final answer.

    Three requirements: the reference must parse as a math expression at all (answers
    like empty or malformed LaTeX raise ``InvalidTask``), a correct boxed restatement
    must grade positive, and a mutated answer must grade zero — an answer that cannot
    be told apart from a wrong one cannot check completions.
    """
    correct = f"The final answer is $\\boxed{{{answer}}}$."
    mutated = f"The final answer is $\\boxed{{{answer}X'}}$."
    try:
        if grade_math_text_candidate(MathSpec(expected=answer), correct).reward <= 0:
            return False
    except InvalidTask:
        return False
    try:
        mutated_reward = grade_math_text_candidate(MathSpec(expected=answer), mutated).reward
    except InvalidTask:
        return True
    return mutated_reward <= 0


def load_r_pool(
    spec: RPoolSpec = R_POOL_SPEC,
    *,
    limit: int | None = None,
    load_dataset: Callable = hf_load_dataset,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Load the pinned pool and adapt its rows to ``{problem, answer, source}``.

    The answer is the last ``\\boxed{}`` of the solution; rows without one have no
    checkable final answer and are dropped, and the drop is counted. Returns the rows
    and the manifest entry that records the pinned revision.
    """
    dataset = load_dataset(spec.repo, spec.config, revision=spec.revision, split=spec.split, streaming=True)
    rows: list[dict[str, str]] = []
    dropped_without_boxed_answer = 0
    for row in dataset:
        problem = str(row.get("problem", "")).strip()
        answer = (extract_boxed(str(row.get("solution", ""))) or "").strip()
        if not problem or not answer:
            dropped_without_boxed_answer += 1
            continue
        rows.append({"problem": problem, "answer": answer, "source": str(row.get("source", ""))})
        if limit is not None and len(rows) >= limit:
            break
    if not rows:
        raise ValueError(f"{spec.repo}@{spec.revision}: every row was dropped; no R tasks can be drawn")
    manifest_entry = {
        "repo": spec.repo,
        "revision": spec.revision,
        "config": spec.config,
        "split": spec.split,
        "rows": len(rows),
        "dropped_without_boxed_answer": dropped_without_boxed_answer,
    }
    return rows, manifest_entry


def _json_row(line: str) -> dict | None:
    line = line.strip()
    return json.loads(line) if line else None


def draw_r_tasks(
    pool_rows: list[dict[str, str]],
    *,
    seed: int,
    reserved_pool_path: str | Path,
    probe_fraction: float = 0.1,
) -> tuple[list[RTask], dict[str, int]]:
    """Draw verified R tasks from pool rows of ``{problem, answer, source}``.

    The reserved-benchmark file is a required build input: a missing file raises rather
    than silently skipping the exclusion. Rows are dropped — each reason counted —
    when their source tag names a P_think-reserved pool, when any 13-gram of the
    problem overlaps the P_think pool (which catches rephrased reservations), when the
    problem duplicates an earlier row, or when the answer is not verifier-parsable.
    Splits are seeded hashes of the normalized problem, so one seed reproduces a
    split exactly and a different seed re-draws it.
    """
    path = Path(reserved_pool_path)
    if not path.exists():
        raise FileNotFoundError(f"required reserved-benchmark exclusion input is missing: {path}")
    p_think_ngrams: frozenset[tuple[str, ...]] = frozenset()
    for line in path.read_text().splitlines():
        row = _json_row(line)
        if row:
            p_think_ngrams |= ngrams(normalize_problem(str(row.get("problem", ""))))

    counts = {
        "kept": 0,
        "dropped_malformed": 0,
        "dropped_reserved_source": 0,
        "dropped_p_think_overlap": 0,
        "dropped_duplicate": 0,
        "dropped_unparsable_answer": 0,
    }
    tasks: list[RTask] = []
    seen: set[str] = set()
    for row in pool_rows:
        problem = str(row.get("problem", "")).strip()
        answer = str(row.get("answer", "")).strip()
        source = str(row.get("source", ""))
        if not problem or not answer:
            counts["dropped_malformed"] += 1
            continue
        if is_reserved_for_probe(source):
            counts["dropped_reserved_source"] += 1
            continue
        normalized = normalize_problem(problem)
        if ngrams(normalized) & p_think_ngrams:
            counts["dropped_p_think_overlap"] += 1
            continue
        if normalized in seen:
            counts["dropped_duplicate"] += 1
            continue
        if not answer_is_verifier_parsable(answer):
            counts["dropped_unparsable_answer"] += 1
            continue
        seen.add(normalized)
        digest = hashlib.sha256(f"{seed}:{normalized}".encode()).hexdigest()
        split = PROBE_SPLIT if (int(digest[:8], 16) % 10_000) < probe_fraction * 10_000 else TRAIN_SPLIT
        tasks.append(
            RTask(
                task_id=f"r-{digest[:16]}",
                problem=problem,
                answer=answer,
                source=source,
                split=split,
                prompt=r_prompt(problem),
            )
        )
        counts["kept"] += 1
    return tasks, counts


def verify_final_answer(problem_answer: str, completion: str) -> bool:
    """Grade one completion's final answer against the task's stored answer with the
    math verifier MarinSkyRL uses."""
    reward = grade_math_text_candidate(MathSpec(expected=problem_answer), completion)
    return reward.reward > 0


def write_split(tasks: list[RTask], output_dir: Path) -> dict[str, int]:
    """Write train and probe jsonl files; returns the per-split counts."""
    counts = {TRAIN_SPLIT: 0, PROBE_SPLIT: 0}
    for split in (TRAIN_SPLIT, PROBE_SPLIT):
        with (output_dir / f"r_{split}.jsonl").open("w") as handle:
            for task in tasks:
                if task.split != split:
                    continue
                handle.write(json.dumps(asdict(task), sort_keys=True) + "\n")
                counts[split] += 1
    if counts[TRAIN_SPLIT] == 0 or counts[PROBE_SPLIT] == 0:
        raise ValueError(f"splits must both be nonempty, got {counts}")
    return counts


def build_r_tasks(
    output_dir: Path | str,
    *,
    seed: int,
    reserved_pool_path: str | Path,
    probe_fraction: float = 0.1,
    spec: RPoolSpec = R_POOL_SPEC,
    limit: int | None = None,
    load_dataset: Callable = hf_load_dataset,
) -> dict[str, Any]:
    """Load the pinned pool, draw the R tasks, write both splits and ``manifest.json``.

    The manifest records the pinned pool revision, the seed, every exclusion count,
    and the split sizes — the build inputs and outcomes the run manifest points at.
    """
    output_path = Path(output_dir)
    pool_rows, pool_entry = load_r_pool(spec, limit=limit, load_dataset=load_dataset)
    tasks, counts = draw_r_tasks(
        pool_rows, seed=seed, reserved_pool_path=reserved_pool_path, probe_fraction=probe_fraction
    )
    split_counts = write_split(tasks, output_path)
    manifest = {
        "pool": pool_entry,
        "seed": seed,
        "probe_fraction": probe_fraction,
        "exclusions": counts,
        "splits": split_counts,
        "grader": "verifyit.modes.grade_math.grade_math_text_candidate",
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
