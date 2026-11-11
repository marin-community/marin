# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Find the NuminaMath-LEAN rows that state a problem of the frozen Lean eval set, and write ``eval_overlap.json``.

A row overlaps an eval problem when their theorem statements are the same text once comments,
whitespace and binder names are removed, or when their informal problems share at least ``THRESHOLD`` of their
five-token shingles (Jaccard similarity). The second test catches the same competition problem stated
differently in Lean. Its threshold is low on purpose: between 0.2 and 0.5 about half the matches are the
same problem reworded and the rest are other problems on the same template, which costs a few training
rows. It cannot see an eval problem without informal text (miniF2F leaves some theorems without a
docstring). Run it from a Marin checkout against the files in ``PINNED_INPUTS``::

    uv run python -m experiments.post_training.task_curation.datasets.lean.eval_overlap \\
        train-00000-of-00001.parquet --minif2f Test.lean Valid.lean --proofnet proofnet-verified.jsonl \\
        --putnam putnam.json

Each output line is one match; the file records each input's URL and SHA-256 and the overlapping uuids.
"""

import argparse
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.datasets.lean.evals import (
    MINIF2F_FILES_URL,
    PUTNAMBENCH_REVISION,
    minif2f_rows,
)
from experiments.post_training.task_curation.datasets.lean.lean_grade import TARGET_NAME, without_comments
from experiments.post_training.task_curation.datasets.lean.numina import (
    EVAL_OVERLAP_FILE,
    NUMINA_FILE,
    NUMINA_REVISION,
    is_human_proof_row,
)
from experiments.post_training.task_curation.datasets.lean.proofs import PROOFNET_FILE, PROOFNET_VERIFIED_REVISION

PINNED_INPUTS = {
    "train-00000-of-00001.parquet": (
        f"https://huggingface.co/datasets/project-numina/NuminaMath-LEAN/resolve/{NUMINA_REVISION}/{NUMINA_FILE}"
    ),
    "Test.lean": f"{MINIF2F_FILES_URL}/Test.lean",
    "Valid.lean": f"{MINIF2F_FILES_URL}/Valid.lean",
    PROOFNET_FILE: (
        f"https://huggingface.co/datasets/marcusm117/ProofNet-Verified/resolve/{PROOFNET_VERIFIED_REVISION}/{PROOFNET_FILE}"
    ),
    "putnam.json": (
        f"https://raw.githubusercontent.com/trishullab/PutnamBench/{PUTNAMBENCH_REVISION}/informal/putnam.json"
    ),
}
"""Where each input comes from, by file name: the dataset, then miniF2F as DeepMind maintains it, the
ProofNet-Verified source and PutnamBench's informal statements."""

THRESHOLD = 0.2
SHINGLE = 5
DECLARATION = re.compile(rf"^[ \t]*(?:theorem|lemma)\s+(?P<name>{TARGET_NAME.pattern})", re.MULTILINE)
PROOF_START = re.compile(r"(?s)^(?P<type>.*?):=")
"""A statement runs through its first ``:=``; a tactic block or a term proof follows."""
BINDER_NAMES = re.compile(r"[({\[⦃]\s*(?P<names>[^:()\[\]{}⦃⦄]+?)\s*:")
"""The names a binder group introduces, as in ``(a b : Nat)``."""
IDENTIFIER = re.compile(r"[^\W\d]\w*'*")
TOKEN = re.compile(r"\\[a-z]+|[^\W_]+|[=+*/<>^-]")
LATEX_LAYOUT = re.compile(r"\\(?:left|right|displaystyle|quad|qquad|textbf|mathrm|textrm|text|operatorname)\b")


def statement_key(declaration: str) -> str | None:
    """The last theorem's binders and type, comment-free, with its binder names numbered in order and no
    whitespace, so that statements that differ only in names and spacing agree."""
    body = without_comments(declaration)
    found = list(DECLARATION.finditer(body))
    match = PROOF_START.match(body[found[-1].end() :]) if found else None
    if match is None:
        return None
    text = match["type"]
    names = [name for group in BINDER_NAMES.finditer(text) for name in group["names"].split()]
    for index, name in enumerate(dict.fromkeys(name for name in names if IDENTIFIER.fullmatch(name))):
        text = re.sub(rf"(?<![\w.']){re.escape(name)}(?![\w'])", f"#{index}", text)
    return "".join(text.split())


def shingles(text: str) -> frozenset[tuple[str, ...]]:
    text = LATEX_LAYOUT.sub(" ", unicodedata.normalize("NFKC", text).casefold().replace(r"\dfrac", r"\frac"))
    tokens = TOKEN.findall(text)
    return frozenset(zip(*(tokens[i:] for i in range(SHINGLE)), strict=False))


def lean_problems(path: Path) -> list[dict]:
    """Each theorem of a miniF2F file with its docstring, the informal problem, as the miniF2F eval reads them."""
    return [
        {"eval": f"miniF2F:{row['name']}", "formal": row["formal_statement"], "informal": row["informal"]}
        for row in minif2f_rows(StoragePath(str(path)))
    ]


def eval_problems(minif2f: list[Path], proofnet: Path, putnam: Path) -> list[dict]:
    problems = [problem for path in minif2f for problem in lean_problems(path)]
    for line in proofnet.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            problems.append(
                {"eval": f"ProofNet:{row['name']}", "formal": row["formal_stmt"], "informal": row["informal_stmt"]}
            )
    for row in json.loads(putnam.read_text()):
        problems.append(
            {"eval": f"PutnamBench:{row['problem_name']}", "formal": "", "informal": row["informal_statement"]}
        )
    return problems


def overlaps(rows: list[dict], problems: list[dict]) -> list[dict]:
    """Every (row, eval problem) pair with the same statement or similar informal text."""
    by_statement, by_shingle, grams = defaultdict(set), defaultdict(set), []
    for index, problem in enumerate(problems):
        if key := statement_key(problem["formal"]):
            by_statement[key].add(index)
        grams.append(shingles(problem["informal"]))
        for gram in grams[-1]:
            by_shingle[gram].add(index)
    found = []
    for row in rows:
        same = by_statement.get(statement_key(row["formal_statement"]) or "", set())
        own = shingles(row["problem"] or "")
        shared = defaultdict(int)
        for gram in own:
            for index in by_shingle.get(gram, ()):
                shared[index] += 1
        similar = {index: count / (len(own) + len(grams[index]) - count) for index, count in shared.items()}
        for index in sorted(same | {index for index, value in similar.items() if value >= THRESHOLD}):
            match = {"uuid": row["uuid"], "eval": problems[index]["eval"], "same_statement": index in same}
            found.append({**match, "jaccard": round(similar.get(index, 0.0), 3)})
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("numina", type=Path)
    parser.add_argument("--minif2f", type=Path, nargs="+", required=True)
    parser.add_argument("--proofnet", type=Path, required=True)
    parser.add_argument("--putnam", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=EVAL_OVERLAP_FILE)
    args = parser.parse_args()
    rows = [row for row in pq.read_table(args.numina).to_pylist() if is_human_proof_row(row, None)]
    found = overlaps(rows, eval_problems(args.minif2f, args.proofnet, args.putnam))
    for match in found:
        print(json.dumps(match))
    inputs = [args.numina, *args.minif2f, args.proofnet, args.putnam]
    if unknown := [path.name for path in inputs if path.name not in PINNED_INPUTS]:
        parser.error(f"not a pinned input: {unknown}")
    record = {
        "inputs": {
            path.name: {"url": PINNED_INPUTS[path.name], "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in inputs
        },
        "threshold": THRESHOLD,
        "overlapping": sorted({match["uuid"] for match in found}),
    }
    args.output.write_text(json.dumps(record, indent=1) + "\n")


if __name__ == "__main__":
    main()
