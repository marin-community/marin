# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate the NuminaMath-LEAN auto-informalization duplicate and split manifest.

This command reads the pinned source parquet, identifies removable exact duplicates, groups strong
Lean/text near matches into the same train or evaluation split, and writes the manifest consumed by
``autoinformalization.py``. It does not create TaskSpecs or grade model responses.

Exact copies share a source UUID or both normalized statements. Strong near-matches require at least
``NEAR_THRESHOLD`` Jaccard similarity in both informal and formal five-token n-grams. Exact copies
keep one preferred row; near-match variants remain, but their connected component stays in one split.
Identical normalized Lean statements stay in one split even when the human references differ.
Rows sharing a UUID with a passing Numina proof task and their entire connected components stay in
training, because proof prompts expose the human problem alongside the Lean statement.

Run against the pinned NuminaMath parquet::

    uv run python -m experiments.post_training.task_curation.datasets.lean.autoinformalization_manifest \
        train-00000-of-00001.parquet
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from experiments.post_training.task_curation.datasets.lean.lean_problem_similarity import (
    jaccard,
    ngrams,
    normalized_text,
    row_key,
    statement_key,
    statement_ngrams,
)
from experiments.post_training.task_curation.datasets.lean.numina import PASSING_PROOFS

NEAR_THRESHOLD = 0.9
RARE_NGRAMS = 8
MAX_NGRAM_FREQUENCY = 128
EVAL_MODULUS = 20
SPLIT_SALT = "numina-autoinformalization-v1"
GROUND_TRUTH_RANK = {"complete": 0, "with_sorry": 1, "statement": 2}
PROOF_TRAIN_UUIDS = frozenset(key.split(":", 1)[0] for key in PASSING_PROOFS)


class Components:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def join(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def split_for(key: str) -> str:
    digest = hashlib.sha256(f"{SPLIT_SALT}:{key}".encode()).digest()
    return "eval" if int.from_bytes(digest[:4]) % EVAL_MODULUS == 0 else "train"


def human_statement_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = []
    for index, row in enumerate(rows):
        indexed = {**row, "_numina_index": index}
        if row.get("author") == "human" and row_key(indexed) is not None:
            selected.append(indexed)
    return selected


def preferred(index: int, rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> tuple[int, str]:
    return GROUND_TRUTH_RANK.get(rows[index].get("ground_truth_type"), 3), keys[index]


def grouped(values: Sequence[str | None]) -> Iterable[list[int]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(values):
        if value:
            buckets[value].append(index)
    return (indices for indices in buckets.values() if len(indices) > 1)


def candidate_pairs(grams: Sequence[frozenset[tuple[str, ...]]]) -> set[tuple[int, int]]:
    frequency = Counter(gram for row in grams for gram in row)
    postings: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(grams):
        for gram in row:
            postings[gram].append(index)
    pairs = set()
    for left, row in enumerate(grams):
        selected = sorted(row, key=lambda gram: (frequency[gram], gram))[:RARE_NGRAMS]
        for gram in selected:
            if frequency[gram] > MAX_NGRAM_FREQUENCY:
                continue
            pairs.update((left, right) for right in postings[gram] if left < right)
    return pairs


def exact_edges(
    rows: Sequence[dict[str, Any]], formal: Sequence[str | None], informal: Sequence[str]
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    seen = set()
    for rule, groups in (
        ("source_id", grouped([row["uuid"] for row in rows])),
        (
            "exact_pair",
            grouped([f"{left}\0{right}" if left else None for left, right in zip(formal, informal, strict=True)]),
        ),
    ):
        for indices in groups:
            first = indices[0]
            for right in indices[1:]:
                pair = (first, right, rule)
                if pair not in seen:
                    edges.append({"left": first, "right": right, "rule": rule})
                    seen.add(pair)
    return edges


def near_edges(
    informal: Sequence[frozenset[tuple[str, ...]]], formal: Sequence[frozenset[tuple[str, ...]]]
) -> list[dict[str, Any]]:
    candidates = candidate_pairs(informal) | candidate_pairs(formal)
    edges: list[dict[str, Any]] = []
    for left, right in sorted(candidates):
        informal_score = jaccard(informal[left], informal[right])
        formal_score = jaccard(formal[left], formal[right])
        if informal_score >= NEAR_THRESHOLD and formal_score >= NEAR_THRESHOLD:
            edges.append(
                {
                    "left": left,
                    "right": right,
                    "rule": "joint_near_match",
                    "informal_jaccard": round(informal_score, 4),
                    "formal_jaccard": round(formal_score, 4),
                }
            )
    return edges


def build_manifest(rows: Iterable[dict[str, Any]], input_record: Mapping[str, str] | None = None) -> dict[str, Any]:
    selected = human_statement_rows(rows)
    keys = [row_key(row) for row in selected]
    assert all(keys)
    row_keys = [key for key in keys if key is not None]
    formal_keys = [statement_key(row["formal_statement"]) for row in selected]
    informal_keys = [normalized_text(row["problem"]) for row in selected]
    informal_grams = [ngrams(row["problem"]) for row in selected]
    formal_grams = [statement_ngrams(row["formal_statement"]) for row in selected]

    exact = exact_edges(selected, formal_keys, informal_keys)
    exact_components = Components(len(selected))
    for edge in exact:
        exact_components.join(edge["left"], edge["right"])
    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(selected)):
        groups[exact_components.find(index)].append(index)
    representatives = {
        root: min(indices, key=lambda index: preferred(index, selected, row_keys)) for root, indices in groups.items()
    }

    near = near_edges(informal_grams, formal_grams)
    clusters = Components(len(selected))
    for edge in (*exact, *near):
        clusters.join(edge["left"], edge["right"])
    for indices in grouped(formal_keys):
        for right in indices[1:]:
            if clusters.find(indices[0]) != clusters.find(right):
                near.append({"left": indices[0], "right": right, "rule": "same_formal_statement"})
                clusters.join(indices[0], right)
    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(selected)):
        members[clusters.find(index)].append(index)

    records: list[dict[str, Any]] = []
    proof_train_rows = {row_keys[index] for index, row in enumerate(selected) if row["uuid"] in PROOF_TRAIN_UUIDS}
    for indices in members.values():
        if len(indices) == 1:
            continue
        kept = sorted({representatives[exact_components.find(index)] for index in indices})
        removed = sorted(set(indices) - set(kept))
        member_set = set(indices)
        matches: list[dict[str, Any]] = []
        for edge in (*exact, *near):
            if edge["left"] in member_set and edge["right"] in member_set:
                matches.append({**edge, "left": row_keys[edge["left"]], "right": row_keys[edge["right"]]})
        cluster_id = hashlib.sha256("\0".join(sorted(row_keys[index] for index in indices)).encode()).hexdigest()[:16]
        records.append(
            {
                "id": cluster_id,
                "split": (
                    "train"
                    if any(row_keys[index] in proof_train_rows for index in indices)
                    else split_for(min(row_keys[index] for index in kept))
                ),
                "kept": [row_keys[index] for index in kept],
                "removed": [
                    {
                        "row": row_keys[index],
                        "kept": row_keys[representatives[exact_components.find(index)]],
                    }
                    for index in removed
                ],
                "matches": sorted(matches, key=lambda match: (match["left"], match["right"], match["rule"])),
            }
        )
    records.sort(key=lambda record: record["id"])
    clustered = {
        key: record["split"]
        for record in records
        for key in (*record["kept"], *(item["row"] for item in record["removed"]))
    }
    removed = {item["row"] for record in records for item in record["removed"]}
    split_counts = Counter(
        clustered.get(key, "train" if key in proof_train_rows else split_for(key))
        for key in row_keys
        if key not in removed
    )
    return {
        "input": dict(input_record or {}),
        "settings": {
            "near_threshold": NEAR_THRESHOLD,
            "ngram_size": 5,
            "rare_ngrams": RARE_NGRAMS,
            "max_ngram_frequency": MAX_NGRAM_FREQUENCY,
            "eval_modulus": EVAL_MODULUS,
            "split_salt": SPLIT_SALT,
        },
        "counts": {
            "human_statement_rows": len(selected),
            "kept_rows": len(selected) - len(removed),
            "removed_exact_duplicates": len(removed),
            "train": split_counts["train"],
            "eval": split_counts["eval"],
        },
        "clusters": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("numina_autoinformalization_dedup.json"))
    args = parser.parse_args()
    data = args.parquet.read_bytes()
    manifest = build_manifest(
        pq.read_table(args.parquet).to_pylist(),
        {
            "url": (
                "https://huggingface.co/datasets/project-numina/NuminaMath-LEAN/resolve/"
                "51fa67f1f647ae1ecd81eef9f19306aa8a7b3a94/data/train-00000-of-00001.parquet"
            ),
            "sha256": hashlib.sha256(data).hexdigest(),
        },
    )
    args.output.write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()
