# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Weighted subsampling of the hero data mixture to a fixed token budget.

To retokenize ~100B tokens (~0.5% of the 23T store) at the step-108000 mixture distribution
without a global two-pass count, keep each surviving document independently with a per-cell
Bernoulli probability

    p[cell] = min(1, target_tokens * weight[cell] / available_tokens[cell])

where ``weight`` is the step-108k ("main") phase weight over the 200 ``(cluster, quality)`` cells
and ``available_tokens`` is the store's per-cell token count. The expected sampled tokens per cell
is then ``p[cell] * available_tokens[cell] ≈ target_tokens * weight[cell]``, i.e. the mixture
distribution, and the expected total is ``target_tokens``. This is embarrassingly parallel: each
shard decides independently, no coordination.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Iterator

import pyarrow.compute as pc
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath

from experiments.datakit.store.datakit_store import (
    _load_cluster_table,
    _load_decon_table,
    _load_exact_duplicates,
    _load_quality_table,
    _load_verified_duplicates,
)


def cell_name(cluster: int, quality: int) -> str:
    return f"c{cluster:02d}q{quality}"


def keep_probabilities(mix_json_path: str, target_tokens: int, phase: str = "main") -> dict[str, float]:
    """Per-cell Bernoulli keep probability for a ``target_tokens`` weighted sample.

    ``phase`` selects the phase whose weights define the target distribution (the step-108k
    distribution is the "main" phase of the September mixture).
    """
    with open(mix_json_path) as handle:
        spec = json.load(handle)
    available = spec["available_tokens"]
    phases = {p["name"]: p["weights"] for p in spec["phases"]}
    if phase not in phases:
        raise ValueError(f"phase {phase!r} not in {sorted(phases)}")
    weights = phases[phase]
    if set(weights) != set(available):
        raise ValueError("weight cells do not match available_tokens cells")
    return {
        cell: 0.0 if available[cell] <= 0 else min(1.0, target_tokens * weights[cell] / available[cell])
        for cell in available
    }


def expected_sampled_tokens(keep_probs: dict[str, float], mix_json_path: str) -> float:
    with open(mix_json_path) as handle:
        available = json.load(handle)["available_tokens"]
    return sum(keep_probs[c] * available[c] for c in keep_probs)


def _iter_normalized_docs(path: str) -> Iterator[tuple[str, str]]:
    """Yield ``(id, text)`` per document from one normalized parquet shard, in stored order."""
    with StoragePath(path).open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        for batch in parquet.iter_batches(columns=["id", "text"]):
            ids = batch.column("id").to_pylist()
            texts = batch.column("text").to_pylist()
            yield from zip(ids, texts, strict=True)


def align_source_shards(source_name: str, dirs: dict[str, str]) -> list[dict[str, str]]:
    """Build per-shard specs keyed on the normalized shards, aligning attrs by basename.

    ``dirs`` maps {normalized, decontam, cluster, quality, exact_dedup, dedup} -> directory.
    Mirrors the datakit store's basename co-partitioning, but enumerates normalized (text) shards
    since the sampler works on raw text rather than tokenized attributes.
    """
    norm_dir = dirs["normalized"].rstrip("/")
    shards = sorted(str(m) for m in StoragePath(f"{norm_dir}/*.parquet").glob())
    if not shards:
        raise FileNotFoundError(f"{source_name}: no normalized shards under {norm_dir}")
    specs = []
    for shard in shards:
        base = os.path.basename(shard)
        specs.append(
            {
                "normalized": shard,
                "decontam": f"{dirs['decontam'].rstrip('/')}/{base}",
                "cluster": f"{dirs['cluster'].rstrip('/')}/{base}",
                "quality": f"{dirs['quality'].rstrip('/')}/{base}",
                "exact_dedup": f"{dirs['exact_dedup'].rstrip('/')}/{base}",
                "dedup": f"{dirs['dedup'].rstrip('/')}/{base}",
                "source_name": source_name,
                "basename": base,
            }
        )
    return specs


def sample_shard(
    spec: dict[str, str], cluster_col: str, keep_probs: dict[str, float], seed: int
) -> Iterator[tuple[str, str]]:
    """Yield ``(id, text)`` for surviving docs kept by the per-cell Bernoulli draw.

    Survival matches the datakit store (not contaminated, not exact/verified duplicate); routing is
    positional against the co-partitioned dense attribute tables, verified by id equality.
    """
    decon_ids, contaminated = _load_decon_table(spec["decontam"])
    cluster_ids, cluster_vals = _load_cluster_table(spec["cluster"], cluster_col)
    quality_ids, quality_buckets = _load_quality_table(spec["quality"])
    n = len(decon_ids)
    where = f"{spec['source_name']}/{spec['basename']}"
    if not (len(cluster_ids) == len(quality_ids) == n):
        raise RuntimeError(f"{where}: dense-table row count mismatch -- co-partitioning broken")
    if not pc.all(pc.equal(decon_ids, cluster_ids)).as_py() or not pc.all(pc.equal(decon_ids, quality_ids)).as_py():
        raise RuntimeError(f"{where}: attribute id mismatch -- co-partitioning broken")
    expected_ids = decon_ids.to_pylist()
    exact_dups = _load_exact_duplicates(spec["exact_dedup"])
    verified_dups = _load_verified_duplicates(spec["dedup"])
    rng = random.Random(f"{spec['source_name']}/{spec['basename']}/{seed}")
    position = 0
    for doc_id, text in _iter_normalized_docs(spec["normalized"]):
        if position >= n:
            raise RuntimeError(f"{where}: normalized has more docs than attr rows ({n}) -- co-partitioning broken")
        if doc_id != expected_ids[position]:
            raise RuntimeError(
                f"{where}: normalized/attr id mismatch at {position}: {doc_id!r} != {expected_ids[position]!r}"
            )
        pos, position = position, position + 1
        if contaminated[pos] or doc_id in verified_dups or doc_id in exact_dups:
            continue
        cell = cell_name(int(cluster_vals[pos]), int(quality_buckets[pos]))
        if rng.random() < keep_probs.get(cell, 0.0):
            yield doc_id, text
    if position != n:
        raise RuntimeError(f"{where}: normalized docs ({position}) != attr rows ({n}) -- co-partitioning broken")
