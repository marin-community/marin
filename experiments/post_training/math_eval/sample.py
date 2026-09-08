# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Seeded sampling that enforces the pool's persisted membership lock."""

import hashlib
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from experiments.post_training.math_eval.pool import SPLIT_PRIORITY, canonical_json


@dataclass(frozen=True)
class Sample:
    rows: list[dict[str, Any]]
    receipt: dict[str, Any]


def sample(
    manifest: Sequence[dict[str, Any]],
    selection: Mapping[str, Any],
    bins: Sequence[str],
    n: int,
    seed: int,
    *,
    model: str,
    split: str,
    exclude: frozenset[str] = frozenset(),
) -> Sample:
    """Return a deterministic selection and its call/result hashes.

    Any edited split assignment or gold fails the manifest hash check. Rejected
    audit items never enter a sample; unknown bins fail rather than silently
    substituting another split or duplicating available examples.
    """
    if split not in SPLIT_PRIORITY or model not in selection["prompt_template_ids"]:
        raise ValueError("Unknown model or split")
    if n <= 0 or not bins or len(set(bins)) != len(bins):
        raise ValueError("A sample requires positive n and distinct bins")
    ordered = sorted(manifest, key=lambda row: row["prompt_sha256"])
    manifest_hash = hashlib.sha256(canonical_json(ordered).encode()).hexdigest()
    if manifest_hash != selection["manifest_sha256"]:
        raise ValueError("Manifest does not match the frozen selection")
    locks = {name: [row["prompt_sha256"] for row in ordered if row["split"] == name] for name in SPLIT_PRIORITY}
    if locks != selection["rows"]:
        raise ValueError("Split membership does not match the frozen selection")
    if hashlib.sha256(canonical_json(locks).encode()).hexdigest() != selection["heldout_lock_sha256"]:
        raise ValueError("Heldout lock hash changed")
    rows = [
        row
        for row in ordered
        if row["split"] == split
        and row["bin"] in bins
        and row["prompt_sha256"] not in exclude
        and row["audit_status"] != "reject"
    ]
    if set(bins) != {row["bin"] for row in rows}:
        raise ValueError("Requested bin has no eligible rows in this split")
    if len(rows) < n:
        raise ValueError(f"Requested {n} rows but only {len(rows)} remain in {split}")
    chosen = random.Random(seed).sample(rows, n)
    call = {
        "pool_version": selection["pool_version"],
        "manifest_sha256": manifest_hash,
        "bins": sorted(bins),
        "n": n,
        "seed": seed,
        "model": model,
        "split": split,
        "exclude": sorted(exclude),
    }
    ids = [row["prompt_sha256"] for row in chosen]
    return Sample(
        chosen,
        {
            "call": call,
            "call_sha256": hashlib.sha256(canonical_json(call).encode()).hexdigest(),
            "prompt_sha256": ids,
            "result_sha256": hashlib.sha256(canonical_json(ids).encode()).hexdigest(),
        },
    )
