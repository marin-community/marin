# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize keep-id lists for the paranoid short-doc math val sets.

"Paranoid" filter, per cutoff: keep a val doc iff it is fully contained in
validation windows (zero verbatim train leakage from window-split spill) AND
its max verified train Jaccard across all three subset scans (3, 4plus,
4plus_mind) is < cutoff. Docs must fit in one 4096-token val window, so these
sets are short-doc biased by construction.

Runs locally against the replay artifacts in scratch/ and writes one JSON per
cutoff. Upload the outputs to
gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/keep_ids/
for the build driver (scripts/analysis/build_decon_val_sets.py).

    .venv/bin/python scripts/analysis/build_paranoid_val_keep_ids.py
"""

import json
import logging
from pathlib import Path

import numpy as np
import xxhash

logger = logging.getLogger(__name__)

SCRATCH = Path("scratch")
FULLY_CONTAINED = SCRATCH / "nemotron_math_val_fully_contained_doc_indices.npy"
MAX_JACCARD_UNION = SCRATCH / "nemotron_math_val_doc_max_jaccard_union.npy"
ALL_IDS = SCRATCH / "nemotron_math_all_ids.npy"
DOC_OFFSETS = SCRATCH / "nemotron_math_doc_offsets.npy"
VAL_DOC_INDICES = SCRATCH / "nemotron_math_val_doc_indices.npy"

# Precomputed in the corrected 2026-06-07 paranoid matrix; hard asserts so a
# drifted input artifact fails loudly instead of producing a silently
# different set. NOTE: nemotron_math_doc_offsets.npy holds doc END offsets
# (len == num docs, offsets[-1] == total tokens); doc d spans
# [offsets[d-1], offsets[d]) with offsets[-1] meaning 0 for d == 0. The first
# build shipped keep lists computed as if it were a boundaries array — every
# span belonged to doc d+1 — and the driver's exact-count gate caught it.
EXPECTED = {
    0.5: {"docs": 13_947, "tokens": 10_282_799},
    0.75: {"docs": 28_089, "tokens": 20_782_728},
    0.9: {"docs": 33_196, "tokens": 25_346_090},
}
CUTOFF_TAG = {0.5: "j050", 0.75: "j075", 0.9: "j090"}


def keep_doc_indices(fully: np.ndarray, max_j: dict[int, float], cutoff: float) -> np.ndarray:
    """Fully-contained docs whose max train Jaccard is below the cutoff."""
    return np.array(sorted(d for d in fully.tolist() if max_j.get(d, 0.0) < cutoff), dtype=np.int64)


def keep_ids_digest(ids: list[str]) -> str:
    return xxhash.xxh3_128_hexdigest("\n".join(sorted(ids)).encode())


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    fully = np.load(FULLY_CONTAINED)
    union = np.load(MAX_JACCARD_UNION)
    max_j = {int(d): float(j) for d, j in zip(union["doc"], union["jaccard"], strict=True)}
    all_ids = np.load(ALL_IDS, mmap_mode="r")
    ends = np.load(DOC_OFFSETS).astype(np.int64)
    starts = np.concatenate([[0], ends[:-1]])
    val_doc_indices = set(np.load(VAL_DOC_INDICES).tolist())
    # Guard against boundary-convention drift: every fully-contained doc must
    # be one of the retokenization-verified val docs.
    assert set(fully.tolist()) <= val_doc_indices, "fully-contained docs not a subset of verified val docs"

    out_dir = SCRATCH / "decon_val_keep_ids"
    out_dir.mkdir(exist_ok=True)

    previous: set[str] | None = None
    for cutoff in sorted(EXPECTED):
        keep = keep_doc_indices(fully, max_j, cutoff)
        tokens = int((ends[keep] - starts[keep]).sum())
        assert len(keep) == EXPECTED[cutoff]["docs"], f"J>={cutoff}: {len(keep)} docs != {EXPECTED[cutoff]['docs']}"
        assert tokens == EXPECTED[cutoff]["tokens"], f"J>={cutoff}: {tokens} tokens != {EXPECTED[cutoff]['tokens']}"

        id_rows = np.asarray(all_ids[keep])
        ids = [f"{int(hi):016x}{int(lo):016x}" for hi, lo in id_rows]
        id_set = set(ids)
        assert len(id_set) == len(ids), "duplicate ids in keep set"
        if previous is not None:
            assert previous < id_set, "keep sets must be strictly nested across ascending cutoffs"
        previous = id_set

        payload = {
            "name": f"nemotron_math_paranoid_val_{CUTOFF_TAG[cutoff]}",
            "filter": "fully_contained_in_val_windows_and_max_train_jaccard_lt_cutoff",
            "cutoff": cutoff,
            "scan_subsets": ["3_284x71", "4plus_284x71", "4plus_mind_284x71"],
            "expected_docs": EXPECTED[cutoff]["docs"],
            "expected_tokens": EXPECTED[cutoff]["tokens"],
            "keep_ids_xxh3": keep_ids_digest(ids),
            "doc_indices": keep.tolist(),
            "max_jaccard_by_doc": {str(int(d)): max_j[int(d)] for d in keep if int(d) in max_j},
            "ids": ids,
        }
        out = out_dir / f"keep_ids_{CUTOFF_TAG[cutoff]}.json"
        with open(out, "w") as f:
            json.dump(payload, f)
        logger.info("J>=%s: %d docs, %d tokens -> %s", cutoff, len(keep), tokens, out)


if __name__ == "__main__":
    main()
