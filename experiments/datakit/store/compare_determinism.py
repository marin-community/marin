# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Diff two clustered-store builds for bit-level determinism.

Reads two :class:`ClusteredStoreData` artifacts (two independent builds over the
same input) and reports whether their per-bucket doc/token counts, totals, and
dedup/decontam counters are identical. Used to verify datakit build determinism
(marin#6798). Run on-cluster where the store bucket's creds are present::

    python -m experiments.datakit.store.compare_determinism <store_a> <store_b>
"""

import sys

from marin.execution.artifact import read_artifact

from experiments.datakit.store.datakit_store import ClusteredStoreData

_COUNTERS = (
    "datakit_store/records_in",
    "datakit_store/records_out",
    "datakit_store/contaminated_dropped",
    "datakit_store/dedup_noncanonical_dropped",
)


def _bucket_map(store: ClusteredStoreData) -> dict[tuple[int, int], tuple[int, int]]:
    return {(b.cluster_id, b.quality_bucket): (b.total_elements, b.total_tokens) for b in store.buckets}


def main() -> None:
    path_a, path_b = sys.argv[1], sys.argv[2]
    a = read_artifact(path_a, ClusteredStoreData)
    b = read_artifact(path_b, ClusteredStoreData)

    ba, bb = _bucket_map(a), _bucket_map(b)
    docs_a = sum(v[0] for v in ba.values())
    docs_b = sum(v[0] for v in bb.values())
    toks_a = sum(v[1] for v in ba.values())
    toks_b = sum(v[1] for v in bb.values())

    print(f"A = {path_a}")
    print(f"B = {path_b}")
    print(f"total_docs   A={docs_a} B={docs_b} {'MATCH' if docs_a == docs_b else 'DIFFER Δ=' + str(docs_b - docs_a)}")
    print(f"total_tokens A={toks_a} B={toks_b} {'MATCH' if toks_a == toks_b else 'DIFFER Δ=' + str(toks_b - toks_a)}")
    print(f"n_buckets    A={len(ba)} B={len(bb)} {'MATCH' if len(ba) == len(bb) else 'DIFFER'}")

    print("counters:")
    for k in _COUNTERS:
        va, vb = a.counters.get(k), b.counters.get(k)
        print(f"  {k}: A={va} B={vb} {'MATCH' if va == vb else 'DIFFER'}")

    keys = sorted(set(ba) | set(bb))
    diffs = [(k, ba.get(k), bb.get(k)) for k in keys if ba.get(k) != bb.get(k)]
    print(f"per-bucket: {len(keys)} keys, {len(diffs)} differ")
    for k, va, vb in diffs[:40]:
        print(f"  cluster={k[0]} quality={k[1]}: A={va} B={vb}")

    identical = docs_a == docs_b and toks_a == toks_b and not diffs and ba == bb
    print(f"\nRESULT: {'IDENTICAL — deterministic' if identical else 'NON-IDENTICAL'}")


if __name__ == "__main__":
    main()
