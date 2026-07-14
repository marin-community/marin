# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stress fuzzy-dedup determinism on synthetic heavy-duplicate data (marin#6798).

The curated testbed samples contain few/no fuzzy duplicates, so they do not
exercise the canonical-selection path. This job builds synthetic sources with
*many* near-duplicate clusters whose members have deliberately different token
counts (repetition-based), runs ``normalize → minhash → fuzzy_dups`` twice
against the distributed backend, and asserts the two runs pick the identical
canonical for every cluster.

Sensitivity: within a cluster, variant ``i`` is the shared core repeated
``i+1`` times, so all variants share the same MinHash signature (identical
shingle *set*) and cluster together, but their token counts differ by up to
``Vx``. If canonical selection depended on shard/reduce/merge order, the two
runs would keep different-length survivors and the per-cluster canonical map
would diverge -- exactly the drift #6798 is about, amplified.

Run on-cluster (needs the store bucket's creds + a cluster fleet)::

    iris --cluster=cw-rno2a job run --enable-extra-resources \\
        -e MARIN_PREFIX s3://marin-us-east-02a/marin \\
        -- python -m experiments.datakit.store.synth_dedup_determinism \\
            --prefix s3://marin-us-east-02a/marin/user/rav/datakit/synth_dups_6798 \\
            --sources 3 --clusters 3000 --variants 5 --max-parallelism 32
"""

import argparse
import logging
import random

import pyarrow.parquet as pq
from marin.datakit.normalize import NormalizedData, normalize_to_parquet
from marin.processing.classification.deduplication.fuzzy_dups import compute_fuzzy_dups_attrs
from marin.processing.classification.deduplication.fuzzy_minhash import MinHashAttrData, compute_minhash_attrs
from rigging.filesystem import StoragePath, prefix_join
from rigging.log_setup import configure_logging
from zephyr.writers import write_jsonl_file

logger = logging.getLogger(__name__)

_VOCAB = [f"w{n:04d}" for n in range(2000)]


def _core(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(_VOCAB) for _ in range(n_words))


def _gen_source_docs(rng: random.Random, n_clusters: int, variants: int, core_words: int) -> list[dict]:
    """One synthetic source: ``n_clusters`` near-dup groups + one singleton each.

    Cluster ``c`` variant ``i`` = core repeated ``i+1`` times plus a unique tag,
    so all variants share the core's shingle set (cluster together) but differ
    in length and are not byte-identical (survive exact dedup). Singletons are
    unique cores that must stay unclustered.
    """
    docs: list[dict] = []
    for c in range(n_clusters):
        core = _core(rng, core_words)
        for i in range(variants):
            text = " ".join([core] * (i + 1)) + f" tag_{c}_{i}"
            docs.append({"id": f"c{c}_v{i}", "text": text})
        docs.append({"id": f"c{c}_single", "text": _core(rng, core_words)})
    rng.shuffle(docs)
    return docs


def _build_sources(
    prefix: str, n_sources: int, n_clusters: int, variants: int, core_words: int, shard_bytes: int
) -> dict[str, NormalizedData]:
    """Generate + normalize synthetic sources under ``{prefix}/<source>``."""
    sources: dict[str, NormalizedData] = {}
    for s in range(n_sources):
        name = f"synth_{s:02d}"
        rng = random.Random(1000 + s)
        docs = _gen_source_docs(rng, n_clusters, variants, core_words)
        raw_dir = prefix_join(prefix, f"raw/{name}")
        # A few raw shards so normalize fans across workers.
        n_raw = max(1, len(docs) // 4000)
        for k in range(n_raw):
            write_jsonl_file(docs[k::n_raw], prefix_join(raw_dir, f"shard_{k:03d}.jsonl.gz"))
        out = prefix_join(prefix, name)
        nd = normalize_to_parquet(input_path=raw_dir, output_path=out, target_partition_bytes=shard_bytes)
        sources[name] = nd
        logger.info("built source %s: %d docs -> %s", name, len(docs), nd.main_output_dir)
    return sources


def _run_dedup(sources: dict[str, NormalizedData], out_root: str, max_parallelism: int) -> dict[str, dict]:
    """Run minhash + fuzzy_dups; return ``{source -> {id: (cluster_id, is_canonical)}}``."""
    minhashes: list[MinHashAttrData] = []
    src_by_main = {}
    for name, nd in sources.items():
        mh = compute_minhash_attrs(source=nd, output_path=prefix_join(out_root, f"minhash/{name}"))
        minhashes.append(mh)
        src_by_main[nd.main_output_dir] = name

    dups = compute_fuzzy_dups_attrs(
        inputs=minhashes, output_path=prefix_join(out_root, "dups"), max_parallelism=max_parallelism
    )
    result: dict[str, dict] = {}
    for main_dir, per in dups.sources.items():
        name = src_by_main[main_dir]
        rows: dict[str, tuple] = {}
        for m in sorted(str(x) for x in StoragePath(prefix_join(per.attr_dir, "*.parquet")).glob()):
            with StoragePath(m).open("rb") as fh:
                table = pq.read_table(fh)
            if table.num_rows == 0:
                continue
            ids = table.column("id").to_pylist()
            attrs = table.column("attributes").combine_chunks()
            cids = attrs.field("dup_cluster_id").to_pylist()
            cans = attrs.field("is_cluster_canonical").to_pylist()
            for i, c, ca in zip(ids, cids, cans, strict=True):
                rows[i] = (c, ca)
        result[name] = rows
        logger.info("run dedup: source %s -> %d cluster-member rows", name, len(rows))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True, help="synthetic sample root (writable, in-region)")
    parser.add_argument("--sources", type=int, default=3)
    parser.add_argument("--clusters", type=int, default=3000, help="near-dup clusters per source")
    parser.add_argument("--variants", type=int, default=5, help="members per cluster")
    parser.add_argument("--core-words", type=int, default=60)
    parser.add_argument("--shard-bytes", type=int, default=1_000_000, help="normalize target partition bytes")
    parser.add_argument("--max-parallelism", type=int, default=32)
    args = parser.parse_args()

    configure_logging(logging.INFO)
    prefix = args.prefix.rstrip("/")

    sources = _build_sources(prefix, args.sources, args.clusters, args.variants, args.core_words, args.shard_bytes)

    run_a = _run_dedup(sources, prefix_join(prefix, "dedup_a"), args.max_parallelism)
    run_b = _run_dedup(sources, prefix_join(prefix, "dedup_b"), args.max_parallelism)

    total_members = sum(len(r) for r in run_a.values())
    total_dropped = sum(1 for r in run_a.values() for v in r.values() if not v[1])
    total_canonical = total_members - total_dropped
    print(
        f"cluster members: {total_members}, canonical(kept): {total_canonical}, non-canonical(dropped): {total_dropped}"
    )

    identical = run_a == run_b
    if not identical:
        for name in sorted(run_a):
            a, b = run_a[name], run_b.get(name, {})
            diff = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
            if diff:
                print(f"  source {name}: {len(diff)} differing ids, e.g. {diff[:5]}")
    assert total_dropped > 0, "synthetic data did not exercise dedup (no non-canonicals dropped)"
    print(f"\nRESULT: {'IDENTICAL — fuzzy dedup deterministic' if identical else 'NON-IDENTICAL — NON-DETERMINISTIC'}")
    if not identical:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
