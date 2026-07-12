# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-source document frequency of eval n-grams over the 100B sample (marin#6852).

The precision fix that survives recall: filter n-grams that are *ubiquitous
within a source* (a congressional enacting clause across every ``cp/usgpo`` bill,
a license header across a code source), not globally-common and not eval-common.
Such n-grams carry no contamination signal for docs from that source, while a
genuine leak's distinctive n-grams stay rare-in-source and survive.

For every ``<source>`` subdir of the 100B sample (``<source>/outputs/main/*.parquet``)
count, per eval n-gram (membership via the decon eval bloom), how many docs of
that source contain it. Emits ``source, hash, df`` for df >= --min-df plus the
per-source doc totals, from which a mark-time common-set is thresholded.

    uv run iris --cluster=cw-rno2a job run --cpu 16 --memory 48GB --enable-extra-resources \\
        -e MARIN_PREFIX s3://marin-us-east-02a/marin \\
        -- python experiments/datakit/decontam/ops/per_source_df.py \\
           --sample-root datakit/sample_100b_e273e96d \\
           --out s3://marin-us-east-02a/marin/user/rav/decon_viewer/per_source_df_100b.parquet
"""

import argparse
import logging
from collections import Counter, defaultdict
from multiprocessing import Pool

import dupekit
import pyarrow as pa
import pyarrow.parquet as pq
from marin.datakit.decon import _bloom_hash, _extract_ngrams, bloom_paths
from rigging.filesystem import marin_prefix, url_to_fs

from experiments.datakit.testbed.decon_arm import NGRAM_LENGTH, PARAGRAPH_DELIMITER, build_testbed_decon_steps

logger = logging.getLogger(__name__)
_READ_BATCH_ROWS = 20000
_SOURCE_SEP = "/outputs/main/"
_BLOOM: dupekit.Bloom | None = None
_ROOT_URL = ""
_ROOT_RESOLVED = ""


def _init(bloom_path: str, root_url: str, root_resolved: str) -> None:
    global _BLOOM, _ROOT_URL, _ROOT_RESOLVED
    _ROOT_URL = root_url
    _ROOT_RESOLVED = root_resolved
    fs, resolved = url_to_fs(bloom_path)
    with fs.open(resolved, "rb") as fh:
        _BLOOM = dupekit.Bloom.load_bytes(fh.read())


def _source_of(path: str) -> str:
    """``<root>/<source>/outputs/main/part-*.parquet`` -> ``<source>``."""
    return path.split(_ROOT_RESOLVED + "/", 1)[-1].split(_SOURCE_SEP, 1)[0]


def _count_file(path: str) -> tuple[str, Counter, int]:
    """(source, hash->doc-frequency in this file, n_docs)."""
    assert _BLOOM is not None
    source = _source_of(path)
    c: Counter = Counter()
    n = 0
    try:
        fs, _ = url_to_fs(_ROOT_URL)
        with fs.open(path, "rb") as fh:
            for batch in pq.ParquetFile(fh).iter_batches(batch_size=_READ_BATCH_ROWS, columns=["text"]):
                for text in batch.column("text").to_pylist():
                    if not text:
                        continue
                    n += 1
                    seen: set[int] = set()
                    for para in str(text).split(PARAGRAPH_DELIMITER):
                        for ng in _extract_ngrams(para, NGRAM_LENGTH, 0):
                            h = _bloom_hash(ng)
                            if h in _BLOOM:
                                seen.add(h)
                    for h in seen:
                        c[h] += 1
    except Exception as e:  # a bad shard shouldn't kill the pool
        logger.warning("skip %s: %s %s", path, type(e).__name__, str(e)[:100])
    return source, c, n


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-df", type=int, default=2, help="only emit hashes with per-source df >= this")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    bloom_step = next(
        s
        for s in build_testbed_decon_steps(target_total_tokens_b=0.1, exclude_sources=frozenset())
        if s.name.startswith("datakit/bloom/")
    )
    bloom_path, _ = bloom_paths(bloom_step.output_path)

    root_url = f"{marin_prefix()}/{args.sample_root}"
    fs, root = url_to_fs(root_url)
    files = sorted(f for f in fs.find(root) if f.endswith(".parquet"))
    logger.info("bloom=%s  root=%s  files=%d  workers=%d", bloom_path, root, len(files), args.workers)

    per_source: dict[str, Counter] = defaultdict(Counter)
    doc_counts: Counter = Counter()
    done = 0
    with Pool(args.workers, initializer=_init, initargs=(bloom_path, root_url, root)) as pool:
        for source, c, n in pool.imap_unordered(_count_file, files):
            per_source[source].update(c)
            doc_counts[source] += n
            done += 1
            if done % 40 == 0:
                logger.info("files %d/%d  sources=%d", done, len(files), len(per_source))

    logger.info("=== per-source doc counts + #common hashes (df>=%d / df>=0.5%%) ===", args.min_df)
    src_col, hash_col, df_col = [], [], []
    for source in sorted(per_source):
        n = doc_counts[source]
        counter = per_source[source]
        frac_thr = max(args.min_df, int(0.005 * n))
        n_common = sum(1 for v in counter.values() if v >= frac_thr)
        logger.info("  %-40s docs=%-8d common(df>=%d)=%d", source, n, frac_thr, n_common)
        for h, df in counter.items():
            if df >= args.min_df:
                src_col.append(source)
                hash_col.append(h)
                df_col.append(df)

    tbl = pa.table(
        {
            "source": pa.array(src_col, pa.string()),
            "hash": pa.array(hash_col, pa.uint64()),
            "df": pa.array(df_col, pa.int64()),
        }
    )
    ofs, opath = url_to_fs(args.out)
    with ofs.open(opath, "wb") as fh:
        pq.write_table(tbl, fh, compression="zstd")
    # per-source doc totals alongside
    dfs = pa.table(
        {"source": pa.array(list(doc_counts), pa.string()), "n_docs": pa.array(list(doc_counts.values()), pa.int64())}
    )
    dpath = opath.rsplit(".", 1)[0] + "_doccounts.parquet"
    with ofs.open(dpath, "wb") as fh:
        pq.write_table(dfs, fh, compression="zstd")
    logger.info("wrote %s (%d rows) + %s", args.out, len(src_col), dpath)


if __name__ == "__main__":
    main()
