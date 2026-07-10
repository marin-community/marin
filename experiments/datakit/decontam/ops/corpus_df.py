# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Corpus document-frequency of eval n-grams over the 100B testbed sample (marin#6852).

Tests whether a *corpus*-DF filter separates non-distinctive boilerplate (legal
enacting clauses, license headers, shared math identities, famous quotes) from
genuine leaks. For every eval n-gram (membership via the decon eval bloom), count
how many 100B-sample documents contain it. Boilerplate → DF≫1; a real leak → DF≈1.

The 100B sample (~256 GB / 766 parquet files) is too big for one process, so we
fan out over files with a process pool. Each worker counts document frequency
(distinct docs per hash) for its files; the main process merges and writes
``hash, df`` parquet. A handful of known boilerplate n-grams are probed live so
DF separation is visible while it runs.

    uv run iris --cluster=cw-rno2a job run --cpu 32 --memory 48GB --enable-extra-resources \\
        -e MARIN_PREFIX s3://marin-us-east-02a/marin \\
        -- python experiments/datakit/decontam/ops/corpus_df.py \\
           --sample-root datakit/sample_100b_e273e96d \\
           --out s3://marin-us-east-02a/marin/user/rav/decon_viewer/corpus_df_100b.parquet
"""

import argparse
import logging
import random
from collections import Counter
from multiprocessing import Pool

import dupekit
import pyarrow as pa
import pyarrow.parquet as pq
from marin.datakit.decon import _bloom_hash, _extract_ngrams, bloom_paths
from rigging.filesystem import marin_prefix, url_to_fs

from experiments.datakit.testbed.decon_arm import NGRAM_LENGTH, PARAGRAPH_DELIMITER, build_testbed_decon_steps

logger = logging.getLogger(__name__)

# Known boilerplate / FP n-grams to watch DF climb on (sanity while running).
_PROBES = {
    "enacting_clause": "Be it enacted by the Senate and House of Representatives of the United States",
    "sum_of_cubes": "y^3 + z^3 - 3xyz = (x + y + z)(x^2 +",
    "gettysburg": "score and seven years ago our fathers brought forth on this continent a",
}
_BLOOM: dupekit.Bloom | None = None
_ROOT_URL = ""


def _init(bloom_path: str, root_url: str) -> None:
    global _BLOOM, _ROOT_URL
    _ROOT_URL = root_url
    fs, resolved = url_to_fs(bloom_path)
    with fs.open(resolved, "rb") as fh:
        _BLOOM = dupekit.Bloom.load_bytes(fh.read())


_READ_BATCH_ROWS = 20000


def _count_file(path: str) -> Counter:
    """Document frequency (distinct docs per eval-hash) contributed by one file.

    Streams the parquet in row batches so a big shard never fully materializes;
    a bad shard is logged and skipped rather than killing the whole pool."""
    assert _BLOOM is not None
    c: Counter = Counter()
    try:
        fs, _ = url_to_fs(_ROOT_URL)
        with fs.open(path, "rb") as fh:
            for batch in pq.ParquetFile(fh).iter_batches(batch_size=_READ_BATCH_ROWS, columns=["text"]):
                for text in batch.column("text").to_pylist():
                    if not text:
                        continue
                    seen: set[int] = set()
                    for para in str(text).split(PARAGRAPH_DELIMITER):
                        for ng in _extract_ngrams(para, NGRAM_LENGTH, 0):
                            h = _bloom_hash(ng)
                            if h in _BLOOM:
                                seen.add(h)
                    for h in seen:
                        c[h] += 1
    except Exception as e:
        logger.warning("skip %s: %s %s", path, type(e).__name__, str(e)[:100])
    return c


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-root", required=True, help="path under MARIN_PREFIX, e.g. datakit/sample_100b_e273e96d")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-files", type=int, default=0, help="0 = all; else a random subsample")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    bloom_step = next(
        s
        for s in build_testbed_decon_steps(target_total_tokens_b=0.1, exclude_sources=frozenset())
        if s.name.startswith("datakit/bloom/")
    )
    bloom_path, _ = bloom_paths(bloom_step.output_path)
    probe_hashes = {k: _bloom_hash(v) for k, v in _PROBES.items()}

    root_url = f"{marin_prefix()}/{args.sample_root}"
    fs, root = url_to_fs(root_url)
    files = sorted(f for f in fs.find(root) if f.endswith(".parquet"))
    if args.max_files and len(files) > args.max_files:
        files = random.Random(0).sample(files, args.max_files)
    logger.info("bloom=%s  files=%d  workers=%d", bloom_path, len(files), args.workers)

    total: Counter = Counter()
    n_docs_files = 0
    with Pool(args.workers, initializer=_init, initargs=(bloom_path, root_url)) as pool:
        for c in pool.imap_unordered(_count_file, files):
            total.update(c)
            n_docs_files += 1
            if n_docs_files % 20 == 0:
                probes = {k: total.get(h, 0) for k, h in probe_hashes.items()}
                logger.info("files %d/%d  distinct_hashes=%d  probe_DF=%s", n_docs_files, len(files), len(total), probes)

    logger.info("FINAL probe DF: %s", {k: total.get(h, 0) for k, h in probe_hashes.items()})
    for thr in (1, 2, 5, 10, 50, 100, 500):
        logger.info("eval n-grams with DF >= %d: %d", thr, sum(1 for v in total.values() if v >= thr))

    tbl = pa.table({"hash": pa.array(list(total.keys()), pa.uint64()), "df": pa.array(list(total.values()), pa.int64())})
    ofs, opath = url_to_fs(args.out)
    with ofs.open(opath, "wb") as fh:
        pq.write_table(tbl, fh, compression="zstd")
    logger.info("wrote %s (%d hashes)", args.out, len(total))


if __name__ == "__main__":
    main()
