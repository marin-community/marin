# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the per-source common-n-gram filter (marin#6852).

Given per-source eval-n-gram document frequencies (``per_source_df.py``), build a
mark-time common-set per source (n-grams in >= a fraction of that source's docs)
and check:

* PRECISION — recompute the current flagged docs' paragraph overlap with the
  source common-set removed from both numerator and denominator. Boilerplate
  paragraphs (all-common) collapse to zero n-grams -> no flag.
* RECALL — for a sample of eval items injected verbatim, the post-filter overlap
  stays 1.0 unless *every* n-gram is common, so an item is only lost if some
  source's common-set covers it entirely. Report that worst-case coverage.

    uv run iris --cluster=cw-rno2a job run --cpu 2 --memory 3GB \\
        -e MARIN_PREFIX s3://marin-us-east-02a/marin \\
        -- python experiments/datakit/decontam/ops/validate_per_source_filter.py \\
           --df s3://marin-us-east-02a/marin/user/rav/decon_viewer/per_source_df_100b.parquet
"""

import argparse
import logging
import random
import statistics
from collections import defaultdict

import dupekit
import pyarrow.parquet as pq
from marin.datakit.decon import _bloom_hash, _discover_eval_files, _extract_ngrams, bloom_paths
from rigging.filesystem import marin_prefix, url_to_fs
from zephyr.readers import load_file

from experiments.datakit.decontam.prepare_eval_corpus import DECON_EXCLUDED_EVAL_TASKS
from experiments.datakit.testbed.decon_arm import NGRAM_LENGTH, PARAGRAPH_DELIMITER, build_testbed_decon_steps

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)
THR = 0.5


def _read(path):
    fs, resolved = url_to_fs(path)
    with fs.open(resolved, "rb") as fh:
        return pq.read_table(fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--df", required=True)
    ap.add_argument("--frac", type=float, default=0.005, help="common if df/n_docs >= frac")
    ap.add_argument("--min-abs", type=int, default=5, help="and df >= this absolute floor")
    ap.add_argument("--global-df", default=None, help="optional global corpus-DF parquet to union in")
    ap.add_argument("--global-min-df", type=int, default=50, help="global DF >= this is also common")
    args = ap.parse_args()

    global_common: set[int] = set()
    if args.global_df:
        gt = _read(args.global_df)
        global_common = {
            h
            for h, d in zip(gt.column("hash").to_pylist(), gt.column("df").to_pylist(), strict=True)
            if d >= args.global_min_df
        }
        log.info("global common hashes (df>=%d): %d", args.global_min_df, len(global_common))

    dt = _read(args.df)
    doc = _read(args.df.rsplit(".", 1)[0] + "_doccounts.parquet")
    n_docs = dict(zip(doc.column("source").to_pylist(), doc.column("n_docs").to_pylist(), strict=True))
    per_source_df: dict[str, dict[int, int]] = defaultdict(dict)
    for s, h, d in zip(
        dt.column("source").to_pylist(), dt.column("hash").to_pylist(), dt.column("df").to_pylist(), strict=True
    ):
        per_source_df[s][h] = d
    common: dict[str, set[int]] = defaultdict(set)
    for s, hd in per_source_df.items():
        thr = max(args.min_abs, int(args.frac * n_docs.get(s, 0)))
        common[s] = {h for h, d in hd.items() if d >= thr} | global_common
    log.info(
        "per-source common hashes: %d; global-common: %d",
        sum(len(v - global_common) for v in common.values()),
        len(global_common),
    )

    steps = build_testbed_decon_steps(
        target_total_tokens_b=0.1, exclude_sources=frozenset({"finetranslations", "ghalogs/public"})
    )
    decons = {
        s.name.removeprefix("datakit/testbed_decon/"): s for s in steps if s.name.startswith("datakit/testbed_decon/")
    }
    bloom_path, _ = bloom_paths(next(s for s in steps if s.name.startswith("datakit/bloom/")).output_path)
    bfs, bres = url_to_fs(bloom_path)
    with bfs.open(bres, "rb") as fh:
        bloom = dupekit.Bloom.load_bytes(fh.read())

    # ---- PRECISION: the 3 currently-flagged sources ----
    log.info("=== PRECISION: flagged-doc overlap, standard vs per-source-filtered ===")
    for src in ("cp/usgpo", "nemotron_sft/sft_math", "nemotron_cc_v2/medium_quality"):
        ds = decons[src]
        samp = next(d for d in ds.deps if d.name.startswith("data/datakit/normalized/"))
        fo, ro = url_to_fs(ds.output_path)
        fid = None
        for f in sorted(x for x in fo.find(ro) if x.endswith(".parquet")):
            with fo.open(f, "rb") as fh:
                t = pq.read_table(fh, columns=["id", "attributes"])
            for i, a in zip(t.column("id").to_pylist(), t.column("attributes").to_pylist(), strict=True):
                if a and a.get("contaminated"):
                    fid = i
                    break
            if fid:
                break
        fs2, rs2 = url_to_fs(samp.output_path)
        text = None
        for f in sorted(x for x in fs2.find(rs2) if x.endswith(".parquet")):
            with fs2.open(f, "rb") as fh:
                t = pq.read_table(fh, columns=["id", "text"])
            m = dict(zip(t.column("id").to_pylist(), t.column("text").to_pylist(), strict=True))
            if fid in m:
                text = str(m[fid])
                break
        cs = common.get(src, set()) | global_common
        best_std = best_filt = 0.0
        for para in text.split(PARAGRAPH_DELIMITER):
            ngs = [ng for ng in _extract_ngrams(para, NGRAM_LENGTH, 0)]
            if not ngs:
                continue
            hs = [_bloom_hash(ng) for ng in ngs]
            std = sum(h in bloom for h in hs) / len(hs)
            kept = [h for h in hs if h not in cs]
            filt = (sum(h in bloom for h in kept) / len(kept)) if kept else 0.0
            best_std = max(best_std, std)
            best_filt = max(best_filt, filt)
        log.info("  %-30s std=%.3f  filtered=%.3f  common_set=%d", src, best_std, best_filt, len(cs))

    # ---- RECALL: verbatim eval items vs worst-source coverage ----
    rng = random.Random(0)
    files = rng.sample(
        list(_discover_eval_files([f"{marin_prefix()}/datakit/decontam/evals"], DECON_EXCLUDED_EVAL_TASKS)), 120
    )
    all_common = (set().union(*common.values()) if common else set()) | global_common
    covers = []
    n_items = 0
    for path in files:
        recs = [str(r.get("text") or "") for r in load_file(path)]
        for text in rng.sample(recs, min(6, len(recs))):
            hs = [
                _bloom_hash(ng)
                for p in text.split(PARAGRAPH_DELIMITER)
                if p
                for ng in _extract_ngrams(p, NGRAM_LENGTH, 0)
            ]
            if not hs:
                continue
            n_items += 1
            # worst case: fraction of the item's n-grams that live in ANY source's common-set
            covers.append(sum(h in all_common for h in hs) / len(hs))
    fully = sum(1 for c in covers if c >= 0.999)
    half = sum(1 for c in covers if c >= THR)
    log.info("=== RECALL: %d verbatim eval items ===", n_items)
    log.info("  worst-case common coverage: median=%.3f mean=%.3f", statistics.median(covers), statistics.fmean(covers))
    log.info("  fully covered (>=99.9%%, could be lost): %d/%d = %.2f%%", fully, n_items, 100 * fully / n_items)
    log.info("  >=50%% covered: %d/%d = %.2f%%", half, n_items, 100 * half / n_items)


if __name__ == "__main__":
    main()
