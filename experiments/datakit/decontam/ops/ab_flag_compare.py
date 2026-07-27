# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Before/after flag-rate compare for the absolute-count recall path (marin#6852).

A single fixed decon run (``--min-abs-hits N``) already stores per-doc
``max_overlap`` and ``contaminated``, so both operating points are recoverable
from one pass — no second run needed:

* baseline (pre-fix, fraction-only) flag  = ``max_overlap >= overlap_threshold``
* fixed (fraction OR absolute-count) flag  = ``contaminated`` (as marked)
* newly flagged by the abs-count path       = ``contaminated AND max_overlap < threshold``

Rebuilds the decon DAG (``sample-root`` / ``exclude`` / ``min-abs-hits`` MUST match
the run) to locate each source's output, counts per source (vectorized over the
``attributes`` struct), and pulls the *newly* flagged docs from the flagged-sample
sidecar with their text + the literal overlapping eval n-grams — the evidence for
eyeballing whether the recall gain is genuine embedded contamination or noise.

    uv run iris --cluster=cw-us-east-02a job run --cpu 2 --memory 6GB --enable-extra-resources \\
        -e MARIN_PREFIX s3://marin-us-east-02a/marin \\
        -- python experiments/datakit/decontam/ops/ab_flag_compare.py \\
           --sample-root datakit/sample_100b_8ae7a94f --min-abs-hits 8 \\
           --exclude <sources absent from the sample> \\
           --out s3://marin-us-east-02a/marin/user/rav/decon6852/ab_100b.json
"""

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import pyarrow.compute as pc
import pyarrow.parquet as pq
from marin.datakit.decon import _bloom_hash, _extract_ngrams
from rigging.filesystem import prefix_join, url_to_fs

from experiments.datakit.testbed.decon_arm import (
    NGRAM_LENGTH,
    OVERLAP_THRESHOLD,
    PARAGRAPH_DELIMITER,
    build_testbed_decon_steps,
)

logger = logging.getLogger(__name__)

_DECON_PREFIX = "datakit/testbed_decon/"


def _overlapping_ngrams(text: str, matched: set[int], limit: int = 8) -> list[str]:
    """The literal doc n-grams that hit the eval bloom — honest evidence for a flag."""
    out: list[str] = []
    for para in text.split(PARAGRAPH_DELIMITER):
        for ng in _extract_ngrams(para, NGRAM_LENGTH, 0):
            if _bloom_hash(ng) in matched:
                out.append(ng)
                if len(out) >= limit:
                    return out
    return out


def _count_file(fs, f: str, threshold: float) -> tuple[int, int, int]:
    # Project only the two scalar subfields (skip the matched_hashes list column).
    with fs.open(f, "rb") as fh:
        tbl = pq.read_table(fh, columns=["attributes.contaminated", "attributes.max_overlap"])
    contaminated = tbl.column(0).combine_chunks()
    max_overlap = tbl.column(1).combine_chunks()
    fixed = pc.sum(contaminated).as_py() or 0
    base = pc.sum(pc.greater_equal(max_overlap, threshold)).as_py() or 0
    return len(contaminated), base, fixed


def _count_source(main_dir: str, threshold: float, workers: int = 8) -> tuple[int, int, int]:
    """Return (n_docs, baseline_flagged, fixed_flagged) over a source's outputs/main.

    baseline = fraction path only (max_overlap >= threshold); fixed = the marked
    ``contaminated`` flag (fraction OR abs-count). Files are read concurrently — the
    object-store reads dominate, so this is I/O-bound.
    """
    fs, resolved = url_to_fs(main_dir)
    files = sorted(x for x in fs.find(resolved) if x.endswith(".parquet"))
    if not files:
        raise FileNotFoundError(main_dir)
    n = base = fixed = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for fn, b, fx in ex.map(lambda f: _count_file(fs, f, threshold), files):
            n += fn
            base += b
            fixed += fx
    return n, base, fixed


def _newly_flagged_samples(flagged_dir: str, threshold: float, limit: int) -> list[dict]:
    """Docs flagged by the abs-count path only (contaminated, max_overlap < threshold)."""
    fs, resolved = url_to_fs(flagged_dir)
    try:
        files = sorted(x for x in fs.find(resolved) if x.endswith(".parquet"))
    except FileNotFoundError:
        return []
    out: list[dict] = []
    for f in files:
        with fs.open(f, "rb") as fh:
            tbl = pq.read_table(fh)
        for row in tbl.to_pylist():
            if row.get("max_overlap", 1.0) < threshold:
                matched = set(row.get("matched_hashes") or [])
                out.append(
                    {
                        "id": row["id"],
                        "max_overlap": round(row.get("max_overlap", 0.0), 4),
                        "n_matched_hashes": len(matched),
                        "overlapping_ngrams": _overlapping_ngrams(row.get("text", ""), matched),
                    }
                )
                if len(out) >= limit:
                    return out
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-root", required=True, help="pre-materialized sample root under MARIN_PREFIX")
    ap.add_argument("--exclude", nargs="*", default=None, help="sources absent from the sample (match the run)")
    ap.add_argument("--min-abs-hits", type=int, default=8, help="the run's min_abs_hits (to match step hashes)")
    ap.add_argument("--only", nargs="*", default=None, help="restrict analysis to these sources")
    ap.add_argument("--samples", type=int, default=6, help="newly-flagged doc samples to pull per source")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    min_abs_hits = None if args.min_abs_hits < 0 else args.min_abs_hits

    steps = build_testbed_decon_steps(
        sample_root=args.sample_root,
        exclude_sources=frozenset(args.exclude or ()),
        only_sources=args.only,
        min_abs_hits=min_abs_hits,
    )
    decon_steps = {s.name.removeprefix(_DECON_PREFIX): s for s in steps if s.name.startswith(_DECON_PREFIX)}
    logger.info("analyzing %d sources (min_abs_hits=%s)", len(decon_steps), min_abs_hits)

    def analyze(item: tuple[str, object]) -> tuple[str, dict | None]:
        name, step = item
        main_dir = prefix_join(step.output_path, "outputs/main")
        try:
            n, base, fixed = _count_source(main_dir, OVERLAP_THRESHOLD, workers=4)
        except FileNotFoundError:
            logger.warning("no output for %s — skipping (source absent from sample)", name)
            return name, None
        newly = fixed - base
        samples = (
            _newly_flagged_samples(
                prefix_join(step.output_path, "outputs/flagged_sample"), OVERLAP_THRESHOLD, args.samples
            )
            if newly > 0
            else []
        )
        logger.info(
            "%-42s n=%-10d base=%-7d (%.4f%%)  fixed=%-7d (%.4f%%)  new=%d",
            name,
            n,
            base,
            100 * base / n if n else 0,
            fixed,
            100 * fixed / n if n else 0,
            newly,
        )
        return name, {
            "n_docs": n,
            "baseline_flagged": base,
            "fixed_flagged": fixed,
            "newly_flagged": newly,
            "baseline_rate": round(base / n, 6) if n else 0.0,
            "fixed_rate": round(fixed / n, 6) if n else 0.0,
            "newly_rate": round(newly / n, 6) if n else 0.0,
            "newly_samples": samples,
        }

    per_source: dict[str, dict] = {}
    tot_n = tot_base = tot_fixed = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for name, rec in ex.map(analyze, sorted(decon_steps.items())):
            if rec is None:
                continue
            per_source[name] = rec
            tot_n += rec["n_docs"]
            tot_base += rec["baseline_flagged"]
            tot_fixed += rec["fixed_flagged"]

    summary = {
        "total_docs": tot_n,
        "baseline_flagged": tot_base,
        "fixed_flagged": tot_fixed,
        "newly_flagged": tot_fixed - tot_base,
        "baseline_rate": round(tot_base / tot_n, 6) if tot_n else 0.0,
        "fixed_rate": round(tot_fixed / tot_n, 6) if tot_n else 0.0,
    }
    logger.info(
        "TOTAL: %d docs  baseline=%d (%.5f%%)  fixed=%d (%.5f%%)  newly=%d (%.1fx)",
        tot_n,
        tot_base,
        100 * summary["baseline_rate"],
        tot_fixed,
        100 * summary["fixed_rate"],
        summary["newly_flagged"],
        (tot_fixed / tot_base) if tot_base else float("inf"),
    )
    ofs, opath = url_to_fs(args.out)
    with ofs.open(opath, "w") as fh:
        json.dump({"summary": summary, "min_abs_hits": min_abs_hits, "by_source": per_source}, fh, indent=2)
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
