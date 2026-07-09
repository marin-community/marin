# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export a testbed decon run to a compact JSON for the decon viewer app.

Runs on CW/reno (needs read access to the sample + decon + eval-bloom outputs).
Rebuilds the decon DAG (``target`` + ``exclude`` must match the run) to locate
each source's decon output and its sample, then per source computes the flag
rate and samples flagged docs with their text, ``max_overlap``, and matched eval
families (``matched_hashes → eval_hash_index → eval_id → family``).

    uv run iris --cluster=cw-rno2a job run -e MARIN_PREFIX s3://marin-us-east-02a/marin \\
        -- python experiments/datakit/decontam/viewer/export_run.py \\
           --target-tokens-b 0.1 --exclude finetranslations ghalogs/public \\
           --label baseline --out s3://marin-us-east-02a/marin/user/rav/decon_viewer/runs

The JSON is written to ``<out>/<label>.json``; pull it down to feed ``app.py``.
"""

import argparse
import json
import logging
import random
import re
from collections import Counter

import pyarrow.parquet as pq
from marin.datakit.decon import _bloom_hash, _extract_ngrams, bloom_paths
from rigging.filesystem import marin_prefix, prefix_join, url_to_fs

from experiments.datakit.testbed.decon_arm import build_testbed_decon_steps

logger = logging.getLogger(__name__)

_SPLIT_RE = re.compile(r"^(.*)-(validation|test|training|train|dev|eval)-\d+$")
_SAMPLES_PER_SOURCE = 60
_TEXT_CLIP = 4000
_EVAL_TEXT_CLIP = 1200
_MAX_MATCHED_EVALS = 5
# Must match the decon run's NGramConfig (decon_arm uses the ngram_length=13 default).
_NGRAM_LENGTH = 13
_MAX_MATCHED_NGRAMS = 8


def _overlapping_ngrams(text: str, matched_hashes: set[int]) -> list[str]:
    """The literal doc n-grams that hit the eval bloom — the honest evidence for a flag.

    A single hashed n-gram maps to many eval records (a shared template or
    formula recurs across a whole task), so the report's per-eval attribution
    can surface an arbitrary, unrelated eval problem. The overlapping n-gram
    text itself is unambiguous: it is exactly the string the doc and some eval
    share. Re-extract with the same paragraph/n-gram policy as the mark side.
    """
    seen: set[str] = set()
    out: list[str] = []
    for para in text.split("\n"):
        for ng in _extract_ngrams(para, _NGRAM_LENGTH, 0):
            if ng not in seen and _bloom_hash(ng) in matched_hashes:
                seen.add(ng)
                out.append(ng)
                if len(out) >= _MAX_MATCHED_NGRAMS:
                    return out
    return out


def eval_id_to_family(eval_id: str) -> str:
    m = _SPLIT_RE.match(eval_id)
    return m.group(1) if m else eval_id


def _load_eval_texts(eval_ids: set[str]) -> dict[str, str]:
    """Load the text of each matched eval record from the staged eval corpus.

    Only reads the eval files whose task (parent dir, recovered from the eval_id)
    is referenced, so it stays cheap despite the corpus being ~250 MB.
    """
    if not eval_ids:
        return {}
    tasks = {eval_id_to_family(e) for e in eval_ids}
    fs, root = url_to_fs(f"{marin_prefix()}/datakit/decontam/evals")
    files = [f for f in fs.find(root) if f.endswith(".parquet") and f.split("/")[-2] in tasks]
    texts: dict[str, str] = {}
    for f in files:
        with fs.open(f, "rb") as fh:
            tbl = pq.read_table(fh, columns=["id", "text"])
        for i, t in zip(tbl.column("id").to_pylist(), tbl.column("text").to_pylist(), strict=True):
            if i in eval_ids:
                texts[i] = t
    return texts


_READ_BATCH_ROWS = 131072


def _read_parquet(path: str, columns: list[str] | None = None):
    """Yield record batches (not whole tables) so a huge file — e.g. the 36M-row
    eval_hash_index — never materializes at once and OOMs the exporter."""
    fs, resolved = url_to_fs(path)
    files = sorted(f for f in fs.find(resolved) if f.endswith(".parquet"))
    for f in files:
        with fs.open(f, "rb") as fh:
            yield from pq.ParquetFile(fh).iter_batches(batch_size=_READ_BATCH_ROWS, columns=columns)


def _source_rows(decon_out: str, k: int, rng: random.Random) -> tuple[int, int, list[dict]]:
    """Return (n_docs, n_flagged, reservoir[{id, max_overlap, matched_hashes}]).

    Reservoir-samples ``k`` flagged rows in one streaming pass so a precision-poor
    source with millions of flags doesn't materialize them all in memory.
    """
    n_docs = n_flagged = 0
    reservoir: list[dict] = []
    for tbl in _read_parquet(decon_out, columns=["id", "attributes"]):
        ids = tbl.column("id").to_pylist()
        attrs = tbl.column("attributes").to_pylist()
        n_docs += len(ids)
        for did, a in zip(ids, attrs, strict=True):
            if a is None or not a.get("contaminated"):
                continue
            n_flagged += 1
            row = {"id": did, "max_overlap": a.get("max_overlap"), "matched_hashes": a.get("matched_hashes") or []}
            if len(reservoir) < k:
                reservoir.append(row)
            else:
                j = rng.randint(0, n_flagged - 1)
                if j < k:
                    reservoir[j] = row
    return n_docs, n_flagged, reservoir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens-b", type=float, required=True)
    ap.add_argument("--exclude", nargs="*", default=None)
    ap.add_argument("--label", required=True, help="run label (json basename)")
    ap.add_argument("--out", required=True, help="output dir (JSON written to <out>/<label>.json)")
    ap.add_argument("--only", nargs="*", default=None, help="restrict export to these sources")
    ap.add_argument("--samples", type=int, default=_SAMPLES_PER_SOURCE)
    args = ap.parse_args()
    rng = random.Random(0)

    steps = build_testbed_decon_steps(
        target_total_tokens_b=args.target_tokens_b,
        only_sources=args.only,  # validates against the source set (raises on typos)
        exclude_sources=frozenset(args.exclude or ()),
    )
    bloom_step = next(s for s in steps if s.name.startswith("datakit/bloom/"))
    decon_steps = [s for s in steps if s.name.startswith("datakit/testbed_decon/")]
    _, index_path = bloom_paths(bloom_step.output_path)

    # First pass: per-source rate + collect the matched hashes we actually need.
    per_source: list[dict] = []
    needed_hashes: set[int] = set()
    for ds in decon_steps:
        source = ds.name.removeprefix("datakit/testbed_decon/")
        sample_step = next(d for d in ds.deps if d.name.startswith("data/datakit/normalized/"))
        n_docs, n_flagged, sampled = _source_rows(ds.output_path, args.samples, rng)
        for row in sampled:
            needed_hashes.update(row["matched_hashes"])
        per_source.append(
            {
                "name": source,
                "docs": n_docs,
                "flagged": n_flagged,
                "rate": (n_flagged / n_docs) if n_docs else 0.0,
                "sample_path": sample_step.output_path,
                "sampled": sampled,
            }
        )
        logger.info("source %s: %d/%d flagged (%.4f%%)", source, n_flagged, n_docs, 100 * per_source[-1]["rate"])

    # Load needed hash → eval_id rows from the index. Keep the eval ids (not just
    # the family) so we can also surface the matched eval TEXT.
    hash_to_evals: dict[int, set[str]] = {}
    for tbl in _read_parquet(index_path):
        hs = tbl.column("hash").to_pylist()
        eids = tbl.column("eval_id").to_pylist()
        for h, eid in zip(hs, eids, strict=True):
            if h in needed_hashes:
                hash_to_evals.setdefault(h, set()).add(str(eid))
    needed_eval_ids: set[str] = set().union(*hash_to_evals.values()) if hash_to_evals else set()
    eval_texts = _load_eval_texts(needed_eval_ids)
    logger.info(
        "resolved %d hashes -> %d eval ids (%d with text)", len(hash_to_evals), len(needed_eval_ids), len(eval_texts)
    )

    # Second pass: attach doc text + eval attribution (family counts + matched eval records w/ text).
    for src in per_source:
        sampled = src.pop("sampled")
        want_ids = {r["id"] for r in sampled}
        id_to_text: dict[str, str] = {}
        for tbl in _read_parquet(src["sample_path"], columns=["id", "text"]):
            ids = tbl.column("id").to_pylist()
            txt = tbl.column("text").to_pylist()
            for did, t in zip(ids, txt, strict=True):
                if did in want_ids:
                    id_to_text[did] = t
        fam_counter: Counter = Counter()
        docs = []
        for r in sampled:
            fams: Counter = Counter()
            eval_hits: Counter = Counter()
            for h in r["matched_hashes"]:
                for eid in hash_to_evals.get(h, ()):  # a hash can map to >1 eval record
                    fams[eval_id_to_family(eid)] += 1
                    eval_hits[eid] += 1
            fam_counter.update(fams.keys())
            matched_evals = [
                {
                    "eval_id": eid,
                    "family": eval_id_to_family(eid),
                    "hits": hits,
                    "text": (eval_texts.get(eid, "") or "")[:_EVAL_TEXT_CLIP],
                }
                for eid, hits in eval_hits.most_common(_MAX_MATCHED_EVALS)
            ]
            full_text = id_to_text.get(r["id"], "") or ""
            docs.append(
                {
                    "id": r["id"],
                    "max_overlap": r["max_overlap"],
                    "n_matched": len(r["matched_hashes"]),
                    "families": fams.most_common(8),
                    "matched_ngrams": _overlapping_ngrams(full_text, set(r["matched_hashes"])),
                    "matched_evals": matched_evals,
                    "text": full_text[:_TEXT_CLIP],
                }
            )
        src["top_families"] = fam_counter.most_common(12)
        src["samples"] = docs

    run = {
        "label": args.label,
        "target_tokens_b": args.target_tokens_b,
        "exclude": args.exclude or [],
        "root": per_source[0]["sample_path"].rsplit("/", 1)[0] if per_source else None,
        "sources": sorted(per_source, key=lambda s: s["rate"], reverse=True),
    }
    out_path = prefix_join(args.out, f"{args.label}.json")
    fs, _ = url_to_fs(out_path)
    with fs.open(out_path, "w") as f:
        json.dump(run, f)
    logger.info("wrote %s (%d sources)", out_path, len(per_source))


if __name__ == "__main__":
    main()
