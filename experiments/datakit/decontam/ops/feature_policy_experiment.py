# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline precision experiment for decon feature policies (marin#6852).

The 100M testbed sample is small (~104k docs), so we can rebuild the eval bloom
under several candidate feature policies and re-mark the whole sample in one
process — no pipeline re-run per variant. Compares, per policy:

* paragraph delimiter: single ``\\n`` (current) vs ``\\n\\n`` (true paragraph)
* dropping "math-heavy" n-grams (LaTeX / operator runs) — targets the shared
  formula/identity false positives
* dropping all-stopword n-grams — included to show it does NOT help these FPs

Part B measures the document frequency (over the sample) of the n-grams that
triggered the current flags: a distinctive leak has DF≈1, boilerplate has DF≫1.
This is the empirical answer to "do we need TF-IDF/DF stats?".

Run on the cluster (needs CW read for eval corpus + sample):

    uv run iris --cluster=cw-rno2a job run -e MARIN_PREFIX s3://marin-us-east-02a/marin \\
        -- python experiments/datakit/decontam/ops/feature_policy_experiment.py \\
           --out s3://marin-us-east-02a/marin/user/rav/decon_viewer/feature_policy.json
"""

import argparse
import json
import logging

import dupekit
import pyarrow.parquet as pq
from marin.datakit.decon import _bloom_hash, _discover_eval_files, _has_alpha
from rigging.filesystem import marin_prefix, url_to_fs
from zephyr.readers import load_file

from experiments.datakit.decontam.prepare_eval_corpus import DECON_EXCLUDED_EVAL_TASKS
from experiments.datakit.testbed.decon_arm import (
    ESTIMATED_DOC_COUNT,
    FALSE_POSITIVE_RATE,
    build_testbed_decon_steps,
)

logger = logging.getLogger(__name__)

_N = 13
_THRESHOLD = 0.5
_EVALS_RELATIVE = "datakit/decontam/evals"
_POSTFIX_RUN = "user/rav/decon_viewer/runs/decon_100m_postfix.json"

# A minimal English stopword set — enough to test whether all-stopword n-grams
# are the FP driver (they are not; the FPs are formulas and famous quotes).
_STOP = frozenset(
    "a an the of to in and or is are was were be been being for on at by with as from that this it its into "
    "than then so such not no nor only own same too very can will would could should may might must do does "
    "did has have had he she they we you i but if while about over under out up down".split()
)

# (delimiter, extra_filter) per policy variant.
_VARIANTS = {
    "v0_nl": ("\n", None),
    "v1_nlnl": ("\n\n", None),
    "v2_nl_math": ("\n", "math"),
    "v3_nlnl_math": ("\n\n", "math"),
    "v4_nl_stop": ("\n", "stop"),
}


def _is_math_heavy(ngram: str) -> bool:
    """True if ≥half the tokens are LaTeX commands / operator-or-number runs.

    Catches the shared-formula FPs (``\\frac{11 \\times 10 ...}``, ``\\tan (A+B)``,
    ``x^3 + y^3 + z^3 - 3xyz = ...``) without touching prose."""
    ts = ngram.split()
    if not ts:
        return False
    mathy = sum(1 for t in ts if t.startswith("\\") or any(c in t for c in "{}^_$=") or not any(c.isalpha() for c in t))
    return mathy / len(ts) >= 0.5


def _is_all_stop(ngram: str) -> bool:
    words = ["".join(c for c in t.lower() if c.isalpha()) for t in ngram.split()]
    words = [w for w in words if w]
    return bool(words) and all(w in _STOP for w in words)


def _keep(ngram: str, extra: str | None) -> bool:
    if not _has_alpha(ngram):  # cluster-D letterless filter, always on
        return False
    if extra == "math" and _is_math_heavy(ngram):
        return False
    if extra == "stop" and _is_all_stop(ngram):
        return False
    return True


def _paragraph_ngrams(paragraph: str, extra: str | None) -> list[str]:
    ts = paragraph.split()
    out = []
    for i in range(len(ts) - _N + 1):
        ng = " ".join(ts[i : i + _N])
        if _keep(ng, extra):
            out.append(ng)
    return out


def _doc_overlap(text: str, delim: str, extra: str, bloom) -> tuple[float, list[str]]:
    best_ov, best_ngrams = 0.0, []
    for para in text.split(delim):
        ngs = _paragraph_ngrams(para, extra)
        if not ngs:
            continue
        matched = [ng for ng in ngs if _bloom_hash(ng) in bloom]
        ov = len(matched) / len(ngs)
        if ov > best_ov:
            best_ov, best_ngrams = ov, matched[:5]
    return best_ov, best_ngrams


def _build_blooms() -> dict:
    blooms = {k: dupekit.Bloom(ESTIMATED_DOC_COUNT, FALSE_POSITIVE_RATE) for k in _VARIANTS}
    n_records = 0
    for path in _discover_eval_files([f"{marin_prefix()}/{_EVALS_RELATIVE}"], DECON_EXCLUDED_EVAL_TASKS):
        for rec in load_file(path):
            text = rec.get("text")
            if not text:
                continue
            text = str(text)
            n_records += 1
            for key, (delim, extra) in _VARIANTS.items():
                for para in text.split(delim):
                    for ng in _paragraph_ngrams(para, extra):
                        blooms[key].add(_bloom_hash(ng))
        if n_records and n_records % 20000 == 0:
            logger.info("bloom build: %d eval records", n_records)
    logger.info("built %d variant blooms from %d eval records", len(blooms), n_records)
    return blooms


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Known current flags + their FP n-grams (probe set for the DF measurement).
    fs, postfix = url_to_fs(f"{marin_prefix()}/{_POSTFIX_RUN}")
    with fs.open(postfix) as fh:
        run = json.load(fh)
    known_fp = set()
    probe: dict[int, str] = {}
    for s in run["sources"]:
        for d in s.get("samples", []):
            known_fp.add((s["name"], d["id"]))
            for ng in d.get("matched_ngrams", []):
                probe[_bloom_hash(ng)] = ng
    logger.info("known flags: %d, probe n-grams: %d", len(known_fp), len(probe))

    blooms = _build_blooms()

    steps = build_testbed_decon_steps(
        target_total_tokens_b=0.1, exclude_sources=frozenset({"finetranslations", "ghalogs/public"})
    )
    decons = [s for s in steps if s.name.startswith("datakit/testbed_decon/")]

    flags = {k: [] for k in _VARIANTS}
    probe_df = dict.fromkeys(probe, 0)  # sample document frequency of each probe n-gram
    n_docs = 0
    for ds in decons:
        src = ds.name.removeprefix("datakit/testbed_decon/")
        samp = next(d for d in ds.deps if d.name.startswith("data/datakit/normalized/"))
        sfs, sroot = url_to_fs(samp.output_path)
        for f in sorted(x for x in sfs.find(sroot) if x.endswith(".parquet")):
            with sfs.open(f, "rb") as fh:
                tbl = pq.read_table(fh, columns=["id", "text"])
            for did, text in zip(tbl.column("id").to_pylist(), tbl.column("text").to_pylist(), strict=True):
                if not text:
                    continue
                n_docs += 1
                # DF: which probe n-grams occur in this doc (v0 policy)
                doc_hashes = {_bloom_hash(ng) for para in text.split("\n") for ng in _paragraph_ngrams(para, None)}
                for h in probe:
                    if h in doc_hashes:
                        probe_df[h] += 1
                for key, (delim, extra) in _VARIANTS.items():
                    ov, ngs = _doc_overlap(text, delim, extra, blooms[key])
                    if ov >= _THRESHOLD:
                        flags[key].append({"source": src, "id": did, "overlap": ov, "ngrams": ngs})
        logger.info("marked source %s (running docs=%d)", src, n_docs)

    summary = {}
    for key in _VARIANTS:
        fl = flags[key]
        ids = {(x["source"], x["id"]) for x in fl}
        summary[key] = {
            "total_flagged": len(fl),
            "known_fp_surviving": len(ids & known_fp),
            "new_flags": len(ids - known_fp),
            "flags": sorted(fl, key=lambda x: -x["overlap"])[:40],
        }
        logger.info(
            "%s: total=%d  known_fp_surviving=%d/%d  new=%d",
            key,
            len(fl),
            len(ids & known_fp),
            len(known_fp),
            len(ids - known_fp),
        )

    df_report = sorted(({"ngram": probe[h], "sample_df": probe_df[h]} for h in probe), key=lambda x: -x["sample_df"])
    for r in df_report:
        logger.info("DF %5d  %s", r["sample_df"], r["ngram"][:70])

    out = {"n_docs": n_docs, "n_sources": len(decons), "variants": summary, "probe_df": df_report}
    ofs, opath = url_to_fs(args.out)
    with ofs.open(opath, "w") as fh:
        json.dump(out, fh)
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
