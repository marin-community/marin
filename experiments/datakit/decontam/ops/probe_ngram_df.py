# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lean document-frequency probe for the decon FP n-grams (marin#6852).

Answers "do we need TF-IDF/DF stats?": counts, over the 100M testbed sample,
how many docs contain each n-gram that triggered a current flag. A distinctive
leak has DF≈1; boilerplate (a shared formula, a famous quote) has DF≫1, so a
document-frequency cut cleanly separates them.

Cheap: single sample pass, membership against the small probe set only.

    uv run iris --cluster=cw-rno2a job run --cpu 2 -e MARIN_PREFIX s3://marin-us-east-02a/marin \\
        -- python experiments/datakit/decontam/ops/probe_ngram_df.py \\
           --out s3://marin-us-east-02a/marin/user/rav/decon_viewer/probe_df.json
"""

import argparse
import json
import logging

import pyarrow.parquet as pq
from marin.datakit.decon import _bloom_hash, _has_alpha
from rigging.filesystem import marin_prefix, url_to_fs

from experiments.datakit.testbed.decon_arm import build_testbed_decon_steps

logger = logging.getLogger(__name__)

_N = 13
_POSTFIX_RUN = "user/rav/decon_viewer/runs/decon_100m_postfix.json"


def _doc_ngram_hashes(text: str) -> set[int]:
    out = set()
    for para in text.split("\n"):
        ts = para.split()
        for i in range(len(ts) - _N + 1):
            ng = " ".join(ts[i : i + _N])
            if _has_alpha(ng):
                out.add(_bloom_hash(ng))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fs, postfix = url_to_fs(f"{marin_prefix()}/{_POSTFIX_RUN}")
    with fs.open(postfix) as fh:
        run = json.load(fh)
    probe: dict[int, str] = {}
    for s in run["sources"]:
        for d in s.get("samples", []):
            for ng in d.get("matched_ngrams", []):
                probe[_bloom_hash(ng)] = ng
    logger.info("probe n-grams: %d", len(probe))

    steps = build_testbed_decon_steps(
        target_total_tokens_b=0.1, exclude_sources=frozenset({"finetranslations", "ghalogs/public"})
    )
    decons = [s for s in steps if s.name.startswith("datakit/testbed_decon/")]

    df = dict.fromkeys(probe, 0)
    n_docs = 0
    for ds in decons:
        samp = next(d for d in ds.deps if d.name.startswith("data/datakit/normalized/"))
        sfs, sroot = url_to_fs(samp.output_path)
        for f in sorted(x for x in sfs.find(sroot) if x.endswith(".parquet")):
            with sfs.open(f, "rb") as fh:
                tbl = pq.read_table(fh, columns=["text"])
            for text in tbl.column("text").to_pylist():
                if not text:
                    continue
                n_docs += 1
                hs = _doc_ngram_hashes(text)
                for h in probe:
                    if h in hs:
                        df[h] += 1
        logger.info("scanned %d docs", n_docs)

    report = sorted(({"ngram": probe[h], "sample_df": df[h]} for h in probe), key=lambda x: -x["sample_df"])
    for r in report:
        logger.info("DF %6d  %s", r["sample_df"], r["ngram"][:70])
    ofs, opath = url_to_fs(args.out)
    with ofs.open(opath, "w") as fh:
        json.dump({"n_docs": n_docs, "probe_df": report}, fh)
    logger.info("wrote %s (%d docs)", args.out, n_docs)


if __name__ == "__main__":
    main()
