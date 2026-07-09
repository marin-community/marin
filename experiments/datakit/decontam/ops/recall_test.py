# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recall harness for decon (marin#6852).

Injects real eval items into synthetic corpus docs in three forms — verbatim,
re-wrapped to short lines, and embedded in filler — builds a bloom over a sample
of the staged eval corpus, marks the injected docs, and reports the flag rate
per form. Quantifies recall and the known short-line / embedded recall gaps so a
regression is visible if the algorithm changes.

    python experiments/datakit/decontam/ops/recall_test.py [--tasks 80] [--items 200]

Reads the eval corpus from R2 (marin-na) using ``R2_*`` env creds — no cluster.
"""

import argparse
import os
import random

import dupekit
import fsspec
from marin.datakit.decon import NGramConfig, _bloom_hash, _extract_features, _paragraph_overlap_and_matches
from zephyr.readers import load_file

EVALS = "s3://marin-na/marin/datakit/decontam/evals"
NGRAM = NGramConfig(ngram_length=13, stride=0, overlap_threshold=0.5)

# zephyr's load_file resolves the S3 client from ambient AWS_* env; point it at R2.
os.environ.setdefault("AWS_ACCESS_KEY_ID", os.environ.get("R2_ACCESS_KEY_ID", ""))
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", os.environ.get("R2_SECRET_ACCESS_KEY", ""))
os.environ.setdefault("AWS_ENDPOINT_URL", os.environ.get("R2_ENDPOINT_URL", ""))
FILLER = (
    "This section contains general background commentary unrelated to any benchmark, "
    "written to pad the surrounding document with ordinary prose so the injected span "
    "is diluted among many other sentences that share no ngrams with the eval item. "
)


def _r2():
    return fsspec.core.url_to_fs(
        EVALS,
        key=os.environ["R2_ACCESS_KEY_ID"],
        secret=os.environ["R2_SECRET_ACCESS_KEY"],
        endpoint_url=os.environ["R2_ENDPOINT_URL"],
        client_kwargs={"region_name": "auto"},
    )[0]


def _short_line_wrap(text: str, words_per_line: int = 8) -> str:
    """Re-wrap to short lines (< ngram_length tokens each) — the MMLU/ARC-option shape."""
    words = text.split()
    return "\n".join(" ".join(words[i : i + words_per_line]) for i in range(0, len(words), words_per_line))


def _flagged(doc_text: str, bloom: dupekit.Bloom) -> bool:
    return any(
        _paragraph_overlap_and_matches(p, bloom, NGRAM)[0] >= NGRAM.overlap_threshold for p in doc_text.split("\n") if p
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=80, help="how many eval task files to sample")
    ap.add_argument("--items", type=int, default=200, help="how many eval items to inject")
    args = ap.parse_args()
    rng = random.Random(0)

    fs = _r2()
    files = sorted(f if f.startswith("s3://") else "s3://" + f for f in fs.find(EVALS) if f.endswith(".parquet"))
    files = rng.sample(files, min(args.tasks, len(files)))

    # Build a bloom over the sampled eval tasks + collect their items (with >= 1 ngram).
    bloom = dupekit.Bloom(5_000_000, 1e-9)
    items: list[str] = []
    for f in files:
        for rec in load_file(f):
            text = str(rec.get("text") or "")
            feats = list(_extract_features(text, NGRAM))
            if not feats:
                continue
            for feat in feats:
                bloom.add(_bloom_hash(feat))
            items.append(text)
    inject = rng.sample(items, min(args.items, len(items)))

    forms = {
        "verbatim": lambda t: t,
        "verbatim_in_doc": lambda t: FILLER + "\n" + t + "\n" + FILLER,  # eval as its own paragraph among others
        "short_line_wrapped": _short_line_wrap,
        "embedded_1x_filler": lambda t: FILLER + " " + t.replace("\n", " ") + " " + FILLER,  # same paragraph
    }
    print(f"bloom over {len(files)} tasks, {len(items)} indexable items; injecting {len(inject)}\n")
    print(f"{'form':22s} {'recall':>8}   flagged/total")
    for name, fn in forms.items():
        hits = sum(_flagged(fn(t), bloom) for t in inject)
        print(f"{name:22s} {100 * hits / len(inject):>7.1f}%   {hits}/{len(inject)}")


if __name__ == "__main__":
    main()
