# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recall harness for decon (marin#6852).

Injects real eval items into synthetic corpus docs in four forms — verbatim,
verbatim as its own paragraph, re-wrapped to short lines, and embedded in filler
— builds a bloom over a sample of the staged eval corpus, marks the injected
docs, and reports the flag rate per form. Quantifies recall and the known
short-line / embedded recall gaps so a regression is visible if the algorithm
changes.

Reads the eval corpus at ``{marin_prefix()}/datakit/decontam/evals`` — relative
to ``MARIN_PREFIX`` — so it runs against whichever store that points at (R2 / CW /
GCS). ``MARIN_PREFIX`` and the store's credentials are assumed present in the
environment (as on a cluster worker):

    python experiments/datakit/decontam/ops/recall_test.py [--tasks 80] [--items 200]
"""

import argparse
import random

import dupekit
import pyarrow.parquet as pq
from marin.datakit.decon import NGramConfig, _bloom_hash, _extract_features, _paragraph_overlap_and_matches
from rigging.filesystem import marin_prefix, url_to_fs

from experiments.datakit.decontam.all_sources_decon import NGRAM_LENGTH, OVERLAP_THRESHOLD
from experiments.datakit.decontam.prepare_eval_corpus import DECON_EXCLUDED_EVAL_TASKS

_EVALS_RELATIVE = "datakit/decontam/evals"
NGRAM = NGramConfig(ngram_length=NGRAM_LENGTH, stride=0, overlap_threshold=OVERLAP_THRESHOLD)
FILLER = (
    "This section contains general background commentary unrelated to any benchmark, "
    "written to pad the surrounding document with ordinary prose so the injected span "
    "is diluted among many other sentences that share no ngrams with the eval item. "
)


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

    fs, root = url_to_fs(f"{marin_prefix()}/{_EVALS_RELATIVE}")
    files = sorted(f for f in fs.find(root) if f.endswith(".parquet"))
    # Mirror the production bloom population: skip the tasks decon excludes at read time.
    files = [f for f in files if f.split("/")[-2] not in DECON_EXCLUDED_EVAL_TASKS]
    files = rng.sample(files, min(args.tasks, len(files)))

    # Build a bloom over the sampled eval tasks + collect their items (with >= 1 ngram).
    bloom = dupekit.Bloom(5_000_000, 1e-9)
    items: list[str] = []
    for f in files:
        with fs.open(f, "rb") as fh:
            texts = pq.read_table(fh, columns=["text"]).column("text").to_pylist()
        for text in texts:
            text = str(text or "")
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
