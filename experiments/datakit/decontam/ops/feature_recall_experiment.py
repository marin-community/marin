# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recall cost of candidate decon feature policies (marin#6852 follow-up).

Extends the injection recall harness to compare, side by side:

* P0 ``\\n``            — current
* P1 ``\\n\\n``          — true-paragraph split
* P2 ``\\n\\n`` + per-ngram math filter   — drop math-heavy 13-grams only
* P3 ``\\n\\n`` + whole-math-paragraph    — drop a paragraph if it is math-heavy

Injects real eval items (verbatim / own-paragraph / short-line / embedded) and
reports recall per form AND split by whether the eval item is itself math-heavy
— that split is the whole point: math filtering only costs recall on math items,
and whole-paragraph removal costs strictly more than the per-ngram filter.

    uv run iris --cluster=cw-rno2a job run --cpu 2 --memory 4GB --enable-extra-resources \\
        -e MARIN_PREFIX s3://marin-us-east-02a/marin \\
        -- python experiments/datakit/decontam/ops/feature_recall_experiment.py --tasks 200 --items 400
"""

import argparse
import random

import dupekit
import pyarrow.parquet as pq
from marin.datakit.decon import _bloom_hash, _has_alpha
from rigging.filesystem import marin_prefix, url_to_fs

from experiments.datakit.decontam.ops.feature_policy_experiment import _is_math_heavy
from experiments.datakit.decontam.prepare_eval_corpus import DECON_EXCLUDED_EVAL_TASKS

_N = 13
_THRESHOLD = 0.5
_EVALS_RELATIVE = "datakit/decontam/evals"
_FILLER = (
    "This section contains general background commentary unrelated to any benchmark, "
    "written to pad the surrounding document with ordinary prose so the injected span "
    "is diluted among many other sentences that share no ngrams with the eval item. "
)

# (delimiter, math_mode): math_mode in {None, "perngram", "wholepara"}.
_POLICIES = {
    "P0_nl": ("\n", None),
    "P1_nlnl": ("\n\n", None),
    "P2_nlnl_perngram_math": ("\n\n", "perngram"),
    "P3_nlnl_wholepara_math": ("\n\n", "wholepara"),
}


def _para_ngrams(paragraph: str) -> list[str]:
    ts = paragraph.split()
    return [ng for i in range(len(ts) - _N + 1) if _has_alpha(ng := " ".join(ts[i : i + _N]))]


def _kept(ngrams: list[str], math_mode: str | None) -> list[str]:
    """Apply the math policy to a paragraph's n-grams (whole-paragraph drop returns [])."""
    if not ngrams or math_mode is None:
        return ngrams
    if math_mode == "perngram":
        return [ng for ng in ngrams if not _is_math_heavy(ng)]
    if sum(_is_math_heavy(ng) for ng in ngrams) / len(ngrams) >= _THRESHOLD:  # wholepara
        return []
    return ngrams


def _features(text: str, delim: str, math_mode: str | None):
    for para in text.split(delim):
        if para:
            yield from _kept(_para_ngrams(para), math_mode)


def _flagged(text: str, delim: str, math_mode: str | None, bloom) -> bool:
    for para in text.split(delim):
        if not para:
            continue
        ngs = _kept(_para_ngrams(para), math_mode)
        if not ngs:
            continue
        if sum(_bloom_hash(ng) in bloom for ng in ngs) / len(ngs) >= _THRESHOLD:
            return True
    return False


def _short_line_wrap(text: str, words_per_line: int = 8) -> str:
    words = text.split()
    return "\n".join(" ".join(words[i : i + words_per_line]) for i in range(0, len(words), words_per_line))


def _is_math_item(text: str) -> bool:
    """Item is 'math' if ≥1/3 of its baseline n-grams are math-heavy."""
    ngs = [ng for para in text.split("\n") if para for ng in _para_ngrams(para)]
    return bool(ngs) and sum(_is_math_heavy(ng) for ng in ngs) / len(ngs) >= 1 / 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=200)
    ap.add_argument("--items", type=int, default=400)
    args = ap.parse_args()
    rng = random.Random(0)

    fs, root = url_to_fs(f"{marin_prefix()}/{_EVALS_RELATIVE}")
    files = sorted(f for f in fs.find(root) if f.endswith(".parquet"))
    files = [f for f in files if f.split("/")[-2] not in DECON_EXCLUDED_EVAL_TASKS]
    files = rng.sample(files, min(args.tasks, len(files)))

    blooms = {k: dupekit.Bloom(20_000_000, 1e-9) for k in _POLICIES}
    items: list[str] = []
    for f in files:
        with fs.open(f, "rb") as fh:
            texts = pq.read_table(fh, columns=["text"]).column("text").to_pylist()
        for text in texts:
            text = str(text or "")
            base_feats = list(_features(text, "\n", None))
            if not base_feats:
                continue
            items.append(text)
            for key, (delim, mm) in _POLICIES.items():
                for feat in _features(text, delim, mm):
                    blooms[key].add(_bloom_hash(feat))

    inject = rng.sample(items, min(args.items, len(items)))
    math_inj = [t for t in inject if _is_math_item(t)]
    prose_inj = [t for t in inject if not _is_math_item(t)]
    forms = {
        "verbatim": lambda t: t,
        "verbatim_in_doc": lambda t: _FILLER + "\n" + t + "\n" + _FILLER,
        "short_line_wrapped": _short_line_wrap,
        "embedded_1x_filler": lambda t: _FILLER + " " + t.replace("\n", " ") + " " + _FILLER,
    }
    print(
        f"bloom over {len(files)} tasks, {len(items)} indexable items; inject {len(inject)} "
        f"(math={len(math_inj)}, prose={len(prose_inj)})\n"
    )

    def recall(sub, fn, delim, mm, bloom):
        return 100 * sum(_flagged(fn(t), delim, mm, bloom) for t in sub) / len(sub) if sub else 0.0

    for key, (delim, mm) in _POLICIES.items():
        print(f"=== {key} ===")
        bloom = blooms[key]
        for name, fn in forms.items():
            allr = recall(inject, fn, delim, mm, bloom)
            mathr = recall(math_inj, fn, delim, mm, bloom)
            proser = recall(prose_inj, fn, delim, mm, bloom)
            print(f"  {name:20s} all={allr:5.1f}%  math={mathr:5.1f}%  prose={proser:5.1f}%")


if __name__ == "__main__":
    main()
