# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controlled data validation for cluster B (marin#6852) on real RC eval docs.

Loads the real reading-comprehension datasets straight from HF (no lm-eval —
avoids its transformers pin), and for each builds two blooms: BASELINE (old eval
text = rendered prompt incl. passage + target + all raw fields) and FIXED
(``_lmh_doc_text`` = question+answer, passage dropped). Then, over the real docs:

  * passage-only FP: a corpus doc that is just the public passage — should flag
    in BASELINE (false positive) and NOT in FIXED, and
  * Q+A leakage recall: a corpus doc carrying the question+answer — should flag
    in BOTH (the fix must not lose genuine-leakage detection).

    HF_TOKEN=... uv run python experiments/datakit/decontam/ops/b_validate.py
"""

import logging

import dupekit
from datasets import load_dataset
from marin.datakit.decon import NGramConfig, _bloom_hash, _extract_features, _paragraph_overlap_and_matches

from experiments.datakit.decontam.prepare_eval_corpus import _PASSAGE_FIELDS, _concat_strings, _lmh_doc_text

logging.basicConfig(level=logging.WARNING)
NGRAM = NGramConfig(ngram_length=13, stride=0, overlap_threshold=0.5)
_MAX = 300

# (label, hf_id, config, split, prompt_fn, target_fn) — passage field is auto-detected.
_TASKS = [
    (
        "anli_r3",
        "facebook/anli",
        None,
        "test_r3",
        lambda d: f"{d['premise']}\nHypothesis: {d['hypothesis']}",
        lambda d: str(d.get("label", "")),
    ),
    (
        "race",
        "ehovy/race",
        "all",
        "test",
        lambda d: f"Article: {d['article']}\nQuestion: {d['question']}",
        lambda d: str(d.get("answer", "")),
    ),
    (
        "boolq",
        "google/boolq",
        None,
        "validation",
        lambda d: f"{d['passage']}\nQuestion: {d['question']}",
        lambda d: str(d.get("answer", "")),
    ),
    (
        "squad_v2",
        "rajpurkar/squad_v2",
        None,
        "validation",
        lambda d: f"{d['context']}\nQuestion: {d['question']}",
        lambda d: " ".join((d.get("answers") or {}).get("text", [])),
    ),
]


def _baseline_text(doc: dict, prompt_fn, target_fn) -> str:
    parts = []
    try:
        p = prompt_fn(doc) or ""
    except Exception:
        p = ""
    if p:
        parts.append(str(p))
    t = str(target_fn(doc) or "")
    if t:
        parts.append(t)
    parts.append(_concat_strings(doc))
    return "\n\n".join(x for x in parts if x.strip())


def _bloom(texts) -> dupekit.Bloom:
    bf = dupekit.Bloom(5_000_000, 1e-9)
    for tx in texts:
        for feat in _extract_features(tx, NGRAM):
            bf.add(_bloom_hash(feat))
    return bf


def _flagged(text: str, bf: dupekit.Bloom) -> bool:
    return any(_paragraph_overlap_and_matches(p, bf, NGRAM)[0] >= NGRAM.overlap_threshold for p in text.split("\n") if p)


def _stringify(doc: dict) -> dict:
    """Coerce non-str/list scalars to str so _concat_strings/_lmh see them as raw fields."""
    out = {}
    for k, v in doc.items():
        if isinstance(v, (str, list)):
            out[k] = v
        elif isinstance(v, (int, float, bool)):
            out[k] = str(v)
    return out


def main() -> None:
    print(f"{'task':12s} {'passage-only FP (base→fixed)':30s} {'Q+A recall (base→fixed)':26s}")
    for label, hf_id, cfg, split, prompt_fn, target_fn in _TASKS:
        try:
            ds = load_dataset(hf_id, cfg, split=split)
            docs = [_stringify(d) for d in ds.select(range(min(_MAX, len(ds))))]
        except Exception as e:
            print(f"{label:12s} LOAD FAIL: {type(e).__name__} {str(e)[:70]}")
            continue

        base = _bloom(_baseline_text(d, prompt_fn, target_fn) for d in docs)
        fixed = _bloom(_lmh_doc_text(d, prompt_fn, target_fn) for d in docs)

        passages = [
            d[k] for d in docs for k in d if k.lower() in _PASSAGE_FIELDS and isinstance(d[k], str) and d[k].strip()
        ]
        qa = [t for d in docs if (t := _lmh_doc_text(d, prompt_fn, target_fn)).strip()]

        fp_b = sum(_flagged(p, base) for p in passages)
        fp_f = sum(_flagged(p, fixed) for p in passages)
        tp_b = sum(_flagged(q, base) for q in qa)
        tp_f = sum(_flagged(q, fixed) for q in qa)
        fp = f"{fp_b}/{len(passages)} → {fp_f}/{len(passages)}"
        tp = f"{tp_b}/{len(qa)} → {tp_f}/{len(qa)}"
        print(f"{label:12s} {fp:30s} {tp:26s}")
    print("DONE")


if __name__ == "__main__":
    main()
