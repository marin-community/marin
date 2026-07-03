# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase-1 intrinsic pre-filter for the tokenizer bake-off: fertility + serving cost.

Measures tokens/byte per tokenizer arm over a fixed multi-domain held-out corpus and
turns it into the deployment-scale cost signature (serving FLOPs/byte at the ~250B/20B
target model). This needs no training and runs in minutes; it ranks arms cheaply so
only the promising ones spend GPU time in later phases. Fertility is an intrinsic
property of the tokenizer + corpus, so these numbers hold regardless of model scale;
only the FLOP multipliers depend on the target model.

Run:  uv run python -m experiments.tokenize.fertility_report [--max-mb 4] [--out report.json]
"""

from __future__ import annotations

import argparse
import json
import logging

import datasets
from levanter.tokenizers import load_tokenizer

from experiments.tokenize.bakeoff_tokenizers import ALL_ARMS, TokenizerArm
from experiments.tokenize.flop_equivalent import DEFAULT_SERVING, arm_cost, fertility_of

logger = logging.getLogger(__name__)

# Held-out eval domains, each a streamable HuggingFace source. Kept small and diverse so
# the fertility profile spans the regimes where tokenizers differ most (prose, code,
# numerics, non-Latin script). (dataset, config, split, text_field).
DomainSpec = tuple[str, str | None, str, str]
EVAL_DOMAINS: dict[str, DomainSpec] = {
    "english_web": ("DKYoon/SlimPajama-6B", None, "validation", "text"),
    "code": ("codeparrot/github-code-clean", "all-all", "train", "code"),
    "math": ("HuggingFaceTB/finemath", "finemath-3plus", "train", "text"),
    "multilingual_zh": ("wikimedia/wikipedia", "20231101.zh", "train", "text"),
}


def _stream_text(spec: DomainSpec, max_bytes: int) -> str:
    """Stream up to ``max_bytes`` of UTF-8 text from a HF dataset, concatenated with newlines."""
    name, config, split, field = spec
    ds = datasets.load_dataset(name, config, split=split, streaming=True)
    chunks: list[str] = []
    total = 0
    for row in ds:
        text = row.get(field) or ""
        if not text:
            continue
        chunks.append(text)
        total += len(text.encode("utf-8"))
        if total >= max_bytes:
            break
    return "\n".join(chunks)


def _load_corpus(max_bytes: int) -> dict[str, str]:
    """Fetch each eval domain, skipping (with a warning) any that fail to stream."""
    corpus: dict[str, str] = {}
    for domain, spec in EVAL_DOMAINS.items():
        try:
            text = _stream_text(spec, max_bytes)
        except Exception as e:  # a flaky/gated source shouldn't sink the whole report
            logger.warning("skipping domain %s (%s): %s", domain, spec[0], str(e)[:160])
            continue
        if text:
            corpus[domain] = text
            logger.info("domain %s: %.2f MB", domain, len(text.encode("utf-8")) / 1e6)
    if not corpus:
        raise RuntimeError("no eval domains loaded; cannot produce a fertility report")
    return corpus


def measure_arm(arm: TokenizerArm, corpus: dict[str, str]) -> dict:
    """Raw per-domain token/byte counts for one arm — enough to re-score under any cost model.

    Stores counts (not just ratios) so a different domain weighting or a different serving
    cost model can be recomputed offline without re-tokenizing (replayability).
    """
    # Load through levanter (tokenizers.Tokenizer.from_file), the exact path marin's tokenize
    # pipeline uses. This matters for SuperBPE: AutoTokenizer.from_pretrained would honor the
    # repo's GPT2Tokenizer class and overwrite the superword pretokenizer, silently measuring a
    # worse-than-baseline fertility; from_file preserves it so the measured cost matches training.
    tok = load_tokenizer(arm.ref)
    real_vocab = len(tok.get_vocab())
    if real_vocab != arm.vocab_size:
        logger.warning("%s: registered vocab %d != loaded %d; using loaded", arm.name, arm.vocab_size, real_vocab)

    def encode(s: str) -> list[int]:
        return tok.encode(s, add_special_tokens=False)

    by_domain = {d: fertility_of(encode, [text]) for d, text in corpus.items()}
    overall = fertility_of(encode, list(corpus.values()))
    return {
        "name": arm.name,
        "ref": arm.ref,
        "axis": str(arm.axis),
        "vocab_size": real_vocab,
        "fertility_overall": overall.fertility,
        "total_tokens": overall.total_tokens,
        "total_bytes": overall.total_bytes,
        "by_domain": {
            d: {"tokens": m.total_tokens, "bytes": m.total_bytes, "fertility": m.fertility} for d, m in by_domain.items()
        },
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-mb", type=float, default=4.0, help="MB of text to stream per domain")
    ap.add_argument("--out", type=str, default=None, help="write the full report as JSON here")
    ap.add_argument("--arms", type=str, default=None, help="comma-separated arm names (default: all)")
    args = ap.parse_args()

    arms = ALL_ARMS
    if args.arms:
        wanted = set(args.arms.split(","))
        arms = tuple(a for a in ALL_ARMS if a.name in wanted)

    corpus = _load_corpus(int(args.max_mb * 1e6))
    rows = [measure_arm(a, corpus) for a in arms]
    domains = list(corpus.keys())

    # Always persist the raw measurements so the cost can be recomputed later under a
    # different ServingCostModel without re-tokenizing (see experiments.tokenize.bakeoff_analysis).
    out_path = args.out or "fertility_raw.json"
    with open(out_path, "w") as f:
        json.dump({"domains": domains, "arms": rows}, f, indent=2)

    # Price at the default deployment serving model (16k context, 5:1 local:global). This
    # is a view, not the source of truth — the raw JSON above can be re-priced any way.
    costs = {r["name"]: arm_cost(r["name"], r["vocab_size"], r["fertility_overall"], DEFAULT_SERVING) for r in rows}
    ref = costs.get("marin-128k") or next(iter(costs.values()))
    attn_share = DEFAULT_SERVING.attention_flop_fraction(ref.vocab_size) * 100

    print(f"\n=== Phase 1: fertility + serving cost @ {DEFAULT_SERVING.context_len} ctx ===")
    print(f"(attention = {attn_share:.1f}% of forward FLOPs at this context; 5:1 local:global)")
    header = f"{'arm':14s} {'vocab':>7s} {'B/tok':>6s} " + " ".join(f"{d[:8]:>8s}" for d in domains)
    header += f" {'infFLOP/B':>10s} {'rel_serve':>9s} {'head%':>5s}"
    print(header)
    for r in sorted(rows, key=lambda x: costs[x["name"]].infer_flops_per_byte):
        c = costs[r["name"]]
        per = " ".join(f"{r['by_domain'][d]['bytes'] / r['by_domain'][d]['tokens']:8.2f}" for d in domains)
        print(
            f"{r['name']:14s} {r['vocab_size']:7d} {1.0 / r['fertility_overall']:6.2f} {per} "
            f"{c.infer_flops_per_byte:10.3e} {c.infer_flops_per_byte / ref.infer_flops_per_byte:9.3f} "
            f"{c.lm_head_flop_fraction * 100:4.1f}%"
        )
    print("\n(B/tok and per-domain columns are bytes/token — higher = fewer tokens = cheaper.")
    print(f" rel_serve is serving FLOPs/byte vs marin-128k at the target model. Raw data: {out_path})")


if __name__ == "__main__":
    main()
