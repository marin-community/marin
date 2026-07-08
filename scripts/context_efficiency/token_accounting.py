#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "pyarrow"]
# ///
"""Denominators, the output-fidelity diagnostic, and the per-session amplifier.

Three outputs the rest of the pipeline needs:

1. **Denominators.** The budget expressed three ways: raw distinct-input tokens,
   base-price input-equivalents (the headline denominator), and full
   dollar-equivalent including output. Any uplift % must name which it divides by.
2. **Output-fidelity diagnostic.** `real_output / est(text+thinking+tool_use)`
   comes out ~13x. That is not a tokenization factor — it is evidence that thinking
   blocks are billed but largely absent from the stored transcript, which is why we
   anchor the budget decomposition on usage records, not the chars/4 proxy.
3. **Per-session amplifier** `A_S = cache_read_S / cache_creation_S`, exported to
   `session_amplifier.parquet` and used to price each saved chunk with its own
   session's realized re-read multiple.

The by-class budget split, prelude decomposition, and eviction measurement live in
`budget_decomposition.py`.
"""
import argparse
import json
import os

import pandas as pd

# Anthropic prompt-cache price multipliers (relative to base input = 1.0).
P_WRITE = 1.25  # 5m cache write
P_READ = 0.10  # cache read
P_IN = 1.0  # uncached input
P_OUT = 5.0  # output (Opus list ratio; parameterized for sensitivity)


def load(data_dir):
    b = pd.read_parquet(os.path.join(data_dir, "blocks.parquet"))
    t = pd.read_parquet(os.path.join(data_dir, "turns.parquet"))
    return b, t


def denominators(turns):
    cc = turns.cache_creation.sum()
    cr = turns.cache_read.sum()
    inp = turns.input_tokens.sum()
    out = turns.output_tokens.sum()
    raw_input = cc + inp  # distinct new input tokens
    base_price = P_WRITE * cc + P_READ * cr + P_IN * inp  # input-equivalents
    dollar = base_price + P_OUT * out  # full, incl output
    return {
        "sum_cache_creation": int(cc),
        "sum_cache_read": int(cr),
        "sum_input": int(inp),
        "sum_output": int(out),
        "D_raw_distinct_input": int(raw_input),
        "D_base_price_input_equiv": int(base_price),
        "D_dollar_equiv_input_units": int(dollar),
        "observed_amplifier_read_over_create": round(cr / max(cc, 1), 2),
    }


def output_fidelity(blocks, turns):
    """Diagnostic: how much generated (billed) output is missing from the transcript."""
    gen = blocks[blocks.block_type.isin(["text", "thinking", "tool_use"])]
    est_out = gen.est_tokens.sum()
    real_out = turns.output_tokens.sum()
    return {
        "output_ratio_real_over_est": round(real_out / max(est_out, 1), 2),
        "est_visible_generated_tokens": int(est_out),
        "real_output_tokens": int(real_out),
        "note": "ratio >> 1 => thinking billed but not persisted; output-side re-derivation savings are under-observed",
    }


def session_amplifier(turns):
    """Per-session realized re-read multiple A_S = cache_read / cache_creation."""
    per = (
        turns.groupby("session_id")
        .agg(a_create=("cache_creation", "sum"), a_read=("cache_read", "sum"), n_turns=("turn_idx", "max"))
        .reset_index()
    )
    per["observed_amplifier"] = per.a_read / per.a_create.clip(lower=1)
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "_data"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "_data", "token_accounting.json"))
    args = ap.parse_args()
    blocks, turns = load(args.data)

    per = session_amplifier(turns)
    result = {
        "n_sessions": int(turns.session_id.nunique()),
        "n_turns": len(turns),
        "price_multipliers": {"write": P_WRITE, "read": P_READ, "input": P_IN, "output": P_OUT},
        "denominators": denominators(turns),
        "output_fidelity": output_fidelity(blocks, turns),
        "amplifier_median_session": round(float(per.observed_amplifier.median()), 2),
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    amp_path = os.path.join(args.data, "session_amplifier.parquet")
    per[["session_id", "n_turns", "a_create", "a_read", "observed_amplifier"]].to_parquet(amp_path, index=False)

    print(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}\nwrote {amp_path}")


if __name__ == "__main__":
    main()
