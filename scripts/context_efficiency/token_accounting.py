#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "pyarrow", "numpy"]
# ///
"""Token accounting: denominators, cost-model validation, calibration.

Answers three questions the uplift estimate depends on:

1. **Denominators.** Total budget expressed three ways (codex review #1):
   raw distinct-input tokens, base-price input-equivalents, full dollar-equivalent
   (incl. output). The headline uplift % must name which it divides by.
2. **Cost-model validation.** Does `billed(C,t,T)=1.25C+0.10C(T-t)` describe the
   data? We compare the *observed* amortization ratio (Σcache_read/Σcache_creation)
   to the ratio implied by the visible body blocks under the turn-lifetime model.
   The chars/4 proxy cancels in that ratio, so it is a clean structural check.
3. **Calibration + prelude residual.** Use chars/4 (k=1.0) as the content proxy;
   the output_tokens ratio is reported only as a diagnostic (it reveals that
   thinking is not persisted). Estimate the fixed-prelude share of cache_creation
   as a residual — reported with the TTL-recreation and warm-start caveats, not
   asserted.
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

BODY_BLOCKS = {"tool_result", "user_text", "text", "thinking", "tool_use"}


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


def calibrate(blocks, turns):
    """Content-token proxy factor + a diagnostic that reveals missing thinking.

    We use k=1.0 (chars/4 is the standard proxy; ~3.5-4 chars/token for code+prose)
    with a sensitivity band. We deliberately do NOT anchor k on output_tokens: the
    ratio real_output / est(text+thinking+tool_use) comes out ~13x, which is not a
    tokenization factor — it is evidence that **thinking blocks are largely absent
    from the stored transcript** while the model was still billed for generating
    them. We report that ratio as a diagnostic (and threat-to-validity), not a
    calibration.
    """
    gen = blocks[blocks.block_type.isin(["text", "thinking", "tool_use"])]
    est_out = gen.est_tokens.sum()
    real_out = turns.output_tokens.sum()
    return {
        "k_chars_to_tokens": 1.0,
        "k_sensitivity_band": [0.9, 1.2],
        "output_ratio_diagnostic": round(real_out / max(est_out, 1), 2),
        "est_visible_generated_tokens": int(est_out),
        "real_output_tokens": int(real_out),
        "note": (
            "output_ratio>>1 => thinking not fully persisted; output-token "
            "re-derivation savings are therefore under-observed, not measurable here."
        ),
    }


def validate_model(blocks, turns, k):
    """Structural check of the amortization law, per session then aggregated.

    For each session: modeled_create = k*Σ est_tokens(body); modeled_read =
    k*Σ est_tokens*(n_turns - first_paid_turn) capped at >=0. Compare the ratio
    modeled_read/modeled_create against observed cache_read/cache_creation. If the
    law holds, the two ratios track (k cancels within the ratio).
    """
    body = blocks[blocks.block_type.isin(BODY_BLOCKS)].copy()
    nt = turns.groupby("session_id").turn_idx.max().rename("n_turns")
    body = body.join(nt, on="session_id")
    body["remaining"] = (body["n_turns"] - body["first_paid_turn"]).clip(lower=0)
    body["m_create"] = body["est_tokens"]
    body["m_read"] = body["est_tokens"] * body["remaining"]
    per = body.groupby("session_id").agg(m_create=("m_create", "sum"), m_read=("m_read", "sum")).reset_index()
    act = (
        turns.groupby("session_id")
        .agg(a_create=("cache_creation", "sum"), a_read=("cache_read", "sum"), n_turns=("turn_idx", "max"))
        .reset_index()
    )
    per = per.merge(act, on="session_id")
    per["modeled_ratio"] = per.m_read / per.m_create.clip(lower=1)
    per["observed_ratio"] = per.a_read / per.a_create.clip(lower=1)
    # aggregate (token-weighted) ratios
    agg = {
        "modeled_full_retention_read_over_create": round(per.m_read.sum() / max(per.m_create.sum(), 1), 2),
        "observed_read_over_create": round(per.a_read.sum() / max(per.a_create.sum(), 1), 2),
        "interpretation": (
            "Full-retention model overcounts reads ~12x because it "
            "assumes every chunk survives to session end (ignores "
            "compaction + 1h-TTL eviction). Spearman shows it RANKS "
            "sessions correctly, so we keep per-session shape but "
            "GROUND the read multiple on each session's OBSERVED "
            "amplifier in the uplift model, not on (T-t)."
        ),
    }
    # correlation across sessions with >=5 turns (where amortization is meaningful)
    sub = per[per.n_turns >= 5]
    if len(sub) > 10:
        # Spearman = Pearson of ranks (avoids a scipy dependency).
        agg["ratio_spearman_ge5turns"] = round(sub.modeled_ratio.rank().corr(sub.observed_ratio.rank()), 3)
        agg["n_sessions_ge5turns"] = len(sub)
    per["observed_amplifier"] = per.a_read / per.a_create.clip(lower=1)
    return agg, per


def prelude_residual(blocks, turns, k):
    """cache_creation not explained by visible body = prelude + TTL re-creation.

    body_create_est = k * Σ est_tokens(body). residual = actual_creation - that.
    Also measure warm starts: first-turn cache_read>0 means the prelude prefix was
    already cached (codex #6). eph_1h vs eph_5m indicates long-TTL prelude caching.
    """
    body = blocks[blocks.block_type.isin(BODY_BLOCKS)]
    body_est = k * body.est_tokens.sum()
    actual_create = turns.cache_creation.sum()
    residual = actual_create - body_est
    # warm-start: fraction of sessions whose first turn already reads cache
    first = turns.sort_values("turn_idx").groupby("session_id").first()
    warm = (first.cache_read > 0).mean()
    return {
        "actual_cache_creation": int(actual_create),
        "body_creation_est_calibrated": int(body_est),
        "residual_prelude_plus_recreation": int(residual),
        "residual_frac_of_creation": round(residual / max(actual_create, 1), 3),
        "eph_1h_frac_of_creation": round(turns.eph_1h.sum() / max(actual_create, 1), 3),
        "eph_5m_frac_of_creation": round(turns.eph_5m.sum() / max(actual_create, 1), 3),
        "warm_start_session_frac": round(float(warm), 3),
        "mean_prelude_per_session": int(residual / max(turns.session_id.nunique(), 1)),
    }


def tool_result_breakdown(blocks, k):
    tr = blocks[blocks.block_type == "tool_result"]
    by = (tr.groupby("tool_name").est_tokens.sum() * k).sort_values(ascending=False)
    total = by.sum()
    return {name: {"tokens": int(v), "pct": round(100 * v / max(total, 1), 1)} for name, v in by.head(12).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "_data"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "_data", "token_accounting.json"))
    args = ap.parse_args()
    blocks, turns = load(args.data)

    cal = calibrate(blocks, turns)
    k = cal["k_chars_to_tokens"]
    dn = denominators(turns)
    val, per = validate_model(blocks, turns, k)
    pre = prelude_residual(blocks, turns, k)
    trb = tool_result_breakdown(blocks, k)

    result = {
        "n_sessions": int(turns.session_id.nunique()),
        "n_turns": len(turns),
        "price_multipliers": {"write": P_WRITE, "read": P_READ, "input": P_IN, "output": P_OUT},
        "denominators": dn,
        "calibration": cal,
        "cost_model_validation": val,
        "prelude": pre,
        "tool_result_tokens_by_tool": trb,
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    # per-session observed amplifier, consumed by uplift.py to price saved chunks.
    amp_path = os.path.join(args.data, "session_amplifier.parquet")
    per[["session_id", "n_turns", "a_create", "a_read", "observed_amplifier"]].to_parquet(amp_path, index=False)

    print(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}\nwrote {amp_path}")


if __name__ == "__main__":
    main()
