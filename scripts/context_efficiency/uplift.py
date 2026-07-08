#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "pyarrow", "numpy"]
# ///
"""Per-intervention token-budget uplift, grounded and bounded.

Turns the redundancy pools into net savings as a fraction of the base-price
input-equivalent budget, with three defensible layers:

1. **Addressable ceiling.** What fraction of the *whole* budget is even touched by
   the surface memory/RAG/docs can act on (tool-result + exploration content)?
   If that ceiling is small, no memory system can save much — this is the headline
   framing.
2. **Grounded hit-rates.** Instead of priors, hit-rate bands are tied to the
   *measured* content-stability split (byte-identical vs changed) from
   `redundancy.json`, and to an LLM-audit cross-check (`_data/audit_result.json`).
3. **Net of costs + masking.** Each served item still costs a replacement read;
   overlapping levers (wiki-summary vs docs/repo-map) are unioned, not summed.

Also runs the falsification ablation: how concentrated modeled savings are in the
heaviest sessions, and the sensitivity of the headline to the one swing variable
(the changed-content hit-rate).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

P_WRITE, P_READ = 1.25, 0.10
REPLACEMENT_FRAC = 0.15  # a served item costs ~15% of the original (compact entry read)

# Hit-rate bands (conservative, likely, optimistic). The changed-content band is
# grounded in the LLM audit of the 15 hottest re-read files (audit_result.json):
# token-weighted replaceable fraction ~0.17, held down by large volatile
# edit-traced files whose re-reads a summary cannot replace.
BANDS = {
    "wiki_cache_byte_identical": (0.60, 0.80, 0.95),  # unchanged file -> content cache is reliable
    "wiki_summary_changed": (0.10, 0.17, 0.30),  # audit-grounded (SWING var)
    "rag_exploration": (0.20, 0.45, 0.70),  # exact-repeat stable discovery (tiny pool)
}


def load(data_dir):
    turns = pd.read_parquet(os.path.join(data_dir, "turns.parquet"))
    blocks = pd.read_parquet(os.path.join(data_dir, "blocks.parquet"))
    amp = pd.read_parquet(os.path.join(data_dir, "session_amplifier.parquet"))
    with open(os.path.join(data_dir, "redundancy.json")) as fh:
        red = json.load(fh)
    return turns, blocks, amp, red


def base_price_budget(turns):
    return P_WRITE * turns.cache_creation.sum() + P_READ * turns.cache_read.sum() + turns.input_tokens.sum()


def addressable_ceiling(blocks, amp, budget):
    """Cost of ALL tool-result + tool-use body content = the surface memory can touch."""
    body = blocks[blocks.block_type.isin(["tool_result", "tool_use"])].copy()
    body = body.merge(amp[["session_id", "n_turns", "observed_amplifier"]], on="session_id", how="left")
    body["n_turns"] = body["n_turns"].fillna(1)
    body["observed_amplifier"] = body["observed_amplifier"].fillna(1.0)
    body["remaining"] = (body["n_turns"] - body["first_paid_turn"]).clip(lower=0)
    body["cost"] = body.est_tokens * (P_WRITE + P_READ * np.minimum(body.remaining, body.observed_amplifier))
    total = body.cost.sum()
    return {
        "all_toolresult_tooluse_cost_input_equiv": int(total),
        "pct_of_budget": round(100 * total / budget, 2),
        "by_tool": {
            k: {"cost": int(v), "pct_budget": round(100 * v / budget, 2)}
            for k, v in body.groupby("tool_name").cost.sum().sort_values(ascending=False).head(8).items()
        },
    }


def net(pool, band, replacement=REPLACEMENT_FRAC):
    lo, mid, hi = band
    keep = 1 - replacement
    return {"conservative": int(pool * lo * keep), "likely": int(pool * mid * keep), "optimistic": int(pool * hi * keep)}


def interventions(red, budget):
    xs = red["read_redundancy"]["cross_session"]
    bi = xs["byte_identical_saved"]
    ch = xs["changed_saved"]
    rag = red["exploration_redundancy"]["stable_addressable"]["gross_repeat_saved_input_equiv"]

    wiki_cache = net(bi, BANDS["wiki_cache_byte_identical"])
    wiki_summary = net(ch, BANDS["wiki_summary_changed"])
    rag_net = net(rag, BANDS["rag_exploration"])

    def pct(d):
        return {k: round(100 * v / budget, 3) for k, v in d.items()}

    # Shared wiki/memory = cache win + summary win (disjoint pools, additive).
    wiki_total = {k: wiki_cache[k] + wiki_summary[k] for k in wiki_cache}
    # Docs/repo-map delivers the SAME changed-file summary pool by another route,
    # so the combined estimate unions it with wiki_summary (counts it once) rather
    # than adding a second copy. rag is a disjoint tiny pool, so it is additive.
    combined_memory_plus_docs = {k: wiki_cache[k] + wiki_summary[k] + rag_net[k] for k in wiki_cache}
    return {
        "shared_wiki_memory": {
            "pool_input_equiv": bi + ch,
            "net": wiki_total,
            "pct_budget": pct(wiki_total),
            "components": {"cache_unchanged": wiki_cache, "summary_changed": wiki_summary},
        },
        "rag_semantic_search": {
            "pool_input_equiv": rag,
            "net": rag_net,
            "pct_budget": pct(rag_net),
            "note": "exact-repeat pool is tiny; semantic generalization unmeasurable here",
        },
        "better_docs_repo_map": {
            "note": (
                "delivers the changed-file summary pool by another route; " "unioned with wiki_summary, not additive"
            ),
            "net": wiki_summary,
            "pct_budget": pct(wiki_summary),
        },
        "combined_non_double_counted": {"net": combined_memory_plus_docs, "pct_budget": pct(combined_memory_plus_docs)},
    }


def ablation(blocks, amp):
    """Concentration of modeled read+explore savings in the heaviest sessions."""
    body = blocks[blocks.block_type == "tool_result"].copy()
    body = body.merge(amp[["session_id", "n_turns", "observed_amplifier"]], on="session_id", how="left")
    body["n_turns"] = body["n_turns"].fillna(1)
    body["observed_amplifier"] = body["observed_amplifier"].fillna(1.0)
    body["remaining"] = (body["n_turns"] - body["first_paid_turn"]).clip(lower=0)
    body["cost"] = body.est_tokens * (P_WRITE + P_READ * np.minimum(body.remaining, body.observed_amplifier))
    per = body.groupby("session_id").cost.sum().sort_values(ascending=False)
    total = per.sum()
    top100 = per.head(100).sum()
    return {
        "toolresult_cost_total": int(total),
        "top100_sessions_share": round(float(top100 / total), 3),
        "top1pct_sessions_share": round(float(per.head(max(1, len(per) // 100)).sum() / total), 3),
    }


def prelude_carry(turns, budget):
    """Cost of carrying the fixed prelude (tools+skills+AGENTS.md+MEMORY.md+first
    prompt), re-read every turn via cache_read. This is where an always-on
    wiki/memory lives — so growing it *adds* to this, taxed by the amplifier.
    """
    first = turns.sort_values("turn_idx").groupby("session_id").first()
    first["prefix0"] = first.cache_creation + first.cache_read + first.input_tokens
    nt = turns.groupby("session_id").turn_idx.max().rename("nt")
    first = first.join(nt)
    carry = (P_READ * first.prefix0 * first.nt).sum()
    top100 = (P_READ * first.prefix0 * first.nt).sort_values(ascending=False).head(100).sum()
    return {
        "median_initial_prefix_tokens": int(first.prefix0.median()),
        "mean_initial_prefix_tokens": int(first.prefix0.mean()),
        "prelude_carry_input_equiv": int(carry),
        "pct_of_budget": round(100 * carry / budget, 1),
        "top100_sessions_share": round(float(top100 / carry), 2),
        "note": (
            "always-on wiki/memory/docs ride in this prefix; each added token is "
            "re-read every turn. Lever = TRIM/lazy-load the prelude, not grow it. "
            "prefix0 includes the first user prompt so this is an upper-ish bound."
        ),
    }


def sensitivity(red, budget):
    """Headline vs the swing variable (changed-content hit-rate)."""
    ch = red["read_redundancy"]["cross_session"]["changed_saved"]
    bi = red["read_redundancy"]["cross_session"]["byte_identical_saved"]
    keep = 1 - REPLACEMENT_FRAC
    base_bi = bi * BANDS["wiki_cache_byte_identical"][1] * keep
    rows = {}
    for hr in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5):
        rows[f"changed_hit_{hr}"] = round(100 * (base_bi + ch * hr * keep) / budget, 3)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "_data"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "_data", "uplift.json"))
    ap.add_argument(
        "--audit",
        default=os.path.join(os.path.dirname(__file__), "_data", "audit_result.json"),
        help="optional LLM-audit result to fold in the measured changed-file hit-rate",
    )
    args = ap.parse_args()
    turns, blocks, amp, red = load(args.data)
    budget = base_price_budget(turns)

    result = {
        "base_price_budget_input_equiv": int(budget),
        "addressable_ceiling": addressable_ceiling(blocks, amp, budget),
        "prelude_carry": prelude_carry(turns, budget),
        "interventions": interventions(red, budget),
        "ablation_heaviest_sessions": ablation(blocks, amp),
        "sensitivity_changed_hit_rate": sensitivity(red, budget),
        "hit_rate_bands": BANDS,
        "replacement_frac": REPLACEMENT_FRAC,
    }
    if os.path.exists(args.audit):
        with open(args.audit) as fh:
            result["llm_audit"] = json.load(fh)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
