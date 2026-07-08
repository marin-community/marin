#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "pyarrow", "numpy"]
# ///
"""Ground-truth decomposition of the token budget by where the cost lives.

The budget is exact from usage records: `1.25*cache_creation + 0.10*cache_read +
1.0*input`. Every token in the context window is written once (create) and then
re-read (cache_read) on each later turn it survives. So a tool result is *part of
the re-read stream*: cutting it shrinks both its one-time surface and its
downstream carry. This script anchors on ground-truth usage rather than the
chars/4 content proxy, because the stored transcript is NOT the billed token
stream (tool results are truncated in storage, thinking is stripped, images cost
tokens with ~0 chars). We therefore:

1. Split the budget into read / write / input (exact).
2. Split the read stream into prelude-carry (re-read every turn) vs conversation-
   carry (the accumulated tool/assistant/user content re-read every turn). Prelude
   size is the first-turn prefix minus the first user prompt; the rest is exact.
3. Decompose the prelude into harness-fixed (system prompt + tool schemas + skill
   catalog, not persisted) vs marin-controlled (AGENTS.md + MEMORY.md index).
4. Bound the tool-addressable surface: tool content's share of conversation-carry
   and new-content-write, using the faithful-proxy class mass as a LOWER bound
   (tool results are truncated in storage, so their true share is higher).
5. Measure the real eviction tax: cache_creation on turns following a >5min gap
   (the 5min cache TTL is harness-fixed and uncontrollable; prefix SIZE is the
   lever, since a smaller prefix re-materialises cheaper on every eviction).
"""
import argparse
import json
import os

import pandas as pd

P_WRITE, P_READ, P_IN = 1.25, 0.10, 1.0
TTL = 300  # 5-minute cache TTL (harness-fixed; we cannot change it)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MEMORY_MD = "/home/power/.claude/projects/-home-power-code-marin/memory/MEMORY.md"


def _tok(path):
    # marin-controlled prelude files; a missing one degrades that component to 0
    # (visible in the output) rather than failing the whole decomposition.
    try:
        return os.path.getsize(path) / 4.0
    except OSError:
        return 0.0


def load(data_dir):
    b = pd.read_parquet(os.path.join(data_dir, "blocks.parquet"))
    t = pd.read_parquet(os.path.join(data_dir, "turns.parquet"))
    return b, t


def prelude_per_session(blocks, turns):
    f = turns.sort_values("turn_idx").groupby("session_id").first()
    f["prefix0"] = f.cache_creation + f.cache_read + f.input_tokens
    ut = blocks[blocks.block_type == "user_text"].sort_values("ts").groupby("session_id").est_tokens.first()
    f = f.join(ut.rename("first_prompt"))
    f["first_prompt"] = f["first_prompt"].fillna(0.0).clip(upper=f["prefix0"])
    f["prelude"] = (f["prefix0"] - f["first_prompt"]).clip(lower=0)
    f = f.join(turns.groupby("session_id").turn_idx.max().rename("nt"))
    return f


def top_split(turns, prelude):
    CC, CR, IN = turns.cache_creation.sum(), turns.cache_read.sum(), turns.input_tokens.sum()
    budget = P_WRITE * CC + P_READ * CR + P_IN * IN
    prelude_read = (P_READ * prelude.prelude * prelude.nt.clip(lower=0)).sum()
    prelude_write = (P_WRITE * prelude.prelude).sum()
    conv_read = P_READ * CR - prelude_read
    newcontent_write = P_WRITE * CC - prelude_write

    def pc(x):
        return round(100 * x / budget, 1)

    return budget, {
        "observed_budget_input_equiv": int(budget),
        "exact_price_split": {
            "read_0.10x": pc(P_READ * CR),
            "write_1.25x": pc(P_WRITE * CC),
            "input_1.0x": pc(P_IN * IN),
        },
        "where_the_cost_lives": {
            "conversation_carry_reread": pc(conv_read),
            "new_content_write": pc(newcontent_write),
            "prelude_carry_reread": pc(prelude_read),
            "prelude_write": pc(prelude_write),
            "uncached_input": pc(P_IN * IN),
        },
        "prelude_total_pct": pc(prelude_read + prelude_write),
        "conversation_total_pct": pc(conv_read + newcontent_write),
    }


def prelude_decomp(prelude):
    fixed_med = float(prelude.prelude.median())
    claudemd = _tok(os.path.join(REPO_ROOT, "AGENTS.md"))
    memory = _tok(MEMORY_MD)
    marin = claudemd + memory
    harness = max(fixed_med - marin, 0.0)
    return {
        "median_prelude_tokens": int(fixed_med),
        "mean_prelude_tokens": int(prelude.prelude.mean()),
        "components": {
            "harness_system_tools_skills": {"tokens": int(harness), "pct": round(100 * harness / max(fixed_med, 1), 1)},
            "claudeMd_AGENTS_md": {"tokens": int(claudemd), "pct": round(100 * claudemd / max(fixed_med, 1), 1)},
            "MEMORY_md_index": {"tokens": int(memory), "pct": round(100 * memory / max(fixed_med, 1), 1)},
        },
        "marin_controlled_tokens": int(marin),
        "marin_controlled_pct_of_prelude": round(100 * marin / max(fixed_med, 1), 1),
        "note": (
            "harness part (system prompt + tool schemas + skill catalog) is not in the transcript; "
            "measured as prelude residual. Only AGENTS.md + MEMORY.md are marin-controllable, and "
            "MEMORY.md grows with every saved memory (taxed at the full carry rate)."
        ),
    }


def tool_surface(blocks, split):
    """Tool content's share of conversation-carry + new-content-write (lower bound)."""
    mass = blocks.groupby("block_type").est_tokens.sum()
    conv = {c: float(mass.get(c, 0)) for c in ["tool_result", "tool_use", "text", "user_text"]}
    conv_tot = sum(conv.values())
    tool_share = (conv["tool_result"] + conv["tool_use"]) / max(conv_tot, 1)
    conv_carry_pct = split["where_the_cost_lives"]["conversation_carry_reread"]
    newwrite_pct = split["where_the_cost_lives"]["new_content_write"]
    tool_pct = tool_share * (conv_carry_pct + newwrite_pct)
    return {
        "within_conversation_proxy_shares": {c: round(100 * v / conv_tot, 1) for c, v in conv.items()},
        "tool_share_of_conversation": round(tool_share, 3),
        "tool_addressable_pct_of_budget_lower_bound": round(tool_pct, 1),
        "note": (
            "tool_result is truncated in storage, so its proxy share (and thus the tool-addressable "
            "surface) is a LOWER bound. This is the surface a wiki/index/memory/compaction can act on; "
            "coverage (how much is actually forestallable/compressible) is measured by the semantic labels."
        ),
    }


def eviction(turns, budget):
    t = turns.sort_values(["session_id", "turn_idx"])
    t = t[t.turn_idx > 0]
    g = t.gap_sec.fillna(0)
    evicted_create = t[g > TTL].cache_creation.sum()
    total_create = turns.cache_creation.sum()
    return {
        "ttl_seconds": TTL,
        "evicted_creation_after_gt_ttl_gap": int(evicted_create),
        "eviction_tax_input_equiv": int(P_WRITE * evicted_create),
        "eviction_pct_of_budget": round(100 * P_WRITE * evicted_create / budget, 1),
        "frac_of_all_creation": round(evicted_create / max(total_create, 1), 3),
        "turns_after_gt_ttl_gap_pct": round(100 * (g > TTL).mean(), 1),
        "note": (
            "The 5min TTL is harness-fixed and uncontrollable. Eviction re-materialises the prefix at "
            "1.25x instead of 0.10x (12.5x). It is small here (~1% of budget) because most turns are "
            "<1min apart, but it makes prefix SIZE the lever: a smaller carried prefix costs less on "
            "every eviction as well as every read."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "_data"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "_data", "budget_decomposition.json"))
    args = ap.parse_args()
    blocks, turns = load(args.data)
    prelude = prelude_per_session(blocks, turns)
    budget, split = top_split(turns, prelude)
    result = {
        "n_sessions": int(turns.session_id.nunique()),
        "n_turns": len(turns),
        "aggregate_read_over_create_amplifier": round(turns.cache_read.sum() / max(turns.cache_creation.sum(), 1), 1),
        **split,
        "prelude_decomposition": prelude_decomp(prelude),
        "tool_addressable_surface": tool_surface(blocks, split),
        "eviction": eviction(turns, budget),
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
