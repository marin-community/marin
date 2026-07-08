# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ground-truth token accounting: denominators, the per-session amplifier, and the
by-class budget decomposition.

Two steps read the normalized tables from :mod:`transcripts`:

- :func:`run_accounting` writes ``token_accounting.json`` (denominators + the
  output-fidelity diagnostic) and ``session_amplifier.parquet`` (the per-session
  re-read multiple ``A_S = cache_read / cache_creation`` that prices each saved
  chunk downstream).
- :func:`run_budget` writes ``budget_decomposition.json``: the exact price split
  (read/write/input), where the cost lives (conversation-carry vs prelude-carry vs
  new-content-write), the prelude component breakdown, the tool-addressable surface
  (a lower bound, since tool results are truncated in storage), and the eviction tax.

The budget is exact from usage records: ``1.25*cache_creation + 0.10*cache_read +
1.0*input``. Every token is written once (create) and re-read (cache_read) on each
later turn it survives, so a tool result is part of the re-read stream: cutting it
shrinks both its one-time surface and its downstream carry.
"""

import json
import logging
import os
from dataclasses import dataclass

import pandas as pd
from rigging.filesystem import StoragePath, prefix_join

from experiments.context_efficiency.transcripts import CHARS_PER_TOK

logger = logging.getLogger(__name__)

# Anthropic prompt-cache price multipliers (relative to base input = 1.0).
P_WRITE = 1.25  # 5m cache write
P_READ = 0.10  # cache read
P_IN = 1.0  # uncached input
P_OUT = 5.0  # output (Opus list ratio; parameterized for sensitivity)
TTL = 300  # 5-minute cache TTL (harness-fixed; we cannot change it)


def load(sessions_path: str):
    b = pd.read_parquet(prefix_join(sessions_path, "blocks.parquet"))
    t = pd.read_parquet(prefix_join(sessions_path, "turns.parquet"))
    return b, t


def _tok(path: str) -> float:
    """Token proxy (bytes/4) of a marin-controlled prelude file; a missing or
    unset one degrades that component to 0 (visible in the output) rather than
    failing the whole decomposition."""
    if not path:
        return 0.0
    try:
        return os.path.getsize(path) / CHARS_PER_TOK
    except OSError:
        return 0.0


# ---- denominators + amplifier (token_accounting.json, session_amplifier.parquet) ----


def denominators(turns) -> dict:
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


def output_fidelity(blocks, turns) -> dict:
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
    """Per-session realized re-read multiple ``A_S = cache_read / cache_creation``."""
    per = (
        turns.groupby("session_id")
        .agg(a_create=("cache_creation", "sum"), a_read=("cache_read", "sum"), n_turns=("turn_idx", "max"))
        .reset_index()
    )
    per["observed_amplifier"] = per.a_read / per.a_create.clip(lower=1)
    return per


@dataclass(frozen=True)
class AccountingConfig:
    sessions_path: str
    output_path: str


def run_accounting(cfg: AccountingConfig) -> None:
    blocks, turns = load(cfg.sessions_path)
    per = session_amplifier(turns)
    result = {
        "n_sessions": int(turns.session_id.nunique()),
        "n_turns": len(turns),
        "price_multipliers": {"write": P_WRITE, "read": P_READ, "input": P_IN, "output": P_OUT},
        "denominators": denominators(turns),
        "output_fidelity": output_fidelity(blocks, turns),
        "amplifier_median_session": round(float(per.observed_amplifier.median()), 2),
    }
    StoragePath(cfg.output_path).mkdirs()
    with StoragePath(prefix_join(cfg.output_path, "token_accounting.json")).open("w") as fh:
        json.dump(result, fh, indent=2)
    per[["session_id", "n_turns", "a_create", "a_read", "observed_amplifier"]].to_parquet(
        prefix_join(cfg.output_path, "session_amplifier.parquet"), index=False
    )
    logger.info(
        "token accounting: %d sessions, median amplifier %.2f", result["n_sessions"], per.observed_amplifier.median()
    )


# ---- by-class budget decomposition (budget_decomposition.json) ----


def prelude_per_session(blocks, turns):
    """First-turn prefix minus the first user prompt is the always-carried prelude."""
    f = turns.sort_values("turn_idx").groupby("session_id").first()
    f["prefix0"] = f.cache_creation + f.cache_read + f.input_tokens
    ut = blocks[blocks.block_type == "user_text"].sort_values("ts").groupby("session_id").est_tokens.first()
    f = f.join(ut.rename("first_prompt"))
    f["first_prompt"] = f["first_prompt"].fillna(0.0).clip(upper=f["prefix0"])
    f["prelude"] = (f["prefix0"] - f["first_prompt"]).clip(lower=0)
    f = f.join(turns.groupby("session_id").turn_idx.max().rename("nt"))
    return f


def top_split(turns, prelude):
    """Split the exact budget into where the cost lives, all shares summing to it."""
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


def prelude_decomp(prelude, agents_md_path: str, memory_md_path: str) -> dict:
    """Split the median prelude into harness-fixed vs marin-controlled components."""
    fixed_med = float(prelude.prelude.median())
    agentsmd = _tok(agents_md_path)
    memory = _tok(memory_md_path)
    marin = agentsmd + memory
    harness = max(fixed_med - marin, 0.0)
    return {
        "median_prelude_tokens": int(fixed_med),
        "mean_prelude_tokens": int(prelude.prelude.mean()),
        "components": {
            "harness_system_tools_skills": {"tokens": int(harness), "pct": round(100 * harness / max(fixed_med, 1), 1)},
            "agents_md": {"tokens": int(agentsmd), "pct": round(100 * agentsmd / max(fixed_med, 1), 1)},
            "memory_md_index": {"tokens": int(memory), "pct": round(100 * memory / max(fixed_med, 1), 1)},
        },
        "marin_controlled_tokens": int(marin),
        "marin_controlled_pct_of_prelude": round(100 * marin / max(fixed_med, 1), 1),
        "note": (
            "harness part (system prompt + tool schemas + skill catalog) is not in the transcript; "
            "measured as prelude residual. Only AGENTS.md + MEMORY.md are marin-controllable, and "
            "MEMORY.md grows with every saved memory (taxed at the full carry rate)."
        ),
    }


def tool_surface(blocks, split) -> dict:
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


def eviction(turns, budget) -> dict:
    """Cache re-creation on turns following a gap longer than the TTL."""
    t = turns.sort_values(["session_id", "turn_idx"])
    t = t[t.turn_idx > 0]
    g = t.gap.fillna(0)
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


@dataclass(frozen=True)
class BudgetConfig:
    sessions_path: str
    agents_md_path: str
    memory_md_path: str
    output_path: str


def run_budget(cfg: BudgetConfig) -> None:
    blocks, turns = load(cfg.sessions_path)
    prelude = prelude_per_session(blocks, turns)
    budget, split = top_split(turns, prelude)
    result = {
        "n_sessions": int(turns.session_id.nunique()),
        "n_turns": len(turns),
        "aggregate_read_over_create_amplifier": round(turns.cache_read.sum() / max(turns.cache_creation.sum(), 1), 1),
        **split,
        "prelude_decomposition": prelude_decomp(prelude, cfg.agents_md_path, cfg.memory_md_path),
        "tool_addressable_surface": tool_surface(blocks, split),
        "eviction": eviction(turns, budget),
    }
    StoragePath(cfg.output_path).mkdirs()
    with StoragePath(prefix_join(cfg.output_path, "budget_decomposition.json")).open("w") as fh:
        json.dump(result, fh, indent=2)
    logger.info(
        "budget: %d input-equiv, tool surface >= %.1f%%",
        budget,
        result["tool_addressable_surface"]["tool_addressable_pct_of_budget_lower_bound"],
    )
