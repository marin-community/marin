# ruff: noqa: E501
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""compute_api_costs.py — Reconstruct API spend from raw_api_logger artifacts.

Walks `results/raw/**` and aggregates token usage from every persisted call
into a per-(date, provider, model, service_tier, role) ledger, then applies
published per-million-token pricing to produce a USD cost estimate.

Supports three schemas, all written by `raw_api_logger.RawAPILogger.call()`
or its batch siblings:

1. **Sync per-call JSON** (`<exp>/<ts>/<role>/<seq>__...__<nonce>.json`):
   - top-level `{"timestamp_utc", "role", "key", "status", "response", "wall_time_s"}`
   - `response.usage.*` for Claude, `response.usage_metadata.*` for Gemini,
     `response.usage.{prompt_tokens, completion_tokens, ...}` for OpenAI sync.

2. **Anthropic batch results** (`*_results.jsonl`): one line per call, with
   `result.message.usage.*` and `result.message.model` set; `service_tier="batch"`.

3. **OpenAI batch outputs** (`output_*.jsonl`): one line per call, with
   `response.body.usage.*`, `response.body.usage.prompt_tokens_details.cached_tokens`,
   and `response.body.model` set.

Pricing references (per million tokens, USD; current as of 2026-05-13):

- **claude-sonnet-4-6** — standard $3 in / $15 out; cache write 5m $3.75 / 1h $6;
  cache read $0.30; **batch is 50% off** ($1.50 / $7.50; cache write $1.875;
  cache read $0.15). Source: https://docs.anthropic.com/en/docs/about-claude/pricing
- **gpt-5.1 (2025-11-13)** — standard $1.25 in / $0.125 cached in / $10 out;
  **batch is 50% off** ($0.625 / $0.0625 / $5). Source: https://openai.com/api/pricing/
- **gemini-3.1-pro-preview** — standard $2 in / $12 out (≤200k); cached read $0.20;
  **batch is 50% off** ($1 / $6). Thinking tokens are billed as output.
  Source: https://ai.google.dev/gemini-api/docs/pricing

Run:
    python compute_api_costs.py                  # full ledger
    python compute_api_costs.py --since 2026-05-08
    python compute_api_costs.py --by-experiment  # group by top-level exp dir
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pricing tables — USD per 1,000,000 tokens
# ---------------------------------------------------------------------------

ANTHROPIC_STD = {
    "claude-sonnet-4-6": {
        "input": 3.00,
        "cache_write_5m": 3.75,
        "cache_write_1h": 6.00,
        "cache_read": 0.30,
        "output": 15.00,
    },
    "claude-sonnet-4-5": {
        "input": 3.00,
        "cache_write_5m": 3.75,
        "cache_write_1h": 6.00,
        "cache_read": 0.30,
        "output": 15.00,
    },
    "claude-haiku-4-5": {
        "input": 1.00,
        "cache_write_5m": 1.25,
        "cache_write_1h": 2.00,
        "cache_read": 0.10,
        "output": 5.00,
    },
    "claude-opus-4-7": {
        "input": 15.00,
        "cache_write_5m": 18.75,
        "cache_write_1h": 30.00,
        "cache_read": 1.50,
        "output": 75.00,
    },
}
ANTHROPIC_BATCH = {  # half off standard for input and output and cache writes (cache reads also half off)
    model: {k: v * 0.5 for k, v in rates.items()} for model, rates in ANTHROPIC_STD.items()
}

OPENAI_STD = {
    "gpt-5.1-2025-11-13": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5.1": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-1-2025-11-13": {"input": 1.25, "cached_input": 0.125, "output": 10.00},  # alias
}
OPENAI_BATCH = {model: {k: v * 0.5 for k, v in rates.items()} for model, rates in OPENAI_STD.items()}

GEMINI_STD = {
    "gemini-3.1-pro-preview": {"input": 2.00, "cached_read": 0.20, "output": 12.00},  # ≤200k tier
    "gemini-3-pro-preview": {"input": 2.00, "cached_read": 0.20, "output": 12.00},
    "gemini-3.1-pro": {"input": 2.00, "cached_read": 0.20, "output": 12.00},
    "gemini-3-flash-preview": {"input": 0.50, "cached_read": 0.05, "output": 3.00},
    "gemini-3.1-flash-preview": {"input": 0.50, "cached_read": 0.05, "output": 3.00},
}
GEMINI_BATCH = {model: {k: v * 0.5 for k, v in rates.items()} for model, rates in GEMINI_STD.items()}

# Older model line-items observed in pre-Run-7 raw outputs.
OPENAI_LEGACY = {
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5-2025-08-07": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
}
GEMINI_LEGACY = {
    "gemini-2.5-flash": {"input": 0.30, "cached_read": 0.075, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "cached_read": 0.025, "output": 0.40},
    "gemini-2.5-pro": {"input": 1.25, "cached_read": 0.31, "output": 10.00},
    "gemini-flash-latest": {"input": 0.30, "cached_read": 0.075, "output": 2.50},
}


def _resolve_pricing(provider: str, model: str, service_tier: str) -> dict | None:
    """Return per-million rate dict, or None if unknown."""
    is_batch = service_tier == "batch"
    if provider == "anthropic":
        table = ANTHROPIC_BATCH if is_batch else ANTHROPIC_STD
        return table.get(model)
    if provider == "openai":
        if is_batch:
            return OPENAI_BATCH.get(model) or OPENAI_LEGACY.get(model)
        return OPENAI_STD.get(model) or OPENAI_LEGACY.get(model)
    if provider == "google":
        if is_batch:
            return GEMINI_BATCH.get(model) or {k: v * 0.5 for k, v in (GEMINI_LEGACY.get(model) or {}).items()}
        return GEMINI_STD.get(model) or GEMINI_LEGACY.get(model)
    return None


# ---------------------------------------------------------------------------
# Usage extraction
# ---------------------------------------------------------------------------


@dataclass
class CallUsage:
    """Normalized usage record for a single LM API call."""

    provider: str  # anthropic / openai / google
    model: str
    service_tier: str  # "batch" or "standard"
    input_tokens: int = 0
    cached_input_tokens: int = 0  # already-cached read (90% off for OpenAI; 90% off for Anthropic)
    cache_write_5m_tokens: int = 0  # Anthropic only
    cache_write_1h_tokens: int = 0  # Anthropic only
    output_tokens: int = 0  # includes thinking_tokens for Gemini
    timestamp_utc: str | None = None
    role: str | None = None
    experiment: str | None = None


def _parse_anthropic_message(msg: dict) -> CallUsage | None:
    usage = msg.get("usage")
    if not usage:
        return None
    return CallUsage(
        provider="anthropic",
        model=msg.get("model", "unknown"),
        service_tier=usage.get("service_tier", "standard"),
        input_tokens=usage.get("input_tokens", 0) or 0,
        cached_input_tokens=usage.get("cache_read_input_tokens", 0) or 0,
        cache_write_5m_tokens=(usage.get("cache_creation", {}) or {}).get("ephemeral_5m_input_tokens", 0)
        or (usage.get("cache_creation_input_tokens", 0) or 0),
        cache_write_1h_tokens=(usage.get("cache_creation", {}) or {}).get("ephemeral_1h_input_tokens", 0) or 0,
        output_tokens=usage.get("output_tokens", 0) or 0,
    )


def _parse_openai_body(body: dict, service_tier_hint: str = "standard") -> CallUsage | None:
    usage = body.get("usage")
    if not usage:
        return None
    prompt = usage.get("prompt_tokens", 0) or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    completion = usage.get("completion_tokens", 0) or 0
    return CallUsage(
        provider="openai",
        model=body.get("model", "unknown"),
        service_tier=service_tier_hint,
        input_tokens=prompt - cached,  # uncached prompt portion
        cached_input_tokens=cached,
        output_tokens=completion,
    )


def _parse_gemini_response(response: dict, service_tier_hint: str = "standard") -> CallUsage | None:
    meta = response.get("usage_metadata")
    if not meta:
        return None
    prompt = meta.get("prompt_token_count", 0) or 0
    cached = meta.get("cached_content_token_count", 0) or 0
    candidates = meta.get("candidates_token_count", 0) or 0
    thoughts = meta.get("thoughts_token_count", 0) or 0
    return CallUsage(
        provider="google",
        model=response.get("model_version", "unknown"),
        service_tier=service_tier_hint,
        input_tokens=prompt - cached,
        cached_input_tokens=cached,
        output_tokens=candidates + thoughts,
    )


def _parse_synced_file(path: Path) -> CallUsage | None:
    """Per-call JSON file written by RawAPILogger.call (sync mode)."""
    try:
        with open(path) as f:
            obj = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if obj.get("status") != "ok":
        return None
    response = obj.get("response")
    if not isinstance(response, dict):
        return None
    ts = obj.get("timestamp_utc")
    # Heuristic: which provider?
    if "usage_metadata" in response:
        u = _parse_gemini_response(response, "standard")
    elif "usage" in response and (response.get("model", "").startswith("claude") or "stop_reason" in response):
        u = _parse_anthropic_message(response)
    elif "choices" in response and "usage" in response:
        u = _parse_openai_body(response, "standard")
    else:
        return None
    if u is None:
        return None
    u.timestamp_utc = ts
    u.role = obj.get("role")
    return u


def _iter_jsonl(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _parse_jsonl_file(path: Path) -> list[CallUsage]:
    out: list[CallUsage] = []
    # Heuristic: Anthropic batch result lines have `result.message`; OpenAI batch output lines have `response.body`.
    for obj in _iter_jsonl(path):
        if "result" in obj and isinstance(obj["result"], dict) and "message" in obj["result"]:
            msg = obj["result"]["message"]
            u = _parse_anthropic_message(msg)
            if u is not None:
                out.append(u)
        elif "response" in obj and isinstance(obj["response"], dict) and "body" in obj["response"]:
            body = obj["response"]["body"]
            u = _parse_openai_body(body, "batch")
            if u is not None:
                out.append(u)
        else:
            # Unknown jsonl line shape; skip silently
            pass
    return out


# ---------------------------------------------------------------------------
# Walk + aggregate
# ---------------------------------------------------------------------------


TIMESTAMP_DIR_RX = re.compile(r"(\d{4})-(\d{2})-(\d{2})T(\d{2})-(\d{2})-(\d{2})")


def _date_from_path(path: Path) -> str | None:
    # Search every path component for a timestamp substring (handles prefixed
    # directory names like "round_1_2026-05-09T07-27-11" as well as bare
    # "2026-05-09T07-27-11").
    for part in path.parts:
        m = TIMESTAMP_DIR_RX.search(part)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def _experiment_from_path(path: Path, raw_root: Path) -> str:
    rel = path.relative_to(raw_root)
    return rel.parts[0] if rel.parts else "unknown"


@dataclass
class Aggregator:
    raw_root: Path
    since: str | None = None  # YYYY-MM-DD inclusive
    rows: list[dict] = field(default_factory=list)
    counters: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))

    def _accept_date(self, date_str: str | None) -> bool:
        if self.since is None or date_str is None:
            return True
        return date_str >= self.since

    def ingest(self, u: CallUsage, path: Path):
        exp = _experiment_from_path(path, self.raw_root)
        date = _date_from_path(path) or (u.timestamp_utc or "")[:10] or "unknown"
        if not self._accept_date(date):
            return
        u.experiment = exp
        if u.timestamp_utc is None:
            u.timestamp_utc = date
        key = (date, u.provider, u.model, u.service_tier, exp, u.role or "")
        c = self.counters[key]
        c["calls"] += 1
        c["input_tokens"] += u.input_tokens
        c["cached_input_tokens"] += u.cached_input_tokens
        c["cache_write_5m_tokens"] += u.cache_write_5m_tokens
        c["cache_write_1h_tokens"] += u.cache_write_1h_tokens
        c["output_tokens"] += u.output_tokens

    def walk(self):
        # First pass: sync per-call JSONs under <exp>/<ts>/<role>/*.json
        for p in self.raw_root.rglob("*.json"):
            name = p.name
            if name in ("_manifest.json",):
                continue
            if name.endswith("_state.json") or "custom_id_map" in name:
                continue
            # Per-call sync JSON files live 4 levels deep from the raw root:
            # <raw_root>/<exp>/<ts>/<role>/<file>.json
            try:
                rel = p.relative_to(self.raw_root)
            except ValueError:
                continue
            if len(rel.parts) < 3:
                continue
            u = _parse_synced_file(p)
            if u is None:
                continue
            self.ingest(u, p)
        # Second pass: batch jsonl files (results / outputs)
        for p in self.raw_root.rglob("*.jsonl"):
            name = p.name
            if "requests" in name or "input_" in name:
                continue
            if not ("results" in name or "output" in name):
                continue
            for u in _parse_jsonl_file(p):
                self.ingest(u, p)

    def compute_cost(self) -> list[dict]:
        rows = []
        for (date, provider, model, tier, exp, role), c in self.counters.items():
            rates = _resolve_pricing(provider, model, tier)
            cost = None
            if rates is not None:
                if provider == "anthropic":
                    cost = (
                        c["input_tokens"] * rates["input"]
                        + c["cached_input_tokens"] * rates.get("cache_read", 0)
                        + c["cache_write_5m_tokens"] * rates.get("cache_write_5m", 0)
                        + c["cache_write_1h_tokens"] * rates.get("cache_write_1h", 0)
                        + c["output_tokens"] * rates["output"]
                    ) / 1_000_000
                elif provider == "openai":
                    cost = (
                        c["input_tokens"] * rates["input"]
                        + c["cached_input_tokens"] * rates.get("cached_input", 0)
                        + c["output_tokens"] * rates["output"]
                    ) / 1_000_000
                elif provider == "google":
                    cost = (
                        c["input_tokens"] * rates["input"]
                        + c["cached_input_tokens"] * rates.get("cached_read", 0)
                        + c["output_tokens"] * rates["output"]
                    ) / 1_000_000
            rows.append(
                {
                    "date": date,
                    "provider": provider,
                    "model": model,
                    "service_tier": tier,
                    "experiment": exp,
                    "role": role,
                    "calls": c["calls"],
                    "input_tokens": c["input_tokens"],
                    "cached_input_tokens": c["cached_input_tokens"],
                    "cache_write_5m_tokens": c["cache_write_5m_tokens"],
                    "cache_write_1h_tokens": c["cache_write_1h_tokens"],
                    "output_tokens": c["output_tokens"],
                    "cost_usd": round(cost, 4) if cost is not None else None,
                }
            )
        return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _fmt(n: float, w: int = 10, decimals: int = 2) -> str:
    if n is None:
        return "?".rjust(w)
    return f"{n:>{w},.{decimals}f}"


def _print_pivot(rows: list[dict], key_cols: list[str], out=sys.stdout):
    pivot: dict[tuple, dict] = defaultdict(lambda: defaultdict(float))
    totals = defaultdict(float)
    for r in rows:
        k = tuple(r[c] for c in key_cols)
        pivot[k]["calls"] += r["calls"]
        pivot[k]["input"] += r["input_tokens"]
        pivot[k]["cached"] += r["cached_input_tokens"]
        pivot[k]["output"] += r["output_tokens"]
        if r["cost_usd"] is not None:
            pivot[k]["cost"] += r["cost_usd"]
            totals["cost"] += r["cost_usd"]
        totals["calls"] += r["calls"]
        totals["input"] += r["input_tokens"]
        totals["cached"] += r["cached_input_tokens"]
        totals["output"] += r["output_tokens"]
    header = (
        " | ".join(c.ljust(28) for c in key_cols)
        + " | "
        + "calls".rjust(8)
        + " | "
        + "in_tok".rjust(13)
        + " | "
        + "cached_in".rjust(13)
        + " | "
        + "out_tok".rjust(13)
        + " | "
        + "cost_usd".rjust(10)
    )
    print(header, file=out)
    print("-" * len(header), file=out)
    for k in sorted(pivot.keys()):
        v = pivot[k]
        cols = (
            " | ".join(str(x).ljust(28) for x in k)
            + f" | {int(v['calls']):>8,} | {int(v['input']):>13,} | {int(v['cached']):>13,} | {int(v['output']):>13,} | {_fmt(v['cost'], 10, 2)}"
        )
        print(cols, file=out)
    print("-" * len(header), file=out)
    print(
        f"{'TOTAL'.ljust(28 * len(key_cols) + (len(key_cols) - 1) * 3)} | {int(totals['calls']):>8,} | {int(totals['input']):>13,} | {int(totals['cached']):>13,} | {int(totals['output']):>13,} | {_fmt(totals['cost'], 10, 2)}",
        file=out,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", default="/lfs/skampere3/0/ahmedah/code/marin/.claude/worktrees/align/results/raw")
    p.add_argument("--since", default=None, help="YYYY-MM-DD inclusive lower bound")
    p.add_argument("--by-experiment", action="store_true", help="pivot by date x experiment x provider x model x tier")
    p.add_argument("--by-provider-day", action="store_true", help="pivot by date x provider")
    p.add_argument("--csv", default=None, help="write detailed per-row CSV to this path")
    args = p.parse_args()

    agg = Aggregator(raw_root=Path(args.raw_root), since=args.since)
    agg.walk()
    rows = agg.compute_cost()

    if args.csv:
        import csv

        with open(args.csv, "w", newline="") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        print(f"wrote {len(rows)} rows -> {args.csv}", file=sys.stderr)

    if args.by_provider_day:
        _print_pivot(rows, key_cols=["date", "provider"])
    elif args.by_experiment:
        _print_pivot(rows, key_cols=["date", "experiment", "provider", "model", "service_tier"])
    else:
        _print_pivot(rows, key_cols=["provider", "model", "service_tier"])


if __name__ == "__main__":
    main()
