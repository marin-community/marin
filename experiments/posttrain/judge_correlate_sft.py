#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SFT-only per-statement Spearman + Pearson for all 3 judges.

Purely local — reads from ~/judge_correlations/inputs/{gpt41,gpt51,goss}/sft/
and writes ~/judge_correlations/outputs/sft_spearman_pearson.json. No API
calls.

Computes BOTH Spearman and Pearson per-statement for each of the three judge
pairs (gpt41↔gpt51, gpt41↔goss, gpt51↔goss), restricted to the SFT target
only. Reports them side by side so the conservativeness gap between the two
metrics is visible per-statement.

Why both metrics side by side:
- Spearman is the primary metric for LLM judge correlation on ordinal 1–10
  scales because it only cares about rank order, not numeric spacing.
- Pearson gets inflated when the score distribution is top-heavy (most items
  piled at 9 or 10) because a linear fit through narrow variance looks like
  proportional agreement. EXP-026 reported Pearson numbers; this script
  shows the Pearson and Spearman side by side so the inflation is obvious
  where it happens.

Usage:
    uv run python experiments/posttrain/judge_correlate_sft.py
"""

from __future__ import annotations

import collections
import gzip
import json
import logging
import os
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_ROOT = Path.home() / "judge_correlations"
TARGET = os.environ.get("JUDGE_TARGET", "sft")
JUDGES = ("gpt41", "gpt51", "goss")


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_goss_sharded(target: str) -> list[dict[str, Any]]:
    shard_dir = DATA_ROOT / "inputs" / "goss" / target
    records: list[dict[str, Any]] = []
    for shard in sorted(shard_dir.glob("shard_*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def load_all_for_target(target: str) -> dict[str, list[dict[str, Any]]]:
    gpt41 = _load_jsonl(DATA_ROOT / "inputs" / "gpt41" / target / "judged_results.jsonl")
    gpt51 = _load_jsonl(DATA_ROOT / "inputs" / "gpt51" / target / "judged_results.jsonl")
    goss = load_goss_sharded(target)
    return {"gpt41": gpt41, "gpt51": gpt51, "goss": goss}


# --------------------------------------------------------------------------- #
# Score extraction (same legacy parse-failure rules as judge_spearman.py)
# --------------------------------------------------------------------------- #


def extract_score(record: dict[str, Any], judge: str) -> int | None:
    j = record.get("judgment", {}) or {}
    score = j.get("score")
    if score is None:
        return None
    explanation = j.get("explanation", "") or ""
    if judge == "gpt41" and score == 5 and "Failed to parse" in explanation:
        return None
    if judge == "goss" and score == 0 and "Parse failure" in explanation:
        return None
    try:
        return int(score)
    except (TypeError, ValueError):
        return None


def join_key(r: dict[str, Any]) -> tuple[str, str, str]:
    return (
        r.get("prompt_id", "") or "",
        r.get("response_text", "") or "",
        r.get("behavior_id", "") or "",
    )


# --------------------------------------------------------------------------- #
# Stats (inline — no scipy dep)
# --------------------------------------------------------------------------- #


def pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    dxs = [x - mx for x in xs]
    dys = [y - my for y in ys]
    num = sum(dx * dy for dx, dy in zip(dxs, dys, strict=True))
    sx = sum(dx * dx for dx in dxs) ** 0.5
    sy = sum(dy * dy for dy in dys) ** 0.5
    if sx == 0 or sy == 0:
        return None
    return num / (sx * sy)


def _average_ranks(vs: list[float]) -> list[float]:
    n = len(vs)
    indexed = sorted(range(n), key=lambda i: vs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vs[indexed[j + 1]] == vs[indexed[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    return pearson(_average_ranks(xs), _average_ranks(ys))


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    xs_sorted = sorted(xs)
    mid = len(xs_sorted) // 2
    if len(xs_sorted) % 2:
        return xs_sorted[mid]
    return (xs_sorted[mid - 1] + xs_sorted[mid]) / 2


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    logger.info("Loading SFT records for all 3 judges (local, no API calls)")
    raw = load_all_for_target(TARGET)
    for j in JUDGES:
        logger.info("  %s: %d raw records", j, len(raw[j]))

    # Index by join key for O(1) pair lookup.
    indexed: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {}
    score_counts: dict[str, collections.Counter] = {j: collections.Counter() for j in JUDGES}
    for j in JUDGES:
        indexed[j] = {}
        for r in raw[j]:
            indexed[j][join_key(r)] = r
            s = extract_score(r, j)
            if s is not None:
                score_counts[j][s] += 1

    pairs: list[tuple[str, str]] = [("gpt41", "gpt51"), ("gpt41", "goss"), ("gpt51", "goss")]

    results: dict[str, Any] = {
        "target": TARGET,
        "raw_record_counts": {j: len(raw[j]) for j in JUDGES},
        "clean_score_counts": {j: sum(score_counts[j].values()) for j in JUDGES},
        "score_histograms": {j: {int(k): int(v) for k, v in sorted(score_counts[j].items())} for j in JUDGES},
        "pairs": {},
    }

    for pair in pairs:
        ja, jb = pair
        shared_keys = set(indexed[ja].keys()) & set(indexed[jb].keys())
        per_stmt_scores: dict[str, tuple[list[int], list[int]]] = collections.defaultdict(lambda: ([], []))
        skipped = collections.Counter()

        for key in shared_keys:
            ra = indexed[ja][key]
            rb = indexed[jb][key]
            sa = extract_score(ra, ja)
            sb = extract_score(rb, jb)
            if sa is None and sb is None:
                skipped["both_parse_fail"] += 1
                continue
            if sa is None:
                skipped[f"{ja}_parse_fail"] += 1
                continue
            if sb is None:
                skipped[f"{jb}_parse_fail"] += 1
                continue
            bid = key[2]
            per_stmt_scores[bid][0].append(sa)
            per_stmt_scores[bid][1].append(sb)

        per_stmt: dict[str, dict[str, Any]] = {}
        for bid, (xs, ys) in sorted(per_stmt_scores.items()):
            if len(xs) < 3:
                continue
            rho = spearman([float(x) for x in xs], [float(y) for y in ys])
            r = pearson([float(x) for x in xs], [float(y) for y in ys])
            per_stmt[bid] = {
                "n": len(xs),
                "spearman": rho,
                "pearson": r,
                "mean_a": _mean(xs),
                "mean_b": _mean(ys),
            }

        spearmans = [v["spearman"] for v in per_stmt.values() if v["spearman"] is not None]
        pearsons = [v["pearson"] for v in per_stmt.values() if v["pearson"] is not None]
        summary = {
            "n_statements": len(per_stmt),
            "n_paired_items": sum(v["n"] for v in per_stmt.values()),
            "shared_keys": len(shared_keys),
            "skipped": dict(skipped),
            "spearman": {
                "mean": _mean(spearmans),
                "median": _median(spearmans),
                "min": min(spearmans) if spearmans else None,
                "max": max(spearmans) if spearmans else None,
                "frac_ge_0.5": sum(1 for v in spearmans if v >= 0.5) / len(spearmans) if spearmans else None,
                "frac_ge_0.7": sum(1 for v in spearmans if v >= 0.7) / len(spearmans) if spearmans else None,
                "frac_ge_0.9": sum(1 for v in spearmans if v >= 0.9) / len(spearmans) if spearmans else None,
            },
            "pearson": {
                "mean": _mean(pearsons),
                "median": _median(pearsons),
                "min": min(pearsons) if pearsons else None,
                "max": max(pearsons) if pearsons else None,
                "frac_ge_0.5": sum(1 for v in pearsons if v >= 0.5) / len(pearsons) if pearsons else None,
                "frac_ge_0.7": sum(1 for v in pearsons if v >= 0.7) / len(pearsons) if pearsons else None,
                "frac_ge_0.9": sum(1 for v in pearsons if v >= 0.9) / len(pearsons) if pearsons else None,
            },
        }
        results["pairs"][f"{ja}_vs_{jb}"] = {"per_statement": per_stmt, "summary": summary}

    output_path = DATA_ROOT / "outputs" / "sft_spearman_pearson.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")
    logger.info("wrote %s", output_path)

    _print_summary(results)
    return 0


def _print_summary(results: dict[str, Any]) -> None:
    print()
    print("=" * 82)
    print(f" SFT-only per-statement Spearman + Pearson (target={results['target']})")
    print("=" * 82)
    print()
    print(" Raw record counts + clean score counts:")
    for j in JUDGES:
        raw = results["raw_record_counts"][j]
        clean = results["clean_score_counts"][j]
        drop_pct = (raw - clean) / raw * 100 if raw else 0
        print(f"   {j:<8}  raw={raw:>6}  clean={clean:>6}  parse_drop={raw - clean:>4} ({drop_pct:.2f}%)")
    print()

    for pair_name, pair_data in results["pairs"].items():
        summary = pair_data["summary"]
        per_stmt = pair_data["per_statement"]
        s = summary["spearman"]
        r = summary["pearson"]

        print("=" * 82)
        print(f" {pair_name}")
        print("=" * 82)
        print(
            f"   shared_keys={summary['shared_keys']}   "
            f"n_statements={summary['n_statements']}   "
            f"n_paired_items={summary['n_paired_items']}"
        )
        if summary["skipped"]:
            print(f"   skipped: {summary['skipped']}")
        print()
        print(f"   {'metric':<10} {'mean':>9} {'median':>9} {'min':>9} {'max':>9} {'≥0.5':>7} {'≥0.7':>7} {'≥0.9':>7}")
        print(f"   {'-'*10} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*7} {'-'*7} {'-'*7}")
        print(
            f"   {'Spearman':<10} "
            f"{s['mean']:>9.4f} {s['median']:>9.4f} {s['min']:>9.4f} {s['max']:>9.4f} "
            f"{s['frac_ge_0.5'] * 100:>6.1f}% {s['frac_ge_0.7'] * 100:>6.1f}% {s['frac_ge_0.9'] * 100:>6.1f}%"
        )
        print(
            f"   {'Pearson':<10} "
            f"{r['mean']:>9.4f} {r['median']:>9.4f} {r['min']:>9.4f} {r['max']:>9.4f} "
            f"{r['frac_ge_0.5'] * 100:>6.1f}% {r['frac_ge_0.7'] * 100:>6.1f}% {r['frac_ge_0.9'] * 100:>6.1f}%"
        )
        print()

        # Per-statement table, sorted by Spearman descending.
        rows = sorted(
            [(bid, st) for bid, st in per_stmt.items() if st["spearman"] is not None],
            key=lambda kv: kv[1]["spearman"],
            reverse=True,
        )
        print("   Per-statement (sorted by Spearman desc):")
        print(f"   {'behavior_id':<42} {'n':>5} {'ρ (Spear)':>10} {'r (Pear)':>10} " f"{'r-ρ':>8} {'μa':>6} {'μb':>6}")
        print(f"   {'-'*42} {'-'*5} {'-'*10} {'-'*10} {'-'*8} {'-'*6} {'-'*6}")
        for bid, st in rows:
            delta = (st["pearson"] or 0) - (st["spearman"] or 0)
            print(
                f"   {bid:<42} {st['n']:>5} "
                f"{st['spearman']:>10.4f} {st['pearson']:>10.4f} {delta:>+8.4f} "
                f"{st['mean_a']:>6.2f} {st['mean_b']:>6.2f}"
            )
        print()


if __name__ == "__main__":
    raise SystemExit(main())
