# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Aggregate engine hits and judge verdicts into the head-to-head scorecard.

Every metric is computed at each k in ``k_values`` by truncating the top-K hits, so
one query pass per engine yields the whole pass@k curve:

- ``recall_at_k`` — a gold file appears among the top-k hit files (deterministic).
- ``mrr`` — mean reciprocal rank of the first gold hit.
- ``judge_hit_at_k`` / ``judge_full_at_k`` — the judge found a satisfying (or exact)
  snippet at rank <= k (independent of gold).
- ``tokens_at_k`` — mean snippet tokens across the top-k, extracted uniformly from the
  repo so answer size is comparable regardless of how an engine chunks.

Also carries index build time/size and query latency. Writes ``results.json`` and a
human-readable ``results.md`` leaderboard.
"""

import json
import logging
import os
from dataclasses import dataclass

from rigging.filesystem import StoragePath, prefix_join

from experiments.code_search_eval.common import est_tokens, read_jsonl, snippet

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScoringConfig:
    benchmark_path: str
    repo_root: str
    k_values: tuple[int, ...]
    engine_hits: tuple[tuple[str, str], ...]  # (engine, hits_dir)
    engine_judge: tuple[tuple[str, str], ...]  # (engine, judge_dir)
    engine_index: tuple[tuple[str, str], ...]  # (engine, index_dir, for build_meta)
    output_path: str


def _recall_and_mrr(gold: dict[str, list[str]], hits: dict[str, list[dict]], k_values):
    recall = {k: 0 for k in k_values}
    rr_sum = 0.0
    n = 0
    for qid, gold_files in gold.items():
        gset = set(gold_files)
        ranked = [h["file"] for h in hits.get(qid, [])]
        n += 1
        first = next((i for i, f in enumerate(ranked, start=1) if f in gset), None)
        if first is not None:
            rr_sum += 1.0 / first
            for k in k_values:
                if first <= k:
                    recall[k] += 1
    return (
        {k: round(recall[k] / max(n, 1), 3) for k in k_values},
        round(rr_sum / max(n, 1), 3),
    )


def _judge_rates(verdicts: dict[str, dict], n_queries: int, k_values):
    hit = {k: 0 for k in k_values}
    full = {k: 0 for k in k_values}
    for v in verdicts.values():
        rank = v.get("best_rank")
        verdict = v.get("verdict")
        if not rank or verdict == "none":
            continue
        for k in k_values:
            if rank <= k:
                if verdict in ("full", "partial"):
                    hit[k] += 1
                if verdict == "full":
                    full[k] += 1
    denom = max(n_queries, 1)
    return {k: round(hit[k] / denom, 3) for k in k_values}, {k: round(full[k] / denom, 3) for k in k_values}


def _tokens_at_k(repo_root: str, gold: dict, hits: dict[str, list[dict]], k_values):
    out = {k: 0.0 for k in k_values}
    n = max(len(gold), 1)
    for qid in gold:
        ranked = hits.get(qid, [])
        prev_k = 0
        running = 0
        for k in sorted(k_values):
            for h in ranked[prev_k:k]:
                running += est_tokens(snippet(repo_root, h["file"], h["start_line"], h["end_line"]))
            out[k] += running
            prev_k = k
    return {k: int(out[k] / n) for k in k_values}


def _build_meta(index_dir: str) -> dict:
    path = os.path.join(index_dir, "build_meta.json")
    if not os.path.exists(path):
        return {"build_seconds": None, "index_bytes": None}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _latency_ms(hit_rows: list[dict]) -> float | None:
    lats = [r["query_latency"] for r in hit_rows if r.get("query_latency") is not None]
    return round(1000 * sum(lats) / len(lats), 1) if lats else None


def _markdown(result: dict) -> str:
    ks = result["k_values"]
    lines = [
        f"# Code search evaluation — {result['n_queries']} queries",
        "",
        "Recall@k (gold file in top-k) / judge-hit@k (agent says a snippet answers the need):",
        "",
        "| engine | "
        + " | ".join(f"R@{k}" for k in ks)
        + " | "
        + " | ".join(f"J@{k}" for k in ks)
        + " | MRR | tok@5 | build s | idx MB | ms/q |",
        "|" + "---|" * (2 * len(ks) + 5),
    ]
    ranked = sorted(result["engines"].items(), key=lambda kv: -(kv[1]["recall_at_k"].get(str(max(ks)), 0) or 0))
    for name, e in ranked:
        r = e["recall_at_k"]
        j = e["judge_hit_at_k"]
        idx_mb = round(e["index_bytes"] / 1e6, 1) if e.get("index_bytes") else "-"
        row = (
            f"| {name} | "
            + " | ".join(f"{r.get(str(k), 0):.2f}" for k in ks)
            + " | "
            + " | ".join(f"{j.get(str(k), 0):.2f}" for k in ks)
            + f" | {e['mrr']:.2f} | {e['tokens_at_k'].get(str(5), '-')} "
            + f"| {e.get('build_seconds') or '-'} | {idx_mb} | {e.get('query_latency_ms') or '-'} |"
        )
        lines.append(row)
    return "\n".join(lines) + "\n"


def run_scoring(cfg: ScoringConfig) -> None:
    benchmark = read_jsonl(prefix_join(cfg.benchmark_path, "benchmark.jsonl"))
    gold = {q["query_id"]: q["gold_files"] for q in benchmark}
    n = len(gold)
    hits_dirs = dict(cfg.engine_hits)
    judge_dirs = dict(cfg.engine_judge)
    index_dirs = dict(cfg.engine_index)
    k_values = list(cfg.k_values)

    engines: dict[str, dict] = {}
    for engine in hits_dirs:
        hit_rows = read_jsonl(prefix_join(hits_dirs[engine], f"{engine}_hits.jsonl"))
        hits = {h["query_id"]: h["hits"] for h in hit_rows}
        verdict_path = prefix_join(judge_dirs[engine], f"{engine}_judge.jsonl")
        verdicts = {v["query_id"]: v for v in read_jsonl(verdict_path)} if StoragePath(verdict_path).exists() else {}

        recall, mrr = _recall_and_mrr(gold, hits, k_values)
        judge_hit, judge_full = _judge_rates(verdicts, n, k_values)
        meta = _build_meta(index_dirs[engine])
        # str keys so the JSON round-trips cleanly and the markdown can index by str(k)
        engines[engine] = {
            "recall_at_k": {str(k): v for k, v in recall.items()},
            "mrr": mrr,
            "judge_hit_at_k": {str(k): v for k, v in judge_hit.items()},
            "judge_full_at_k": {str(k): v for k, v in judge_full.items()},
            "tokens_at_k": {str(k): v for k, v in _tokens_at_k(cfg.repo_root, gold, hits, k_values).items()},
            "build_seconds": meta.get("build_seconds"),
            "index_bytes": meta.get("index_bytes"),
            "query_latency_ms": _latency_ms(hit_rows),
            "n_hits_queries": len(hits),
            "n_judged": len(verdicts),
        }

    kmax = max(k_values)
    leaderboard = sorted(engines, key=lambda e: -(engines[e]["recall_at_k"][str(kmax)]))
    result = {
        "n_queries": n,
        "k_values": k_values,
        "repo_root": cfg.repo_root,
        "engines": engines,
        "leaderboard_by_recall_at_kmax": leaderboard,
    }
    StoragePath(cfg.output_path).mkdirs()
    with StoragePath(prefix_join(cfg.output_path, "results.json")).open("w") as fh:
        json.dump(result, fh, indent=2)
    StoragePath(prefix_join(cfg.output_path, "results.md")).write_text(_markdown(result))
    logger.info("scored %d engines over %d queries: %s", len(engines), n, " > ".join(leaderboard))
