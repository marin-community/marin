# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyzer for the human-apply pilot picks JSONL.

Implements the §7.3.3 SYNTHESIZED PROTOCOL decision rule:

  - Dirichlet-Multinomial posterior on {GPT, Pro, Cla, None} per statement.
  - Two posteriors (clear-only + weighted-all) computed; verdict drops a tier
    if they disagree on the best.
  - Sensitivity check at alpha ∈ {0.5, 1, 2}.
  - Tie-handling: 0.5 weight per tied judge; 1/3 for three-way tie.
  - MULTI_MODAL detection via k-means on reason embeddings (k ∈ {1, 2, 3} by
    silhouette score). Falls back to lexical TF-IDF if sentence-transformers
    isn't installed.
  - contested_lift diagnostic overlay computed from easy-anchor cells.
  - Time-per-cell fatigue signal (median first-half vs second-half).

Inputs:
  picks.jsonl exported from the human-apply viewer (one row per cell pick).

Outputs:
  Per-statement verdict to stdout + JSON report (--output).

Usage:
  python e9_human_apply_analyze.py picks.jsonl --output report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Canonical 3-judge ensemble (must match viewer).
JUDGES = ("gpt", "gemini-pro", "claude")

PICK_TO_JUDGE_WEIGHTS = {
    "A": {"A": 1.0},
    "B": {"B": 1.0},
    "C": {"C": 1.0},
    "TIE_AB": {"A": 0.5, "B": 0.5},
    "TIE_AC": {"A": 0.5, "C": 0.5},
    "TIE_BC": {"B": 0.5, "C": 0.5},
    "TIE_ABC": {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3},
    "NONE": {"NONE": 1.0},
}

# Pre-registered thresholds — locked before pilot.
THRESHOLDS = {
    "adopt_p_best": 0.85,
    "tentative_p_best": 0.65,
    "posterior_mean_floor_adopt": 0.50,
    "posterior_mean_floor_tentative": 0.40,
    "spec_broken_p_none_threshold": 0.30,
    "spec_broken_posterior_prob": 0.50,
    "fatigue_ratio": 0.60,
    "multimodal_within_cluster_dominance": 0.50,
    "retest_consistency": 0.80,
}

N_MC_SAMPLES = 10_000


def load_picks(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def map_pick_to_judge_weights(pick: dict) -> dict[str, float]:
    """Convert a pick (with label-judge mapping) to {judge_name: weight}.

    Returns dict keyed by canonical judge name (gpt/gemini-pro/claude/NONE).
    Multiplies by clear/close weight (clear=1.0, close=0.5).
    """
    label_weights = PICK_TO_JUDGE_WEIGHTS[pick["pick"]]
    mapping = pick["judge_mapping"]  # {"A": "gpt", "B": "gemini-pro", ...}
    cc = pick.get("clear_or_close", "clear")
    cc_weight = 1.0 if cc == "clear" else 0.5
    judge_weights = defaultdict(float)
    for label, w in label_weights.items():
        if label == "NONE":
            judge_weights["NONE"] += w * cc_weight
        else:
            judge_weights[mapping[label]] += w * cc_weight
    return dict(judge_weights)


def dirichlet_posterior(
    counts: dict, alpha: float, n_samples: int = N_MC_SAMPLES, rng: np.random.Generator | None = None
) -> dict:
    """Sample posterior; return P(each is best), posterior means, P(None > 0.30)."""
    rng = rng or np.random.default_rng(0)
    categories = [*JUDGES, "NONE"]
    a = np.array([alpha + counts.get(c, 0.0) for c in categories])
    samples = rng.dirichlet(a, size=n_samples)
    judge_samples = samples[:, :3]
    none_samples = samples[:, 3]
    p_best = {}
    for i, j in enumerate(JUDGES):
        is_best = (judge_samples[:, i] > np.delete(judge_samples, i, axis=1).max(axis=1)) & (
            judge_samples[:, i] > none_samples
        )
        p_best[j] = float(is_best.mean())
    p_best["NONE"] = float((none_samples > judge_samples.max(axis=1)).mean())
    means = {c: float(a[i] / a.sum()) for i, c in enumerate(categories)}
    p_none_dominant = float((none_samples > THRESHOLDS["spec_broken_p_none_threshold"]).mean())
    return {
        "p_best": p_best,
        "means": means,
        "p_none_dominant_above_threshold": p_none_dominant,
        "counts": dict(counts),
        "alpha": alpha,
    }


def aggregate_contested_picks(picks: list[dict], clear_only: bool = False) -> dict[str, float]:
    counts: dict[str, float] = defaultdict(float)
    for p in picks:
        if p["cell_type"] != "contested":
            continue
        if clear_only and p.get("clear_or_close") != "clear":
            continue
        jw = map_pick_to_judge_weights(p)
        for j, w in jw.items():
            counts[j] += w
    return dict(counts)


def aggregate_easy_rates(picks: list[dict]) -> dict[str, float]:
    """Easy-cell pick rates per judge (excluding None)."""
    easy = [p for p in picks if p["cell_type"] == "easy"]
    if not easy:
        return {j: 0.0 for j in JUDGES}
    counts: dict[str, float] = defaultdict(float)
    for p in easy:
        jw = map_pick_to_judge_weights(p)
        for j, w in jw.items():
            if j != "NONE":
                counts[j] += w
    total = sum(counts.values()) or 1.0
    return {j: counts.get(j, 0.0) / total for j in JUDGES}


def fatigue_signal(picks: list[dict]) -> dict:
    """Median time-per-cell first-half vs second-half. Lower 2nd-half = fatigue."""
    timed = []
    for p in picks:
        if p.get("pick_timestamp") and p.get("shown_timestamp"):
            dt = (p["pick_timestamp"] - p["shown_timestamp"]) / 1000.0  # ms → sec
            if 1 <= dt <= 600:  # ignore implausibly fast or slow
                timed.append(dt)
    if len(timed) < 6:
        return {"applicable": False, "n_timed": len(timed)}
    half = len(timed) // 2
    first_med = float(np.median(timed[:half]))
    second_med = float(np.median(timed[half:]))
    ratio = second_med / first_med if first_med > 0 else 1.0
    fatigued = ratio < THRESHOLDS["fatigue_ratio"]
    return {
        "applicable": True,
        "first_half_median_sec": first_med,
        "second_half_median_sec": second_med,
        "ratio": ratio,
        "fatigued": fatigued,
    }


def detect_multimodal(picks: list[dict], min_cells: int = 6) -> dict:
    """k-means on reason embeddings; check if different judges dominate different clusters.

    Uses sentence-transformers if available, else TF-IDF as fallback.
    """
    contested = [p for p in picks if p["cell_type"] == "contested" and (p.get("reason") or "").strip()]
    if len(contested) < min_cells:
        return {"applicable": False, "reason": f"fewer than {min_cells} contested picks with reasons"}

    reasons = [p["reason"] for p in contested]
    backend = None
    embeds = None
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeds = np.asarray(model.encode(reasons, show_progress_bar=False))
        backend = "sentence-transformers/all-MiniLM-L6-v2"
    except Exception:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vec = TfidfVectorizer(max_features=512, stop_words="english")
            embeds = vec.fit_transform(reasons).toarray()
            backend = "tfidf"
        except ImportError:
            return {
                "applicable": False,
                "reason": (
                    "neither sentence-transformers nor sklearn available; install one to enable MULTI_MODAL detection"
                ),
            }

    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except ImportError:
        return {"applicable": False, "reason": "sklearn not available for clustering"}

    best_k, best_score, best_labels = 1, -1.0, np.zeros(len(reasons), dtype=int)
    for k in (2, 3):
        if k >= len(reasons):
            continue
        try:
            km = KMeans(n_clusters=k, random_state=0, n_init=10).fit(embeds)
            score = silhouette_score(embeds, km.labels_)
            if score > best_score:
                best_score = score
                best_k = k
                best_labels = km.labels_
        except Exception:
            continue

    if best_k == 1 or best_score < 0.15:
        return {
            "applicable": True,
            "k_selected": best_k,
            "silhouette": best_score,
            "is_multimodal": False,
            "backend": backend,
        }

    cluster_picks: dict[int, list[dict]] = defaultdict(list)
    for p, label in zip(contested, best_labels, strict=False):
        cluster_picks[int(label)].append(p)

    cluster_summary = []
    judges_dominating = set()
    for cluster_id, cps in cluster_picks.items():
        judge_counts: dict[str, float] = defaultdict(float)
        for p in cps:
            for j, w in map_pick_to_judge_weights(p).items():
                judge_counts[j] += w
        total = sum(judge_counts.values()) or 1.0
        dom_judge, dom_count = max(judge_counts.items(), key=lambda kv: kv[1])
        dom_rate = dom_count / total
        cluster_summary.append(
            {
                "cluster_id": cluster_id,
                "n_cells": len(cps),
                "dominant_judge": dom_judge,
                "dominant_rate": dom_rate,
                "sample_reasons": [p["reason"] for p in cps[:3]],
            }
        )
        if dom_rate >= THRESHOLDS["multimodal_within_cluster_dominance"]:
            judges_dominating.add(dom_judge)

    is_multimodal = len(judges_dominating) >= 2
    return {
        "applicable": True,
        "k_selected": best_k,
        "silhouette": best_score,
        "is_multimodal": is_multimodal,
        "backend": backend,
        "clusters": cluster_summary,
    }


def decide_verdict(posteriors_by_alpha: dict, easy_rates: dict[str, float], multimodal: dict, fatigue: dict) -> dict:
    """Apply §7.3.3 decision rule. Returns enum verdict + diagnostic overlays."""
    main = posteriors_by_alpha["alpha_1.0_weighted_all"]
    clear = posteriors_by_alpha["alpha_1.0_clear_only"]

    if main["p_none_dominant_above_threshold"] >= THRESHOLDS["spec_broken_posterior_prob"]:
        return {
            "verdict": "SPEC_NEEDS_REWRITE",
            "reason": "P(p_None > 0.30) high",
            "p_none_dominant": main["p_none_dominant_above_threshold"],
        }

    if multimodal.get("applicable") and multimodal.get("is_multimodal"):
        return {
            "verdict": "MULTI_MODAL",
            "reason": "different judges dominate different reason clusters",
            "clusters": multimodal["clusters"],
        }

    best_judge = max(JUDGES, key=lambda j: main["p_best"][j])
    p_best_main = main["p_best"][best_judge]
    mean_main = main["means"][best_judge]
    best_judge_clear = max(JUDGES, key=lambda j: clear["p_best"][j])
    agree_on_best = best_judge == best_judge_clear

    sensitivity_robust = all(
        max(
            posteriors_by_alpha[f"alpha_{a}_weighted_all"]["p_best"],
            key=lambda j: posteriors_by_alpha[f"alpha_{a}_weighted_all"]["p_best"][j],
        )
        == best_judge
        for a in ("0.5", "1.0", "2.0")
    )

    contested_lift = None
    if best_judge != "NONE":
        # Approximate contested_rate from posterior means.
        hard_rate = mean_main
        easy_rate = easy_rates.get(best_judge, 0.0)
        contested_lift = hard_rate - easy_rate

    confident = (
        p_best_main >= THRESHOLDS["adopt_p_best"]
        and mean_main >= THRESHOLDS["posterior_mean_floor_adopt"]
        and agree_on_best
        and sensitivity_robust
    )
    tentative = (
        p_best_main >= THRESHOLDS["tentative_p_best"] and mean_main >= THRESHOLDS["posterior_mean_floor_tentative"]
    )

    if confident:
        verdict = f"ROUTE_TO_{best_judge.upper().replace('-', '_')}"
    elif tentative:
        verdict = f"TENTATIVE_ROUTE_TO_{best_judge.upper().replace('-', '_')}"
    else:
        verdict = "NO_CLEAR_JUDGE"

    if fatigue.get("fatigued"):
        # Drop a tier when fatigued.
        if verdict.startswith("ROUTE_TO_"):
            verdict = "TENTATIVE_" + verdict
        elif verdict.startswith("TENTATIVE_"):
            verdict = "NO_CLEAR_JUDGE"

    return {
        "verdict": verdict,
        "best_judge": best_judge,
        "p_best": p_best_main,
        "posterior_mean": mean_main,
        "clear_and_weighted_agree": agree_on_best,
        "sensitivity_robust": sensitivity_robust,
        "contested_lift": contested_lift,
        "fatigued": fatigue.get("fatigued", False),
    }


def analyze_statement(picks: list[dict]) -> dict:
    rng = np.random.default_rng(42)
    counts_weighted = aggregate_contested_picks(picks, clear_only=False)
    counts_clear = aggregate_contested_picks(picks, clear_only=True)
    posteriors = {}
    for alpha in (0.5, 1.0, 2.0):
        posteriors[f"alpha_{alpha}_weighted_all"] = dirichlet_posterior(counts_weighted, alpha, rng=rng)
        if counts_clear:
            posteriors[f"alpha_{alpha}_clear_only"] = dirichlet_posterior(counts_clear, alpha, rng=rng)
        else:
            posteriors[f"alpha_{alpha}_clear_only"] = posteriors[f"alpha_{alpha}_weighted_all"]
    easy_rates = aggregate_easy_rates(picks)
    multimodal = detect_multimodal(picks)
    fatigue = fatigue_signal(picks)
    verdict = decide_verdict(posteriors, easy_rates, multimodal, fatigue)
    return {
        "verdict": verdict,
        "easy_rates": easy_rates,
        "multimodal": multimodal,
        "fatigue": fatigue,
        "posteriors": posteriors,
        "n_contested_picks": sum(1 for p in picks if p["cell_type"] == "contested"),
        "n_easy_picks": sum(1 for p in picks if p["cell_type"] == "easy"),
    }


def print_report(report: dict) -> None:
    print(f"\n{'='*72}")
    print("DART Human-Apply Pilot Analyzer Report")
    print(f"{'='*72}\n")
    for sid, r in report.items():
        v = r["verdict"]
        print(f"--- {sid} ---")
        print(f"  VERDICT: {v['verdict']}")
        if "best_judge" in v:
            print(f"  best judge: {v['best_judge']} (P(best)={v['p_best']:.3f}, mean={v['posterior_mean']:.3f})")
            if v.get("contested_lift") is not None:
                print(
                    f"  contested_lift: {v['contested_lift']:+.3f}  (positive = score-correctness driven; "
                    f"negative = reasoning-style spillover)"
                )
            print(
                f"  clear & weighted agree: {v['clear_and_weighted_agree']}, "
                f"sensitivity_robust: {v['sensitivity_robust']}"
            )
        if r["multimodal"].get("applicable"):
            mm = r["multimodal"]
            print(
                f"  multimodal: k_selected={mm['k_selected']}, silhouette={mm.get('silhouette', 0):.3f}, "
                f"is_multimodal={mm['is_multimodal']}"
            )
            if mm.get("is_multimodal"):
                for c in mm["clusters"]:
                    print(
                        f"    cluster {c['cluster_id']}: {c['n_cells']} cells, "
                        f"dominant judge = {c['dominant_judge']} ({c['dominant_rate']:.0%})"
                    )
        if r["fatigue"].get("applicable"):
            f = r["fatigue"]
            print(
                f"  fatigue: first-half {f['first_half_median_sec']:.1f}s, "
                f"second-half {f['second_half_median_sec']:.1f}s, ratio={f['ratio']:.2f}, "
                f"fatigued={f['fatigued']}"
            )
        print(f"  picks: {r['n_contested_picks']} contested + {r['n_easy_picks']} easy")
        print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("picks_jsonl", type=Path, help="Picks JSONL exported from viewer")
    p.add_argument("--output", "-o", type=Path, help="JSON report output path")
    args = p.parse_args()

    picks = load_picks(args.picks_jsonl)
    by_stmt: dict[str, list] = defaultdict(list)
    for p_row in picks:
        by_stmt[p_row["statement_id"]].append(p_row)

    report = {sid: analyze_statement(rows) for sid, rows in by_stmt.items()}
    print_report(report)

    if args.output:
        args.output.write_text(json.dumps(report, indent=2, default=float))
        print(f"\nWrote full report to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
