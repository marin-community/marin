# ruff: noqa: E501, RUF001
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sampling-noise control: rerun GLM-5.1 v2 rubric writer N times on the base
spec (no edits) and compute pairwise text-change distributions.

Compares to the with-edits Δ for GLM-5.1 to separate edit signal from
sampling noise.

Inputs:
- experiments/posttrain/stage3_output/cross_tier_rubrics_v2_glm51_resample_{1..5}.jsonl
  (5 independent reruns of the writer at temperature=0.2, no spec changes)
- experiments/posttrain/stage3_output/cross_tier_rubrics_v2_glm51.jsonl (baseline)
- experiments/posttrain/stage3_output/cross_tier_rubrics_v2_glm51_with_self_edits.jsonl (R1)

Output:
- experiments/posttrain/stage3_output/exp_glm51_resample_noise.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import statistics
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("exp_glm51_resample_noise")

WORKTREE = Path(__file__).resolve().parents[2]
STAGE3 = WORKTREE / "experiments/posttrain/stage3_output"

FIELDS = [
    ("dominant_rubric.GOOD", ("dominant_rubric", "GOOD")),
    ("dominant_rubric.BAD", ("dominant_rubric", "BAD")),
    ("rationale.alternative_readings_rejected", ("rationale", "alternative_readings_rejected")),
    ("worked_example.spec_compliant", ("worked_example", "spec_compliant")),
]


def load_rows(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {(row["pair_id"], row["tension_point_idx"]): row for row in rows}


def get_field(row: dict[str, Any], path: tuple[str, ...]) -> str:
    obj = row.get("parsed", {})
    for key in path:
        obj = obj.get(key, "") if isinstance(obj, dict) else ""
    return obj if isinstance(obj, str) else ""


def text_change(a: str, b: str) -> float:
    """1 - SequenceMatcher ratio. Symmetric. 0=identical, 1=disjoint."""
    return 1.0 - SequenceMatcher(None, a, b).ratio()


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    pos = q * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else float("nan"),
        "p25": quantile(values, 0.25),
        "p50": quantile(values, 0.50),
        "p75": quantile(values, 0.75),
        "p95": quantile(values, 0.95),
        "max": max(values) if values else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-resamples",
        type=int,
        default=5,
        help="Number of resample files to load (default 5).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=STAGE3 / "cross_tier_rubrics_v2_glm51.jsonl",
    )
    parser.add_argument(
        "--with-edits",
        type=Path,
        default=STAGE3 / "cross_tier_rubrics_v2_glm51_with_self_edits.jsonl",
        help="R1 (with-self-edits) GLM-5.1 file for comparison.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=STAGE3 / "exp_glm51_resample_noise.md",
    )
    args = parser.parse_args()

    # Load resamples.
    runs: dict[int, dict[tuple[str, int], dict[str, Any]]] = {}
    for i in range(1, args.num_resamples + 1):
        path = STAGE3 / f"cross_tier_rubrics_v2_glm51_resample_{i}.jsonl"
        rows = load_rows(path)
        ok = sum(1 for r in rows.values() if r["diag"].get("schema_ok"))
        logger.info("resample %d: %d/%d schema_ok at %s", i, ok, len(rows), path)
        runs[i] = rows

    # Identify the common (pair_id, tp) keys.
    common_keys = set.intersection(*(set(rows.keys()) for rows in runs.values()))
    logger.info("common (pair, tp) keys across all %d runs: %d", args.num_resamples, len(common_keys))

    # Pairwise within resamples: compute per-field text-change.
    run_indices = sorted(runs.keys())
    run_pairs = list(itertools.combinations(run_indices, 2))
    logger.info("pairwise run comparisons: %d", len(run_pairs))

    # noise[field_label] -> list of text_change values, one per (key, run-pair).
    noise: dict[str, list[float]] = {label: [] for label, _ in FIELDS}
    for i, j in run_pairs:
        for key in common_keys:
            a = runs[i][key]
            b = runs[j][key]
            for label, path in FIELDS:
                noise[label].append(text_change(get_field(a, path), get_field(b, path)))

    noise_summary = {label: summarize(values) for label, values in noise.items()}

    # With-edits Δ for GLM-5.1: compare baseline vs R1 (self-edits).
    baseline = load_rows(args.baseline)
    with_edits = load_rows(args.with_edits)
    edit_keys = set(baseline.keys()) & set(with_edits.keys())
    logger.info("baseline rows: %d, with-edits rows: %d, intersect: %d", len(baseline), len(with_edits), len(edit_keys))

    edit_signal: dict[str, list[float]] = {label: [] for label, _ in FIELDS}
    for key in edit_keys:
        a = baseline[key]
        b = with_edits[key]
        for label, path in FIELDS:
            edit_signal[label].append(text_change(get_field(a, path), get_field(b, path)))

    edit_summary = {label: summarize(values) for label, values in edit_signal.items()}

    # Write markdown.
    n_keys = len(common_keys)
    n_pairs = len(run_pairs)
    n_datapoints = n_keys * n_pairs

    lines: list[str] = []
    lines.append("# GLM-5.1 sampling-noise control")
    lines.append("")
    lines.append(
        f"5 independent reruns of `write_cross_tier_rubrics_v2_glm51.py` on the **base spec, no edits**, "
        f"at temperature=0.2 (the writer's default). Pairwise per-field text-change "
        f"(`1 - difflib.SequenceMatcher.ratio`) is computed across all {n_pairs} run-pairs × {n_keys} "
        f"(pair_id, tension_point) tasks = **{n_datapoints} datapoints per field**."
    )
    lines.append("")
    lines.append(f"- Baseline (single run): `{args.baseline.relative_to(WORKTREE)}`")
    lines.append(f"- With-edits R1 (self-edits, single run): `{args.with_edits.relative_to(WORKTREE)}`")
    lines.append(f"- Resample files: `cross_tier_rubrics_v2_glm51_resample_{{1..{args.num_resamples}}}.jsonl`")
    lines.append("")

    lines.append("## No-edit sampling-noise floor (pairwise across 5 reruns)")
    lines.append("")
    lines.append("| Field | n | mean | p25 | p50 | p75 | p95 | max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, _ in FIELDS:
        s = noise_summary[label]
        lines.append(
            f"| `{label}` | {s['n']} | {s['mean']:.3f} | {s['p25']:.3f} | "
            f"{s['p50']:.3f} | {s['p75']:.3f} | {s['p95']:.3f} | {s['max']:.3f} |"
        )
    lines.append("")

    lines.append("## With-edits Δ (GLM-5.1 baseline vs R1 self-edits, single run each)")
    lines.append("")
    lines.append("| Field | n | mean | p25 | p50 | p75 | p95 | max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, _ in FIELDS:
        s = edit_summary[label]
        lines.append(
            f"| `{label}` | {s['n']} | {s['mean']:.3f} | {s['p25']:.3f} | "
            f"{s['p50']:.3f} | {s['p75']:.3f} | {s['p95']:.3f} | {s['max']:.3f} |"
        )
    lines.append("")

    lines.append("## Signal vs noise: edit-Δ minus noise-floor")
    lines.append("")
    lines.append(
        "| Field | edit-Δ mean | noise mean | gap (edit − noise) | edit-Δ p50 | noise p50 | gap (p50) | ratio (edit-mean / noise-mean) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, _ in FIELDS:
        e = edit_summary[label]["mean"]
        n = noise_summary[label]["mean"]
        ep50 = edit_summary[label]["p50"]
        np50 = noise_summary[label]["p50"]
        ratio = (e / n) if n > 0 else float("inf")
        lines.append(
            f"| `{label}` | {e:.3f} | {n:.3f} | {e - n:+.3f} | {ep50:.3f} | {np50:.3f} | "
            f"{ep50 - np50:+.3f} | {ratio:.2f}× |"
        )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The no-edit noise floor measures how much rubric text shifts purely from "
        "stochastic decoding (temperature=0.2) when the writer is given the **same** "
        "prompt and base spec. The with-edits Δ measures the change between baseline and "
        "R1 (self-edits). The gap (edit − noise) is the **edit-attributable signal**; "
        "the ratio tells you how many noise floors the with-edits change clears."
    )
    lines.append("")
    lines.append(
        "If the ratio is near 1× the field is dominated by sampling noise; if it is ≥2× "
        "the edit is shifting text well beyond what re-rolling the dice would produce."
    )
    lines.append("")

    args.output.write_text("\n".join(lines) + "\n")
    logger.info("wrote report to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
