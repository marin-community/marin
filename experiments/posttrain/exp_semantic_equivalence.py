# ruff: noqa: B905, RUF001, RUF002
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quantitative semantic-equivalence analysis on rubric resamples.

We have 5 independent runs of the GPT-5.1 (temperature=0, reasoning_effort=none)
rubric writer on the same 22 (pair_id, tension_point_idx) prompts. Pairwise
text-Δ on the resulting rubrics is ~80% per field, but eyeballing suggests the
two runs are saying the *same things* in different words.

This script answers: across 22 rubrics × 10 (i<j) run-pair combos = 220
datapoints, how often are the rubric pairs *semantically* equivalent?

We ask GPT-5.1 (reasoning_effort="none", temperature=0, json_object) to score
each pair on five 0-10 axes plus a boolean `materially_different`:

- good_equivalence
- bad_equivalence
- worked_example_equivalence
- interpretive_equivalence
- overall_equivalence
- materially_different (bool)
- disagreement_summary (str, only meaningful when materially_different=true)

Outputs:
- experiments/posttrain/stage3_output/exp_semantic_equivalence_raw.jsonl
- experiments/posttrain/stage3_output/exp_semantic_equivalence.md

Usage:
    source .env && python experiments/posttrain/exp_semantic_equivalence.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("exp_semantic_equivalence")

WORKTREE = Path(__file__).resolve().parents[2]
STAGE3 = WORKTREE / "experiments/posttrain/stage3_output"
RESAMPLE_PATHS = [STAGE3 / f"cross_tier_rubrics_v2_gpt51_temp0_resample_{i}.jsonl" for i in range(1, 6)]
RAW_OUT = STAGE3 / "exp_semantic_equivalence_raw.jsonl"
REPORT_OUT = STAGE3 / "exp_semantic_equivalence.md"

JUDGE_MODEL = "gpt-5.1"
MAX_COMPLETION_TOKENS = 1500
MAX_RETRIES = 4
DEFAULT_WORKERS = 8

# Fields of interest in each rubric. (label, dotted-path tuple)
FIELDS: list[tuple[str, tuple[str, ...]]] = [
    ("dominant_rubric.GOOD", ("dominant_rubric", "GOOD")),
    ("dominant_rubric.BAD", ("dominant_rubric", "BAD")),
    ("worked_example.spec_compliant", ("worked_example", "spec_compliant")),
    ("rationale.interpretive_choices_made", ("rationale", "interpretive_choices_made")),
]


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #


def load_resample(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
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


# --------------------------------------------------------------------------- #
# Judge prompt
# --------------------------------------------------------------------------- #


SYSTEM_PROMPT = """You are an expert rubric-evaluation judge. You will be given two \
rubrics that were produced by independent runs of the *same* rubric writer on \
the *same* spec-tension prompt. Your job is to decide whether the two rubrics \
are semantically equivalent — i.e., whether they would produce the same verdict \
on the same model response — even if their wording differs.

You must score each axis on a 0-10 integer scale where:
- 10 = the two rubrics are completely equivalent on this axis (same set of \
behaviors classified the same way)
- 7-9 = minor wording or emphasis differences, no behavioral disagreement \
expected
- 4-6 = real but limited disagreement: some edge cases would be classified \
differently
- 1-3 = substantial disagreement: many model responses would be classified \
differently
- 0 = the two rubrics endorse opposite positions

Be precise. Do not reward identical wording — reward identical *meaning*. \
Two rubrics that cite different example behaviors but converge on the same \
underlying principle should score high.

Return strict JSON with exactly these keys:
- good_equivalence (int 0-10)
- bad_equivalence (int 0-10)
- worked_example_equivalence (int 0-10)
- interpretive_equivalence (int 0-10)
- overall_equivalence (int 0-10)
- materially_different (bool): true iff the two rubrics would produce \
different verdicts on at least one realistic model response
- disagreement_summary (string): if materially_different is true, 1-2 \
sentences naming the disagreement. If false, the empty string.
"""


def build_user_prompt(rubric_a: dict[str, str], rubric_b: dict[str, str]) -> str:
    parts = ["# Rubric A\n"]
    for label, _ in FIELDS:
        parts.append(f"## {label}\n{rubric_a.get(label, '')}\n")
    parts.append("\n# Rubric B\n")
    for label, _ in FIELDS:
        parts.append(f"## {label}\n{rubric_b.get(label, '')}\n")
    parts.append(
        "\nScore semantic equivalence between Rubric A and Rubric B. "
        "Return strict JSON with the exact keys specified."
    )
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# OpenAI call
# --------------------------------------------------------------------------- #


REQUIRED_KEYS = {
    "good_equivalence",
    "bad_equivalence",
    "worked_example_equivalence",
    "interpretive_equivalence",
    "overall_equivalence",
    "materially_different",
    "disagreement_summary",
}


def call_judge(
    client: OpenAI,
    rubric_a: dict[str, str],
    rubric_b: dict[str, str],
    max_retries: int = MAX_RETRIES,
) -> dict[str, Any]:
    user = build_user_prompt(rubric_a, rubric_b)
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                max_completion_tokens=MAX_COMPLETION_TOKENS,
                reasoning_effort="none",
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            data = json.loads(content)
            missing = REQUIRED_KEYS - set(data.keys())
            if missing:
                raise ValueError(f"missing keys: {sorted(missing)}")
            return data
        except Exception as e:
            last_err = e
            sleep_s = 2**attempt
            logger.warning("judge attempt %d failed: %s; retrying in %ds", attempt + 1, e, sleep_s)
            time.sleep(sleep_s)
    raise RuntimeError(f"call_judge failed after {max_retries} attempts: {last_err}")


# --------------------------------------------------------------------------- #
# Pairwise driver
# --------------------------------------------------------------------------- #


def extract_rubric(row: dict[str, Any]) -> dict[str, str]:
    return {label: get_field(row, path) for label, path in FIELDS}


def compute_text_deltas(a: dict[str, str], b: dict[str, str]) -> dict[str, float]:
    return {label: text_change(a[label], b[label]) for label, _ in FIELDS}


def build_jobs(
    resamples: list[dict[tuple[str, int], dict[str, Any]]],
    common_keys: list[tuple[str, int]],
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for key in common_keys:
        for i, j in itertools.combinations(range(len(resamples)), 2):
            row_i = resamples[i][key]
            row_j = resamples[j][key]
            rubric_i = extract_rubric(row_i)
            rubric_j = extract_rubric(row_j)
            jobs.append(
                {
                    "pair_id": key[0],
                    "tension_point_idx": key[1],
                    "run_i": i + 1,  # 1-indexed for human readability
                    "run_j": j + 1,
                    "rubric_i": rubric_i,
                    "rubric_j": rubric_j,
                    "text_deltas": compute_text_deltas(rubric_i, rubric_j),
                }
            )
    return jobs


def run_one(client: OpenAI, job: dict[str, Any]) -> dict[str, Any]:
    judgment = call_judge(client, job["rubric_i"], job["rubric_j"])
    return {
        "pair_id": job["pair_id"],
        "tension_point_idx": job["tension_point_idx"],
        "run_i": job["run_i"],
        "run_j": job["run_j"],
        "text_deltas": job["text_deltas"],
        "judgment": judgment,
    }


# --------------------------------------------------------------------------- #
# Aggregation / report
# --------------------------------------------------------------------------- #


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
        "min": min(values) if values else float("nan"),
        "max": max(values) if values else float("nan"),
    }


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    if sx == 0 or sy == 0:
        return float("nan")
    return num / (sx * sy)


def spearman(xs: list[float], ys: list[float]) -> float:
    def rank(v: list[float]) -> list[float]:
        # average ranks for ties
        order = sorted(range(len(v)), key=lambda k: v[k])
        ranks = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
            i = j + 1
        return ranks

    return pearson(rank(xs), rank(ys))


def write_report(results: list[dict[str, Any]], out_path: Path) -> None:
    n = len(results)
    axes = [
        "good_equivalence",
        "bad_equivalence",
        "worked_example_equivalence",
        "interpretive_equivalence",
        "overall_equivalence",
    ]
    per_axis = {a: [float(r["judgment"][a]) for r in results] for a in axes}
    materially = [bool(r["judgment"]["materially_different"]) for r in results]
    n_mat = sum(materially)
    pct_mat = (n_mat / n * 100) if n else 0.0

    overall_eq = per_axis["overall_equivalence"]
    # text-Δ summary per pair: mean across 4 fields
    mean_text_delta = [statistics.fmean(list(r["text_deltas"].values())) for r in results]
    pearson_r = pearson(mean_text_delta, overall_eq)
    spearman_r = spearman(mean_text_delta, overall_eq)

    # Rank by mean text-Δ for the top-5 highest-Δ pairs
    indexed = sorted(range(n), key=lambda i: mean_text_delta[i], reverse=True)[:5]

    lines: list[str] = []
    lines.append("# Semantic Equivalence of GPT-5.1 Temp=0 Rubric Resamples")
    lines.append("")
    lines.append(f"- **Total samples**: {n}")
    lines.append(f"- **Materially different**: {n_mat} / {n} ({pct_mat:.1f}%)")
    lines.append(f"- **Mean overall_equivalence**: {statistics.fmean(overall_eq):.2f} / 10")
    lines.append(f"- **Pearson r (text-Δ vs overall_equivalence)**: {pearson_r:+.3f}")
    lines.append(f"- **Spearman ρ (text-Δ vs overall_equivalence)**: {spearman_r:+.3f}")
    lines.append("")
    lines.append("## Per-axis distribution")
    lines.append("")
    lines.append("| axis | n | mean | p25 | p50 | p75 | p95 | min | max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for a in axes:
        s = summarize(per_axis[a])
        lines.append(
            f"| {a} | {s['n']} | {s['mean']:.2f} | {s['p25']:.2f} | "
            f"{s['p50']:.2f} | {s['p75']:.2f} | {s['p95']:.2f} | "
            f"{s['min']:.2f} | {s['max']:.2f} |"
        )
    lines.append("")

    # Materially-different list
    lines.append("## Materially-different cases")
    lines.append("")
    if n_mat == 0:
        lines.append("_None._")
    else:
        lines.append("| pair_id | tp | run_i | run_j | overall_eq | summary |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for r in results:
            if not r["judgment"]["materially_different"]:
                continue
            j = r["judgment"]
            summary = j.get("disagreement_summary", "").replace("|", "\\|")
            if len(summary) > 240:
                summary = summary[:237] + "..."
            lines.append(
                f"| {r['pair_id']} | {r['tension_point_idx']} | "
                f"{r['run_i']} | {r['run_j']} | "
                f"{int(j['overall_equivalence'])} | {summary} |"
            )
    lines.append("")

    # Top-5 highest text-Δ
    lines.append("## Top-5 highest text-Δ pairs (sanity check: are they semantically different?)")
    lines.append("")
    lines.append(
        "| pair_id | tp | run_i | run_j | mean text-Δ | good | bad | worked | interp | overall | materially_diff |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for idx in indexed:
        r = results[idx]
        j = r["judgment"]
        lines.append(
            f"| {r['pair_id']} | {r['tension_point_idx']} | "
            f"{r['run_i']} | {r['run_j']} | "
            f"{mean_text_delta[idx]:.2f} | "
            f"{int(j['good_equivalence'])} | {int(j['bad_equivalence'])} | "
            f"{int(j['worked_example_equivalence'])} | "
            f"{int(j['interpretive_equivalence'])} | "
            f"{int(j['overall_equivalence'])} | "
            f"{'YES' if j['materially_different'] else 'no'} |"
        )
    lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")
    if pct_mat < 10:
        verdict = (
            "The rubrics are largely **semantically equivalent** across "
            f"runs; only {pct_mat:.1f}% of pairs were judged materially "
            "different despite ~80% text-level divergence."
        )
    elif pct_mat < 30:
        verdict = (
            f"**Mixed**: {pct_mat:.1f}% of rubric pairs are materially "
            "different. The rubric writer is mostly consistent but has "
            "real run-to-run drift on a non-trivial minority."
        )
    else:
        verdict = (
            f"**Substantial divergence**: {pct_mat:.1f}% of rubric pairs "
            "are materially different. Run-to-run rubric quality is not "
            "reliable for downstream training."
        )
    lines.append(verdict)
    lines.append("")

    out_path.write_text("\n".join(lines))
    logger.info("wrote report to %s", out_path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip jobs whose (pair_id, tp, run_i, run_j) is already in the raw output.",
    )
    ap.add_argument(
        "--resample-pattern",
        default="cross_tier_rubrics_v2_gpt51_temp0_resample_{i}.jsonl",
        help="Filename pattern with {i} placeholder, relative to stage3_output. Default = GPT-5.1 temp=0.",
    )
    ap.add_argument(
        "--num-resamples",
        type=int,
        default=5,
    )
    ap.add_argument(
        "--raw-out",
        default="exp_semantic_equivalence_raw.jsonl",
        help="Raw judgments output (relative to stage3_output).",
    )
    ap.add_argument(
        "--report-out",
        default="exp_semantic_equivalence.md",
        help="Report output (relative to stage3_output).",
    )
    args = ap.parse_args()

    if "OPENAI_API_KEY" not in os.environ:
        logger.error("OPENAI_API_KEY not set. Run with `source .env && ...`.")
        return 2

    # Build resample paths from pattern
    resample_paths = [STAGE3 / args.resample_pattern.format(i=i) for i in range(1, args.num_resamples + 1)]
    raw_out = STAGE3 / args.raw_out
    report_out = STAGE3 / args.report_out

    # Reroute the module-level constants for the rest of the run
    global RAW_OUT, REPORT_OUT
    RAW_OUT = raw_out
    REPORT_OUT = report_out

    resamples = [load_resample(p) for p in resample_paths]
    common = set.intersection(*[set(r.keys()) for r in resamples])
    common_keys = sorted(common)
    logger.info("loaded %d resamples; %d common rubrics", len(resamples), len(common_keys))

    jobs = build_jobs(resamples, common_keys)
    logger.info("built %d pairwise jobs", len(jobs))

    # Optional resume
    done: set[tuple[str, int, int, int]] = set()
    existing_results: list[dict[str, Any]] = []
    if args.resume and RAW_OUT.exists():
        for line in RAW_OUT.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            done.add((r["pair_id"], r["tension_point_idx"], r["run_i"], r["run_j"]))
            existing_results.append(r)
        logger.info("resume: %d already-done jobs in %s", len(done), RAW_OUT)
    todo = [j for j in jobs if (j["pair_id"], j["tension_point_idx"], j["run_i"], j["run_j"]) not in done]
    logger.info("%d jobs to run", len(todo))

    client = OpenAI()
    t_start = time.time()
    new_results: list[dict[str, Any]] = []
    n_ok, n_err = 0, 0
    # append-mode raw log
    mode = "a" if (args.resume and RAW_OUT.exists()) else "w"
    with RAW_OUT.open(mode) as raw_f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            future_to_job = {pool.submit(run_one, client, j): j for j in todo}
            for k, fut in enumerate(as_completed(future_to_job), start=1):
                job = future_to_job[fut]
                try:
                    result = fut.result()
                    raw_f.write(json.dumps(result) + "\n")
                    raw_f.flush()
                    new_results.append(result)
                    n_ok += 1
                except Exception as e:
                    n_err += 1
                    logger.error(
                        "job (%s, tp=%d, %d-%d) FAILED: %s",
                        job["pair_id"],
                        job["tension_point_idx"],
                        job["run_i"],
                        job["run_j"],
                        e,
                    )
                if k % 10 == 0 or k == len(todo):
                    elapsed = time.time() - t_start
                    rate = k / elapsed if elapsed > 0 else 0.0
                    logger.info(
                        "progress: %d/%d (ok=%d err=%d) %.1fs %.2f jobs/s",
                        k,
                        len(todo),
                        n_ok,
                        n_err,
                        elapsed,
                        rate,
                    )

    all_results = existing_results + new_results
    logger.info(
        "complete: %d total results (%d new ok, %d new err)",
        len(all_results),
        n_ok,
        n_err,
    )
    if not all_results:
        logger.error("no results — aborting before report")
        return 1

    write_report(all_results, REPORT_OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
