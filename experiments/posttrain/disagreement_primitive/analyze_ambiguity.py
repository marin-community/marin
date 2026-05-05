# ruff: noqa: E501, RUF001, RUF002
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analyzer for SPEC AMBIGUITY EPIC Tier 1+2 outputs.

Consumes:
- `method_b_readings.jsonl`  (Method B: self-disambiguation)
- `method_a_grades.jsonl`    (Method A: bare-statement grading)
- `method_d_rubrics.jsonl`   (Method D: text-only vs examples-only rubrics)

Produces per-statement typed diagnostic + a markdown report:

- Method B: GPT-5.1 clusters the 3-judge × ~3-reading set into semantic groups.
  Cluster count is the ambiguity signal. ≥3 clusters → `language_ambiguous`.
- Method A: per-statement mean score variance across judges + informing-words
  overlap. High variance + low overlap → `language_ambiguous`.
- Method D: GPT-5.1 scores semantic equivalence between text-only and
  examples-only rubrics. Low equivalence → `internally_inconsistent`.

Synthesis logic (per statement):
- internally_inconsistent if D equiv < 7
- language_ambiguous if (B clusters ≥ 3) OR (A score variance ≥ 1.5 AND informing-words overlap < 0.3)
- language_robust otherwise

Outputs:
- `spec_ambiguity_diagnostic.jsonl` (per-statement typed labels)
- `spec_ambiguity_diagnostic.md` (human-readable report)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("analyze_ambiguity")

WORKTREE = Path(__file__).resolve().parents[3]
SPEC_PATH = WORKTREE / "experiments/posttrain/specs/openai_model_spec.jsonl"
DIR = WORKTREE / "experiments/posttrain/disagreement_primitive"

GPT_MODEL = "gpt-5.1"
DEFAULT_TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 2000
MAX_WORKERS = 6


# --------------------------------------------------------------------------- #
# Method B clustering judge
# --------------------------------------------------------------------------- #

CLUSTER_SYSTEM = """You are a clustering judge. You will be given several candidate "readings" of a single spec statement, produced by different language models. Each reading describes an operational interpretation of the statement.

Your job: cluster the readings into groups by SEMANTIC equivalence — readings in the same group would produce the same operational behavior on the same response, even if their wording differs.

Definition of "semantically equivalent": two readings are equivalent if applying them to a borderline response would produce the same compliant/violating verdict in 9 out of 10 cases.

OUTPUT FORMAT (single JSON, no markdown):
{
  "n_distinct_clusters": <int 1+>,
  "clusters": [
    {
      "cluster_id": 1,
      "centroid_description": "<one-sentence summary of what unifies this cluster>",
      "reading_indices": [<int>, ...]
    },
    ...
  ],
  "interpretation_summary": "<2-3 sentences: is this statement language operationally clear (1 cluster) or ambiguous (≥2 clusters)? What's the central interpretive divergence if any?>"
}

REQUIREMENTS
- Number reading_indices from 1 (matching the input numbering).
- Every input reading goes into exactly one cluster.
- Don't artificially split similar readings; don't artificially merge different ones.
- 1 cluster means the statement is operationally clear. 2 clusters means a real divergence. 3+ means substantial ambiguity.
- No commentary outside JSON.
"""


def build_cluster_prompt(statement: dict[str, Any], readings: list[dict[str, Any]]) -> str:
    parts = [
        f"STATEMENT_ID: {statement['id']}",
        f"STATEMENT TEXT:\n{statement['text']}\n",
        f"\nCandidate readings ({len(readings)} total):",
    ]
    for i, r in enumerate(readings, 1):
        desc = r.get("description", "")
        ex = r.get("example_satisfying_only_this_reading", "")
        parts.append(f"\n[{i}] description: {desc}")
        parts.append(f"    example: {ex}")
    parts.append("\nNow cluster per the schema.")
    return "\n".join(parts)


# --------------------------------------------------------------------------- #
# Method D semantic-equivalence judge
# --------------------------------------------------------------------------- #

EQUIV_SYSTEM = """You compare two rubrics for the same spec statement. Rubric T was generated from the statement TEXT only (no examples). Rubric E was generated from the statement EXAMPLES only (no text). Both rubrics describe what counts as satisfying or violating the statement.

Your job: score whether T and E encode the same operational rule. If they diverge meaningfully, the spec is internally inconsistent — the prose says one thing, the examples model another.

Score on 0-10 axes:
- good_equivalence: would T's GOOD criterion and E's GOOD criterion classify the same responses as compliant?
- bad_equivalence: would T's BAD criterion and E's BAD criterion classify the same responses as violating?
- overall_equivalence: holistic — would the rubrics produce the same verdicts in 9/10 cases?

OUTPUT FORMAT (single JSON, no markdown):
{
  "good_equivalence": <int 0-10>,
  "bad_equivalence": <int 0-10>,
  "overall_equivalence": <int 0-10>,
  "materially_different": <bool>,
  "disagreement_summary": "<1-2 sentences if materially_different=true, naming the specific divergence; empty otherwise>"
}

SCORING GUIDE
- 10: identical operational rules
- 7-9: minor wording differences, no behavioral disagreement expected
- 4-6: real but limited disagreement; some edge cases differ
- 1-3: substantial disagreement; many cases differ
- 0: rubrics endorse opposite positions

Two rubrics that cite different examples but converge on the same underlying principle should score high. Two rubrics that share vocabulary but encode different operational rules should score low.
"""


def build_equiv_prompt(statement: dict[str, Any], rubric_t: dict[str, Any], rubric_e: dict[str, Any]) -> str:
    return (
        f"STATEMENT_ID: {statement['id']}\n\n"
        f"RUBRIC T (compiled from statement text only):\n"
        f"GOOD: {rubric_t.get('good_criterion', '')}\n"
        f"BAD: {rubric_t.get('bad_criterion', '')}\n"
        f"KEY_TENSION: {rubric_t.get('key_tension', '')}\n\n"
        f"RUBRIC E (compiled from statement examples only):\n"
        f"GOOD: {rubric_e.get('good_criterion', '')}\n"
        f"BAD: {rubric_e.get('bad_criterion', '')}\n"
        f"KEY_TENSION: {rubric_e.get('key_tension', '')}\n\n"
        "Score the equivalence per the schema."
    )


# --------------------------------------------------------------------------- #
# Shared GPT-5.1 caller
# --------------------------------------------------------------------------- #


def parse_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


def call_gpt(client: OpenAI, system: str, user: str, max_retries: int = 2) -> dict[str, Any]:
    last_err = ""
    last_content = ""
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=DEFAULT_TEMPERATURE,
                max_completion_tokens=MAX_OUTPUT_TOKENS,
                reasoning_effort="none",
                response_format={"type": "json_object"},
            )
            content = resp.choices[0].message.content or ""
            last_content = content
            return parse_json(content)
        except Exception as exc:
            last_err = str(exc)
            time.sleep(1 + attempt)
    raise RuntimeError(f"gpt failed: {last_err}; last={last_content[:300]}")


# --------------------------------------------------------------------------- #
# Method A offline analysis (no API)
# --------------------------------------------------------------------------- #


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(s.lower().strip() for s in a if s), set(s.lower().strip() for s in b if s)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def analyze_method_a(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per (statement, scenario, generator), compute score variance across judges
    and informing-words overlap. Aggregate per statement."""
    by_key = defaultdict(list)
    for r in rows:
        key = (r["statement_id"], r["scenario_id"], r["generator_model"])
        by_key[key].append(r)
    per_stmt_variance = defaultdict(list)
    per_stmt_overlap = defaultdict(list)
    per_stmt_scores = defaultdict(list)
    for (stmt_id, _sid, _gen), judge_rows in by_key.items():
        if len(judge_rows) < 2:
            continue
        scores = [r["score"] for r in judge_rows]
        try:
            var = statistics.stdev(scores) if len(scores) >= 2 else 0.0
        except statistics.StatisticsError:
            var = 0.0
        per_stmt_variance[stmt_id].append(var)
        per_stmt_scores[stmt_id].extend(scores)
        # Pairwise informing-words jaccard, average over pairs
        pairs = []
        for i in range(len(judge_rows)):
            for j in range(i + 1, len(judge_rows)):
                pairs.append(jaccard(judge_rows[i].get("informing_words") or [],
                                     judge_rows[j].get("informing_words") or []))
        if pairs:
            per_stmt_overlap[stmt_id].append(sum(pairs) / len(pairs))
    out: dict[str, dict[str, Any]] = {}
    for stmt_id in set(list(per_stmt_variance.keys()) + list(per_stmt_overlap.keys())):
        v = per_stmt_variance.get(stmt_id) or []
        o = per_stmt_overlap.get(stmt_id) or []
        s = per_stmt_scores.get(stmt_id) or []
        out[stmt_id] = {
            "n_scenarios": len(v),
            "mean_score_stdev": round(sum(v) / len(v), 3) if v else None,
            "max_score_stdev": round(max(v), 3) if v else None,
            "mean_informing_jaccard": round(sum(o) / len(o), 3) if o else None,
            "mean_score": round(sum(s) / len(s), 2) if s else None,
        }
    return out


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #


def synthesize_label(
    b: dict[str, Any] | None,
    a: dict[str, Any] | None,
    d: dict[str, Any] | None,
    c: dict[str, Any] | None = None,
) -> dict[str, Any]:
    flags = []
    rationale = []

    # Method D (text-vs-examples internal consistency)
    d_overall = d.get("overall_equivalence") if d else None
    if d_overall is not None and d_overall < 7:
        flags.append("internally_inconsistent")
        rationale.append(f"Method D: text-vs-examples overall_equivalence={d_overall} (<7)")

    # Method B (judge enumeration)
    b_clusters = b.get("n_distinct_clusters") if b else None

    # Method C (compiler divergence — independent of judge enumeration bias)
    c_mean_pairwise = c.get("mean_pairwise_equivalence") if c else None
    c_min_pairwise = c.get("min_pairwise_equivalence") if c else None

    # Combined ambiguity signal: prefer Method C as the load-bearing test;
    # use Method B as supplementary. Method B alone is enumeration-biased.
    b_high = b_clusters is not None and b_clusters >= 3
    b_mild = b_clusters == 2
    c_diverge = c_mean_pairwise is not None and c_mean_pairwise < 7
    c_strong_diverge = c_mean_pairwise is not None and c_mean_pairwise < 5

    if c_strong_diverge:
        flags.append("language_ambiguous_C_strong")
        rationale.append(f"Method C: mean pairwise compiler equivalence={c_mean_pairwise} (<5)")
    elif c_diverge:
        flags.append("language_ambiguous_C")
        rationale.append(f"Method C: mean pairwise compiler equivalence={c_mean_pairwise} (<7)")

    if b_high:
        flags.append("readings_diverge_B")
        rationale.append(f"Method B: {b_clusters} distinct semantic clusters of readings (≥3)")
    elif b_mild:
        flags.append("readings_mildly_diverge_B")
        rationale.append("Method B: 2 distinct semantic clusters of readings")

    # Method A (offline)
    if a and a.get("mean_score_stdev") is not None and a["mean_score_stdev"] >= 1.5 and (
        a.get("mean_informing_jaccard") is not None and a["mean_informing_jaccard"] < 0.3
    ):
        flags.append("language_ambiguous_A")
        rationale.append(
            f"Method A: mean score stdev={a['mean_score_stdev']} (≥1.5) "
            f"AND informing-words jaccard={a['mean_informing_jaccard']} (<0.3)"
        )

    # Top-level label — Method D and Method C are the load-bearing signals.
    # Method B is enumeration-biased and is treated as supplementary signal only:
    # B alone never triggers an ambiguity label; it can only corroborate C or A.
    if "internally_inconsistent" in flags:
        primary = "internally_inconsistent"
    elif "language_ambiguous_C_strong" in flags:
        primary = "language_ambiguous"
    elif "language_ambiguous_C" in flags:
        # C is below threshold → real ambiguity. Add severity from B if present.
        primary = "language_ambiguous"
    elif "language_ambiguous_A" in flags and "readings_diverge_B" in flags:
        # A and B both fire (without C) — moderate evidence for ambiguity, not strong.
        primary = "language_mildly_ambiguous"
    elif "language_ambiguous_A" in flags:
        primary = "language_mildly_ambiguous"
    else:
        # B-only signal: known enumeration bias, not enough to flag.
        primary = "language_robust"

    return {"primary_label": primary, "all_flags": flags, "rationale": rationale}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-b", type=Path, default=DIR / "method_b_readings.jsonl")
    parser.add_argument("--method-a", type=Path, default=DIR / "method_a_grades.jsonl")
    parser.add_argument("--method-c", type=Path, default=DIR / "method_c_rubrics.jsonl")
    parser.add_argument("--method-d", type=Path, default=DIR / "method_d_rubrics.jsonl")
    parser.add_argument("--output", type=Path, default=DIR / "spec_ambiguity_diagnostic.jsonl")
    parser.add_argument("--report", type=Path, default=DIR / "spec_ambiguity_diagnostic.md")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    spec = {json.loads(line)["id"]: json.loads(line) for line in SPEC_PATH.open() if line.strip()}

    b_rows = [json.loads(l) for l in args.method_b.open() if l.strip()] if args.method_b.exists() else []
    a_rows = [json.loads(l) for l in args.method_a.open() if l.strip()] if args.method_a.exists() else []
    c_rows = [json.loads(l) for l in args.method_c.open() if l.strip()] if args.method_c.exists() else []
    d_rows = [json.loads(l) for l in args.method_d.open() if l.strip()] if args.method_d.exists() else []
    logger.info("loaded B=%d A=%d C=%d D=%d", len(b_rows), len(a_rows), len(c_rows), len(d_rows))

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # ------- Method B clustering ------- #
    b_by_stmt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in b_rows:
        for reading in (row.get("readings") or []):
            if isinstance(reading, dict):
                b_by_stmt[row["statement_id"]].append(reading)

    b_results: dict[str, dict[str, Any]] = {}
    b_keys = sorted(b_by_stmt.keys())
    logger.info("Method B clustering: %d statements", len(b_keys))

    def cluster_one(stmt_id: str) -> tuple[str, dict[str, Any] | None]:
        readings = b_by_stmt[stmt_id]
        if len(readings) < 2:
            return stmt_id, {"n_distinct_clusters": 1, "clusters": [], "interpretation_summary": "<2 readings"}
        try:
            parsed = call_gpt(client, CLUSTER_SYSTEM, build_cluster_prompt(spec[stmt_id], readings))
            return stmt_id, parsed
        except Exception as exc:
            logger.warning("B cluster failed for %s: %s", stmt_id, exc)
            return stmt_id, None

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for sid, parsed in [f.result() for f in [ex.submit(cluster_one, k) for k in b_keys]]:
            b_results[sid] = parsed
        # Note: small enough we don't need as_completed

    logger.info("Method B clustering done")

    # ------- Method D semantic equivalence ------- #
    d_by_stmt: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in d_rows:
        d_by_stmt[row["statement_id"]][row["channel"]] = row

    d_results: dict[str, dict[str, Any]] = {}
    d_keys = [s for s in d_by_stmt if "text_only" in d_by_stmt[s] and "examples_only" in d_by_stmt[s]]
    logger.info("Method D equivalence: %d statements with both channels", len(d_keys))

    def equiv_one(stmt_id: str) -> tuple[str, dict[str, Any] | None]:
        try:
            parsed = call_gpt(
                client, EQUIV_SYSTEM,
                build_equiv_prompt(spec[stmt_id], d_by_stmt[stmt_id]["text_only"], d_by_stmt[stmt_id]["examples_only"]),
            )
            return stmt_id, parsed
        except Exception as exc:
            logger.warning("D equiv failed for %s: %s", stmt_id, exc)
            return stmt_id, None

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        for sid, parsed in [f.result() for f in [ex.submit(equiv_one, k) for k in d_keys]]:
            d_results[sid] = parsed

    logger.info("Method D equivalence done")

    # ------- Method C 3-way pairwise semantic equivalence ------- #
    c_by_stmt: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in c_rows:
        c_by_stmt[row["statement_id"]][row["compiler_model"]] = row

    c_results: dict[str, dict[str, Any]] = {}
    c_keys = [s for s in c_by_stmt if len(c_by_stmt[s]) >= 2]
    logger.info("Method C 3-way: %d statements with ≥2 compilers", len(c_keys))

    def c_pair_one(args_tuple: tuple[str, str, str]) -> tuple[str, str, str, dict[str, Any] | None]:
        sid, m_a, m_b = args_tuple
        try:
            parsed = call_gpt(
                client, EQUIV_SYSTEM,
                build_equiv_prompt(spec[sid], c_by_stmt[sid][m_a], c_by_stmt[sid][m_b]),
            )
            return sid, m_a, m_b, parsed
        except Exception as exc:
            logger.warning("C pair failed for %s/%s/%s: %s", sid, m_a, m_b, exc)
            return sid, m_a, m_b, None

    pair_jobs: list[tuple[str, str, str]] = []
    for sid in c_keys:
        compilers = sorted(c_by_stmt[sid].keys())
        for i in range(len(compilers)):
            for j in range(i + 1, len(compilers)):
                pair_jobs.append((sid, compilers[i], compilers[j]))
    logger.info("Method C pairs to compare: %d", len(pair_jobs))

    pair_results: dict[tuple[str, str, str], dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(c_pair_one, j) for j in pair_jobs]
        for fut in as_completed(futures):
            sid, m_a, m_b, parsed = fut.result()
            if parsed is not None:
                pair_results[(sid, m_a, m_b)] = parsed

    # Aggregate per statement
    for sid in c_keys:
        per_pair = []
        for key in pair_results:
            if key[0] == sid:
                pe = pair_results[key].get("overall_equivalence")
                if isinstance(pe, (int, float)):
                    per_pair.append((f"{key[1]}~{key[2]}", pe))
        if per_pair:
            scores = [p[1] for p in per_pair]
            c_results[sid] = {
                "n_compilers": len(c_by_stmt[sid]),
                "n_pairs": len(per_pair),
                "pairs": per_pair,
                "mean_pairwise_equivalence": round(sum(scores) / len(scores), 2),
                "min_pairwise_equivalence": min(scores),
                "max_pairwise_equivalence": max(scores),
            }
    logger.info("Method C analysis: %d statements", len(c_results))

    # ------- Method A offline ------- #
    a_results = analyze_method_a(a_rows)
    logger.info("Method A offline analysis: %d statements", len(a_results))

    # ------- Synthesis ------- #
    synth: dict[str, dict[str, Any]] = {}
    all_stmts = sorted(spec.keys())
    for sid in all_stmts:
        b = b_results.get(sid)
        a = a_results.get(sid)
        c = c_results.get(sid)
        d = d_results.get(sid)
        synth[sid] = synthesize_label(b, a, d, c)

    # Aggregate ambiguous phrases from Method B (union across judges)
    phrase_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in b_rows:
        for p in row.get("ambiguous_phrases") or []:
            if isinstance(p, str) and p.strip():
                phrase_counts[row["statement_id"]][p.strip()] += 1

    # Write JSONL
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for sid in all_stmts:
            row = {
                "statement_id": sid,
                "primary_label": synth[sid]["primary_label"],
                "all_flags": synth[sid]["all_flags"],
                "rationale": synth[sid]["rationale"],
                "method_b": b_results.get(sid),
                "method_a": a_results.get(sid),
                "method_c": c_results.get(sid),
                "method_d": d_results.get(sid),
                "ambiguous_phrases_top": sorted(phrase_counts.get(sid, {}).items(), key=lambda kv: -kv[1])[:5],
            }
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("wrote %s", args.output)

    # Write markdown report
    label_count: dict[str, int] = defaultdict(int)
    for sid in all_stmts:
        label_count[synth[sid]["primary_label"]] += 1

    lines: list[str] = []
    lines.append("# Spec ambiguity diagnostic — Tier 1 + 2 (Methods A, B, D)")
    lines.append("")
    lines.append(f"Generated 2026-05-01 from {len(b_rows)} Method-B rows, {len(a_rows)} Method-A rows, {len(d_rows)} Method-D rows.")
    lines.append("")
    lines.append("## Headline label distribution")
    lines.append("")
    lines.append("| label | count |")
    lines.append("|---|--:|")
    for lbl in ["language_robust", "language_mildly_ambiguous", "language_ambiguous", "internally_inconsistent"]:
        lines.append(f"| `{lbl}` | {label_count.get(lbl, 0)} |")
    lines.append("")
    lines.append("## Per-statement diagnostics")
    lines.append("")
    lines.append("| statement | label | B clusters | A score-stdev | A informing-jaccard | C mean equiv | C min equiv | D overall equiv |")
    lines.append("|---|---|--:|--:|--:|--:|--:|--:|")
    for sid in all_stmts:
        s = synth[sid]
        b = b_results.get(sid) or {}
        a = a_results.get(sid) or {}
        c = c_results.get(sid) or {}
        d = d_results.get(sid) or {}
        b_n = b.get("n_distinct_clusters", "—")
        a_v = a.get("mean_score_stdev", "—")
        a_j = a.get("mean_informing_jaccard", "—")
        c_m = c.get("mean_pairwise_equivalence", "—")
        c_min = c.get("min_pairwise_equivalence", "—")
        d_o = d.get("overall_equivalence", "—")
        lines.append(f"| `{sid}` | `{s['primary_label']}` | {b_n} | {a_v} | {a_j} | {c_m} | {c_min} | {d_o} |")

    lines.append("")
    lines.append("## Statements flagged `internally_inconsistent`")
    lines.append("")
    inc = [sid for sid in all_stmts if synth[sid]["primary_label"] == "internally_inconsistent"]
    if not inc:
        lines.append("(none)")
    else:
        for sid in inc:
            d = d_results.get(sid) or {}
            lines.append(f"- **`{sid}`** — overall equiv {d.get('overall_equivalence', '—')}/10. {d.get('disagreement_summary', '')}")
    lines.append("")

    lines.append("## Statements flagged `language_ambiguous`")
    lines.append("")
    amb = [sid for sid in all_stmts if synth[sid]["primary_label"] == "language_ambiguous"]
    if not amb:
        lines.append("(none)")
    else:
        for sid in amb:
            b = b_results.get(sid) or {}
            phrases = phrase_counts.get(sid, {})
            top_phrases = sorted(phrases.items(), key=lambda kv: -kv[1])[:5]
            ph_str = ", ".join(f"`{p}`×{n}" for p, n in top_phrases) if top_phrases else "—"
            lines.append(f"- **`{sid}`** — {b.get('n_distinct_clusters', '—')} clusters. Top ambiguous phrases: {ph_str}")
            if b.get("interpretation_summary"):
                lines.append(f"  - {b['interpretation_summary']}")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- **Method B** clusters were computed by GPT-5.1 (reasoning_effort=none) over the union of 3-judge × 2-3 readings per statement.")
    lines.append("- **Method A** statistics are offline (no API): per-scenario score stdev across 3 judges, Jaccard overlap on cited verbatim informing words.")
    lines.append("- **Method D** equivalence was computed by GPT-5.1 between text-only and examples-only rubrics generated separately.")
    lines.append(f"- {label_count.get('language_robust', 0)} of {len(all_stmts)} statements passed all checks.")
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("wrote %s", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
