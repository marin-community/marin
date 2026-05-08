"""Build subagent input markdown files for the bare-vs-rubric qualitative analysis.

Produces:
  claude_subagents/bare_vs_rubric_2026_05_08/inputs/
    judge_gpt_input.md
    judge_gemini_input.md
    judge_claude_input.md
    rubric_paradox_input.md

Layer 1 (per-judge): each judge file contains that judge's score distribution
across all 46 statements × both conditions, plus 4 representative cells per
(statement, condition):
  - 1 high score (4-5)
  - 1 low score (1-2)
  - 1 mid score (3) (when available)
  - 1 outlier cell where this judge differed from the other 2 by ≥2

Layer 2 (rubric paradox): ~9 statements (6 strongly-hurt + 3 strongly-helped),
with 5-8 diagnostic cells per statement where bare-condition judges agreed
within ±1 but rubric-condition judges spread ≥2.

Each cell has: prompt, response, all judges' scores + reasoning.

No API spend. Reads existing per_judgment + Claude jsonl files.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

DIR = Path("experiments/posttrain/disagreement_primitive")
SPEC_PATH = Path("experiments/posttrain/specs/openai_model_spec.jsonl")
RUBRICS_PATH = DIR / "e8_rubrics.jsonl"
OUT_DIR = Path("claude_subagents/bare_vs_rubric_2026_05_08/inputs")

PER_JUDGMENT_OLD = DIR / "grounding" / "per_judgment.jsonl"
PER_JUDGMENT_NEW = DIR / "per_judgment_opposite.jsonl"
PER_JUDGMENT_REASONING = DIR / "per_judgment_reasoning.jsonl"  # extracted from raw dumps
CLAUDE_OLD_DIR = DIR / "claude_judge_v0"
CLAUDE_NEW_DIR = DIR / "claude_judge_v0_opposite"
EXISTING_RESPONSES = DIR / "e8_responses.jsonl"
OPPOSITE_RESPONSES = DIR / "e9_opposite_mode_responses.jsonl"

GEN_KEYS = [
    ("gpt-5.1", "response_gpt"),
    ("Qwen/Qwen2.5-7B-Instruct-Turbo", "response_weak"),
    ("gemini-3-flash-preview", "response_gemini"),
]
GROK_LABEL = "grok-4-1-fast-non-reasoning-opposite"

random.seed(42)


def load_judgments() -> dict[tuple, dict[str, dict]]:
    """by_cell[(sid, cond_internal, scen, gen)] = {judge: {score, reasoning, ...}}"""
    out: dict[tuple, dict[str, dict]] = defaultdict(dict)

    # Reasoning-enriched extract (gpt + gemini for the existing 3 generators)
    if PER_JUDGMENT_REASONING.exists():
        for line in PER_JUDGMENT_REASONING.open():
            r = json.loads(line)
            if r.get("condition") not in {"variant_A", "rubric_plus_spec"}: continue
            if r.get("judge") not in {"gpt", "gemini"}: continue
            try:
                s = int(r["score"]) if r.get("score") is not None else None
            except (TypeError, ValueError):
                s = None
            if s is None or not 1 <= s <= 5: continue
            cell = (r["statement_id"], r["condition"], r["scenario_idx"], r["generator"])
            out[cell][r["judge"]] = {"score": s, "reasoning": r.get("reasoning") or ""}

    # New per_judgment (gpt + gemini + claude on Grok-opposite for some sids)
    if PER_JUDGMENT_NEW.exists():
        for line in PER_JUDGMENT_NEW.open():
            r = json.loads(line)
            if r.get("condition") not in {"variant_A", "rubric_plus_spec"}: continue
            s = r.get("score")
            if not isinstance(s, int) or not 1 <= s <= 5: continue
            cell = (r["statement_id"], r["condition"], r["scenario_idx"], r["generator"])
            out[cell][r["judge"]] = {"score": s, "reasoning": r.get("reasoning", "")}

    # Claude on existing 3 gens
    for sid_dir in CLAUDE_OLD_DIR.iterdir() if CLAUDE_OLD_DIR.exists() else []:
        if not sid_dir.is_dir(): continue
        sid = sid_dir.name
        for cond_short, cond_internal in [("bare", "variant_A"), ("phase_4", "rubric_plus_spec")]:
            p = sid_dir / f"{cond_short}_claude.jsonl"
            if not p.exists(): continue
            for line in p.open():
                r = json.loads(line)
                s = r.get("score")
                if not isinstance(s, int) or not 1 <= s <= 5: continue
                cell = (sid, cond_internal, r["scenario_idx"], r["generator"])
                out[cell]["claude"] = {"score": s, "reasoning": r.get("reasoning", "")}

    # Claude on Grok-opposite
    for sid_dir in CLAUDE_NEW_DIR.iterdir() if CLAUDE_NEW_DIR.exists() else []:
        if not sid_dir.is_dir(): continue
        sid = sid_dir.name
        for cond_short, cond_internal in [("bare", "variant_A"), ("phase_4", "rubric_plus_spec")]:
            p = sid_dir / f"{cond_short}_opposite_claude.jsonl"
            if not p.exists(): continue
            for line in p.open():
                r = json.loads(line)
                s = r.get("score")
                if not isinstance(s, int) or not 1 <= s <= 5: continue
                cell = (sid, cond_internal, r["scenario_idx"], r["generator"])
                out[cell]["claude"] = {"score": s, "reasoning": r.get("reasoning", "")}

    return out


def load_response_index() -> dict[tuple[str, int, str], dict]:
    """(sid, scen, gen) -> {user_query, response}"""
    out: dict[tuple, dict] = {}
    for r in (json.loads(l) for l in EXISTING_RESPONSES.open()):
        sid, scen = r.get("statement_id"), r.get("scenario_idx")
        for label, col in GEN_KEYS:
            text = r.get(col)
            if text:
                out[(sid, scen, label)] = {"user_query": r["user_query"], "response": text}
    if OPPOSITE_RESPONSES.exists():
        for r in (json.loads(l) for l in OPPOSITE_RESPONSES.open()):
            if "error" in r: continue
            out[(r["statement_id"], r["scenario_idx"], r["generator"])] = {
                "user_query": r["user_query"], "response": r["response"],
            }
    return out


def trunc(s: str, n: int = 800) -> str:
    s = (s or "").replace("\r", "")
    return s if len(s) <= n else s[:n] + "…"


# -------------------- Layer 1: per-judge cell selection --------------------


def select_cells_for_judge(by_cell: dict, judge: str, sid: str, cond_internal: str) -> list:
    """Pick 4 representative cells for this (judge, sid, cond)."""
    cells = []
    for (s, c, scen, gen), jd in by_cell.items():
        if s != sid or c != cond_internal: continue
        if judge not in jd: continue
        if not all(j in jd for j in ("gpt", "gemini", "claude")): continue  # need full-3 for outlier check
        my_score = jd[judge]["score"]
        other_scores = [jd[j]["score"] for j in ("gpt", "gemini", "claude") if j != judge]
        diff = max(abs(my_score - o) for o in other_scores)
        cells.append({"sid": sid, "cond": c, "scen": scen, "gen": gen, "jd": jd,
                      "my_score": my_score, "max_diff": diff})

    if not cells:
        return []

    high = [c for c in cells if c["my_score"] in (4, 5)]
    low = [c for c in cells if c["my_score"] in (1, 2)]
    mid = [c for c in cells if c["my_score"] == 3]
    outlier = sorted(cells, key=lambda c: -c["max_diff"])[:5]

    selected = []
    seen_ids = set()
    def add(pool, n):
        random.shuffle(pool)
        for c in pool:
            key = (c["scen"], c["gen"])
            if key not in seen_ids:
                selected.append(c)
                seen_ids.add(key)
                if len([x for x in selected if x["_pool"] == c.get("_pool")]) >= n:  # noop
                    pass
            if len(selected) >= n + len(seen_ids) - 1:
                pass

    # Simpler: pick at most 1 from each pool, with outlier taking priority
    chosen: list = []
    chosen_keys: set = set()
    for pool in [outlier, high, low, mid]:
        for c in pool:
            key = (c["scen"], c["gen"])
            if key in chosen_keys: continue
            chosen.append(c)
            chosen_keys.add(key)
            break  # take one from each pool
    # Top up to 4 with random others
    others = [c for c in cells if (c["scen"], c["gen"]) not in chosen_keys]
    random.shuffle(others)
    for c in others:
        if len(chosen) >= 4: break
        chosen.append(c)
    return chosen[:4]


def render_cell(cell: dict, response_idx: dict, focal_judge: str | None = None) -> str:
    sid, cond, scen, gen = cell["sid"], cell["cond"], cell["scen"], cell["gen"]
    jd = cell["jd"]
    cond_label = "bare" if cond == "variant_A" else "rubric"
    resp_meta = response_idx.get((sid, scen, gen), {"user_query": "[missing]", "response": "[missing]"})
    parts = []
    parts.append(f"#### Cell: scen={scen}, generator={gen}, condition={cond_label}")
    parts.append("")
    parts.append(f"**User query:** {trunc(resp_meta['user_query'], 400)}")
    parts.append(f"**Assistant response:** {trunc(resp_meta['response'], 600)}")
    parts.append("")
    parts.append("**Scores + reasoning:**")
    for j in ("gpt", "gemini", "claude"):
        if j not in jd: continue
        marker = " ←" if j == focal_judge else ""
        score = jd[j]["score"]
        reasoning = trunc(jd[j].get("reasoning", ""), 600)
        parts.append(f"- **{j}**: {score}{marker}  — _{reasoning}_")
    parts.append("")
    return "\n".join(parts)


def build_layer1_per_judge(judge: str, by_cell: dict, response_idx: dict,
                            statements: list[str], delta_alpha: dict[str, float],
                            score_dist: dict, out_path: Path) -> None:
    parts = []
    parts.append(f"# Layer 1 input — Judge: {judge}\n")
    parts.append(f"You are analyzing this judge's behavior across all 46 spec statements under both")
    parts.append(f"conditions: `bare` (statement only) and `rubric` (statement + rubric).\n")
    parts.append(f"## Your task\n")
    parts.append("""\
Write a markdown analysis of how this judge scores responses, comparing bare vs rubric.

For each statement bucket (strongly_helps / helps / neutral / hurts / strongly_hurts), describe:
1. What does this judge anchor to under bare? (spec text fragments? generic compliance heuristic? specific phrases?)
2. What does this judge anchor to under rubric? (specific anchor language? falls back to bare-style intuition?)
3. Where is this judge an outlier vs the consensus of the other two?
4. What kinds of cases does this judge tend to score harshly vs leniently?

Then write a synthesis section:
- Per-statement-type behavior: how does this judge handle style/tone statements vs content-boundary vs procedural?
- The 3-5 most distinctive patterns about how THIS judge specifically interprets the spec.
- Whether the rubric improves or degrades this judge's reliability per statement type.

Cite worked examples liberally — quote the judge's reasoning, then explain.

Target length: ~3000-6000 words. Concrete > general.
""")
    parts.append(f"## Score distribution summary\n")
    parts.append(f"| condition | 1 | 2 | 3 | 4 | 5 | mean | n |")
    parts.append(f"|---|--:|--:|--:|--:|--:|--:|--:|")
    for cond_internal, label in [("variant_A", "bare"), ("rubric_plus_spec", "rubric")]:
        d = score_dist.get((judge, cond_internal), {})
        n = sum(d.values())
        if n == 0: continue
        mean = sum(s * c for s, c in d.items()) / n
        parts.append(f"| {label} | {d.get(1,0)} | {d.get(2,0)} | {d.get(3,0)} | {d.get(4,0)} | {d.get(5,0)} | {mean:.2f} | {n} |")
    parts.append("")

    # Group statements by Δα bucket
    buckets = {
        "strongly_helps (Δα > +0.10)": [s for s in statements if delta_alpha.get(s, 0) > 0.10],
        "helps (+0.05 < Δα ≤ +0.10)": [s for s in statements if 0.05 < delta_alpha.get(s, 0) <= 0.10],
        "neutral (-0.05 ≤ Δα ≤ +0.05)": [s for s in statements if -0.05 <= delta_alpha.get(s, 0) <= 0.05],
        "hurts (-0.10 ≤ Δα < -0.05)": [s for s in statements if -0.10 <= delta_alpha.get(s, 0) < -0.05],
        "strongly_hurts (Δα < -0.10)": [s for s in statements if delta_alpha.get(s, 0) < -0.10],
    }

    for bucket_name, bucket_sids in buckets.items():
        if not bucket_sids: continue
        parts.append(f"\n## Statement bucket: {bucket_name} ({len(bucket_sids)} statements)\n")
        parts.append("Cells from each statement, both conditions:\n")
        for sid in sorted(bucket_sids, key=lambda s: -delta_alpha.get(s, 0)):
            parts.append(f"\n### Statement: `{sid}`  (Δα = {delta_alpha.get(sid, 0):+.3f})\n")
            for cond_internal, cond_label in [("variant_A", "BARE"), ("rubric_plus_spec", "RUBRIC")]:
                parts.append(f"\n**Condition: {cond_label}**\n")
                cells = select_cells_for_judge(by_cell, judge, sid, cond_internal)
                for c in cells:
                    parts.append(render_cell(c, response_idx, focal_judge=judge))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts))
    print(f"  wrote {out_path} ({len(parts)} lines)")


# -------------------- Layer 2: cross-judge synthesis on the paradox --------------------


def select_paradox_cells(by_cell: dict, sid: str, n_target: int = 7) -> list:
    """Find cells where bare judges agreed (range ≤1) but rubric judges spread (range ≥2)."""
    bare_by_pair: dict[tuple[int, str], dict] = {}
    rubric_by_pair: dict[tuple[int, str], dict] = {}
    for (s, c, scen, gen), jd in by_cell.items():
        if s != sid: continue
        if not all(j in jd for j in ("gpt", "gemini", "claude")): continue
        if c == "variant_A":
            bare_by_pair[(scen, gen)] = jd
        elif c == "rubric_plus_spec":
            rubric_by_pair[(scen, gen)] = jd

    diagnostic = []
    for key in set(bare_by_pair) & set(rubric_by_pair):
        bare_jd = bare_by_pair[key]
        rub_jd = rubric_by_pair[key]
        bare_scores = [bare_jd[j]["score"] for j in ("gpt", "gemini", "claude")]
        rub_scores = [rub_jd[j]["score"] for j in ("gpt", "gemini", "claude")]
        if max(bare_scores) - min(bare_scores) <= 1 and max(rub_scores) - min(rub_scores) >= 2:
            diagnostic.append({
                "scen": key[0], "gen": key[1],
                "bare_jd": bare_jd, "rub_jd": rub_jd,
                "bare_range": max(bare_scores) - min(bare_scores),
                "rub_range": max(rub_scores) - min(rub_scores),
            })

    # Sort by largest rubric_range first
    diagnostic.sort(key=lambda d: -d["rub_range"])
    return diagnostic[:n_target]


def render_paradox_cell(cell: dict, sid: str, response_idx: dict) -> str:
    scen, gen = cell["scen"], cell["gen"]
    bare_jd = cell["bare_jd"]
    rub_jd = cell["rub_jd"]
    resp_meta = response_idx.get((sid, scen, gen), {"user_query": "[missing]", "response": "[missing]"})
    parts = []
    parts.append(f"#### Diagnostic cell: scen={scen}, generator={gen}")
    parts.append(f"  bare scores: gpt={bare_jd['gpt']['score']}, gemini={bare_jd['gemini']['score']}, claude={bare_jd['claude']['score']}  (range={cell['bare_range']})")
    parts.append(f"  rubric scores: gpt={rub_jd['gpt']['score']}, gemini={rub_jd['gemini']['score']}, claude={rub_jd['claude']['score']}  (range={cell['rub_range']})")
    parts.append("")
    parts.append(f"**User query:** {trunc(resp_meta['user_query'], 400)}")
    parts.append(f"**Assistant response:** {trunc(resp_meta['response'], 700)}")
    parts.append("")
    parts.append("**Bare reasonings (judges agreed):**")
    for j in ("gpt", "gemini", "claude"):
        parts.append(f"- _{j}_ (score={bare_jd[j]['score']}): {trunc(bare_jd[j].get('reasoning',''), 500)}")
    parts.append("")
    parts.append("**Rubric reasonings (judges diverged):**")
    for j in ("gpt", "gemini", "claude"):
        parts.append(f"- _{j}_ (score={rub_jd[j]['score']}): {trunc(rub_jd[j].get('reasoning',''), 500)}")
    parts.append("")
    return "\n".join(parts)


def build_layer2(by_cell: dict, response_idx: dict, spec_by_id: dict, rubrics_by_id: dict,
                 paradox_sids: list[tuple[str, str]], out_path: Path) -> None:
    parts = []
    parts.append("# Layer 2 input — Cross-judge synthesis on the rubric paradox\n")
    parts.append("""\
## Your task

Across the 9 statements below (6 strongly-hurt + 3 strongly-helped), explain WHY
adding the rubric changes cross-judge agreement.

For each statement, you have:
- The spec statement text
- The rubric anchors (compiled by GPT-5.1)
- 5-7 diagnostic cells where bare-condition judges agreed (range ≤1) but
  rubric-condition judges diverged (range ≥2)
- All 3 judges' reasoning under both conditions for each diagnostic cell

For each statement, answer:
1. What does the rubric add semantically beyond the spec? (Look at rubric anchor
   language and compare to the spec text.)
2. Where do judges anchor differently under rubric vs bare? Quote specific
   anchor phrases that judges seem to interpret differently.
3. Is there evidence the rubric DRIFTS from the spec semantics during compilation?
   (i.e., does the rubric introduce concepts/criteria not in the spec text?)
4. Does one judge dominate the rubric framing while another sticks with spec
   instincts? Or do all 3 just diverge?

Then synthesize across all 9 statements:
- 3-5 mechanistic CATEGORIES of "why rubric hurts" (or helps).
- For each category, name 2-3 statements that exemplify it and quote a worked example.
- Address: when WOULD adding a rubric be expected to help vs hurt? Form a
  qualitative theory.
- Distinguish: rubric-quality issues (compilation drift) vs rubric-interpretation
  issues (judges read same anchor differently).

Target length: ~5000-8000 words. Be specific. Quote reasonings. Distinguish
hypothesis from evidence.

## Δα table for the 9 selected statements
""")
    parts.append("| status | statement | Δα |")
    parts.append("|---|---|--:|")
    for sid, status in paradox_sids:
        # delta computed elsewhere — but we'll just include the bucket label
        parts.append(f"| {status} | {sid} | (see body) |")
    parts.append("")

    for sid, status in paradox_sids:
        stmt = spec_by_id.get(sid, {})
        rubric = rubrics_by_id.get(sid, {})
        parts.append(f"\n---\n\n## Statement: `{sid}`  ({status})\n")
        parts.append(f"### Spec text\n")
        parts.append("> " + (stmt.get("text", "[missing]")).replace("\n", "\n> ") + "\n")
        parts.append(f"### Rubric anchors (compiled by GPT-5.1)\n")
        anchors = (rubric or {}).get("anchors", {}) if isinstance(rubric, dict) else {}
        for k in ("1", "2", "3", "4", "5"):
            anc = anchors.get(k, {}) or {}
            crit = trunc(anc.get("criterion", ""), 600)
            parts.append(f"**Anchor {k}:** {crit}")
            quotes = anc.get("spec_quotes", []) or []
            if quotes:
                parts.append(f"  - spec_quotes: {', '.join(repr(q[:120]) for q in quotes[:3])}")
            parts.append("")

        cells = select_paradox_cells(by_cell, sid, n_target=7)
        parts.append(f"### Diagnostic cells (n={len(cells)})\n")
        for c in cells:
            parts.append(render_paradox_cell(c, sid, response_idx))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts))
    print(f"  wrote {out_path}")


# -------------------- Main --------------------


def main() -> int:
    print("Loading judgments...")
    by_cell = load_judgments()
    print(f"  loaded {len(by_cell)} cells")

    print("Loading responses...")
    response_idx = load_response_index()
    print(f"  loaded {len(response_idx)} response entries")

    print("Loading spec + rubrics...")
    spec_by_id = {json.loads(line)["id"]: json.loads(line) for line in SPEC_PATH.open() if line.strip()}
    rubrics_by_id = {r["statement_id"]: r["rubric"] for r in (json.loads(l) for l in RUBRICS_PATH.open()) if "error" not in r}

    statements = sorted(spec_by_id.keys())

    # Compute Δα per statement (same logic as e9_kappa_with_opposite)
    print("Computing Δα per statement...")
    sys.path.insert(0, str(Path(__file__).parent))
    from e9_kappa_alternatives import krippendorff_alpha_interval
    delta_alpha: dict[str, float] = {}
    ALL = {"gpt-5.1", "Qwen/Qwen2.5-7B-Instruct-Turbo", "gemini-3-flash-preview", GROK_LABEL}
    for sid in statements:
        tb, tp = [], []
        for (s, c, scen, gen), jd in by_cell.items():
            if s != sid or gen not in ALL or len(jd) < 3: continue
            tup = (jd["gpt"]["score"], jd["gemini"]["score"], jd["claude"]["score"])
            if c == "variant_A": tb.append(tup)
            elif c == "rubric_plus_spec": tp.append(tup)
        if len(tb) >= 2 and len(tp) >= 2:
            ab = krippendorff_alpha_interval(tb)
            ap = krippendorff_alpha_interval(tp)
            if ab is not None and ap is not None:
                delta_alpha[sid] = ap - ab

    # Per-judge score distributions
    print("Computing score distributions per judge...")
    score_dist: dict[tuple[str, str], dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for cell, jd in by_cell.items():
        cond = cell[1]
        for j, info in jd.items():
            score_dist[(j, cond)][info["score"]] += 1

    # Build Layer 1 per-judge files
    print("\nBuilding Layer 1 per-judge inputs...")
    for judge in ("gpt", "gemini", "claude"):
        out = OUT_DIR / f"judge_{judge}_input.md"
        build_layer1_per_judge(judge, by_cell, response_idx, statements, delta_alpha, score_dist, out)

    # Build Layer 2 paradox file
    print("\nBuilding Layer 2 paradox input...")
    # Select 6 strongly-hurt + 3 strongly-helped
    sorted_by_d = sorted(delta_alpha.items(), key=lambda kv: kv[1])
    hurt = [(s, "strongly_hurts") for s, d in sorted_by_d[:6]]
    helped = [(s, "strongly_helps") for s, d in list(reversed(sorted_by_d))[:3]]
    paradox_sids = hurt + helped
    print(f"  hurt: {[s for s, _ in hurt]}")
    print(f"  helped: {[s for s, _ in helped]}")
    build_layer2(by_cell, response_idx, spec_by_id, rubrics_by_id, paradox_sids,
                 OUT_DIR / "rubric_paradox_input.md")

    # Print sizes
    print("\nOutput files:")
    for p in sorted(OUT_DIR.glob("*.md")):
        print(f"  {p.name}: {p.stat().st_size:>9,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
