"""Build a subagent input file for investigating why v2 rubrics didn't fix
3 of 5 strongly-hurt statements.

For each of {no_agenda, comply_with_laws, sexual_content_involving_minors}:
  - spec text
  - v1 rubric anchors (GPT-5.1's original compile)
  - v2 rubric anchors (GPT-5.1's recompile after seeing disagreement evidence)
  - top-K cells where rubric STILL has high pairwise variance under v2
  - For each cell: prompt + response + all 3 judges' rationales under
      (a) bare condition
      (b) v1 rubric (existing GPT + Gemini + Claude judgments)
      (c) v2 rubric (NEW GPT v2 + same Gemini + same Claude)

Plus avoid_abuse as a CONTROL showing what v2 success looks like.

Output:
  claude_subagents/v2_failure_analysis_2026_05_08/inputs/v2_failure_input.md
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from e9_build_qualitative_inputs import load_judgments, load_response_index
from e9_rubric_poison_rank import pairwise_variance, outlier_judge

DIR = Path("experiments/posttrain/disagreement_primitive")
SPEC_PATH = Path("experiments/posttrain/specs/openai_model_spec.jsonl")
RUBRICS_V1_PATH = DIR / "e8_rubrics.jsonl"
RUBRICS_V2_PATH = DIR / "e8_rubrics_v2.jsonl"
PER_JUDGMENT_V2 = DIR / "per_judgment_v2.jsonl"
OUT_DIR = Path("claude_subagents/v2_failure_analysis_2026_05_08/inputs")

UNFIXED = ["no_agenda", "comply_with_laws", "sexual_content_involving_minors"]
CONTROL = ["avoid_abuse"]
ALL = UNFIXED + CONTROL

JUDGES = ("gpt", "gemini", "claude")


def load_v2_gpt_phase4():
    out = {}
    if not PER_JUDGMENT_V2.exists():
        return out
    for line in PER_JUDGMENT_V2.open():
        r = json.loads(line)
        s = r.get("score")
        if not isinstance(s, int) or not 1 <= s <= 5: continue
        if r.get("judge") != "gpt" or r.get("condition") != "rubric_plus_spec": continue
        out[(r["statement_id"], r["scenario_idx"], r["generator"])] = {
            "score": s, "reasoning": r.get("reasoning") or "",
            "spec_quotes": r.get("spec_quotes") or [],
            "rubric_quotes": r.get("rubric_quotes") or [],
        }
    return out


def trunc(s, n=500):
    s = (s or "").replace("\r", "")
    return s if len(s) <= n else s[:n] + "…"


def render_anchor(k, anc):
    crit = anc.get("criterion", "") or ""
    quotes = anc.get("spec_quotes", []) or []
    parts = [f"**Anchor {k}:** {crit}"]
    if quotes:
        parts.append(f"  spec_quotes: {[q[:80] for q in quotes[:3]]}")
    return "\n".join(parts)


def main():
    spec = {json.loads(l)["id"]: json.loads(l) for l in SPEC_PATH.open() if l.strip()}
    rubrics_v1 = {r["statement_id"]: r["rubric"] for r in [json.loads(l) for l in RUBRICS_V1_PATH.open()] if "error" not in r}
    rubrics_v2 = {r["statement_id"]: r["rubric"] for r in [json.loads(l) for l in RUBRICS_V2_PATH.open()] if "error" not in r}
    by_cell = load_judgments()
    response_idx = load_response_index()
    gpt_v2 = load_v2_gpt_phase4()

    parts = []
    parts.append("# Investigation: why v2 rubrics did NOT fix 3 of 5 strongly-hurt statements\n")
    parts.append("""\
## Background

We ran a methodology where (1) we identified rubric-poison cells via Δpwv
ranking, (2) showed those cells with rationales to GPT-5.1 the original rubric
compiler, (3) asked it to diagnose the systemic pattern and revise the rubric,
(4) re-judged ONLY GPT-5.1 under phase_4 with the revised (v2) rubric, keeping
existing Gemini + Claude judgments unchanged.

Result on 5 strongly-hurt statements:

  - **avoid_abuse**: α went from −0.138 → +0.647 (+0.785). DRAMATIC fix.
  - refusal_style: +0.117 (modest improvement)
  - **no_agenda**: −0.005 (UNCHANGED)
  - **comply_with_laws**: −0.021 (slightly WORSE)
  - **sexual_content_involving_minors**: −0.016 (UNCHANGED)

The user is concerned about negative agreement (Krippendorff α < 0 means
"worse than chance" — judges are systematically disagreeing more than random
would predict).

## Your task

You are a Claude Opus subagent looking at the actual judge rationales to
understand WHY v2 didn't help on the 3 unfixed statements.

For each of the 3 unfixed statements, read:
  - The spec text
  - The v1 rubric anchors
  - The v2 rubric anchors (note GPT-5.1's diagnosis + revisions)
  - Top-K cells where v2 STILL has high cross-judge disagreement
    (each cell shows: prompt, response, and rationales under bare / v1 / v2 conditions)

Plus avoid_abuse as a CONTROL showing what v2 success looks like.

Answer:

1. **Per statement (the 3 unfixed)**: what specifically is happening in
   the v2 rationales that's still causing disagreement? Quote specific
   reasoning to back claims. Did GPT v2 actually change its reading? Did
   Gemini or Claude shift in response to v2 (they shouldn't — same rubric,
   we assume their reasoning stays similar — but worth checking)?

2. **Identify failure modes**: classify the v2 failure for each unfixed
   statement. Possible failure modes (use yours if better):
   - **Multi-criterion conflict**: rubric is structurally fine but spec has
     competing criteria (e.g., respect autonomy vs show empathy) and the
     rubric forces judges to pick a winner.
   - **Judge prior locked**: even with explicit v2 carve-outs, GPT-as-judge
     (or another) keeps applying its prior and ignores rubric.
   - **Unfixable spec ambiguity**: the spec text itself is ambiguous in a
     way no rubric structure can resolve.
   - **v2 introduced new asymmetry**: the rewrite shifted disagreement to
     a new axis instead of resolving it.
   - **v2 too narrow / too broad**: the rewrite changed the boundary in a
     way that some judges follow but others reject.

3. **Predict v3 prospects**: for each unfixed statement, would a third
   round of rubric revision help, or is this hopeless? If hopeless,
   should the rubric for this statement be DROPPED (judge bare only)?

4. **Compare to avoid_abuse success**: what's structurally different
   between avoid_abuse (where v2 worked) and the unfixed statements?
   Be specific about the rubric language, the type of spec, the
   distribution of disagreement.

Cite worked examples liberally. Quote specific judge reasoning.

Target length: 4000-6000 words. Concrete > abstract.

Output: write your analysis as a markdown file to:
`/lfs/skampere3/0/ahmedah/code/marin/.claude/worktrees/align/claude_subagents/v2_failure_analysis_2026_05_08/v2_failure_synthesis.md`

When done, return a brief 6-line summary of the key findings.

---

# DATA

""")

    for sid in ALL:
        is_control = sid in CONTROL
        marker = "🟢 CONTROL (v2 SUCCESS)" if is_control else "🔴 UNFIXED"
        parts.append(f"\n{'='*80}")
        parts.append(f"## {sid}  {marker}\n")
        parts.append(f"### Spec text\n")
        parts.append("> " + spec[sid]["text"].replace("\n", "\n> ") + "\n")

        parts.append(f"### v1 rubric anchors (original GPT-5.1 compile)\n")
        v1_anchors = rubrics_v1[sid]["anchors"]
        for k in ("1", "2", "3", "4", "5"):
            parts.append(render_anchor(k, v1_anchors.get(k, {})))
        parts.append("")

        parts.append(f"### v2 rubric anchors (GPT-5.1's recompile after seeing disagreement evidence)\n")
        v2_anchors = rubrics_v2[sid]["anchors"]
        for k in ("1", "2", "3", "4", "5"):
            parts.append(render_anchor(k, v2_anchors.get(k, {})))
        parts.append("")

        # Compute top-K cells where v2 STILL has high pairwise variance
        # We use the v2 rubric scores for GPT, existing for Gemini/Claude
        cell_rows = []
        bare_cells = {(c[2], c[3]): jd for c, jd in by_cell.items()
                      if c[0] == sid and c[1] == "variant_A" and len(jd) == 3}
        rub_cells_v1 = {(c[2], c[3]): jd for c, jd in by_cell.items()
                        if c[0] == sid and c[1] == "rubric_plus_spec" and len(jd) == 3}
        for key in set(bare_cells) & set(rub_cells_v1):
            scen, gen = key
            v2_gpt = gpt_v2.get((sid, scen, gen))
            if v2_gpt is None: continue  # GPT v2 missing for this cell
            bare_jd = bare_cells[key]
            rub_jd_v1 = rub_cells_v1[key]
            # v2 scores: GPT v2 + Gemini v1 + Claude v1
            v2_scores = [v2_gpt["score"], rub_jd_v1["gemini"]["score"], rub_jd_v1["claude"]["score"]]
            v1_scores = [rub_jd_v1[j]["score"] for j in JUDGES]
            bare_scores = [bare_jd[j]["score"] for j in JUDGES]
            cell_rows.append({
                "scen": scen, "gen": gen,
                "bare_scores": bare_scores,
                "v1_scores": v1_scores,
                "v2_scores": v2_scores,
                "bare_pwv": pairwise_variance(bare_scores),
                "v1_pwv": pairwise_variance(v1_scores),
                "v2_pwv": pairwise_variance(v2_scores),
                "bare_jd": bare_jd,
                "v1_jd": rub_jd_v1,
                "v2_gpt": v2_gpt,
            })
        # Rank by v2_pwv (where v2 still has disagreement)
        cell_rows.sort(key=lambda r: -r["v2_pwv"])
        K = 6 if is_control else 8  # show fewer for control
        top = cell_rows[:K]

        parts.append(f"### Top-{K} cells where v2 STILL has high disagreement (sorted by v2_pwv)\n")
        for i, r in enumerate(top):
            rm = response_idx.get((sid, r["scen"], r["gen"]), {})
            parts.append(f"\n#### Cell {i+1}: scen={r['scen']}, generator={r['gen']}")
            parts.append(f"  bare scores (gpt,gemini,claude): {tuple(r['bare_scores'])}  pwv={r['bare_pwv']}")
            parts.append(f"  v1 scores (gpt_v1,gemini,claude): {tuple(r['v1_scores'])}  pwv={r['v1_pwv']}")
            parts.append(f"  v2 scores (gpt_v2,gemini,claude): {tuple(r['v2_scores'])}  pwv={r['v2_pwv']}")
            parts.append(f"\n**User query:** {trunc(rm.get('user_query'), 350)}")
            parts.append(f"**Assistant response:** {trunc(rm.get('response'), 450)}\n")
            parts.append("**Bare reasonings:**")
            for j in JUDGES:
                jr = r["bare_jd"][j]
                parts.append(f"- _{j}_ (score={jr['score']}): {trunc(jr.get('reasoning'), 400)}")
            parts.append("\n**v1 rubric reasonings:**")
            for j in JUDGES:
                jr = r["v1_jd"][j]
                parts.append(f"- _{j}_ (score={jr['score']}): {trunc(jr.get('reasoning'), 400)}")
            parts.append("\n**v2 rubric reasonings:**")
            v2_gpt = r["v2_gpt"]
            parts.append(f"- _gpt_ (score={v2_gpt['score']}, **v2 RUBRIC**): {trunc(v2_gpt.get('reasoning'), 400)}")
            for j in ("gemini", "claude"):
                jr = r["v1_jd"][j]
                parts.append(f"- _{j}_ (score={jr['score']}, _unchanged from v1_): {trunc(jr.get('reasoning'), 400)}")
            parts.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "v2_failure_input.md"
    out.write_text("\n".join(parts))
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
