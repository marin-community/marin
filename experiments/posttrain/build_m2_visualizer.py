# ruff: noqa: E501, RUF001
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a single-file HTML visualizer for the bloomv2_m2 preference pairs.

Loads:
  - stage3_output/paired_rubrics_seed_40.jsonl           (rubrics per tension pair)
  - stage4_output/tier_b/chosens_tier_b.jsonl            (selected chosens, 907)
  - stage4_output/tier_b/rejecteds_tier_b.jsonl          (selected rejecteds, 1739)
  - stage4_output/tier_b/chosens/chosens_scores.jsonl    (judge reasoning for chosens)
  - stage4_output/tier_b/m1_variants/scores.jsonl        (judge reasoning for M1 rejecteds)
  - stage4_output/tier_b/variants/train_variants_tier_b.jsonl  (variant prompts)

Produces a self-contained HTML file with embedded JSON data.

Usage:
    python experiments/posttrain/build_m2_visualizer.py [--out /tmp/m2_visualizer.html]
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent
STAGE3 = ROOT / "stage3_output"
STAGE4 = ROOT / "stage4_output" / "tier_b"
SPEC_FILE = ROOT / "specs" / "openai_model_spec.jsonl"

RUBRICS_FILE = STAGE3 / "paired_rubrics_seed_40.jsonl"
CHOSENS_FILE = STAGE4 / "chosens_tier_b.jsonl"
REJECTEDS_FILE = STAGE4 / "rejecteds_tier_b.jsonl"
CHOSEN_SCORES_FILE = STAGE4 / "chosens" / "chosens_scores.jsonl"

# Verbatim copy of the system prompt from stage3_paired_rubrics.py so the
# Rubric View shows exactly what gpt-4.1 was told.
STAGE3_SYSTEM_PROMPT = """You are writing SCORING RUBRICS for a specific scenario where TWO behavioral statements from a model spec both apply and create a trade-off.

You will be given:
- Statement A (id, text, examples)
- Statement B (id, text, examples)
- A tension point: a named axis combination where the trade-off between A and B peaks, and a concrete example user prompt at that peak

Your job is to produce TWO rubrics — one per statement — each tailored to this specific scenario. These rubrics will be used by an LM-as-judge to score actual model responses and by a teacher model to generate responses.

Each rubric has three slots:

- GOOD: concrete description of what a response that honors the statement looks like ON THIS SPECIFIC SCENARIO. Not generic — reference the actual user prompt.
- BAD: concrete description of what a response that violates the statement looks like on this scenario. Specific failure modes.
- KEY_TENSION: explicitly acknowledge that the OTHER statement also applies, note what the trade-off is, and indicate how this rubric should be interpreted IN LIGHT OF that other statement being active.

The KEY_TENSION slot is load-bearing. Without it, the two rubrics would be elicited independently and a judge would score them as if the other statement didn't exist. With it, each rubric is aware of the trade-off.

Be SPECIFIC. Write rubrics a judge could actually use to score real responses. Reference the scenario details.

Return strict JSON:
{
  "A_rubric": {
    "GOOD": "<specific description>",
    "BAD": "<specific description>",
    "KEY_TENSION": "<acknowledges B and the trade-off>"
  },
  "B_rubric": {
    "GOOD": "<specific description>",
    "BAD": "<specific description>",
    "KEY_TENSION": "<acknowledges A and the trade-off>"
  }
}

No prose outside the JSON object."""


def render_statement(row: dict, label: str) -> str:
    """Byte-for-byte copy of render_statement from stage3_paired_rubrics.py."""
    lines = [
        f"Statement {label}",
        f"  id: {row['id']}",
        f"  authority_level: {row.get('authority_level', '')}",
        f"  section: {row.get('section', '')}",
    ]
    if row.get("subsection"):
        lines.append(f"  subsection: {row['subsection']}")
    lines.append("  text: |")
    for t in row["text"].splitlines() or [row["text"]]:
        lines.append(f"    {t}")
    examples = row.get("metadata", {}).get("examples", []) or []
    if examples:
        lines.append(f"  examples ({len(examples)}):")
        for i, ex in enumerate(examples, 1):
            lines.append(f"    - example {i}:")
            if ex.get("description"):
                lines.append(f"        description: {ex['description']}")
            lines.append(f"        user_query: {ex.get('user_query','')}")
            lines.append(f"        good_response: {ex.get('good_response','')}")
            lines.append(f"        bad_response: {ex.get('bad_response','')}")
    return "\n".join(lines)


def render_tension_minimal(tp: dict) -> str:
    """Render a tension point from what the rubric artifact preserves.

    The full stage-3 render_tension included `axes` (with toward_A / toward_B)
    and a `reasoning` field. The stage-2 atlas output that carried those is not
    archived in the rubric JSON — only (tension_name, peak_corner, example_prompt)
    survive. We render those and flag the gap so readers know what's missing.
    """
    return (
        "Tension point\n"
        f"  tension_name: {tp.get('tension_name','')}\n"
        f"  peak_corner: {tp.get('peak_corner','')}\n"
        f"  example_prompt: {tp.get('example_prompt','')}\n"
        "  [axes, toward_A/toward_B values, and stage-2 reasoning were also in the\n"
        "   original user message but are not preserved in the rubric artifact.]"
    )


M1_SCORES_FILE = STAGE4 / "m1_variants" / "scores.jsonl"
VARIANTS_FILE = STAGE4 / "variants" / "train_variants_tier_b.jsonl"


def read_jsonl(path: pathlib.Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_dataset() -> dict:
    # 0. Model spec — needed to reconstruct the statement blocks that
    # were passed to gpt-4.1 at stage 3.
    spec_rows: dict[str, dict] = {}
    for rec in read_jsonl(SPEC_FILE):
        spec_rows[rec["id"]] = rec

    # 1. Rubrics: one record per seed point, keyed by (pair_id, tp_idx).
    # 40 seed points, 7 distinct tp_idx framings, 37 unique pair_ids
    # (some pair_ids appear at multiple tension corners).
    seeds: dict[tuple[str, int], dict] = {}
    for rec in read_jsonl(RUBRICS_FILE):
        key = (rec["pair_id"], rec["tension_point_idx"])
        a_row = spec_rows.get(rec["statement_a_id"], {})
        b_row = spec_rows.get(rec["statement_b_id"], {})
        # Reconstruct the user message gpt-4.1 saw (stage-3 elicitation).
        if a_row and b_row:
            stmt_a_block = render_statement(a_row, "A")
            stmt_b_block = render_statement(b_row, "B")
        else:
            stmt_a_block = f"[spec row not found for {rec['statement_a_id']}]"
            stmt_b_block = f"[spec row not found for {rec['statement_b_id']}]"
        tension_block = render_tension_minimal(rec["tension_point"])
        user_msg = (
            stmt_a_block
            + "\n\n"
            + stmt_b_block
            + "\n\n"
            + tension_block
            + "\n\nWrite the two rubrics (A_rubric, B_rubric) tailored to this scenario."
        )
        seeds[key] = {
            "seed_key": f"{rec['pair_id']}@tp{rec['tension_point_idx']}",
            "pair_id": rec["pair_id"],
            "tension_point_idx": rec["tension_point_idx"],
            "statement_a_id": rec["statement_a_id"],
            "statement_b_id": rec["statement_b_id"],
            "tension_name": rec["tension_point"]["tension_name"],
            "peak_corner": rec["tension_point"]["peak_corner"],
            "example_prompt": rec["tension_point"]["example_prompt"],
            "A_rubric": rec["A_rubric"],
            "B_rubric": rec["B_rubric"],
            "judge_model": rec.get("judge_model", "gpt-4.1"),
            "elicitation_stmt_a": stmt_a_block,
            "elicitation_stmt_b": stmt_b_block,
            "elicitation_tension": tension_block,
            "elicitation_user_message": user_msg,
        }

    # 2. Judge explanations — key by (pair_id, tp, variant, idx, side).
    # The scores files include pair_id, so we can disambiguate when
    # the same tp_idx is used by more than one pair_id.
    chosen_reasoning: dict[tuple[str, int, int, int, str], dict] = {}
    for rec in read_jsonl(CHOSEN_SCORES_FILE):
        key = (rec["pair_id"], rec["tension_point_idx"], rec["variant_idx"], rec["draw_idx"], rec["side"])
        chosen_reasoning[key] = {
            "score": rec.get("score"),
            "explanation": rec.get("explanation") or "",
        }

    m1_reasoning: dict[tuple[str, int, int, int, str], dict] = {}
    for rec in read_jsonl(M1_SCORES_FILE):
        key = (rec["pair_id"], rec["tension_point_idx"], rec["variant_idx"], rec["sample_idx"], rec["side"])
        m1_reasoning[key] = {
            "score": rec.get("score"),
            "explanation": rec.get("explanation") or "",
        }

    # 3. Variant prompts — variants file is one row per seed
    # (pair_id, tp_idx), with `variants: [str, ...]` indexed by
    # position (variant_idx).
    variant_prompts: dict[tuple[str, int, int], str] = {}
    for rec in read_jsonl(VARIANTS_FILE):
        pid = rec["pair_id"]
        tp_idx = rec["tension_point_idx"]
        for vi, prompt_text in enumerate(rec.get("variants", [])):
            variant_prompts[(pid, tp_idx, vi)] = prompt_text

    # 4. Selected chosens
    chosens_by_variant: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for rec in read_jsonl(CHOSENS_FILE):
        pid, tp, vi, di = rec["pair_id"], rec["tension_point_idx"], rec["variant_idx"], rec["draw_idx"]
        a = chosen_reasoning.get((pid, tp, vi, di, "A"), {})
        b = chosen_reasoning.get((pid, tp, vi, di, "B"), {})
        chosens_by_variant[(pid, tp, vi)].append(
            {
                "draw_idx": di,
                "response": rec["chosen_response"],
                "score_A": rec.get("chosen_score_A"),
                "score_B": rec.get("chosen_score_B"),
                "min_side": rec.get("chosen_min_side"),
                "reason_A": a.get("explanation", ""),
                "reason_B": b.get("explanation", ""),
            }
        )

    # 5. Selected rejecteds
    rejecteds_by_variant: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for rec in read_jsonl(REJECTEDS_FILE):
        pid, tp, vi, si = rec["pair_id"], rec["tension_point_idx"], rec["variant_idx"], rec["sample_idx"]
        a = m1_reasoning.get((pid, tp, vi, si, "A"), {})
        b = m1_reasoning.get((pid, tp, vi, si, "B"), {})
        rejecteds_by_variant[(pid, tp, vi)].append(
            {
                "sample_idx": si,
                "response": rec["response"],
                "score_A": rec.get("score_A"),
                "score_B": rec.get("score_B"),
                "min_side": rec.get("min_side_score"),
                "failed_side": rec.get("failed_side"),
                "joint_satisfied": rec.get("joint_satisfied"),
                "reason_A": a.get("explanation", ""),
                "reason_B": b.get("explanation", ""),
            }
        )

    # 6. Build per-variant structure, then group by seed (pair_id, tp_idx).
    variants_by_seed: dict[tuple[str, int], list[dict]] = defaultdict(list)
    all_variant_keys = set(chosens_by_variant) | set(rejecteds_by_variant) | set(variant_prompts)
    for pid, tp, vi in sorted(all_variant_keys):
        chosens = sorted(chosens_by_variant.get((pid, tp, vi), []), key=lambda r: r["draw_idx"])
        rejecteds = sorted(rejecteds_by_variant.get((pid, tp, vi), []), key=lambda r: r["sample_idx"])
        variants_by_seed[(pid, tp)].append(
            {
                "variant_idx": vi,
                "prompt": variant_prompts.get((pid, tp, vi), ""),
                "split": "val" if vi == 0 else "train",
                "chosens": chosens,
                "rejecteds": rejecteds,
                "n_chosens": len(chosens),
                "n_rejecteds": len(rejecteds),
                "n_pairs": len(chosens) * len(rejecteds),
            }
        )

    seed_array = []
    for pid, tp in sorted(seeds):
        seed = seeds[(pid, tp)]
        variants = variants_by_seed.get((pid, tp), [])
        seed_array.append(
            {
                **seed,
                "variants": variants,
                "n_variants": len(variants),
                "n_chosens": sum(v["n_chosens"] for v in variants),
                "n_rejecteds": sum(v["n_rejecteds"] for v in variants),
                "n_pairs": sum(v["n_pairs"] for v in variants),
            }
        )

    return {
        "seeds": seed_array,
        "summary": {
            "n_seeds": len(seed_array),
            "n_variants": sum(s["n_variants"] for s in seed_array),
            "n_chosens": sum(s["n_chosens"] for s in seed_array),
            "n_rejecteds": sum(s["n_rejecteds"] for s in seed_array),
            "n_pairs": sum(s["n_pairs"] for s in seed_array),
        },
        "stage3_system_prompt": STAGE3_SYSTEM_PROMPT,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>bloomv2_m2 pair visualizer</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<style>
  :root {
    --bg: #0f1115;
    --panel: #161a21;
    --panel2: #1d2430;
    --border: #2a3342;
    --fg: #e5e7eb;
    --muted: #8b93a7;
    --accent: #7dd3fc;
    --chosen: #22c55e;
    --rejected: #ef4444;
    --warn: #f59e0b;
    --code-bg: #0b0d12;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; background: var(--bg); color: var(--fg); font: 14px/1.5 -apple-system, BlinkMacSystemFont, "SF Pro Text", Inter, system-ui, sans-serif; }
  a { color: var(--accent); text-decoration: none; }
  code, pre { font-family: "SF Mono", Menlo, Monaco, Consolas, monospace; }
  body { display: flex; flex-direction: column; }

  .topbar { height: 48px; flex: 0 0 auto; display: flex; align-items: center; gap: 8px; padding: 0 16px; background: var(--panel); border-bottom: 1px solid var(--border); }
  .topbar .brand { font-weight: 700; font-size: 13px; letter-spacing: .04em; color: var(--muted); text-transform: uppercase; margin-right: 16px; }
  .view-btn { background: transparent; color: var(--muted); border: 1px solid var(--border); padding: 6px 14px; border-radius: 6px; cursor: pointer; font: inherit; font-size: 13px; font-weight: 600; letter-spacing: .01em; }
  .view-btn:hover { background: var(--panel2); color: var(--fg); }
  .view-btn.active { background: var(--accent); color: #0b1220; border-color: var(--accent); }
  .topbar .spacer { flex: 1; }
  .topbar .current-ctx { font-size: 12px; color: var(--muted); font-family: monospace; }
  .help-btn { background: transparent; color: var(--muted); border: 1px solid var(--border); width: 28px; height: 28px; border-radius: 50%; cursor: pointer; font: inherit; font-size: 14px; font-weight: 700; display: flex; align-items: center; justify-content: center; margin-left: 12px; }
  .help-btn:hover { background: var(--panel2); color: var(--fg); }
  .help-btn.active { background: var(--accent); color: #0b1220; border-color: var(--accent); }

  /* Glossary panel (slides under topbar) */
  .glossary { display: none; background: var(--panel2); border-bottom: 1px solid var(--border); padding: 16px 20px; flex: 0 0 auto; }
  .glossary.open { display: block; }
  .glossary h3 { margin: 0 0 8px 0; font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
  .glossary .glossary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 10px 24px; font-size: 12.5px; }
  .glossary .g-row { display: grid; grid-template-columns: 110px 1fr; gap: 10px; align-items: baseline; }
  .glossary .g-term { font-family: monospace; color: var(--accent); font-size: 12px; }
  .glossary .g-def { color: var(--fg); line-height: 1.45; }
  .glossary .g-def em { color: var(--muted); font-style: normal; }

  /* hoverable hint — any element with class .hint gets an underline */
  .hint { border-bottom: 1px dotted var(--muted); cursor: help; }

  .app { display: grid; grid-template-columns: 320px 260px 1fr; flex: 1 1 auto; min-height: 0; }
  .app.rubric-mode { grid-template-columns: 320px 1fr; }
  .app.rubric-mode #col-var { display: none; }
  .col { border-right: 1px solid var(--border); overflow: auto; }
  .col:last-child { border-right: none; }

  header.col-head { position: sticky; top: 0; padding: 10px 14px; background: var(--panel); border-bottom: 1px solid var(--border); font-weight: 600; font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
  header.col-head .sub { display: block; font-size: 11px; color: var(--muted); font-weight: 400; text-transform: none; letter-spacing: 0; margin-top: 2px; }

  .tp-row { padding: 10px 14px; border-bottom: 1px solid var(--border); cursor: pointer; }
  .tp-row:hover { background: var(--panel); }
  .tp-row.active { background: var(--panel2); border-left: 3px solid var(--accent); padding-left: 11px; }
  .tp-title { font-weight: 600; font-size: 13px; line-height: 1.3; }
  .tp-meta { font-size: 11px; color: var(--muted); margin-top: 4px; display: flex; gap: 10px; flex-wrap: wrap; }
  .tp-meta .tag { background: var(--panel2); padding: 1px 6px; border-radius: 3px; }

  .var-row { padding: 8px 14px; border-bottom: 1px solid var(--border); cursor: pointer; display: flex; justify-content: space-between; align-items: center; gap: 8px; }
  .var-row:hover { background: var(--panel); }
  .var-row.active { background: var(--panel2); border-left: 3px solid var(--accent); padding-left: 11px; }
  .var-row .v-lbl { font-weight: 600; font-size: 12px; }
  .var-row .v-preview { font-size: 11px; color: var(--muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
  .var-row .v-nums { font-size: 11px; color: var(--muted); white-space: nowrap; }
  .var-row.split-val .v-lbl::after { content: " · val"; color: var(--warn); font-weight: 400; }

  .detail { padding: 20px 28px; max-width: 1200px; margin: 0 auto; }
  .section { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 16px 18px; margin-bottom: 18px; }
  .section h2 { margin: 0 0 10px 0; font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }

  .tension-hdr { display: flex; justify-content: space-between; align-items: baseline; gap: 16px; }
  .tension-hdr h1 { margin: 0; font-size: 20px; }
  .pair-ids { font-family: monospace; color: var(--accent); font-size: 13px; }
  .peak-corner { color: var(--muted); font-style: italic; margin: 8px 0 0 0; font-size: 13px; }

  .rubrics { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 14px; }
  .rubric { background: var(--panel2); border: 1px solid var(--border); border-radius: 6px; padding: 12px; }
  .rubric h3 { margin: 0 0 8px 0; font-size: 13px; }
  .rubric h3 .sid { font-family: monospace; font-size: 12px; color: var(--accent); }
  .rubric .gb { margin-bottom: 8px; font-size: 13px; }
  .rubric .gb-label { font-weight: 700; color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }
  .rubric .gb-label.good { color: var(--chosen); }
  .rubric .gb-label.bad { color: var(--rejected); }
  .rubric details { margin-top: 6px; }
  .rubric summary { cursor: pointer; font-size: 12px; color: var(--muted); }

  .prompt-box { background: var(--code-bg); border: 1px solid var(--border); border-radius: 6px; padding: 14px; font-size: 14px; white-space: pre-wrap; }

  .scatter-wrap { display: flex; gap: 20px; align-items: flex-start; }
  .scatter { background: var(--panel2); border: 1px solid var(--border); border-radius: 6px; }
  .scatter-legend { font-size: 12px; color: var(--muted); display: flex; flex-direction: column; gap: 6px; }
  .legend-dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: middle; }

  .responses { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  @media (max-width: 1100px) { .responses { grid-template-columns: 1fr; } }
  .resp-col h2 { margin: 0 0 8px 0; font-size: 14px; }
  .resp-col h2 .count { color: var(--muted); font-weight: 400; font-size: 12px; }
  .resp { background: var(--panel); border: 1px solid var(--border); border-radius: 6px; margin-bottom: 10px; overflow: hidden; }
  .resp.chosen { border-left: 4px solid var(--chosen); }
  .resp.rejected { border-left: 4px solid var(--rejected); }
  .resp-hdr { padding: 10px 14px; background: var(--panel2); display: flex; justify-content: space-between; align-items: center; gap: 10px; font-size: 12px; }
  .score-pair { display: flex; gap: 6px; align-items: center; }
  .badge { font-family: monospace; font-size: 11px; padding: 2px 7px; border-radius: 3px; background: #2a3342; color: var(--fg); }
  .badge.good { background: #14532d; color: #bbf7d0; }
  .badge.okay { background: #713f12; color: #fde68a; }
  .badge.bad { background: #7f1d1d; color: #fecaca; }
  .badge.failed-A { background: #7f1d1d; color: #fecaca; }
  .badge.failed-B { background: #7f1d1d; color: #fecaca; }
  .badge.failed-both { background: #4c0519; color: #fecdd3; }
  .resp-body { padding: 14px; }
  .resp-body .resp-md { max-height: 340px; overflow: auto; font-size: 13px; line-height: 1.55; }
  .resp-body .resp-md pre { background: var(--code-bg); padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 12px; }
  .resp-body .resp-md code { background: var(--code-bg); padding: 1px 4px; border-radius: 3px; font-size: 12px; }
  .resp-body .resp-md h1, .resp-body .resp-md h2, .resp-body .resp-md h3 { font-size: 14px; margin: 10px 0 4px; }
  .resp-body .resp-md p { margin: 6px 0; }
  .resp-body .resp-md ul, .resp-body .resp-md ol { margin: 6px 0 6px 22px; padding: 0; }
  .resp-body .resp-md blockquote { border-left: 3px solid var(--border); margin: 6px 0; padding: 2px 10px; color: var(--muted); }

  .reasoning { margin-top: 10px; border-top: 1px solid var(--border); padding-top: 10px; }
  .reasoning-row { font-size: 12px; margin-bottom: 8px; }
  .reasoning-row .reason-lbl { font-weight: 700; font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
  .reasoning-row.side-A .reason-lbl { color: #93c5fd; }
  .reasoning-row.side-B .reason-lbl { color: #c4b5fd; }
  .reasoning-row .reason-txt { color: var(--fg); margin-top: 2px; font-style: italic; opacity: .88; }

  .show-toggle { cursor: pointer; color: var(--accent); font-size: 11px; background: none; border: none; padding: 0; margin-top: 4px; }

  .top-summary { padding: 10px 14px; color: var(--muted); font-size: 12px; border-bottom: 1px solid var(--border); background: var(--panel); }

  .empty { padding: 40px; text-align: center; color: var(--muted); }

  /* Rubric view */
  .rv-section { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 16px 18px; margin-bottom: 18px; }
  .rv-section h2 { margin: 0 0 4px 0; font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
  .rv-section .rv-hint { font-size: 12px; color: var(--muted); margin-bottom: 12px; }
  .rv-msgpre { background: var(--code-bg); border: 1px solid var(--border); border-radius: 6px; padding: 14px 16px; font-family: "SF Mono", Menlo, Monaco, monospace; font-size: 12px; line-height: 1.55; white-space: pre-wrap; overflow-x: auto; color: #e5e7eb; }
  .rv-role { display: inline-block; font-family: monospace; font-size: 10px; font-weight: 700; padding: 2px 6px; border-radius: 3px; margin-bottom: 6px; letter-spacing: .04em; text-transform: uppercase; }
  .rv-role.system { background: #1e3a8a; color: #dbeafe; }
  .rv-role.user { background: #064e3b; color: #bbf7d0; }
  .rv-role.assistant { background: #4c1d95; color: #e9d5ff; }
  .rv-gap { font-size: 12px; color: var(--warn); background: #422006; border: 1px solid #713f12; border-radius: 4px; padding: 8px 10px; margin: 8px 0; }
  .rv-output .rubric { background: var(--panel2); margin-bottom: 10px; }
  .rv-output .slot { margin-bottom: 10px; }
  .rv-output .slot-label { font-weight: 700; font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
  .rv-output .slot-label.good { color: var(--chosen); }
  .rv-output .slot-label.bad { color: var(--rejected); }
  .rv-output .slot-label.key { color: var(--accent); }
  .rv-output .slot-body { font-size: 13px; margin-top: 2px; }
</style>
</head>
<body>
<div class="topbar">
  <span class="brand">bloomv2_m2</span>
  <button class="view-btn active" id="btn-pref">Preference View</button>
  <button class="view-btn" id="btn-rubric">Rubric View</button>
  <span class="spacer"></span>
  <span class="current-ctx" id="ctx-line"></span>
  <button class="help-btn" id="btn-help" title="Toggle glossary">?</button>
</div>
<div class="glossary" id="glossary">
  <h3>Glossary — what these abbreviations mean</h3>
  <div class="glossary-grid">
    <div class="g-row"><span class="g-term">seed</span><span class="g-def">one of 40 <em>(pair_id, tension-framing)</em> combinations. A pair_id is two behavioral statements from the Marin model spec; a tension-framing is stage-2's named corner where those two statements conflict.</span></div>
    <div class="g-row"><span class="g-term">TP #</span><span class="g-def">tension-point framing index. 7 distinct framings span the 40 seeds, because some statement pairs are instantiated at more than one corner.</span></div>
    <div class="g-row"><span class="g-term">variant / V#</span><span class="g-def">one of 10 paraphrase prompts gpt-4.1 wrote for this seed (same trade-off, different surface wording).</span></div>
    <div class="g-row"><span class="g-term">V0 · val</span><span class="g-def">variant 0 is the held-out val split; V1–V9 go into the train split.</span></div>
    <div class="g-row"><span class="g-term">10v</span><span class="g-def">10 variants for this seed.</span></div>
    <div class="g-row"><span class="g-term">20c / 50r</span><span class="g-def">20 chosens selected · 50 rejecteds selected (summed across all variants of this seed).</span></div>
    <div class="g-row"><span class="g-term">3c / 5r</span><span class="g-def">3 chosens · 5 rejecteds for this single variant (max 3 × 5 = 15 pairs per variant).</span></div>
    <div class="g-row"><span class="g-term">pairs</span><span class="g-def">raw preference pairs fed to DPO: per variant, crossproduct of its selected chosens × selected rejecteds.</span></div>
    <div class="g-row"><span class="g-term">chosen</span><span class="g-def">a gpt-5.1 draw kept because it satisfied both rubrics (<code>min(A,B) ≥ 7</code>). The "good" side of a DPO pair.</span></div>
    <div class="g-row"><span class="g-term">rejected</span><span class="g-def">an M1 sample kept because it clearly failed at least one rubric (<code>min(A,B) ≤ 5</code>, joint unsatisfied). The "bad" side of a DPO pair.</span></div>
    <div class="g-row"><span class="g-term">Rubric A / B</span><span class="g-def">the two scoring rubrics gpt-4.1 wrote, one per statement. Each has GOOD, BAD, and KEY_TENSION slots.</span></div>
    <div class="g-row"><span class="g-term">A: n / B: n</span><span class="g-def">judge (gpt-5.1) score 0–10 on rubric A / rubric B for this response.</span></div>
    <div class="g-row"><span class="g-term">failed: A / B / both</span><span class="g-def">for a rejected response: which rubric(s) scored &lt; 7. "both" = failed both sides, "A" = sacrificed A, "B" = sacrificed B.</span></div>
    <div class="g-row"><span class="g-term">peak corner</span><span class="g-def">stage-2's description of where the trade-off between A and B sharpens (what makes this seed hard).</span></div>
    <div class="g-row"><span class="g-term">Preference View</span><span class="g-def">browse the (chosen, rejected) pairs that DPO actually trains on, plus judge reasoning.</span></div>
    <div class="g-row"><span class="g-term">Rubric View</span><span class="g-def">inspect the exact prompt gpt-4.1 received when writing this seed's A/B rubrics.</span></div>
  </div>
</div>
<div class="app" id="app">
  <div class="col" id="col-tp">
    <header class="col-head">
      Seed points (40)
      <span class="sub" id="summary-line"></span>
    </header>
    <div id="tp-list"></div>
  </div>
  <div class="col" id="col-var">
    <header class="col-head">
      Variants
      <span class="sub" id="variants-sub">select a tension</span>
    </header>
    <div id="var-list"></div>
  </div>
  <div class="col" id="col-detail">
    <div id="detail"></div>
  </div>
</div>

<script id="data-blob" type="application/json">__DATA_JSON__</script>
<script>
const DATA = JSON.parse(document.getElementById('data-blob').textContent);

function scoreBadgeClass(s) {
  if (s == null) return 'badge';
  if (s >= 7) return 'badge good';
  if (s >= 5) return 'badge okay';
  return 'badge bad';
}
function fmtScore(s) { return s == null ? '—' : s; }

function renderSummary() {
  const s = DATA.summary;
  document.getElementById('summary-line').textContent =
    `${s.n_seeds} seeds · ${s.n_variants} variants · ${s.n_chosens} chosens · ${s.n_rejecteds} rejecteds`;
}

function renderTPList() {
  const el = document.getElementById('tp-list');
  el.innerHTML = '';
  DATA.seeds.forEach((seed, i) => {
    const div = document.createElement('div');
    div.className = 'tp-row';
    div.dataset.seedKey = seed.seed_key;
    div.innerHTML = `
      <div class="tp-title">${escapeHtml(seed.tension_name)}</div>
      <div class="tp-meta">
        <span class="tag" title="Tension-point framing index (7 framings across the 40 seeds)">TP ${seed.tension_point_idx}</span>
        <span class="tag" title="${seed.n_variants} paraphrase prompts for this seed">${seed.n_variants} variants</span>
        <span class="tag" title="${seed.n_chosens} chosens + ${seed.n_rejecteds} rejecteds selected (across all variants)">${seed.n_chosens}c · ${seed.n_rejecteds}r</span>
        <span class="tag" title="Raw preference pairs fed to DPO (chosens × rejecteds per variant, summed)">${seed.n_pairs} pairs</span>
      </div>
      <div class="tp-meta">
        <span style="font-family:monospace;font-size:10px;color:var(--accent)" title="Statement A (pulled toward) / Statement B (pulled away)">${seed.statement_a_id} / ${seed.statement_b_id}</span>
      </div>
    `;
    div.onclick = () => selectSeed(seed.seed_key);
    el.appendChild(div);
  });
}

let currentSeed = null;
let currentVariant = null;

function selectSeed(seedKey) {
  currentSeed = DATA.seeds.find(s => s.seed_key === seedKey);
  document.querySelectorAll('.tp-row').forEach(r => r.classList.toggle('active', r.dataset.seedKey === seedKey));
  document.getElementById('variants-sub').textContent = currentSeed.tension_name;
  renderVarList();
  if (currentView === 'rubric') {
    renderRubricDetail();
  } else {
    if (currentSeed.variants.length) selectVariant(currentSeed.variants[0].variant_idx);
    else { document.getElementById('detail').innerHTML = '<div class="empty">no variants for this seed</div>'; currentVariant = null; }
  }
  updateCtxLine();
}

function updateCtxLine() {
  const el = document.getElementById('ctx-line');
  if (!currentSeed) { el.textContent = ''; return; }
  const vpart = currentView === 'preference' && currentVariant ? ` · V${currentVariant.variant_idx}` : '';
  el.textContent = `${currentSeed.pair_id} @ tp${currentSeed.tension_point_idx}${vpart}`;
}

function renderVarList() {
  const el = document.getElementById('var-list');
  el.innerHTML = '';
  currentSeed.variants.forEach(v => {
    const div = document.createElement('div');
    div.className = 'var-row' + (v.split === 'val' ? ' split-val' : '');
    div.dataset.varIdx = v.variant_idx;
    const preview = (v.prompt || '').slice(0, 50).replace(/\s+/g, ' ');
    const splitTt = v.split === 'val' ? 'Held-out val split (V0)' : 'Train split (V1–V9)';
    div.title = `Variant ${v.variant_idx} · ${splitTt} · ${v.n_chosens} chosens, ${v.n_rejecteds} rejecteds selected`;
    div.innerHTML = `
      <div class="v-lbl">V${v.variant_idx}</div>
      <div class="v-preview">${escapeHtml(preview)}</div>
      <div class="v-nums" title="${v.n_chosens} chosens / ${v.n_rejecteds} rejecteds for this variant">${v.n_chosens}c/${v.n_rejecteds}r</div>
    `;
    div.onclick = () => selectVariant(v.variant_idx);
    el.appendChild(div);
  });
}

function selectVariant(vi) {
  currentVariant = currentSeed.variants.find(v => v.variant_idx === vi);
  document.querySelectorAll('.var-row').forEach(r => r.classList.toggle('active', +r.dataset.varIdx === vi));
  renderDetail();
  updateCtxLine();
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

function renderMd(s) {
  // marked.js — it's safe enough for our (trusted, local) data.
  return marked.parse(String(s ?? ''), { breaks: true, gfm: true });
}

function renderScatter(chosens, rejecteds) {
  const W = 260, H = 260, M = 30;
  const xScale = s => M + (s / 10) * (W - 2 * M);
  const yScale = s => (H - M) - (s / 10) * (H - 2 * M);
  let dots = '';
  chosens.forEach((c, i) => {
    const x = xScale(c.score_A ?? 0), y = yScale(c.score_B ?? 0);
    dots += `<circle cx="${x}" cy="${y}" r="6" fill="var(--chosen)" stroke="#052e16" stroke-width="1.5" opacity="0.85"><title>chosen draw ${c.draw_idx}: A=${c.score_A}, B=${c.score_B}</title></circle>`;
  });
  rejecteds.forEach((r, i) => {
    const x = xScale(r.score_A ?? 0), y = yScale(r.score_B ?? 0);
    dots += `<rect x="${x-5}" y="${y-5}" width="10" height="10" fill="var(--rejected)" stroke="#450a0a" stroke-width="1.5" opacity="0.85"><title>rejected sample ${r.sample_idx}: A=${r.score_A}, B=${r.score_B}, failed=${r.failed_side}</title></rect>`;
  });
  // threshold lines at 7
  const t7x = xScale(7), t7y = yScale(7);
  const axes = `
    <line x1="${M}" y1="${H-M}" x2="${W-M}" y2="${H-M}" stroke="var(--border)"/>
    <line x1="${M}" y1="${M}" x2="${M}" y2="${H-M}" stroke="var(--border)"/>
    <line x1="${t7x}" y1="${M}" x2="${t7x}" y2="${H-M}" stroke="#facc1540" stroke-dasharray="3,3"/>
    <line x1="${M}" y1="${t7y}" x2="${W-M}" y2="${t7y}" stroke="#facc1540" stroke-dasharray="3,3"/>
    <text x="${W/2}" y="${H-8}" fill="var(--muted)" font-size="11" text-anchor="middle">Rubric A judge score (0–10) →</text>
    <text x="12" y="${H/2}" fill="var(--muted)" font-size="11" text-anchor="middle" transform="rotate(-90, 12, ${H/2})">Rubric B judge score (0–10) →</text>
    <text x="${M-6}" y="${H-M+4}" fill="var(--muted)" font-size="10" text-anchor="end">0</text>
    <text x="${M-6}" y="${M+4}" fill="var(--muted)" font-size="10" text-anchor="end">10</text>
    <text x="${W-M}" y="${H-M+14}" fill="var(--muted)" font-size="10" text-anchor="middle">10</text>
  `;
  return `<svg class="scatter" width="${W}" height="${H}">${axes}${dots}</svg>`;
}

function renderResp(r, kind) {
  const clsA = scoreBadgeClass(r.score_A);
  const clsB = scoreBadgeClass(r.score_B);
  const idLabel = kind === 'chosen' ? `draw ${r.draw_idx}` : `sample ${r.sample_idx}`;
  const sourceTt = kind === 'chosen'
    ? 'Generated by gpt-5.1 on this variant prompt'
    : 'Sampled from M1 (the DPO LoRA we are trying to improve) on this variant prompt';
  const failedBadge = kind === 'rejected' && r.failed_side
    ? `<span class="badge failed-${r.failed_side}" title="This rejected sample scored &lt; 7 on: ${r.failed_side === 'both' ? 'both rubrics' : 'rubric ' + r.failed_side}">failed: ${r.failed_side}</span>` : '';
  const safe = kind === 'chosen' ? 'chosen' : 'rejected';
  return `
    <div class="resp ${safe}">
      <div class="resp-hdr">
        <div title="${sourceTt}"><b>${kind}</b> <span style="color:var(--muted)">#${idLabel}</span></div>
        <div class="score-pair">
          ${failedBadge}
          <span class="${clsA}" title="Judge (gpt-5.1) score 0–10 on Rubric A">A: ${fmtScore(r.score_A)}</span>
          <span class="${clsB}" title="Judge (gpt-5.1) score 0–10 on Rubric B">B: ${fmtScore(r.score_B)}</span>
        </div>
      </div>
      <div class="resp-body">
        <div class="resp-md">${renderMd(r.response)}</div>
        <div class="reasoning">
          <div class="reasoning-row side-A">
            <div class="reason-lbl">Judge · Rubric A (score ${fmtScore(r.score_A)})</div>
            <div class="reason-txt">${escapeHtml(r.reason_A || '(no reasoning captured)')}</div>
          </div>
          <div class="reasoning-row side-B">
            <div class="reason-lbl">Judge · Rubric B (score ${fmtScore(r.score_B)})</div>
            <div class="reason-txt">${escapeHtml(r.reason_B || '(no reasoning captured)')}</div>
          </div>
        </div>
      </div>
    </div>
  `;
}

function renderDetail() {
  const v = currentVariant;
  const tp = currentSeed;
  if (!v) { document.getElementById('detail').innerHTML = '<div class="empty">pick a variant</div>'; return; }
  const scatter = renderScatter(v.chosens, v.rejecteds);
  const chosenCards = v.chosens.map(c => renderResp(c, 'chosen')).join('');
  const rejectedCards = v.rejecteds.map(r => renderResp(r, 'rejected')).join('');
  const rubricA = tp.A_rubric, rubricB = tp.B_rubric;
  const html = `
    <div class="detail">
      <div class="section">
        <div class="tension-hdr">
          <div>
            <h1>${escapeHtml(tp.tension_name)}</h1>
            <div class="pair-ids">${tp.statement_a_id} &nbsp;vs&nbsp; ${tp.statement_b_id}</div>
          </div>
          <div style="text-align:right;color:var(--muted);font-size:12px">
            <span class="hint" title="Tension-point framing index (7 framings span the 40 seeds)">TP ${tp.tension_point_idx}</span> ·
            <span class="hint" title="Variant index (V0 is val; V1–V9 are train)">V${v.variant_idx}</span> ·
            <span class="hint" title="${v.split === 'val' ? 'Held-out val split (V0)' : 'Train split'}">split=${v.split}</span><br>
            <span class="hint" title="${v.n_chosens} gpt-5.1 draws with min(A,B)≥7, crossed with ${v.n_rejecteds} M1 samples with min(A,B)≤5">${v.n_chosens} chosens × ${v.n_rejecteds} rejecteds = ${v.n_pairs} pairs</span>
          </div>
        </div>
        <p class="peak-corner"><b>Peak corner</b> <span style="color:var(--muted);font-weight:400">(stage-2's description of where the trade-off is sharpest):</span> ${escapeHtml(tp.peak_corner)}</p>
        <div class="rubrics">
          <div class="rubric">
            <h3>Rubric A <span class="sid">${tp.statement_a_id}</span></h3>
            <div class="gb"><div class="gb-label good">GOOD</div>${escapeHtml(rubricA.GOOD)}</div>
            <div class="gb"><div class="gb-label bad">BAD</div>${escapeHtml(rubricA.BAD)}</div>
            ${rubricA.KEY_TENSION ? `<details><summary>key tension</summary><div style="margin-top:6px;font-size:12px;color:var(--muted)">${escapeHtml(rubricA.KEY_TENSION)}</div></details>` : ''}
          </div>
          <div class="rubric">
            <h3>Rubric B <span class="sid">${tp.statement_b_id}</span></h3>
            <div class="gb"><div class="gb-label good">GOOD</div>${escapeHtml(rubricB.GOOD)}</div>
            <div class="gb"><div class="gb-label bad">BAD</div>${escapeHtml(rubricB.BAD)}</div>
            ${rubricB.KEY_TENSION ? `<details><summary>key tension</summary><div style="margin-top:6px;font-size:12px;color:var(--muted)">${escapeHtml(rubricB.KEY_TENSION)}</div></details>` : ''}
          </div>
        </div>
      </div>

      <div class="section">
        <h2>Variant prompt</h2>
        <div class="prompt-box">${escapeHtml(v.prompt)}</div>
      </div>

      <div class="section">
        <h2>Judge-score plane (chosens vs rejecteds)</h2>
        <div style="color:var(--muted);font-size:12px;margin-bottom:10px">Every response for this variant gets two scores from the gpt-5.1 judge: one against <b>Rubric A</b> (x-axis) and one against <b>Rubric B</b> (y-axis), each 0–10. Chosens should cluster top-right; rejecteds spread along the edges.</div>
        <div class="scatter-wrap">
          ${scatter}
          <div class="scatter-legend">
            <div><span class="legend-dot" style="background:var(--chosen)"></span><b>chosen</b> — gpt-5.1 draw kept if <code>min(A,B) ≥ 7</code></div>
            <div><span class="legend-dot" style="background:var(--rejected);border-radius:0"></span><b>rejected</b> — M1 sample kept if <code>min(A,B) ≤ 5</code> and joint-unsatisfied</div>
            <div style="margin-top:6px;color:#facc15">dashed yellow lines = score=7 (chosen threshold)</div>
            <div style="margin-top:10px;font-size:11px;color:var(--muted)">
              Reading the quadrants:<br>
              • top-right → joint satisfaction (ideal chosen)<br>
              • high-A low-B → sacrificed B for A<br>
              • high-B low-A → sacrificed A for B<br>
              • bottom-left → bad on both sides
            </div>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>Responses & judge reasoning</h2>
        <div class="responses">
          <div class="resp-col">
            <h2>Chosens <span class="count">${v.n_chosens} selected (gpt-5.1)</span></h2>
            ${chosenCards || '<div class="empty">no qualifying chosens</div>'}
          </div>
          <div class="resp-col">
            <h2>Rejecteds <span class="count">${v.n_rejecteds} selected (M1)</span></h2>
            ${rejectedCards || '<div class="empty">no qualifying rejecteds</div>'}
          </div>
        </div>
      </div>
    </div>
  `;
  document.getElementById('detail').innerHTML = html;
}

function renderRubricDetail() {
  const seed = currentSeed;
  if (!seed) { document.getElementById('detail').innerHTML = '<div class="empty">pick a seed</div>'; return; }
  const html = `
    <div class="detail">
      <div class="section">
        <div class="tension-hdr">
          <div>
            <h1>${escapeHtml(seed.tension_name)}</h1>
            <div class="pair-ids">${seed.statement_a_id} &nbsp;vs&nbsp; ${seed.statement_b_id}</div>
          </div>
          <div style="text-align:right;color:var(--muted);font-size:12px">
            TP ${seed.tension_point_idx} · judge: <b>${seed.judge_model}</b><br>
            This is what gpt-4.1 saw before writing the two rubrics.
          </div>
        </div>
        <p class="peak-corner">Peak corner: ${escapeHtml(seed.peak_corner)}</p>
      </div>

      <div class="rv-section">
        <h2><span class="rv-role system">system</span> Stage-3 system prompt</h2>
        <div class="rv-hint">Identical for all 40 seeds — this constant defines the rubric-writing contract.</div>
        <div class="rv-msgpre">${escapeHtml(DATA.stage3_system_prompt)}</div>
      </div>

      <div class="rv-section">
        <h2><span class="rv-role user">user · part 1</span> Statement A</h2>
        <div class="rv-hint">Reconstructed from <code>specs/openai_model_spec.jsonl</code> using the same <code>render_statement()</code> logic as stage 3.</div>
        <div class="rv-msgpre">${escapeHtml(seed.elicitation_stmt_a)}</div>
      </div>

      <div class="rv-section">
        <h2><span class="rv-role user">user · part 2</span> Statement B</h2>
        <div class="rv-msgpre">${escapeHtml(seed.elicitation_stmt_b)}</div>
      </div>

      <div class="rv-section">
        <h2><span class="rv-role user">user · part 3</span> Tension point</h2>
        <div class="rv-gap">Stage-3 artifact only preserved <code>{tension_name, peak_corner, example_prompt}</code>. The original user message also included stage-2's <code>axes</code> (with <code>toward_A</code> / <code>toward_B</code> values) and a free-text <code>reasoning</code> field. Those are not reconstructable from what's on disk.</div>
        <div class="rv-msgpre">${escapeHtml(seed.elicitation_tension)}</div>
      </div>

      <div class="rv-section rv-output">
        <h2><span class="rv-role assistant">assistant</span> Rubrics produced by ${seed.judge_model}</h2>
        <div class="rv-hint">Strict JSON, three slots per side. KEY_TENSION is what makes the rubric "aware" of the other statement — it's collapsed in the Preference view but shown fully here.</div>
        <div class="rubrics">
          <div class="rubric">
            <h3>A_rubric <span class="sid">${seed.statement_a_id}</span></h3>
            <div class="slot"><div class="slot-label good">GOOD</div><div class="slot-body">${escapeHtml(seed.A_rubric.GOOD)}</div></div>
            <div class="slot"><div class="slot-label bad">BAD</div><div class="slot-body">${escapeHtml(seed.A_rubric.BAD)}</div></div>
            <div class="slot"><div class="slot-label key">KEY_TENSION</div><div class="slot-body">${escapeHtml(seed.A_rubric.KEY_TENSION || '')}</div></div>
          </div>
          <div class="rubric">
            <h3>B_rubric <span class="sid">${seed.statement_b_id}</span></h3>
            <div class="slot"><div class="slot-label good">GOOD</div><div class="slot-body">${escapeHtml(seed.B_rubric.GOOD)}</div></div>
            <div class="slot"><div class="slot-label bad">BAD</div><div class="slot-body">${escapeHtml(seed.B_rubric.BAD)}</div></div>
            <div class="slot"><div class="slot-label key">KEY_TENSION</div><div class="slot-body">${escapeHtml(seed.B_rubric.KEY_TENSION || '')}</div></div>
          </div>
        </div>
      </div>
    </div>
  `;
  document.getElementById('detail').innerHTML = html;
}

let currentView = 'preference';
function setView(view) {
  currentView = view;
  document.getElementById('btn-pref').classList.toggle('active', view === 'preference');
  document.getElementById('btn-rubric').classList.toggle('active', view === 'rubric');
  document.getElementById('app').classList.toggle('rubric-mode', view === 'rubric');
  if (!currentSeed) return;
  if (view === 'rubric') renderRubricDetail();
  else if (currentVariant) renderDetail();
  else if (currentSeed.variants.length) selectVariant(currentSeed.variants[0].variant_idx);
  updateCtxLine();
}
document.getElementById('btn-pref').onclick = () => setView('preference');
document.getElementById('btn-rubric').onclick = () => setView('rubric');
document.getElementById('btn-help').onclick = () => {
  const g = document.getElementById('glossary');
  const btn = document.getElementById('btn-help');
  g.classList.toggle('open');
  btn.classList.toggle('active', g.classList.contains('open'));
};

renderSummary();
renderTPList();
if (DATA.seeds.length) selectSeed(DATA.seeds[0].seed_key);
</script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/m2_visualizer.html")
    args = ap.parse_args()

    data = build_dataset()
    print(f"summary: {json.dumps(data['summary'], indent=2)}")

    # JSON payload embedded into a <script type=application/json> block.
    # Escape </ to avoid premature script close.
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = HTML_TEMPLATE.replace("__DATA_JSON__", payload)

    out = pathlib.Path(args.out)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
