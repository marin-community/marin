# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501, RUF001, RUF003  -- long LM-prompt strings + intentional unicode (α, ×, −, –) used in DART notation

"""Run 9 Round 2 — re-compile on IMPROVING-but-not-CONVERGED statements with
cumulative-history block (per §1.8.4). Fresh poison cells from R1 v9 judgments.
"""

from __future__ import annotations
import hashlib
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).parent))
import batch_anthropic as ba
from e8_paired_indirection import SPEC_PATH, get_examples, render_anchors, render_examples
from e9_claude_judge import ANTHROPIC_MODEL
from e9_dart_run5 import COMPILER_SYSTEM_V5
from e9_dart_run9_compile import DART_COMPILER_TOOL_V9
from raw_api_logger import RawAPILogger

DIR = Path("experiments/posttrain/disagreement_primitive")
OUT_DIR = DIR / "dart_run9"

# IMPROVING-but-not-CONVERGED from R1 analysis
R2_TARGETS = ["formatting", "no_topic_off_limits"]


def load_jsonl(path):
    return [json.loads(l) for l in path.open() if l.strip()]


def load_v9_r1_judgments():
    """Per-cell scores from R1 judging."""
    by_cell = defaultdict(dict)
    for r in load_jsonl(OUT_DIR / "per_judgment_run9_r1.jsonl"):
        if r.get("score") is None:
            continue
        j = r["judge"]
        ck = (r["statement_id"], r["scenario_idx"], r["generator"], r["condition"])
        by_cell[ck][j] = r["score"]
    return by_cell


def rank_poison_v9(by_cell, sid, condition, top_k=10):
    rows = []
    for ck, scores in by_cell.items():
        s, scen, gen, cond = ck
        if s != sid or cond != condition:
            continue
        ss = list(scores.values())
        if len(ss) < 2:
            continue
        pwv = sum((ss[i] - ss[j]) ** 2 for i in range(len(ss)) for j in range(i + 1, len(ss)))
        rows.append({"scen": scen, "generator": gen, "scores": dict(scores), "pwv": pwv})
    rows.sort(key=lambda r: -r["pwv"])
    # Run 10 (Decision #4): drop unanimous cells (pwv=0) before dedup.
    rows = [r for r in rows if r["pwv"] > 0]
    # Dedupe to one cell per scenario_idx — keeps highest-pwv generator per scenario.
    seen, deduped = set(), []
    for r in rows:
        if r["scen"] in seen:
            continue
        seen.add(r["scen"])
        deduped.append(r)
    return deduped[:top_k], sum(r["pwv"] for r in rows)


def fetch_judge_reasoning_for_v9_cells(cells, sid, condition):
    keys_wanted = {(c["scen"], c["generator"]) for c in cells}
    reasoning = defaultdict(dict)
    for r in load_jsonl(OUT_DIR / "per_judgment_run9_r1.jsonl"):
        if r.get("statement_id") != sid:
            continue
        if r.get("condition") != condition:
            continue
        k = (r["scenario_idx"], r["generator"])
        if k not in keys_wanted:
            continue
        rs = (r.get("reasoning") or "")[:600]
        if rs:
            reasoning[k][r["judge"]] = rs
    return reasoning


def render_history_block(sid, r1_alpha_after, r1_diagnosis, r1_adopted, r1_delta):
    parts = ["=== EDIT HISTORY FOR THIS STATEMENT ===", ""]
    parts.append(
        "The rubric and spec text shown below already incorporate Round 1's adopted edits. "
        "The poison cells shown after this section are computed under the CURRENT (post-R1) state.\n"
    )
    parts.append("Round 1:")
    parts.append(f"  Operative diagnosis: {r1_diagnosis}")
    parts.append(f"  Adopted edits: {r1_adopted}")
    parts.append(f"  Empirical: α_p4 changed by Δ={r1_delta:+.3f} → α_p4_after = {r1_alpha_after:+.3f}")
    parts.append("  Status: IMPROVING but α below T₁=0.5 — needs more work\n")
    parts.append(
        "Given this history:\n"
        "- If α gain is decelerating, propose a different KIND of edit (switch from rubric to examples, or vice versa).\n"
        "- If the same disagreements persist on the new poison cells, declare irreducible.\n"
        "- If your prior edit moved α the wrong way, propose a reversal.\n"
        "- Otherwise, refine.\n"
        "- THIS IS THE FINAL ROUND. If convergence is not imminent, declare irreducible."
    )
    return "\n".join(parts)


def format_cells(cells, response_idx, reasoning_by_key, sid):
    parts = []
    for i, c in enumerate(cells, 1):
        scen, gen = c["scen"], c["generator"]
        s_str = ", ".join(f"{j}={v}" for j, v in c["scores"].items())
        parts.append(f"  #{i} scen={scen} gen={gen} pwv={c['pwv']} scores=[{s_str}]")
        rec = response_idx.get((sid, scen, gen))
        if rec:
            uq, resp = rec
            parts.append(f"     user_query: {uq[:200]}")
            parts.append(f"     response: {resp[:300]}")
        rsn = reasoning_by_key.get((scen, gen), {})
        for j in ("gpt", "gemini-pro", "claude"):
            if j in rsn:
                parts.append(f"     {j}: {rsn[j][:300]}")
    return "\n".join(parts)


def load_response_index():
    out = {}
    for r in load_jsonl(DIR / "e8_responses.jsonl"):
        sid = r.get("statement_id")
        if not sid:
            continue
        for col, label in [
            ("response_gpt", "gpt-5.1"),
            ("response_weak", "Qwen/Qwen2.5-7B-Instruct-Turbo"),
            ("response_gemini", "gemini-3-flash-preview"),
        ]:
            if r.get(col):
                out[(sid, r["scenario_idx"], label)] = (r["user_query"], r[col])
    for r in load_jsonl(DIR / "e9_opposite_mode_responses.jsonl"):
        if "error" in r:
            continue
        sid = r.get("statement_id")
        if not sid:
            continue
        out[(sid, r["scenario_idx"], r["generator"])] = (r["user_query"], r["response"])
    return out


def main():
    summary_r1 = json.loads((OUT_DIR / "run9_synthesis_summary.json").read_text())
    spec = {r["id"]: r for r in load_jsonl(SPEC_PATH)}
    rubrics_v1 = {r["statement_id"]: r["rubric"] for r in load_jsonl(DIR / "e8_rubrics_v1.jsonl") if "error" not in r}

    by_cell_v9 = load_v9_r1_judgments()
    response_idx = load_response_index()

    # Build R2 prompts with cumulative history
    prompts = {}
    for sid in R2_TARGETS:
        info = summary_r1["per_statement"][sid]
        r1_diag = info["operative_diagnosis"]
        r1_adopted = (
            f"rubric_edits={info['n_rubric_adopted']}, "
            f"spec_edits={info['n_spec_adopted']}, "
            f"examples={info['n_examples_adopted']}"
        )

        # Compute R1 final α and Δ for the prompt
        # (rough — load both v1 and v9 α from prior analysis)
        # For prompt purposes, hard-code from earlier analysis output
        if sid == "formatting":
            r1_alpha = 0.450
            r1_delta = 0.090
        elif sid == "no_topic_off_limits":
            r1_alpha = 0.309
            r1_delta = -0.013
        else:
            r1_alpha = 0.0
            r1_delta = 0.0

        # Determine current rubric / spec (v9 state if rubric_drift; v1 + examples if RID)
        if r1_diag == "rubric_drift":
            current_rubric = json.loads((OUT_DIR / sid / "rubric_v9.json").read_text())
            current_spec = spec[sid]
        else:  # RID — current spec has v9 examples appended
            current_rubric = rubrics_v1.get(sid, {})
            spec_with_ex_path = OUT_DIR / sid / "spec_with_examples_v9.json"
            if spec_with_ex_path.exists():
                current_spec = json.loads(spec_with_ex_path.read_text())
            else:
                current_spec = spec[sid]

        # New poison cells under v9 R1 judgments
        bare_cells, bare_total = rank_poison_v9(by_cell_v9, sid, "variant_A", top_k=10)
        rub_cells, rub_total = rank_poison_v9(by_cell_v9, sid, "rubric_plus_spec", top_k=10)
        bare_reasoning = fetch_judge_reasoning_for_v9_cells(bare_cells, sid, "variant_A")
        rub_reasoning = fetch_judge_reasoning_for_v9_cells(rub_cells, sid, "rubric_plus_spec")

        examples_now = get_examples(current_spec)

        history = render_history_block(sid, r1_alpha, r1_diag, r1_adopted, r1_delta)
        prompt = (
            f"=== SPEC STATEMENT TEXT (post-R1 state) ===\n{current_spec['text']}\n\n"
            f"=== SPEC EXAMPLES (post-R1, includes any v9 example additions) ===\n{render_examples(examples_now)}\n\n"
            f"=== CURRENT RUBRIC (post-R1) ===\n{render_anchors(current_rubric)}\n\n"
            f"{history}\n\n"
            f"=== TOP-K BARE-POISON CELLS UNDER v9 (Σ bare_pwv = {bare_total}) ===\n"
            f"{format_cells(bare_cells, response_idx, bare_reasoning, sid)}\n\n"
            f"=== TOP-K RUBRIC-POISON CELLS UNDER v9 (Σ rubric_pwv = {rub_total}) ===\n"
            f"{format_cells(rub_cells, response_idx, rub_reasoning, sid)}\n\n"
            "Diagnose. Propose edits OR declare irreducible. THIS IS THE FINAL ROUND."
        )
        prompts[sid] = prompt
        prompt_path = OUT_DIR / sid / "round_2_compile_prompt.txt"
        prompt_path.write_text(prompt)
        print(f"  built R2 prompt for {sid}: {len(prompt):,} chars")

    print("\n=== Submitting R2 batches ===")

    api_key_oai = os.environ["OPENAI_API_KEY"]
    api_key_anth = os.environ["ANTHROPIC_API_KEY"]
    api_key_gem = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]

    job_dir = Path(f"results/raw/e9_dart_run9_compile_r2/{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S')}")
    job_dir.mkdir(parents=True, exist_ok=True)
    state = {
        "submitted_at_iso": datetime.now(timezone.utc).isoformat(),
        "job_dir": str(job_dir),
        "statements": R2_TARGETS,
        "batches": {},
    }

    # GPT batch
    oai = OpenAI(api_key=api_key_oai)
    cmap_g = {}
    gpt_input_path = job_dir / "input_gpt.jsonl"
    with gpt_input_path.open("w") as f:
        for sid in R2_TARGETS:
            cid = "c2_" + hashlib.md5(f"r2gpt::{sid}".encode()).hexdigest()[:32]
            cmap_g[cid] = sid
            f.write(
                json.dumps(
                    {
                        "custom_id": cid,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": "gpt-5.1",
                            "messages": [
                                {"role": "system", "content": COMPILER_SYSTEM_V5},
                                {"role": "user", "content": prompts[sid]},
                            ],
                            "temperature": 0,
                            "max_completion_tokens": 8000,
                            "reasoning_effort": "none",
                            "response_format": {"type": "json_object"},
                        },
                    }
                )
                + "\n"
            )
    with gpt_input_path.open("rb") as fh:
        up = oai.files.create(file=fh, purpose="batch")
    gpt_batch = oai.batches.create(
        input_file_id=up.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "DART Run 9 R2 compile GPT"},
    )
    state["batches"]["gpt"] = {"batch_id": gpt_batch.id, "n_requests": len(R2_TARGETS), "custom_id_map": cmap_g}
    print(f"  GPT: {gpt_batch.id}")

    # Claude batch
    cmap_c = {}
    cla_reqs = []
    for sid in R2_TARGETS:
        cid = "c2_" + hashlib.md5(f"r2cla::{sid}".encode()).hexdigest()[:32]
        cmap_c[cid] = sid
        cla_reqs.append(
            ba.build_request(
                custom_id=cid,
                model=ANTHROPIC_MODEL,
                system=COMPILER_SYSTEM_V5,
                messages=[{"role": "user", "content": prompts[sid]}],
                max_tokens=8000,
                tools=[DART_COMPILER_TOOL_V9],
                tool_choice={"type": "tool", "name": "submit_dart_diagnosis"},
                thinking={"type": "disabled"},
                temperature=0,
            )
        )
    cla_result = ba.submit(api_key_anth, cla_reqs, job_dir=job_dir, name="run9_r2_compile_cla")
    state["batches"]["claude"] = {
        "batch_id": cla_result["batch_id"],
        "name": "run9_r2_compile_cla",
        "n_requests": len(R2_TARGETS),
        "custom_id_map": cmap_c,
    }
    print(f"  Claude: {cla_result['batch_id']}")

    # Gemini sync
    gem = genai.Client(api_key=api_key_gem)
    log = RawAPILogger("e9_dart_run9_r2_compile_gem")
    gem_results = {}

    def call_gem(sid):
        cfg = types.GenerateContentConfig(
            system_instruction=COMPILER_SYSTEM_V5,
            max_output_tokens=8000,
            temperature=0,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
            response_mime_type="application/json",
        )
        try:
            raw = log.call(
                role="run9_r2_compile_gem",
                key={"sid": sid},
                fn=lambda: gem.models.generate_content(
                    model="gemini-3.1-pro-preview", contents=prompts[sid], config=cfg
                ),
            )
            text = raw.text or ""
            if text.startswith("```"):
                text = text.split("```")[1].lstrip("json\n").strip()
            return sid, json.loads(text) if text.strip() else {}
        except Exception as e:
            return sid, {"error": str(e)[:300]}

    with ThreadPoolExecutor(max_workers=2) as ex:
        for f in [ex.submit(call_gem, s) for s in R2_TARGETS]:
            sid, data = f.result()
            gem_results[sid] = data
            print(f"  Gemini {sid}: diag={data.get('diagnosis', '?')}")

    (OUT_DIR / "diagnoses_gem_r2.jsonl").write_text(
        "\n".join(json.dumps({"statement_id": s, **r}) for s, r in gem_results.items())
    )

    state_path = OUT_DIR / "run9_r2_batches.json"
    state_path.write_text(json.dumps(state, indent=2))
    print(f"\nWrote {state_path}")


if __name__ == "__main__":
    main()
