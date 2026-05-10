"""DART Run 9 Phase 3 — judge v9 conditions on the 6 statements with adopted edits.

Conditions (per statement, branching on diagnosis):
  - rubric_drift statements (formatting, no_erotica_or_gore):
      C_RUBRIC: spec_v1 + rubric_v9 + v1 examples (= original spec.metadata.examples)
  - RID statements with adopted examples (assume_best_intentions, avoid_hateful_content,
      comply_with_laws, no_topic_off_limits):
      C_EXAMPLES: spec_v1 + v1 rubric + v9 examples appended to spec.metadata.examples

3 judges per cell × 80 cells/statement × 6 statements = 1,440 calls per judge.
Submits OpenAI batch + Anthropic batch + Gemini sync.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).parent))
import batch_anthropic as ba
from e8_paired_indirection import (
    SPEC_PATH, get_examples, render_anchors, render_examples, JUDGE_A_SYSTEM,
)
from e8_phase4_rubric_plus_spec import JUDGE_RUBRIC_PLUS_SPEC_SYSTEM
from e9_claude_judge import ANTHROPIC_MODEL
from e9_judge_opposite_mode import JUDGMENT_TOOL_1_5
from raw_api_logger import RawAPILogger

DIR = Path("experiments/posttrain/disagreement_primitive")
OUT_DIR = DIR / "dart_run9"
RUBRICS_V1_PATH = DIR / "e8_rubrics_v1.jsonl"
EXISTING_RESPONSES = DIR / "e8_responses.jsonl"
OPPOSITE_RESPONSES = DIR / "e9_opposite_mode_responses.jsonl"

GEN_KEYS = [
    ("gpt-5.1", "response_gpt"),
    ("Qwen/Qwen2.5-7B-Instruct-Turbo", "response_weak"),
    ("gemini-3-flash-preview", "response_gemini"),
]


def load_jsonl(path):
    return [json.loads(l) for l in path.open() if l.strip()]


def load_cells_for_sids(target_sids):
    """Returns list of (sid, scen, gen, user_query, response) for all 4 generators."""
    cells = []
    for r in load_jsonl(EXISTING_RESPONSES):
        sid = r.get("statement_id")
        if sid not in target_sids: continue
        for label, col in GEN_KEYS:
            text = r.get(col)
            if text:
                cells.append((sid, r["scenario_idx"], label, r["user_query"], text))
    for r in load_jsonl(OPPOSITE_RESPONSES):
        if "error" in r: continue
        sid = r.get("statement_id")
        if sid not in target_sids: continue
        cells.append((sid, r["scenario_idx"], r["generator"], r["user_query"], r["response"]))
    cells.sort(key=lambda c: (c[0], c[1], c[2]))
    return cells


def build_prompt_bare(stmt, examples, user_q, response_text):
    return (
        f"SPEC STATEMENT:\n{stmt['text']}\n\n"
        f"SPEC EXAMPLES:\n{render_examples(examples)}\n\n"
        f"USER QUERY: {user_q}\n\n"
        f"ASSISTANT RESPONSE: {response_text}\n\n"
        "Score per the schema."
    )


def build_prompt_p4(stmt, examples, rubric, user_q, response_text):
    return (
        f"SPEC STATEMENT:\n{stmt['text']}\n\n"
        f"SPEC EXAMPLES:\n{render_examples(examples)}\n\n"
        f"RUBRIC:\n{render_anchors(rubric)}\n\n"
        f"USER QUERY: {user_q}\n\n"
        f"ASSISTANT RESPONSE: {response_text}\n\n"
        "Score per the schema."
    )


def determine_condition_for_sid(sid):
    """Return ('C_RUBRIC' | 'C_EXAMPLES', spec_record, rubric_dict, examples_list)."""
    summary = json.loads((OUT_DIR / "run9_synthesis_summary.json").read_text())
    op_diag = summary["per_statement"][sid]["operative_diagnosis"]

    spec = {r["id"]: r for r in load_jsonl(SPEC_PATH)}
    rubrics_v1 = {r["statement_id"]: r["rubric"]
                  for r in load_jsonl(RUBRICS_V1_PATH) if "error" not in r}
    spec_record = spec[sid]
    v1_examples = get_examples(spec_record)

    if op_diag == "rubric_drift":
        # Use rubric_v9
        rubric_v9_path = OUT_DIR / sid / "rubric_v9.json"
        rubric_v9 = json.loads(rubric_v9_path.read_text())
        return "C_RUBRIC", spec_record, rubric_v9, v1_examples
    elif op_diag == "response_interpretation_disagreement":
        # Use v1 rubric + spec with appended examples
        spec_with_ex_path = OUT_DIR / sid / "spec_with_examples_v9.json"
        spec_with_ex = json.loads(spec_with_ex_path.read_text())
        new_examples = get_examples(spec_with_ex)
        return "C_EXAMPLES", spec_with_ex, rubrics_v1.get(sid, {}), new_examples
    else:
        return None, None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    args = ap.parse_args()

    summary = json.loads((OUT_DIR / "run9_synthesis_summary.json").read_text())
    # Find statements with adopted edits
    target_sids = []
    for sid, info in summary["per_statement"].items():
        if info["operative_diagnosis"] == "rubric_drift" and info["n_rubric_adopted"] > 0:
            target_sids.append(sid)
        elif info["operative_diagnosis"] == "response_interpretation_disagreement" and info["n_examples_adopted"] > 0:
            target_sids.append(sid)

    print(f"Run 9 Phase 3 (Round {args.round}) — judging {len(target_sids)} statements with adopted edits")
    for sid in target_sids:
        cond, _, _, _ = determine_condition_for_sid(sid)
        print(f"  {sid:40s}  condition: {cond}")

    cells = load_cells_for_sids(set(target_sids))
    print(f"Cells: {len(cells)} (expected ~{len(target_sids) * 80})")

    # Pre-build all (cell × condition) entries
    api_key_oai = os.environ["OPENAI_API_KEY"]
    api_key_anth = os.environ["ANTHROPIC_API_KEY"]
    api_key_gem = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]

    job_dir = Path(f"results/raw/e9_dart_run9_judge_r{args.round}/{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S')}")
    job_dir.mkdir(parents=True, exist_ok=True)

    # Build prompts — both variant_A and rubric_plus_spec for each cell
    cmap_gpt = {}
    cmap_cla = {}
    gpt_input = []
    cla_reqs = []
    gem_jobs = []  # (sid, scen, gen, cond, condition_label, prompt_a, prompt_p4)

    for sid in target_sids:
        cond_label, spec_record, rubric, examples = determine_condition_for_sid(sid)
        if cond_label is None:
            continue
        sid_cells = [c for c in cells if c[0] == sid]
        for sid2, scen, gen, uq, resp in sid_cells:
            # variant_A judge call
            user_a = build_prompt_bare(spec_record, examples, uq, resp)
            user_p = build_prompt_p4(spec_record, examples, rubric, uq, resp)
            for cond_name, system, user in [("variant_A", JUDGE_A_SYSTEM, user_a),
                                            ("rubric_plus_spec", JUDGE_RUBRIC_PLUS_SPEC_SYSTEM, user_p)]:
                # GPT batch
                cid_g = "j_" + hashlib.md5(f"r{args.round}::{sid}::{cond_name}::{scen}::{gen}::gpt".encode()).hexdigest()[:32]
                cmap_gpt[cid_g] = (sid, cond_name, scen, gen, cond_label)
                gpt_input.append({
                    "custom_id": cid_g,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-5.1",
                        "messages": [{"role": "system", "content": system},
                                     {"role": "user", "content": user}],
                        "temperature": 0,
                        "max_completion_tokens": 1500,
                        "reasoning_effort": "none",
                        "response_format": {"type": "json_object"},
                    },
                })
                # Claude batch
                cid_c = "j_" + hashlib.md5(f"r{args.round}::{sid}::{cond_name}::{scen}::{gen}::cla".encode()).hexdigest()[:32]
                cmap_cla[cid_c] = (sid, cond_name, scen, gen, cond_label)
                cla_reqs.append(ba.build_request(
                    custom_id=cid_c, model=ANTHROPIC_MODEL,
                    system=system, messages=[{"role": "user", "content": user}],
                    max_tokens=1500, tools=[JUDGMENT_TOOL_1_5],
                    tool_choice={"type": "tool", "name": "submit_judgment"},
                    thinking={"type": "disabled"}, temperature=0,
                ))
                # Gemini sync (batched into job list)
                gem_jobs.append((sid, scen, gen, cond_name, cond_label, system, user))

    print(f"\nBatch sizes: GPT {len(gpt_input)}, Claude {len(cla_reqs)}, Gemini {len(gem_jobs)}")

    state = {"round": args.round,
             "submitted_at_iso": datetime.now(timezone.utc).isoformat(),
             "job_dir": str(job_dir), "target_sids": target_sids,
             "batches": {}}

    # Submit GPT batch
    print("\n=== Submit GPT batch ===")
    oai = OpenAI(api_key=api_key_oai)
    gpt_input_path = job_dir / "input_gpt.jsonl"
    with gpt_input_path.open("w") as f:
        for r in gpt_input:
            f.write(json.dumps(r) + "\n")
    with gpt_input_path.open("rb") as fh:
        up = oai.files.create(file=fh, purpose="batch")
    gpt_batch = oai.batches.create(input_file_id=up.id, endpoint="/v1/chat/completions",
                                    completion_window="24h",
                                    metadata={"description": f"DART Run 9 R{args.round} judge GPT"})
    state["batches"]["gpt"] = {"batch_id": gpt_batch.id, "input_file_id": up.id,
                                "n_requests": len(gpt_input), "custom_id_map": cmap_gpt}
    (job_dir / "custom_id_map_gpt.json").write_text(json.dumps(cmap_gpt, indent=2))
    print(f"  submitted {gpt_batch.id}")

    # Submit Claude batch
    print("\n=== Submit Claude batch ===")
    cla_result = ba.submit(api_key_anth, cla_reqs, job_dir=job_dir,
                            name=f"run9_judge_r{args.round}_cla")
    state["batches"]["claude"] = {"batch_id": cla_result["batch_id"],
                                   "name": f"run9_judge_r{args.round}_cla",
                                   "n_requests": len(cla_reqs), "custom_id_map": cmap_cla}
    (job_dir / "custom_id_map_cla.json").write_text(json.dumps(cmap_cla, indent=2))
    print(f"  submitted {cla_result['batch_id']}")

    # Run Gemini sync (in parallel with batches)
    print("\n=== Gemini Pro sync ===")
    gem = genai.Client(api_key=api_key_gem)
    log = RawAPILogger(f"e9_dart_run9_r{args.round}_judge_gem")
    gem_results = []

    def call_gem(job):
        sid, scen, gen, cond_name, cond_label, system, user = job
        cfg = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            max_output_tokens=1500, temperature=0,
            thinking_config=types.ThinkingConfig(thinking_level="low"),
        )
        try:
            raw = log.call(role=f"r{args.round}_judge_gem_{cond_name}",
                           key={"sid": sid, "scen": scen, "gen": gen, "cond": cond_name},
                           fn=lambda: gem.models.generate_content(
                               model="gemini-3.1-pro-preview", contents=user, config=cfg))
            text = raw.text or ""
            if text.startswith("```"):
                text = text.split("```")[1].lstrip("json\n").strip()
            data = json.loads(text) if text.strip() else {}
            score = data.get("score")
            try:
                score = int(score) if score is not None else None
                if score is not None and not 1 <= score <= 5:
                    score = None
            except (TypeError, ValueError):
                score = None
            return {"judge": "gemini-pro", "statement_id": sid, "scenario_idx": scen,
                    "generator": gen, "condition": cond_name, "v9_condition": cond_label,
                    "score": score, "reasoning": data.get("reasoning")}
        except Exception as e:
            return {"judge": "gemini-pro", "statement_id": sid, "scenario_idx": scen,
                    "generator": gen, "condition": cond_name, "v9_condition": cond_label,
                    "error": str(e)[:200]}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=32) as ex:
        for r in ex.map(call_gem, gem_jobs):
            gem_results.append(r)
    print(f"  Gemini sync done: {len(gem_results)} rows in {time.time()-t0:.0f}s")

    gem_path = OUT_DIR / f"per_judgment_run9_r{args.round}_gem.jsonl"
    gem_path.write_text("\n".join(json.dumps(r) for r in gem_results))
    print(f"  wrote {gem_path}")

    state_path = OUT_DIR / f"run9_judge_r{args.round}_batches.json"
    state_path.write_text(json.dumps(state, indent=2))
    print(f"\nWrote {state_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
