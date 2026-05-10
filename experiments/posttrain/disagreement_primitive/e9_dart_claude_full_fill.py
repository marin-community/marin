"""Fill missing Claude baseline judgments on the 3 non-grok generators
(gpt-5.1, Qwen, gemini-3-flash-preview) × all 46 statements × 2 conditions.

Mirrors e9_dart_gpt_baseline_fill.py but for Anthropic batch.
Target volume: 5,516 calls. Forecast cost via cost_estimate.py: ~$48.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import batch_anthropic as ba
from e8_paired_indirection import (
    SPEC_PATH, get_examples, render_anchors, render_examples, JUDGE_A_SYSTEM,
)
from e8_phase4_rubric_plus_spec import JUDGE_RUBRIC_PLUS_SPEC_SYSTEM
from e9_claude_judge import ANTHROPIC_MODEL
from e9_judge_opposite_mode import JUDGMENT_TOOL_1_5

DIR = Path("experiments/posttrain/disagreement_primitive")
EXISTING_RESPONSES = DIR / "e8_responses.jsonl"
RUBRICS_V1_PATH = DIR / "e8_rubrics_v1.jsonl"
OUT_PATH = DIR / "per_judgment_opposite.jsonl"
BATCH_TRACKER = DIR / "claude_full_fill_batches.json"

GEN_KEYS = [
    ("gpt-5.1", "response_gpt"),
    ("Qwen/Qwen2.5-7B-Instruct-Turbo", "response_weak"),
    ("gemini-3-flash-preview", "response_gemini"),
]


def load_jsonl(path):
    return [json.loads(l) for l in path.open() if l.strip()]


def build_user_prompt_bare(stmt, examples, user_q, response_text):
    return (
        f"SPEC STATEMENT:\n{stmt['text']}\n\n"
        f"SPEC EXAMPLES:\n{render_examples(examples)}\n\n"
        f"USER QUERY: {user_q}\n\n"
        f"ASSISTANT RESPONSE: {response_text}\n\n"
        "Score per the schema."
    )


def build_user_prompt_phase4(stmt, examples, rubric, user_q, response_text):
    return (
        f"SPEC STATEMENT:\n{stmt['text']}\n\n"
        f"SPEC EXAMPLES:\n{render_examples(examples)}\n\n"
        f"RUBRIC:\n{render_anchors(rubric)}\n\n"
        f"USER QUERY: {user_q}\n\n"
        f"ASSISTANT RESPONSE: {response_text}\n\n"
        "Score per the schema."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    spec = {r["id"]: r for r in load_jsonl(SPEC_PATH)}
    rubrics = {r["statement_id"]: r["rubric"]
               for r in load_jsonl(RUBRICS_V1_PATH) if "error" not in r}

    existing = load_jsonl(OUT_PATH)
    existing_keys = {(r["statement_id"], r["scenario_idx"], r["generator"], r["condition"])
                     for r in existing
                     if r.get("judge") == "claude" and r.get("score") is not None}

    # Build cells from non-grok generators
    cells = []
    for r in load_jsonl(EXISTING_RESPONSES):
        sid = r.get("statement_id")
        if not sid or sid not in spec: continue
        for label, col in GEN_KEYS:
            text = r.get(col)
            if text:
                cells.append((sid, r["scenario_idx"], label, r["user_query"], text))
    cells.sort(key=lambda c: (c[0], c[1], c[2]))
    print(f"Total non-grok cells: {len(cells)}")

    api_key = os.environ["ANTHROPIC_API_KEY"]
    job_dir = Path(f"results/raw/e9_dart_claude_full_fill/{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S')}")
    job_dir.mkdir(parents=True, exist_ok=True)

    custom_id_map = {}
    state = {"submitted_at_iso": datetime.now(timezone.utc).isoformat(),
             "job_dir": str(job_dir), "batches": {}}

    for cond, system_prompt, build_fn in [
        ("variant_A", JUDGE_A_SYSTEM,
         lambda stmt, ex, rb, uq, resp: build_user_prompt_bare(stmt, ex, uq, resp)),
        ("rubric_plus_spec", JUDGE_RUBRIC_PLUS_SPEC_SYSTEM,
         lambda stmt, ex, rb, uq, resp: build_user_prompt_phase4(stmt, ex, rb, uq, resp)),
    ]:
        reqs = []
        for sid, scen, gen, uq, resp in cells:
            if (sid, scen, gen, cond) in existing_keys:
                continue
            stmt = spec[sid]
            ex = get_examples(stmt)
            rb = rubrics.get(sid, {})
            user_text = build_fn(stmt, ex, rb, uq, resp)
            raw_id = f"claude_full::{sid}::{cond}::{scen}::{gen}"
            cid = "j_" + hashlib.md5(raw_id.encode()).hexdigest()
            custom_id_map[cid] = (sid, cond, scen, gen)
            req = ba.build_request(
                custom_id=cid,
                model=ANTHROPIC_MODEL,
                system=system_prompt,
                messages=[{"role": "user", "content": user_text}],
                max_tokens=1500,
                tools=[JUDGMENT_TOOL_1_5],
                tool_choice={"type": "tool", "name": "submit_judgment"},
                thinking={"type": "disabled"},
                temperature=0,
            )
            reqs.append(req)
        if not reqs:
            print(f"  {cond}: no work")
            continue
        if args.dry_run:
            print(f"  {cond}: would submit {len(reqs)} reqs")
            continue
        name = f"claude_full_fill_{cond}"
        result = ba.submit(api_key, reqs, job_dir=job_dir, name=name)
        state["batches"][cond] = {
            "batch_id": result["batch_id"], "name": name, "n_requests": len(reqs),
            "submitted_at": time.time(),
        }
        print(f"  {cond}: submitted {result['batch_id']} ({len(reqs)} reqs)")

    if args.dry_run:
        return 0

    cmap_path = job_dir / "custom_id_map.json"
    cmap_path.write_text(json.dumps(custom_id_map, indent=2))
    BATCH_TRACKER.write_text(json.dumps(state, indent=2))
    print(f"\nBatch tracker: {BATCH_TRACKER}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
