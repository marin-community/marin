"""Fill missing Claude baseline judgments — for the 38 of 46 statements where Claude
never judged. Match the existing baseline cell universe: grok-opposite generator only,
20 scenarios per statement.

Output: appends rows to per_judgment_opposite.jsonl with judge="claude" matching the
existing schema for the 8 statements that already have Claude data.
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
OPPOSITE_RESPONSES = DIR / "e9_opposite_mode_responses.jsonl"
RUBRICS_V1_PATH = DIR / "e8_rubrics_v1.jsonl"
OUT_PATH = DIR / "per_judgment_opposite.jsonl"
BATCH_TRACKER = DIR / "claude_baseline_fill_batches.json"


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
    ap.add_argument("--statements", default="auto",
                    help="'auto' = identify missing-Claude statements; comma-sep list otherwise")
    args = ap.parse_args()

    spec = {r["id"]: r for r in load_jsonl(SPEC_PATH)}
    rubrics = {r["statement_id"]: r["rubric"]
               for r in load_jsonl(RUBRICS_V1_PATH) if "error" not in r}

    # Identify which statements lack Claude data
    existing = load_jsonl(OUT_PATH)
    has_claude = {r["statement_id"] for r in existing
                  if r.get("judge") == "claude" and r.get("score") is not None}
    if args.statements == "auto":
        target_sids = [sid for sid in spec.keys() if sid not in has_claude]
    else:
        target_sids = args.statements.split(",")
    print(f"Filling Claude on {len(target_sids)} statements (missing from {len(has_claude)} existing).")

    # Load grok-opposite responses (20 per statement) for the target sids
    cells = []
    for r in load_jsonl(OPPOSITE_RESPONSES):
        if "error" in r: continue
        sid = r.get("statement_id")
        if sid not in target_sids: continue
        cells.append((sid, r["scenario_idx"], r["generator"], r["user_query"], r["response"]))
    cells.sort(key=lambda c: (c[0], c[1]))
    print(f"Cells to judge: {len(cells)} (× 2 conditions = {len(cells)*2} judgments)")

    # Already-done dedupe (idempotent)
    existing_keys = {(r["statement_id"], r["scenario_idx"], r["generator"], r["condition"])
                     for r in existing
                     if r.get("judge") == "claude" and r.get("score") is not None}

    api_key = os.environ["ANTHROPIC_API_KEY"]
    job_dir = Path(f"results/raw/e9_dart_claude_baseline_fill/{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S')}")
    job_dir.mkdir(parents=True, exist_ok=True)

    custom_id_map = {}
    state = {"submitted_at_iso": datetime.now(timezone.utc).isoformat(),
             "job_dir": str(job_dir), "batches": {},
             "n_target_statements": len(target_sids)}

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
            raw_id = f"baseline::{sid}::{cond}::{scen}::{gen}"
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
        name = f"claude_baseline_fill_{cond}"
        result = ba.submit(api_key, reqs, job_dir=job_dir, name=name)
        state["batches"][cond] = {
            "batch_id": result["batch_id"], "name": name, "n_requests": len(reqs),
            "submitted_at": time.time(),
        }
        print(f"  {cond}: submitted batch {result['batch_id']} ({len(reqs)} reqs)")

    cmap_path = job_dir / "custom_id_map.json"
    cmap_path.write_text(json.dumps(custom_id_map, indent=2))
    BATCH_TRACKER.write_text(json.dumps(state, indent=2))
    print(f"\nWrote batch tracker: {BATCH_TRACKER}")
    print(f"Custom ID map: {cmap_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
