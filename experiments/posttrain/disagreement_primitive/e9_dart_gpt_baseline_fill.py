"""Fill missing GPT-5.1 baseline judgments for the 3 non-grok generators
(gpt-5.1, Qwen/Qwen2.5-7B-Instruct-Turbo, gemini-3-flash-preview) across all 46
spec statements × 2 conditions, using OpenAI Batch API for cost.

Goal: bring GPT to match Gemini-3.1-Pro's 80-cell coverage so the canonical
3-judge ensemble can use the full 4-generator universe.

Uses:
  - reasoning_effort="none" (HARD project rule)
  - temperature=0
  - response_format={"type":"json_object"}
  - JUDGE_A_SYSTEM (variant_A) / JUDGE_RUBRIC_PLUS_SPEC_SYSTEM (phase_4)
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from e8_paired_indirection import (
    SPEC_PATH, get_examples, render_anchors, render_examples, JUDGE_A_SYSTEM,
)
from e8_phase4_rubric_plus_spec import JUDGE_RUBRIC_PLUS_SPEC_SYSTEM

DIR = Path("experiments/posttrain/disagreement_primitive")
EXISTING_RESPONSES = DIR / "e8_responses.jsonl"
RUBRICS_V1_PATH = DIR / "e8_rubrics_v1.jsonl"
OUT_PATH = DIR / "per_judgment_opposite.jsonl"
BATCH_TRACKER = DIR / "gpt_baseline_fill_batches.json"

GPT_MODEL = "gpt-5.1"
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

    # Existing GPT keys to skip (idempotency)
    existing = load_jsonl(OUT_PATH)
    existing_keys = {(r["statement_id"], r["scenario_idx"], r["generator"], r["condition"])
                     for r in existing
                     if r.get("judge") == "gpt" and r.get("score") is not None}

    # Build cells for the 3 non-grok generators only
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

    custom_id_map = {}
    requests_by_cond = {"variant_A": [], "rubric_plus_spec": []}

    for cond, system_prompt, build_fn in [
        ("variant_A", JUDGE_A_SYSTEM,
         lambda stmt, ex, rb, uq, resp: build_user_prompt_bare(stmt, ex, uq, resp)),
        ("rubric_plus_spec", JUDGE_RUBRIC_PLUS_SPEC_SYSTEM,
         lambda stmt, ex, rb, uq, resp: build_user_prompt_phase4(stmt, ex, rb, uq, resp)),
    ]:
        for sid, scen, gen, uq, resp in cells:
            if (sid, scen, gen, cond) in existing_keys:
                continue
            stmt = spec[sid]
            ex = get_examples(stmt)
            rb = rubrics.get(sid, {})
            user_text = build_fn(stmt, ex, rb, uq, resp)
            raw_id = f"gpt_baseline::{sid}::{cond}::{scen}::{gen}"
            cid = "j_" + hashlib.md5(raw_id.encode()).hexdigest()[:32]
            custom_id_map[cid] = (sid, cond, scen, gen)
            req = {
                "custom_id": cid,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": GPT_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_text},
                    ],
                    "temperature": 0,
                    "max_completion_tokens": 1500,
                    "reasoning_effort": "none",
                    "response_format": {"type": "json_object"},
                },
            }
            requests_by_cond[cond].append(req)

    total = sum(len(v) for v in requests_by_cond.values())
    print(f"Requests to submit: {total} ({len(requests_by_cond['variant_A'])} variant_A + "
          f"{len(requests_by_cond['rubric_plus_spec'])} rubric_plus_spec)")
    if args.dry_run or total == 0:
        return 0

    # Write batch input files
    job_dir = Path(f"results/raw/e9_dart_gpt_baseline_fill/{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S')}")
    job_dir.mkdir(parents=True, exist_ok=True)
    cmap_path = job_dir / "custom_id_map.json"
    cmap_path.write_text(json.dumps(custom_id_map, indent=2))

    oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    state = {"submitted_at_iso": datetime.now(timezone.utc).isoformat(),
             "job_dir": str(job_dir), "batches": {},
             "custom_id_map_path": str(cmap_path)}

    for cond, reqs in requests_by_cond.items():
        if not reqs: continue
        input_file = job_dir / f"input_{cond}.jsonl"
        with input_file.open("w") as f:
            for r in reqs:
                f.write(json.dumps(r) + "\n")
        # Upload input file
        with input_file.open("rb") as fh:
            up = oai.files.create(file=fh, purpose="batch")
        # Create batch (24h SLA)
        batch = oai.batches.create(
            input_file_id=up.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"DART gpt baseline fill {cond}"},
        )
        state["batches"][cond] = {
            "batch_id": batch.id, "input_file_id": up.id,
            "input_file": str(input_file), "n_requests": len(reqs),
            "submitted_at": time.time(),
        }
        print(f"  {cond}: submitted batch {batch.id} ({len(reqs)} reqs, file {up.id})")

    BATCH_TRACKER.write_text(json.dumps(state, indent=2))
    print(f"\nBatch tracker: {BATCH_TRACKER}")
    print(f"Custom ID map: {cmap_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
