"""DART Pro-judge audit: re-judge all 46 statements × all generators × both conditions
with gemini-3.1-pro-preview as a stronger replacement for gemini-3-flash-preview judge.

Tests the hypothesis that "Gemini-judge degeneracy" we kept seeing across DART runs
(scoring 5 in 67-79/80 cells) was Flash's resolution limit, not a Gemini-family bias.

Configuration per dart.md Gotcha 17:
  - model: gemini-3.1-pro-preview
  - thinking_level="low" (minimal not supported on Pro)
  - temperature=0
  - response_mime_type="application/json"
  - safety_settings: BLOCK_NONE

Two conditions per cell:
  - variant_A: spec text + examples + scenario + response (no rubric) — JUDGE_A_SYSTEM
  - rubric_plus_spec: spec + examples + rubric + scenario + response — JUDGE_RUBRIC_PLUS_SPEC_SYSTEM

Output: per_judgment_pro_audit.jsonl (judge="gemini-pro" to distinguish from Flash data).

Idempotent: dedupe on (sid, scen, gen, condition) before submitting.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).parent))
from e8_paired_indirection import (
    SPEC_PATH, get_examples, render_anchors, render_examples,
    JUDGE_A_SYSTEM,
)
from e8_phase4_rubric_plus_spec import JUDGE_RUBRIC_PLUS_SPEC_SYSTEM
from raw_api_logger import RawAPILogger

DIR = Path("experiments/posttrain/disagreement_primitive")
RUBRICS_V1_PATH = DIR / "e8_rubrics_v1.jsonl"
EXISTING_RESPONSES = DIR / "e8_responses.jsonl"
OPPOSITE_RESPONSES = DIR / "e9_opposite_mode_responses.jsonl"
OUT_PATH = DIR / "per_judgment_pro_audit.jsonl"

GEMINI_MODEL = "gemini-3.1-pro-preview"
MAX_TOKENS = 1500

GEN_KEYS = [
    ("gpt-5.1", "response_gpt"),
    ("Qwen/Qwen2.5-7B-Instruct-Turbo", "response_weak"),
    ("gemini-3-flash-preview", "response_gemini"),
]

_GEMINI_SAFETY_BLOCK_NONE = [
    types.SafetySetting(category=c, threshold=types.HarmBlockThreshold.BLOCK_NONE)
    for c in (
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        types.HarmCategory.HARM_CATEGORY_HARASSMENT,
    )
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def parse_json_strict(text: str) -> dict:
    s = (text or "").strip()
    if s.startswith("```"):
        nl = s.find("\n")
        if nl != -1:
            first = s[:nl].strip("`").strip()
            if first == "" or first.lower() == "json":
                s = s[nl + 1:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()
    return json.loads(s)


def build_user_prompt_bare(stmt: dict, examples: list, user_q: str, response_text: str) -> str:
    """variant_A: spec statement + examples + scenario + response (no rubric)."""
    return (
        f"SPEC STATEMENT:\n{stmt['text']}\n\n"
        f"SPEC EXAMPLES:\n{render_examples(examples)}\n\n"
        f"USER QUERY: {user_q}\n\n"
        f"ASSISTANT RESPONSE: {response_text}\n\n"
        "Score per the schema."
    )


def build_user_prompt_phase4(stmt: dict, examples: list, rubric: dict,
                             user_q: str, response_text: str) -> str:
    """rubric_plus_spec: spec + examples + rubric + scenario + response."""
    return (
        f"SPEC STATEMENT:\n{stmt['text']}\n\n"
        f"SPEC EXAMPLES:\n{render_examples(examples)}\n\n"
        f"RUBRIC:\n{render_anchors(rubric)}\n\n"
        f"USER QUERY: {user_q}\n\n"
        f"ASSISTANT RESPONSE: {response_text}\n\n"
        "Score per the schema."
    )


def call_gemini_pro_judge(client: genai.Client, log: RawAPILogger,
                          role: str, key: dict, system: str, user: str,
                          max_retries: int = 3) -> dict:
    cfg = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=MAX_TOKENS,
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_level="low"),
        response_mime_type="application/json",
        safety_settings=_GEMINI_SAFETY_BLOCK_NONE,
    )
    last_err = None
    for attempt in range(max_retries):
        try:
            raw = log.call(role=role, key={**key, "attempt": attempt},
                           fn=lambda: client.models.generate_content(
                               model=GEMINI_MODEL, contents=user, config=cfg))
            return parse_json_strict(raw.text or "")
        except Exception as e:
            msg = str(e)
            last_err = msg[:300]
            # 429 rate limit → exponential backoff
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                wait = 2 ** attempt + (attempt * 0.5)
                time.sleep(wait)
                continue
            # JSON parse error or transient — quick retry
            time.sleep(0.5 * (attempt + 1))
            continue
    raise RuntimeError(f"3 retries failed: {last_err}")


def build_cells() -> list[tuple]:
    """Returns list of (sid, scenario_idx, generator, user_query, response_text)."""
    cells = []
    for r in load_jsonl(EXISTING_RESPONSES):
        sid = r.get("statement_id")
        if not sid: continue
        for label, col in GEN_KEYS:
            text = r.get(col)
            if text:
                cells.append((sid, r["scenario_idx"], label, r["user_query"], text))
    if OPPOSITE_RESPONSES.exists():
        for r in load_jsonl(OPPOSITE_RESPONSES):
            if "error" in r: continue
            sid = r.get("statement_id")
            if not sid: continue
            cells.append((sid, r["scenario_idx"], r["generator"], r["user_query"], r["response"]))
    cells.sort(key=lambda c: (c[0], c[1], c[2]))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-workers", type=int, default=64)
    ap.add_argument("--statements", default="all", help="comma-separated sids or 'all'")
    ap.add_argument("--conditions", default="variant_A,rubric_plus_spec")
    args = ap.parse_args()

    spec = {r["id"]: r for r in load_jsonl(SPEC_PATH)}
    rubrics = {r["statement_id"]: r["rubric"] for r in load_jsonl(RUBRICS_V1_PATH) if "error" not in r}

    sids_filter = None if args.statements == "all" else set(args.statements.split(","))
    cells = build_cells()
    if sids_filter:
        cells = [c for c in cells if c[0] in sids_filter]
    conditions = [c.strip() for c in args.conditions.split(",")]

    # Build work units: (sid, scen, gen, condition)
    work = []
    for sid, scen, gen, uq, resp in cells:
        for cond in conditions:
            work.append((sid, scen, gen, cond, uq, resp))

    # Idempotency: skip already-done
    existing_keys = set()
    if OUT_PATH.exists():
        for r in load_jsonl(OUT_PATH):
            if r.get("score") is not None or "error" not in r:
                existing_keys.add((r.get("statement_id"), r.get("scenario_idx"),
                                   r.get("generator"), r.get("condition")))
    work = [w for w in work if (w[0], w[1], w[2], w[3]) not in existing_keys]

    print(f"DART Pro-judge audit")
    print(f"  model: {GEMINI_MODEL}, thinking_level=low, temperature=0")
    print(f"  statements: {len(set(c[0] for c in cells))}")
    print(f"  cells: {len(cells)}")
    print(f"  conditions: {conditions}")
    print(f"  total work units: {len(work)} (after dedupe; {len(existing_keys)} already done)")
    print(f"  max_workers: {args.max_workers}")
    print(f"  output: {OUT_PATH}\n")

    if not work:
        print("Nothing to do. Exiting.")
        return 0

    client = genai.Client(api_key=(os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]))
    log = RawAPILogger("e9_dart_pro_judge_audit")
    print(f"  raw log dir: {log.run_dir}\n")

    def process(wu):
        sid, scen, gen, cond, uq, resp = wu
        if sid not in spec:
            return {"judge": "gemini-pro", "statement_id": sid, "scenario_idx": scen,
                    "generator": gen, "condition": cond, "error": "spec_not_found"}
        stmt = spec[sid]
        examples = get_examples(stmt)
        if cond == "variant_A":
            user = build_user_prompt_bare(stmt, examples, uq, resp)
            system = JUDGE_A_SYSTEM
        elif cond == "rubric_plus_spec":
            if sid not in rubrics:
                return {"judge": "gemini-pro", "statement_id": sid, "scenario_idx": scen,
                        "generator": gen, "condition": cond, "error": "rubric_v1_not_found"}
            user = build_user_prompt_phase4(stmt, examples, rubrics[sid], uq, resp)
            system = JUDGE_RUBRIC_PLUS_SPEC_SYSTEM
        else:
            return {"judge": "gemini-pro", "statement_id": sid, "scenario_idx": scen,
                    "generator": gen, "condition": cond, "error": f"unknown_condition_{cond}"}

        try:
            data = call_gemini_pro_judge(
                client, log,
                role=f"pro_judge_{cond}",
                key={"sid": sid, "scen": scen, "gen": gen, "cond": cond},
                system=system, user=user,
            )
            score = data.get("score")
            try:
                score = int(score) if score is not None else None
                if score is not None and not 1 <= score <= 5:
                    score = None
            except (TypeError, ValueError):
                score = None
            return {
                "judge": "gemini-pro",
                "statement_id": sid, "scenario_idx": scen,
                "generator": gen, "condition": cond,
                "score": score,
                "reasoning": data.get("reasoning"),
                "spec_quotes": data.get("spec_quotes") or [],
                "rubric_quotes": data.get("rubric_quotes") or [],
                "example_refs": data.get("example_refs") or [],
            }
        except Exception as e:
            return {"judge": "gemini-pro", "statement_id": sid, "scenario_idx": scen,
                    "generator": gen, "condition": cond, "error": str(e)[:300]}

    t0 = time.time()
    n_done = 0
    n_err = 0
    n_scored = 0
    BATCH_SIZE = 50  # flush to disk every 50 results

    pending_writes = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(process, wu): wu for wu in work}
        for fut in as_completed(futs):
            r = fut.result()
            pending_writes.append(r)
            n_done += 1
            if "error" in r:
                n_err += 1
            elif r.get("score") is not None:
                n_scored += 1
            if len(pending_writes) >= BATCH_SIZE:
                with OUT_PATH.open("a") as f:
                    for row in pending_writes:
                        f.write(json.dumps(row) + "\n")
                pending_writes = []
            if n_done % 100 == 0:
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (len(work) - n_done) / rate / 60 if rate > 0 else 0
                print(f"  [{n_done}/{len(work)}] scored={n_scored} err={n_err} "
                      f"elapsed={elapsed:.0f}s rate={rate:.1f}/s eta={eta:.1f}min")

    if pending_writes:
        with OUT_PATH.open("a") as f:
            for row in pending_writes:
                f.write(json.dumps(row) + "\n")

    elapsed = time.time() - t0
    print(f"\n=== DONE ===")
    print(f"  total: {n_done}, scored: {n_scored}, errors: {n_err}")
    print(f"  wall: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  out: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
