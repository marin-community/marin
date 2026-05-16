# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recover from e9_dart_run5_judge.py failures: submit Claude batches with correct API,
retry Gemini sync errors with fallback parsing.
"""

from __future__ import annotations
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from google import genai

sys.path.insert(0, str(Path(__file__).parent))
import batch_anthropic as ba
from e8_phase4_rubric_plus_spec import JUDGE_RUBRIC_PLUS_SPEC_SYSTEM
from e9_claude_judge import ANTHROPIC_MODEL
from e9_judge_opposite_mode import JUDGMENT_TOOL_1_5
from e9_dart_run5_judge import (
    SID,
    ITER_DIR,
    load_jsonl,
    load_responses,
    build_user_prompt,
    make_conditions,
    call_gemini_judge,
)
from raw_api_logger import RawAPILogger


def main():
    round_n = 1
    sid_dir = ITER_DIR / SID
    out_path = sid_dir / f"run5_per_judgment_round_{round_n}.jsonl"

    conditions = make_conditions(round_n)
    cells = load_responses()

    log = RawAPILogger(f"e9_dart_run5_round_{round_n}_judge_recover")
    print(f"raw log dir: {log.run_dir}")

    # ============ Retry Gemini sync errors (skip if no errors remain) ============
    existing = load_jsonl(out_path)
    err_keys = {
        (r["statement_id"], r["scenario_idx"], r["generator"], r["condition"])
        for r in existing
        if r.get("judge") == "gemini" and "error" in r
    }
    print(f"\nGemini errors to retry: {len(err_keys)}")
    if not err_keys:
        print("  (skipping retry — no errors remain)")

    gem = genai.Client(api_key=(os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]))
    retried = []

    def retry_gem_one(cell, cond):
        sid, scen, gen, uq, resp = cell
        if (sid, scen, gen, cond) not in err_keys:
            return None
        stmt_text, examples, rubric = conditions[cond]
        user = build_user_prompt(stmt_text, examples, rubric, uq, resp)
        # Try a couple times with progressively larger max_tokens / explicit "be brief"
        for attempt, max_t in [(1, 1500), (2, 2500), (3, 4000)]:
            try:
                data = call_gemini_judge(
                    gem,
                    log,
                    role=f"run5_round_{round_n}_judge_gem_{cond}_retry{attempt}",
                    key={"sid": sid, "cond": cond, "scen": scen, "gen": gen, "attempt": attempt},
                    system=JUDGE_RUBRIC_PLUS_SPEC_SYSTEM,
                    user=user,
                    max_tokens=max_t,
                )
                score = int(data["score"]) if data.get("score") is not None else None
                return {
                    "judge": "gemini",
                    "statement_id": sid,
                    "scenario_idx": scen,
                    "generator": gen,
                    "condition": cond,
                    "score": score,
                    "reasoning": data.get("reasoning"),
                }
            except Exception as e:
                last_err = str(e)[:150]
                continue
        return {
            "judge": "gemini",
            "statement_id": sid,
            "scenario_idx": scen,
            "generator": gen,
            "condition": cond,
            "error": f"3 retries failed: {last_err}",
        }

    if err_keys:
        print("Retrying...")
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = []
            for cond in conditions:
                for cell in cells:
                    futs.append(ex.submit(retry_gem_one, cell, cond))
            for f in as_completed(futs):
                r = f.result()
                if r:
                    retried.append(r)
        print(f"  retried: {len(retried)} in {time.time()-t0:.0f}s")
        n_recovered = sum(1 for r in retried if r.get("score") is not None)
        print(f"  recovered (now scored): {n_recovered}")

        # Append recovered rows; remove old error rows
        kept = [
            r
            for r in existing
            if not (
                r.get("judge") == "gemini"
                and "error" in r
                and (r["statement_id"], r["scenario_idx"], r["generator"], r["condition"]) in err_keys
            )
        ]
        kept.extend(retried)
        out_path.write_text("\n".join(json.dumps(r) for r in kept) + "\n")
        print(f"Wrote {out_path} ({len(kept)} total rows)")

    # ============ Submit Claude batches ============
    print("\n=== Submit Claude batches ===")
    anthropic_key = os.environ["ANTHROPIC_API_KEY"]
    job_dir = Path(
        f"results/raw/e9_dart_run5_claude_batch/round_{round_n}_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S')}"
    )
    job_dir.mkdir(parents=True, exist_ok=True)
    custom_id_map = {}
    batch_state = {
        "round": round_n,
        "conditions": list(conditions.keys()),
        "submitted_at_iso": datetime.now(timezone.utc).isoformat(),
        "job_dir": str(job_dir),
        "batches": {},
    }

    import hashlib

    for cond, (stmt_text, examples, rubric) in conditions.items():
        reqs = []
        for sid, scen, gen, uq, resp in cells:
            raw = f"r{round_n}::{sid}::{cond}::{scen}::{gen}"
            cid = "j_" + hashlib.md5(raw.encode()).hexdigest()
            custom_id_map[cid] = (sid, cond, scen, gen)
            user_text = build_user_prompt(stmt_text, examples, rubric, uq, resp)
            req = ba.build_request(
                custom_id=cid,
                model=ANTHROPIC_MODEL,
                system=JUDGE_RUBRIC_PLUS_SPEC_SYSTEM,
                messages=[{"role": "user", "content": user_text}],
                max_tokens=1500,
                tools=[JUDGMENT_TOOL_1_5],
                tool_choice={"type": "tool", "name": "submit_judgment"},
                thinking={"type": "disabled"},
                cache_user_prefix=ba.prefix_before(user_text),
            )
            reqs.append(req)
        name = f"run5_round_{round_n}_{cond}"
        result = ba.submit(anthropic_key, reqs, job_dir=job_dir, name=name)
        batch_state["batches"][cond] = {
            "batch_id": result["batch_id"],
            "name": name,
            "n_requests": len(reqs),
            "submitted_at": time.time(),
        }
        print(f"  {cond}: submitted {result['batch_id']} ({len(reqs)} reqs)")

    cmap_path = job_dir / "custom_id_map.json"
    cmap_path.write_text(json.dumps(custom_id_map, indent=2))

    pointer = sid_dir / f"run5_round_{round_n}_batches.json"
    pointer.write_text(json.dumps(batch_state, indent=2))
    print(f"Wrote pointer: {pointer}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
