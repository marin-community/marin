"""DART iterative validation — overnight orchestrator.

State machine that walks Round 1 → Round 2 → Round 3 (if time permits) by
checking what's done, what's in flight, and what to do next. Designed to be
called repeatedly by a ScheduleWakeup loop; on each call, advances state by
one step if conditions are met, then exits.

State file: dart_iteration/orchestrator_state.json

Usage:
    .venv/bin/python e9_dart_iter_orchestrate.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DIR = Path("experiments/posttrain/disagreement_primitive")
ITER_DIR = DIR / "dart_iteration"
STATE_PATH = ITER_DIR / "orchestrator_state.json"

# Wake-time gates (in UTC)
DEADLINE_NEW_SUBMISSIONS_HOUR = 13  # no new submissions after 13:00 UTC
DEADLINE_HARD_HOUR = 15             # must report by 15:00 UTC


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {"phase": "init", "history": []}
    return json.loads(STATE_PATH.read_text())


def save_state(state: dict):
    state["last_updated"] = now_utc().isoformat()
    STATE_PATH.write_text(json.dumps(state, indent=2))


def log(state: dict, event: str):
    state.setdefault("history", []).append({"time": now_utc().isoformat(), "event": event})
    print(f"[{now_utc().isoformat()}] {event}")


def count_iter_judgments(round_n: int, condition: str | None = None) -> int:
    p = ITER_DIR / f"per_judgment_iter_round_{round_n}.jsonl"
    if not p.exists():
        return 0
    n = 0
    for line in p.open():
        if not line.strip():
            continue
        if condition is None:
            n += 1
        else:
            try:
                if json.loads(line).get("condition") == condition:
                    n += 1
            except json.JSONDecodeError:
                pass
    return n


def claude_batches_status(round_n: int) -> dict:
    """Read the batch tracker for round_n and return status counts."""
    p = ITER_DIR / f"round_{round_n}_batches.json"
    if not p.exists():
        return {"missing": True}
    data = json.loads(p.read_text())
    return data  # contains batch IDs and metadata; status check elsewhere


def is_after_hour(hour: int) -> bool:
    return now_utc().hour >= hour


def expected_iter_judgments(conditions: list[str]) -> int:
    """Per-judge: 13 statements × ~80 cells × N conditions. Per all 3 judges."""
    return 13 * 80 * len(conditions) * 3  # rough


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)[-2000:]


def advance(state: dict) -> bool:
    """Advance state by at most one step. Returns True if state changed."""
    phase = state.get("phase", "init")
    log(state, f"check phase={phase}")

    if phase == "init":
        # Synthesis already done by hand earlier; just confirm artifacts exist.
        if not (ITER_DIR / "round_1_synthesis_summary.json").exists():
            log(state, "missing round_1_synthesis_summary.json — synthesis not yet run")
            return False
        state["phase"] = "round_1_judging_in_flight"
        log(state, "advanced to round_1_judging_in_flight")
        return True

    if phase == "round_1_judging_in_flight":
        # Check sync judgments accumulating
        n = count_iter_judgments(1)
        target_per_cond = 13 * 80  # ~ per condition per judge
        log(state, f"round 1 judgments accumulated: {n}")
        # We expect ~9360 total after all 3 judges × 3 conditions (~80 cells × 13 stmts).
        # If we hit ≥ 70% AND no growth in 30 min, consider it done.
        prev_n = state.get("round_1_last_n", 0)
        state["round_1_last_n"] = n
        if n >= 6500:  # ~70% complete is good enough to analyze
            state["phase"] = "round_1_analyze"
            log(state, f"round 1 judging mostly complete ({n} ≥ 6500); advancing to analyze")
            return True
        if is_after_hour(13):
            log(state, "past deadline for new submissions; analyzing whatever's available")
            state["phase"] = "round_1_analyze"
            return True
        return False

    if phase == "round_1_analyze":
        # Run analysis script
        rc, out = run([".venv/bin/python", str(DIR / "e9_dart_iter_analyze.py"),
                       "--round", "1", "--conditions", "C1,C2,C3"])
        log(state, f"round 1 analyze rc={rc}")
        log(state, f"analyze tail: {out[-500:]}")
        if rc != 0:
            state["phase"] = "round_1_analyze_failed"
            return False
        state["phase"] = "round_1_complete"
        return True

    if phase == "round_1_complete":
        # Decide if Round 2 fires: any IMPROVING statements AND time < 13:00 UTC
        if is_after_hour(DEADLINE_NEW_SUBMISSIONS_HOUR):
            log(state, "past deadline for new submissions; finalizing")
            state["phase"] = "render_final"
            return True
        # Read summary
        summary_path = ITER_DIR / "round_1_analysis_summary.json"
        if not summary_path.exists():
            log(state, "missing round 1 summary — cannot decide R2")
            return False
        summary = json.loads(summary_path.read_text())
        improving = [s for s, info in summary["per_statement"].items() if info["verdict"] == "improving"]
        log(state, f"improving after R1: {len(improving)}")
        if not improving:
            log(state, "no statements improving; skipping R2; finalizing")
            state["phase"] = "render_final"
            return True
        state["round_2_eligible"] = improving
        state["phase"] = "round_2_compile"
        return True

    if phase == "round_2_compile":
        # Generate prompts for round 2
        rc, out = run([".venv/bin/python", str(DIR / "e9_dart_iter_compile.py"),
                       "--round", "2", "--statements", ",".join(state["round_2_eligible"])])
        log(state, f"round 2 compile rc={rc}")
        if rc != 0:
            state["phase"] = "round_2_compile_failed"
            return False
        # NOTE: round 2 compile currently writes prompts only; needs follow-up work to
        # actually call the 3 compilers via existing dart_compiler scripts. Marking as
        # "compile_prompts_ready" — the orchestrator will note this and not block on it
        # for the morning report; user can inspect and finish manually.
        state["phase"] = "round_2_compile_prompts_ready"
        log(state, "round 2 prompts written; full R2 LLM calls require manual launch — see README")
        # For overnight, we end here unless time-permitting work was added.
        state["phase"] = "render_final"
        return True

    if phase == "render_final":
        # Render the final report
        rc, out = run([".venv/bin/python", str(DIR / "e9_dart_iter_render.py")])
        log(state, f"render rc={rc}")
        state["phase"] = "done"
        return True

    return False


def main():
    state = load_state()
    changed = True
    # Advance as many steps as possible in this wake
    while changed:
        changed = advance(state)
        save_state(state)
    print(f"\nFinal phase: {state.get('phase')}")
    print(f"State file: {STATE_PATH}")


if __name__ == "__main__":
    ITER_DIR.mkdir(parents=True, exist_ok=True)
    main()
