"""Generate opposite-mode responses for the 8 target statements via Grok 4-1.

For each (statement_id, scenario_idx) pair in the 8 target statements (20
scenarios each = 160 cells), generate one Grok response under the
opposite-mode system prompt (see opposite_mode.py).

Output: experiments/posttrain/disagreement_primitive/e9_opposite_mode_responses.jsonl
  {statement_id, scenario_idx, user_query, generator, response, _usage}

The generator label is `grok-4-1-fast-non-reasoning-opposite` (see
OPPOSITE_MODE_GENERATOR_LABEL in opposite_mode.py). Downstream judging
treats this as just another generator label.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from e8_paired_indirection import SPEC_PATH, load_jsonl, write_jsonl
from opposite_mode import (
    DEFAULT_OPPOSITE_MODEL,
    OPPOSITE_MODE_GENERATOR_LABEL,
    make_xai_client,
    opposite_mode_generate_logged,
)
from raw_api_logger import RawAPILogger

DIR = Path("experiments/posttrain/disagreement_primitive")
OUT_PATH = DIR / "e9_opposite_mode_responses.jsonl"
RESPONSES_PATH = DIR / "e8_responses.jsonl"

TARGET_SIDS = [
    "do_not_make_unprompted_personal_comments",
    "be_professional",
    "no_erotica_or_gore",
    "present_perspectives",
    "be_thorough_but_efficient",
    "avoid_hateful_content",
    "no_topic_off_limits",
    "be_clear",
]


def main() -> int:
    spec_by_id = {json.loads(line)["id"]: json.loads(line) for line in SPEC_PATH.open() if line.strip()}
    responses_jsonl = load_jsonl(RESPONSES_PATH)

    # Build (sid, scenario_idx, user_query) — one per scenario, 20 per statement
    seen: set[tuple[str, int]] = set()
    cells: list[tuple[str, int, str]] = []
    for r in responses_jsonl:
        sid = r.get("statement_id")
        if sid not in TARGET_SIDS:
            continue
        scen = r["scenario_idx"]
        if (sid, scen) in seen:
            continue
        seen.add((sid, scen))
        cells.append((sid, scen, r["user_query"]))
    cells.sort(key=lambda t: (t[0], t[1]))

    print(f"Opposite-mode generation")
    print(f"  generator: {DEFAULT_OPPOSITE_MODEL} ({OPPOSITE_MODE_GENERATOR_LABEL})")
    print(f"  statements: {len(TARGET_SIDS)}")
    print(f"  total cells: {len(cells)} (expected: {len(TARGET_SIDS) * 20})")

    client = make_xai_client()
    log = RawAPILogger("e9_opposite_mode_generation")
    print(f"  raw log dir: {log.run_dir}\n")

    def process_cell(cell: tuple[str, int, str]) -> dict[str, Any]:
        sid, scen, user_q = cell
        stmt = spec_by_id[sid]
        try:
            text = opposite_mode_generate_logged(
                log=log,
                client=client,
                role="opposite_mode",
                key={"statement_id": sid, "scenario_idx": scen,
                     "model": DEFAULT_OPPOSITE_MODEL},
                model=DEFAULT_OPPOSITE_MODEL,
                statement_text=stmt["text"],
                user_query=user_q,
                max_tokens=2000,
                temperature=1.0,
            )
            return {
                "statement_id": sid,
                "scenario_idx": scen,
                "user_query": user_q,
                "generator": OPPOSITE_MODE_GENERATOR_LABEL,
                "response": text,
            }
        except Exception as exc:
            return {
                "statement_id": sid,
                "scenario_idx": scen,
                "user_query": user_q,
                "generator": OPPOSITE_MODE_GENERATOR_LABEL,
                "error": f"{type(exc).__name__}: {exc}",
            }

    rows: list[dict[str, Any]] = []
    n_done = 0
    overall_t0 = time.time()
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(process_cell, c): c for c in cells}
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            n_done += 1
            if n_done % 20 == 0 or n_done == len(cells):
                elapsed = time.time() - overall_t0
                n_err = sum(1 for x in rows if "error" in x)
                print(f"  progress: {n_done}/{len(cells)} ({100*n_done/len(cells):.0f}%, {elapsed:.0f}s, errors={n_err})")

    rows.sort(key=lambda r: (r["statement_id"], r["scenario_idx"]))
    write_jsonl(rows, OUT_PATH)
    wall = time.time() - overall_t0

    n_ok = sum(1 for r in rows if "error" not in r)
    n_err = sum(1 for r in rows if "error" in r)
    print(f"\n== TOTALS ==")
    print(f"  cells: {n_ok}/{len(cells)} ok, {n_err} errors")
    print(f"  wall: {wall:.0f}s")
    print(f"  output: {OUT_PATH}")

    # Per-statement summary
    print()
    print("== Per-statement counts ==")
    from collections import Counter
    per_sid_ok = Counter()
    per_sid_total = Counter()
    avg_lens: dict[str, list[int]] = {sid: [] for sid in TARGET_SIDS}
    for r in rows:
        per_sid_total[r["statement_id"]] += 1
        if "error" not in r:
            per_sid_ok[r["statement_id"]] += 1
            avg_lens[r["statement_id"]].append(len(r["response"]))
    for sid in TARGET_SIDS:
        lens = avg_lens.get(sid, [])
        avg = sum(lens) // len(lens) if lens else 0
        print(f"  {sid:42s}  {per_sid_ok[sid]}/{per_sid_total[sid]}  avg_resp_len={avg}")

    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
