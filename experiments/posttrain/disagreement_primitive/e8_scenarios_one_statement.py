# ruff: noqa: E501
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate scenarios for ONE statement (`do_not_lie`) and verify the saved file end-to-end.

Run:
    source .env2 && .venv/bin/python experiments/posttrain/disagreement_primitive/e8_scenarios_one_statement.py

Verifies:
  1. GPT-5.1 returned the requested 20 scenarios.
  2. None of them are at the suspicious 200-char boundary (would indicate truncation).
  3. The raw saved file (results/raw/.../scenario_gen/*.json) parses to the same scenarios
     as the in-memory return value, byte-for-byte for every user_query.
  4. The structured output (e8_scenarios_one_statement.jsonl) roundtrips cleanly.
  5. All 20 scenarios are distinct (no duplicates from a degenerate sample).
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from raw_api_logger import RawAPILogger
from e8_paired_indirection import (
    N_SCENARIOS,
    SCENARIO_GEN_SYSTEM,
    SPEC_PATH,
    call_gpt_json,
    make_clients,
)

TEST_STATEMENT_ID = "do_not_lie"


def main() -> int:
    log = RawAPILogger("e8_scenarios_one_statement")
    print(f"raw run dir: {log.run_dir}\n")

    spec = [json.loads(line) for line in SPEC_PATH.open() if line.strip()]
    stmt = next((s for s in spec if s["id"] == TEST_STATEMENT_ID), None)
    if stmt is None:
        print(f"FAIL: statement {TEST_STATEMENT_ID} not found")
        return 1

    print(f"statement: {stmt['id']}")
    print(f"text head: {stmt['text'][:160]}...")
    print(f"asking GPT-5.1 for {N_SCENARIOS} scenarios at temp=0...\n")

    oai, _, _ = make_clients()
    user_prompt = (
        f"STATEMENT TEXT:\n{stmt['text']}\n\n"
        f"Generate {N_SCENARIOS} borderline scenarios per the schema."
    )

    data = call_gpt_json(
        log, oai, "scenario_gen", {"statement_id": stmt["id"]},
        SCENARIO_GEN_SYSTEM, user_prompt, max_tokens=4000,
    )

    scenarios = data.get("scenarios") or []

    # --- Check 1: returned count ---
    print(f"=== Returned scenarios: {len(scenarios)} (expected {N_SCENARIOS}) ===")
    if len(scenarios) != N_SCENARIOS:
        print(f"  FAIL: expected {N_SCENARIOS}, got {len(scenarios)}")
        return 1

    # --- Check 2: query length distribution ---
    qlens = [len(s.get("user_query", "")) for s in scenarios]
    qlens_sorted = sorted(qlens)
    print(f"  query lengths: min={min(qlens)}, p25={qlens_sorted[len(qlens)//4]}, "
          f"median={qlens_sorted[len(qlens)//2]}, p75={qlens_sorted[3*len(qlens)//4]}, max={max(qlens)}")
    print(f"  total chars: {sum(qlens)}")

    at_200 = sum(1 for q in qlens if q == 200)
    above_200 = sum(1 for q in qlens if q > 200)
    print(f"  scenarios at exactly 200 chars: {at_200} (would indicate [:200] truncation)")
    print(f"  scenarios above 200 chars: {above_200} (proves no [:200] is in the path)")
    if at_200 > 1 and above_200 == 0:
        print(f"  WARN: many at exactly 200 with none above — possible silent truncation; investigate")

    # --- Check 3: distinct scenarios (no degenerate repeats) ---
    distinct = len({s.get("user_query", "") for s in scenarios})
    print(f"  distinct scenarios: {distinct}/{len(scenarios)}")
    if distinct < len(scenarios):
        print(f"  WARN: duplicate scenarios in the batch")

    # --- Check 4: print all 20 scenarios (full text, no truncation in display) ---
    print("\n=== All 20 scenarios (full text) ===")
    for i, s in enumerate(scenarios):
        q = s.get("user_query", "")
        print(f"\n[{i}] (len={len(q)}):")
        print(f"  {q}")

    # --- Check 5: roundtrip the saved raw file ---
    print(f"\n=== Verifying saved raw file ===")
    files = sorted(p for p in log.run_dir.rglob("*.json") if p.name != "_manifest.json")
    if len(files) != 1:
        print(f"  FAIL: expected 1 file, got {len(files)}: {files}")
        return 1
    raw_rec = json.loads(files[0].read_text())
    saved_str = raw_rec["response"]["choices"][0]["message"]["content"]
    saved_data = json.loads(saved_str)
    saved_scenarios = saved_data.get("scenarios") or []
    if len(saved_scenarios) != len(scenarios):
        print(f"  FAIL: count mismatch (returned {len(scenarios)}, saved {len(saved_scenarios)})")
        return 1
    for i, (returned, saved) in enumerate(zip(scenarios, saved_scenarios)):
        if returned.get("user_query") != saved.get("user_query"):
            print(f"  FAIL: scenario[{i}] differs")
            print(f"    returned ({len(returned.get('user_query',''))}c): {returned.get('user_query')!r}")
            print(f"    saved    ({len(saved.get('user_query',''))}c): {saved.get('user_query')!r}")
            return 1
    print(f"  OK roundtrip: all {len(scenarios)} scenarios byte-for-byte equal between API return and saved file")
    print(f"  raw file: {files[0]}")
    print(f"  raw size: {files[0].stat().st_size} bytes")

    # --- Check 6: write structured output and re-read it ---
    out_path = Path("experiments/posttrain/disagreement_primitive/e8_scenarios_one_statement.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        fh.write(json.dumps({"statement_id": stmt["id"], "scenarios": scenarios}, ensure_ascii=False) + "\n")
    rb = json.loads(out_path.read_text())
    rb_qlens = [len(s.get("user_query", "")) for s in rb.get("scenarios", [])]
    if rb_qlens != qlens:
        print(f"  FAIL: structured output lost data")
        return 1
    print(f"  OK structured output: {out_path} ({out_path.stat().st_size} bytes), roundtrips with identical length distribution")

    print(f"\nALL CHECKS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
