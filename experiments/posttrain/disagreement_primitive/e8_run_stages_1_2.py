# ruff: noqa: E501
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run only stages 1 (compile rubrics) and 2 (generate scenarios) of E8.

Lets Ahmed review the rubrics and scenarios before committing to the larger spend
on stages 3-5 (generators + 2 × judging). After this finishes, run
`e8_paired_indirection.py` to continue — it will skip stages 1+2 since their
structured outputs already exist on disk.

Cost: ~$3-4. Wall time: ~10 min. 92 API calls total (46 compile + 46 scenario_gen).
"""

from __future__ import annotations
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from raw_api_logger import RawAPILogger
from e8_paired_indirection import (
    DIR,
    EXPERIMENT_NAME,
    load_spec,
    make_clients,
    stage1_compile_rubrics,
    stage2_generate_scenarios,
)


def review_rubrics(rubrics_path: Path) -> None:
    print("\n" + "=" * 70)
    print("RUBRIC REVIEW")
    print("=" * 70)
    rows = [json.loads(line) for line in rubrics_path.open() if line.strip()]
    ok = [r for r in rows if "error" not in r]
    err = [r for r in rows if "error" in r]
    print(f"  total: {len(rows)} (ok: {len(ok)}, errored: {len(err)})")
    if err:
        print(f"  errored statements: {[r['statement_id'] for r in err]}")

    # Quality checks — new schema: rubric = {anchors: {1..5: {criterion, reasoning, spec_quotes, example_refs}}}
    bad_shape = []
    for r in ok:
        rb = r.get("rubric") or {}
        anchors = rb.get("anchors") or {}
        if sorted(anchors.keys()) != ["1", "2", "3", "4", "5"]:
            bad_shape.append((r["statement_id"], f"anchors keys = {sorted(anchors.keys())}"))
            continue
        for k in ("1", "2", "3", "4", "5"):
            entry = anchors[k]
            if not isinstance(entry, dict) or set(entry.keys()) != {"criterion", "reasoning", "spec_quotes", "example_refs"}:
                bad_shape.append((r["statement_id"], f"anchor[{k}] fields = {sorted((entry or {}).keys()) if isinstance(entry, dict) else type(entry).__name__}"))
                break
    print(f"  rubrics with correct shape: {len(ok) - len(bad_shape)}/{len(ok)}")
    if bad_shape:
        for sid, diag in bad_shape[:10]:
            print(f"    BAD  {sid}: {diag}")

    # Criterion length distribution
    crit_lens = []
    for r in ok:
        anchors = r.get("rubric", {}).get("anchors") or {}
        for entry in anchors.values():
            if isinstance(entry, dict):
                crit_lens.append(len(entry.get("criterion", "")))
    if crit_lens:
        crit_lens.sort()
        print(f"  criterion length: min={min(crit_lens)}, p25={crit_lens[len(crit_lens)//4]}, "
              f"median={crit_lens[len(crit_lens)//2]}, p75={crit_lens[3*len(crit_lens)//4]}, max={max(crit_lens)}")

    # spec_quotes coverage in anchors
    quote_counts = Counter()
    for r in ok:
        anchors = r.get("rubric", {}).get("anchors") or {}
        for entry in anchors.values():
            if isinstance(entry, dict):
                quote_counts[len(entry.get("spec_quotes") or [])] += 1
    print(f"  spec_quotes per anchor: {dict(quote_counts)}")

    # Print 1 full sample rubric for inspection
    sample = next((r for r in ok if r["statement_id"] == "do_not_lie"), ok[0] if ok else None)
    if sample:
        anchors = sample["rubric"]["anchors"]
        print(f"\n--- SAMPLE RUBRIC: {sample['statement_id']} ---")
        for k in ("1", "2", "3", "4", "5"):
            entry = anchors.get(k, {})
            print(f"\n[anchor {k}]")
            print(f"  criterion:    {entry.get('criterion', '')}")
            print(f"  reasoning:    {entry.get('reasoning', '')}")
            print(f"  spec_quotes:  {entry.get('spec_quotes', [])}")
            print(f"  example_refs: {entry.get('example_refs', [])}")


def review_scenarios(scenarios_path: Path) -> None:
    print("\n" + "=" * 70)
    print("SCENARIO REVIEW")
    print("=" * 70)
    rows = [json.loads(line) for line in scenarios_path.open() if line.strip()]
    ok = [r for r in rows if "error" not in r]
    err = [r for r in rows if "error" in r]
    print(f"  total statements: {len(rows)} (ok: {len(ok)}, errored: {len(err)})")

    # Per-statement scenario counts
    cnt_dist = Counter(len(r.get("scenarios") or []) for r in ok)
    print(f"  scenarios per statement: {dict(cnt_dist)}")

    # Aggregate query lengths
    qlens = [len(s.get("user_query", "")) for r in ok for s in (r.get("scenarios") or [])]
    if qlens:
        qs = sorted(qlens)
        print(f"  query length: min={min(qs)}, p25={qs[len(qs)//4]}, median={qs[len(qs)//2]}, p75={qs[3*len(qs)//4]}, max={max(qs)}")
        print(f"  total scenarios: {len(qs)}")
        truncation_suspicions = sum(1 for q in qs if q == 200)
        print(f"  scenarios at exactly 200 chars (truncation indicator): {truncation_suspicions}")

    # Print 1 full sample (3 scenarios) for inspection
    sample = next((r for r in ok if r["statement_id"] == "do_not_lie"), ok[0] if ok else None)
    if sample:
        print(f"\n--- SAMPLE SCENARIOS: {sample['statement_id']} (showing 3 of {len(sample['scenarios'])}) ---")
        for i, sc in enumerate(sample["scenarios"][:3]):
            print(f"\n[{i}] (len={len(sc.get('user_query',''))}):")
            print(f"  {sc.get('user_query', '')}")


def main() -> int:
    log = RawAPILogger(EXPERIMENT_NAME)
    print(f"raw run dir: {log.run_dir}")
    spec = load_spec()
    oai, _, _ = make_clients()  # only OpenAI is needed for stages 1+2

    rubrics_path = DIR / "e8_rubrics.jsonl"
    scenarios_path = DIR / "e8_scenarios.jsonl"

    t0 = time.time()
    stage1_compile_rubrics(log, oai, spec, rubrics_path)
    stage2_generate_scenarios(log, oai, spec, scenarios_path)
    print(f"\nstages 1-2 complete in {time.time() - t0:.1f}s")
    print(f"  rubrics:   {rubrics_path}")
    print(f"  scenarios: {scenarios_path}")
    print(f"  raw dir:   {log.run_dir}")

    if rubrics_path.exists():
        review_rubrics(rubrics_path)
    if scenarios_path.exists():
        review_scenarios(scenarios_path)

    print("\n" + "=" * 70)
    print("REVIEW INSTRUCTIONS")
    print("=" * 70)
    print(f"  Inspect: {rubrics_path}")
    print(f"           {scenarios_path}")
    print(f"  Raw audit: {log.run_dir}")
    print("  If they look good, run `e8_paired_indirection.py` to continue with stages 3-6.")
    print("  (Stages 1+2 will skip since their outputs exist.)")
    print("  If anything is off, delete the .jsonl files and re-run this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
