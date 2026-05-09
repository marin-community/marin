"""Fetch Claude batch results for DART iter round N and append to per_judgment_iter_round_{N}.jsonl.

Reads dart_iteration/round_{N}_batches.json, downloads results from Anthropic, maps
custom_ids back to (statement, condition, scenario, generator), extracts scores via
the submit_judgment tool, and appends to the per_judgment file.

Usage:
    source .env && source .env2 && .venv/bin/python e9_dart_iter_fetch_claude.py --round 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import batch_anthropic as ba

DIR = Path("experiments/posttrain/disagreement_primitive")
ITER_DIR = DIR / "dart_iteration"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True)
    args = ap.parse_args()

    api_key = os.environ["ANTHROPIC_API_KEY"]
    batches_path = ITER_DIR / f"round_{args.round}_batches.json"
    if not batches_path.exists():
        raise SystemExit(f"missing {batches_path}")
    info = json.loads(batches_path.read_text())

    out_path = ITER_DIR / f"per_judgment_iter_round_{args.round}.jsonl"
    job_dir = Path(info["job_dir"])

    # Read existing rows so we can dedupe
    existing_keys = set()
    if out_path.exists():
        for line in out_path.open():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("judge") == "claude":
                existing_keys.add((r.get("statement_id"), r.get("scenario_idx"),
                                   r.get("generator"), r.get("condition")))

    new_rows = []
    counts: Counter = Counter()
    for cond, batch_meta in info["batches"].items():
        name = batch_meta["name"]
        cmap_path = Path(batch_meta["custom_id_map"])
        cmap = json.loads(cmap_path.read_text())
        print(f"\n[{cond}] fetching {name} (batch_id={batch_meta['batch_id']})")
        try:
            entries = ba.collect(api_key, job_dir, name=name)
        except Exception as exc:
            print(f"  ERROR fetching {name}: {exc}")
            continue
        print(f"  {len(entries)} result entries")

        for entry in entries:
            cid = entry.get("custom_id")
            mapping = cmap.get(cid)
            if mapping is None:
                counts["custom_id_not_in_map"] += 1
                continue
            sid, condition_from_map, scen, gen = mapping
            assert condition_from_map == cond, f"mismatch: {condition_from_map} vs {cond}"
            args_dict = ba.extract_tool_args(entry)
            if args_dict is None:
                counts["no_tool_args"] += 1
                continue
            score = args_dict.get("score")
            try:
                score = int(score) if score is not None else None
                if score is not None and not 1 <= score <= 5:
                    score = None
            except (TypeError, ValueError):
                score = None
            row_key = (sid, scen, gen, cond)
            if row_key in existing_keys:
                counts["dedupe_skip"] += 1
                continue
            new_rows.append({
                "judge": "claude",
                "statement_id": sid,
                "scenario_idx": scen,
                "generator": gen,
                "condition": cond,
                "score": score,
                "reasoning": args_dict.get("reasoning"),
                "spec_quotes": args_dict.get("spec_quotes") or [],
                "rubric_quotes": args_dict.get("rubric_quotes") or [],
                "example_refs": args_dict.get("example_refs") or [],
                "rubric_spec_tension": args_dict.get("rubric_spec_tension"),
                "rubric_version": "v2" if cond in ("C1", "C3") else "v1",
            })
            counts["scored"] += 1 if score is not None else 0
            counts["null_score"] += 0 if score is not None else 1

    # Append to file
    if new_rows:
        with out_path.open("a") as f:
            for r in new_rows:
                f.write(json.dumps(r) + "\n")
    print(f"\nAppended {len(new_rows)} new claude rows to {out_path}")
    print(f"Counters: {dict(counts)}")


if __name__ == "__main__":
    main()
