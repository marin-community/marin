# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-stage GLM phase_4 retry: primary repair + score-with-partial-reasoning fallback.

Wraps `e9_repair_glm_phase4.py`'s repair pipeline with a fallback stage
that handles the unrepairable-by-`repair_glm_json` records via
`score_and_reasoning_partial` from `e9_glm_json_score_extract`.

This script is forward-looking and opt-in only: production scripts and
the existing `e9_repair_glm_phase4.py` are unchanged. It exists because
the empirical recovery rate of `repair_glm_json` alone on the disk
corpus was 121/315 (38%) — far below the ~80% target — driven by a
secondary failure mode (`","quoted phrase","` corruption tails) that
the original strategies don't catch.

INPUTS / OUTPUTS
----------------
Same CLI as `e9_repair_glm_phase4.py`:

    --raw-dir PATH/TO/RUN/role/         (RawAPILogger directory)
    --raw-jsonl PATH/TO/judgments.jsonl (flat jsonl with raw_text)
    --out-dir PATH                      (where to write outputs)

Outputs:

    <out_dir>/repaired_judgments.jsonl
        All recovered records (valid + repaired + partial-extract).
        Records carry `_repair_strategy`, `_partial_parse`, and (for
        partial extracts) `_partial_extract_truncated_at`.
    <out_dir>/repair_summary.json
        Aggregate counts.

USAGE
-----
    .venv/bin/python experiments/posttrain/disagreement_primitive/e9_repair_glm_phase4_v2.py \
        --raw-dir results/raw/e8_phase4_glm/2026-05-06T01-02-07/judge_rubric_plus_spec_glm \
        --out-dir experiments/posttrain/disagreement_primitive/phase4_glm_repaired_v2/
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from e9_glm_json_repair import repair_glm_json
from e9_glm_json_score_extract import score_and_reasoning_partial
from e9_repair_glm_phase4 import _iter_from_jsonl, _iter_from_raw_dir


def _strategy_for(raw_text: str) -> tuple[dict[str, Any] | None, str, bool, int | None]:
    """Try primary repair, then partial-extract fallback. Returns (data,
    strategy, partial, truncated_at)."""
    primary = repair_glm_json(raw_text)
    if primary.data is not None:
        return primary.data, primary.strategy, primary.partial, None
    if primary.strategy == "empty_body":
        return None, "empty_body", False, None
    fallback = score_and_reasoning_partial(raw_text)
    if fallback.ok:
        return fallback.data, "score_and_reasoning_partial", True, fallback.truncated_at
    return None, "unrepairable", False, None


def repair_iter(records):
    out: list[dict[str, Any]] = []
    counts: Counter = Counter()
    for rec in records:
        raw = rec.get("raw_text", "") or ""
        data, strategy, partial, truncated_at = _strategy_for(raw)
        counts[strategy] += 1
        if data is None:
            continue
        row = dict(rec.get("key") or {})
        row.update(data)
        if strategy != "valid":
            row["_repair_strategy"] = strategy
        if partial:
            row["_partial_parse"] = True
        if truncated_at is not None:
            row["_partial_extract_truncated_at"] = truncated_at
        row["_source_path"] = rec.get("source_path")
        out.append(row)
    return out, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--raw-dir", type=Path)
    src.add_argument("--raw-jsonl", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.raw_dir is not None:
        if not args.raw_dir.is_dir():
            raise SystemExit(f"raw-dir not found: {args.raw_dir}")
        records = list(_iter_from_raw_dir(args.raw_dir))
    else:
        if not args.raw_jsonl.is_file():
            raise SystemExit(f"raw-jsonl not found: {args.raw_jsonl}")
        records = list(_iter_from_jsonl(args.raw_jsonl))

    rows, counts = repair_iter(records)

    out_path = args.out_dir / "repaired_judgments.jsonl"
    with out_path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    repaired = sum(v for k, v in counts.items() if k not in {"valid", "unrepairable", "empty_body"})
    summary = {
        "input_count": sum(counts.values()),
        "output_count": len(rows),
        "by_strategy": dict(counts),
        "repaired_count": repaired,
        "unrepairable_count": counts.get("unrepairable", 0) + counts.get("empty_body", 0),
        "out_path": str(out_path),
    }
    summary_path = args.out_dir / "repair_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"input rows:   {summary['input_count']}")
    print(f"valid:        {counts.get('valid', 0)}")
    print(f"repaired:     {repaired}")
    for k in sorted(counts):
        if k in {"valid", "unrepairable", "empty_body"}:
            continue
        print(f"  via {k}: {counts[k]}")
    print(f"unrepairable: {summary['unrepairable_count']}")
    print(f"  empty_body: {counts.get('empty_body', 0)}")
    print(f"  unrepairable: {counts.get('unrepairable', 0)}")
    print(f"output:       {out_path}")
    print(f"summary:      {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
