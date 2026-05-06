# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline retry: re-parse GLM phase_4 raw responses through repair_glm_json.

Forward-looking utility. When the original raw GLM phase_4 SDK responses
are available (under `results/raw/e8_phase2_glm/<run-ts>/judge_*` or a
restored equivalent), this script reads the raw text, runs it through the
repair pass, and writes a new judgments.jsonl that includes any rows we
were able to recover.

INPUTS
------
The script accepts EITHER:
  --raw-dir PATH/TO/RUN/role/        (a single role directory with per-call .json files)
  --raw-jsonl PATH/TO/judgments.jsonl  (a flat jsonl where each row has 'raw_text' or
                                        'choices[0].message.content')

OUTPUTS
-------
  <out_dir>/repaired_judgments.jsonl
      All successfully-parsed records (valid + repaired). Each repaired
      record carries _repair_strategy and (when relevant) _partial_parse.
  <out_dir>/repair_summary.json
      Aggregate counts: valid, repaired (per strategy), unparseable.

NOT RUN BY THIS REPO. The script is forward-looking — see report at
glm_json_repair_report.md for what's needed to actually drive a retry.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from collections.abc import Iterable

sys.path.insert(0, str(Path(__file__).parent))
from e9_glm_json_repair import RepairResult, repair_glm_json

# -------------------- Input adapters --------------------


def _iter_from_raw_dir(raw_dir: Path) -> Iterable[dict[str, Any]]:
    """Iterate dump files under a RawAPILogger run/role directory.

    Each file is the SDK-serialised response (model_dump() result). We
    look for the chat-completion shape:
        {"choices": [{"message": {"content": "..."}}, ...], ...}
    Plus the metadata under a sibling `key` (filename-encoded).
    """
    for path in sorted(raw_dir.glob("*.json")):
        with path.open() as fh:
            blob = json.load(fh)
        # Pull metadata from the manifest if present; fall back to filename.
        key = blob.get("key") or {}
        # Pull raw text. RawAPILogger stores either the full SDK response
        # under "response" or a top-level model_dump.
        sdk = blob.get("response") or blob
        try:
            content = sdk["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            content = blob.get("text") or ""
        yield {
            "source_path": str(path),
            "key": key,
            "raw_text": content or "",
        }


def _iter_from_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Iterate from a flat jsonl input. Each row may carry the raw text
    directly (`raw_text`), under `content`, or as a nested SDK-shaped dict.
    Rows without recoverable raw text are reported with raw_text="".
    """
    with path.open() as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            raw = row.get("raw_text")
            if raw is None:
                raw = row.get("content")
            if raw is None:
                # Try SDK shape.
                try:
                    raw = row["choices"][0]["message"]["content"]
                except (KeyError, IndexError, TypeError):
                    raw = ""
            yield {
                "source_path": f"{path}:{line_no}",
                "key": {k: row[k] for k in ("statement_id", "scenario_idx", "generator") if k in row},
                "raw_text": raw or "",
            }


# -------------------- Repair runner --------------------


def repair_iter(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter]:
    """Run repair_glm_json over each record. Returns the repaired-or-valid
    rows plus a counter keyed by strategy.

    `unrepairable` rows are NOT emitted into the returned list (so the
    output file matches what a working judge would have produced).
    """
    out: list[dict[str, Any]] = []
    counts: Counter = Counter()
    for rec in records:
        result: RepairResult = repair_glm_json(rec.get("raw_text", "") or "")
        counts[result.strategy] += 1
        if result.data is None:
            continue
        row = dict(rec.get("key") or {})
        row.update(result.data)
        if result.strategy != "valid":
            row["_repair_strategy"] = result.strategy
        if result.partial:
            row["_partial_parse"] = True
        row["_source_path"] = rec.get("source_path")
        out.append(row)
    return out, counts


# -------------------- CLI --------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--raw-dir",
        type=Path,
        help="A RawAPILogger role/ directory with per-call .json dumps.",
    )
    src.add_argument(
        "--raw-jsonl",
        type=Path,
        help="A flat jsonl where each row carries raw_text/content or a nested choices[].",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Where to write repaired_judgments.jsonl + repair_summary.json.",
    )
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

    summary = {
        "input_count": sum(counts.values()),
        "output_count": len(rows),
        "by_strategy": dict(counts),
        "repaired_count": sum(v for k, v in counts.items() if k not in {"valid", "unrepairable", "empty_body"}),
        "unrepairable_count": counts.get("unrepairable", 0) + counts.get("empty_body", 0),
        "out_path": str(out_path),
    }
    summary_path = args.out_dir / "repair_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(f"input rows: {summary['input_count']}")
    print(f"valid:      {counts.get('valid', 0)}")
    repaired = summary["repaired_count"]
    print(f"repaired:   {repaired}")
    for k in sorted(counts):
        if k in {"valid", "unrepairable", "empty_body"}:
            continue
        print(f"  via {k}: {counts[k]}")
    print(f"unrepairable: {summary['unrepairable_count']}")
    print(f"  empty_body: {counts.get('empty_body', 0)}")
    print(f"  unrepairable: {counts.get('unrepairable', 0)}")
    print(f"output:     {out_path}")
    print(f"summary:    {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
