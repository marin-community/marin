"""Phase 4 — Poll, fetch, and integrate Claude batch results.

Reads pointer from `e9_claude_batch_pointer.json` (written by Phase 3),
polls each batch_id until terminal, downloads results, parses each tool_use
entry, and writes per-statement Claude jsonl rows.

Outputs:
  - claude_judge_v0_opposite/<sid>/{bare,phase_4}_opposite_claude.jsonl  (Batch A)
  - claude_judge_v0/<sid>/{bare,phase_4}_claude.jsonl                    (Batch B)

Idempotent: re-running picks up where it left off (state files persist).
Safe to run while Phase 3 is still submitting (will block on poll).
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from e8_paired_indirection import SPEC_PATH, load_jsonl, write_jsonl
import batch_anthropic as ba

DIR = Path("experiments/posttrain/disagreement_primitive")
POINTER_PATH = DIR / "e9_claude_batch_pointer.json"
CLAUDE_OPPOSITE_DIR = DIR / "claude_judge_v0_opposite"
CLAUDE_EXISTING_DIR = DIR / "claude_judge_v0"
OPPOSITE_RESPONSES = DIR / "e9_opposite_mode_responses.jsonl"
EXISTING_RESPONSES = DIR / "e8_responses.jsonl"


def load_user_q_response_index() -> dict[tuple[str, int, str], dict[str, Any]]:
    """Build (sid, scen, generator) → {user_query, response} index across both response files."""
    out: dict[tuple[str, int, str], dict[str, Any]] = {}
    # Grok-opposite responses
    for r in load_jsonl(OPPOSITE_RESPONSES) if OPPOSITE_RESPONSES.exists() else []:
        if "error" in r:
            continue
        out[(r["statement_id"], r["scenario_idx"], r["generator"])] = {
            "user_query": r["user_query"], "response": r["response"],
        }
    # Existing 3-gen responses
    GEN_KEYS = [("gpt-5.1", "response_gpt"), ("Qwen/Qwen2.5-7B-Instruct-Turbo", "response_weak"),
                ("gemini-3-flash-preview", "response_gemini")]
    for r in load_jsonl(EXISTING_RESPONSES) if EXISTING_RESPONSES.exists() else []:
        sid, scen = r.get("statement_id"), r.get("scenario_idx")
        for gen_label, col in GEN_KEYS:
            text = r.get(col)
            if text:
                out[(sid, scen, gen_label)] = {"user_query": r["user_query"], "response": text}
    return out


def load_custom_id_map(job_dir: Path, batch_name: str) -> dict[str, tuple[str, str, int, str]]:
    """Read sidecar custom_id → (sid, cond_short, scen, gen) map written by Phase 3."""
    p = job_dir / f"{batch_name}_custom_id_map.json"
    if not p.exists():
        raise FileNotFoundError(f"sidecar map missing at {p}")
    raw = json.loads(p.read_text())
    return {cid: (vals[0], vals[1], int(vals[2]), vals[3]) for cid, vals in raw.items()}


def integrate(entries: list[dict[str, Any]], out_dir: Path, response_idx: dict, batch_label: str,
              cid_map: dict[str, tuple[str, str, int, str]]) -> dict[str, int]:
    """Convert batch result entries into per-statement Claude jsonl rows.

    Appends to existing files; deduplicates by (scenario_idx, generator).
    Uses the sidecar custom_id map (Phase 3 writes; Phase 4 reads).
    """
    by_path: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    n_ok = n_fail = n_other = 0
    for entry in entries:
        cid = entry.get("custom_id", "")
        meta = cid_map.get(cid)
        if meta is None:
            n_other += 1
            continue
        sid, cond_short, scen, gen = meta
        args = ba.extract_tool_args(entry)
        if args is None:
            n_fail += 1
            row = {"statement_id": sid, "scenario_idx": scen, "generator": gen,
                   "error": str((entry.get("result") or {}).get("type", "no_tool_use"))}
        else:
            score = args.get("score")
            if not isinstance(score, int) or not 1 <= score <= 5:
                n_fail += 1
                row = {"statement_id": sid, "scenario_idx": scen, "generator": gen,
                       "error": f"bad score: {score!r}"}
            else:
                n_ok += 1
                resp_meta = response_idx.get((sid, scen, gen), {})
                usage = ba.usage_of(entry)
                row = {
                    "statement_id": sid, "scenario_idx": scen, "generator": gen,
                    "user_query": resp_meta.get("user_query", ""),
                    "response": resp_meta.get("response", ""),
                    "score": score,
                    "reasoning": args.get("reasoning") or "",
                    "spec_quotes": args.get("spec_quotes") or [],
                    "rubric_quotes": args.get("rubric_quotes") or [],
                    "example_refs": args.get("example_refs") or [],
                    "_usage": usage,
                    "_repair_strategy": "batch_tool_use",
                    "_batch_label": batch_label,
                }
                if cond_short == "phase_4":
                    row["rubric_spec_tension"] = args.get("rubric_spec_tension")
                    row["tension_description"] = args.get("tension_description") or ""

        suffix = "_opposite_claude.jsonl" if "opposite" in batch_label else "_claude.jsonl"
        target = out_dir / sid / f"{cond_short}{suffix}"
        by_path[target].append(row)

    # Write each per-statement file: merge existing + new (dedup by (scenario_idx, generator)).
    for path, new_rows in by_path.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = []
        if path.exists():
            existing = [json.loads(l) for l in path.open()]
        # Build dict keyed by (scen, gen). New wins over existing only if new has score and existing doesn't.
        merged: dict[tuple[int, str], dict[str, Any]] = {}
        for r in existing:
            merged[(r["scenario_idx"], r["generator"])] = r
        for r in new_rows:
            key = (r["scenario_idx"], r["generator"])
            old = merged.get(key)
            if old is None:
                merged[key] = r
            elif old.get("score") is None and r.get("score") is not None:
                merged[key] = r
        rows_sorted = sorted(merged.values(), key=lambda r: (r["scenario_idx"], r["generator"]))
        with path.open("w") as f:
            for r in rows_sorted:
                f.write(json.dumps(r) + "\n")

    return {"ok": n_ok, "fail": n_fail, "other": n_other, "files_written": len(by_path)}


def main() -> int:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set; run: source .env2")
    if not POINTER_PATH.exists():
        raise SystemExit(f"no pointer at {POINTER_PATH}; run e9_phase3_submit_claude_batches.py first")

    pointer = json.loads(POINTER_PATH.read_text())
    job_dir = Path(pointer["job_dir"])
    print(f"Phase 4 — fetching Claude batches from {job_dir}")

    response_idx = load_user_q_response_index()
    print(f"  loaded response index: {len(response_idx)} (sid, scen, gen) entries")

    summary = {}
    for name, info in pointer["batches"].items():
        if info is None:
            print(f"\n  skipping {name} (was empty at submit time)")
            continue
        print(f"\n  ---- {name} (batch_id={info['batch_id']}, n_requests={info['n_requests']}) ----")
        print(f"  polling...")
        ba.poll(api_key, job_dir, name=name, interval=60.0)
        print(f"  collecting...")
        entries = ba.collect(api_key, job_dir, name=name)
        out_dir = CLAUDE_OPPOSITE_DIR if "opposite" in name else CLAUDE_EXISTING_DIR
        cid_map = load_custom_id_map(job_dir, name)
        result = integrate(entries, out_dir, response_idx, batch_label=name, cid_map=cid_map)
        print(f"  result: {result}")
        summary[name] = result

    print(f"\n== Phase 4 SUMMARY ==")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
