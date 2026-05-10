"""Fetch Claude baseline fill batches and append to per_judgment_opposite.jsonl."""
from __future__ import annotations
import json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import batch_anthropic as ba

DIR = Path("experiments/posttrain/disagreement_primitive")
TRACKER = DIR / "claude_baseline_fill_batches.json"
OUT_PATH = DIR / "per_judgment_opposite.jsonl"


def load_jsonl(path):
    return [json.loads(l) for l in path.open() if l.strip()]


def main():
    api_key = os.environ["ANTHROPIC_API_KEY"]
    info = json.loads(TRACKER.read_text())
    job_dir = Path(info["job_dir"])
    cmap = json.loads((job_dir / "custom_id_map.json").read_text())

    existing = load_jsonl(OUT_PATH)
    existing_keys = {(r.get("statement_id"), r.get("scenario_idx"), r.get("generator"),
                      r.get("condition"), r.get("judge"))
                     for r in existing}

    new_rows = []
    for cond, batch_meta in info["batches"].items():
        entries = ba.collect(api_key, job_dir, name=batch_meta["name"])
        print(f"  {cond}: {len(entries)} entries")
        for entry in entries:
            cid = entry.get("custom_id")
            mapping = cmap.get(cid)
            if mapping is None:
                continue
            sid, condition_from_map, scen, gen = mapping
            args = ba.extract_tool_args(entry)
            if args is None:
                continue
            score = args.get("score")
            try:
                score = int(score) if score is not None else None
                if score is not None and not 1 <= score <= 5:
                    score = None
            except (TypeError, ValueError):
                score = None
            if (sid, scen, gen, cond, "claude") in existing_keys:
                continue
            new_rows.append({
                "judge": "claude",
                "statement_id": sid,
                "scenario_idx": scen,
                "generator": gen,
                "condition": cond,
                "score": score,
                "reasoning": args.get("reasoning"),
                "spec_quotes": args.get("spec_quotes") or [],
                "rubric_quotes": args.get("rubric_quotes") or [],
                "example_refs": args.get("example_refs") or [],
            })

    with OUT_PATH.open("a") as f:
        for r in new_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nAppended {len(new_rows)} new claude rows to {OUT_PATH}")
    n_scored = sum(1 for r in new_rows if r.get("score") is not None)
    print(f"  scored: {n_scored}, null: {len(new_rows) - n_scored}")


if __name__ == "__main__":
    main()
