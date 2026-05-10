"""Run 5 — fetch Claude batches + compute α per condition.

Reads run5_round_{N}_batches.json, fetches results, appends Claude rows to
run5_per_judgment_round_{N}.jsonl. Then computes α per condition (3-judge,
plus drop-each-judge sensitivity) and writes a summary.
"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import batch_anthropic as ba
from e9_kappa_alternatives import krippendorff_alpha_interval

DIR = Path("experiments/posttrain/disagreement_primitive")
ITER_DIR = DIR / "dart_iteration"
SID = "no_topic_off_limits"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.open() if l.strip()]


def fetch_claude(round_n: int):
    api_key = os.environ["ANTHROPIC_API_KEY"]
    sid_dir = ITER_DIR / SID
    info = json.loads((sid_dir / f"run5_round_{round_n}_batches.json").read_text())
    job_dir = Path(info["job_dir"])
    cmap = json.loads((job_dir / "custom_id_map.json").read_text())
    out_path = sid_dir / f"run5_per_judgment_round_{round_n}.jsonl"

    existing = load_jsonl(out_path) if out_path.exists() else []
    existing_keys = {(r.get("judge"), r.get("statement_id"), r.get("scenario_idx"),
                      r.get("generator"), r.get("condition"))
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
            if ("claude", sid, scen, gen, cond) in existing_keys:
                continue
            new_rows.append({
                "judge": "claude", "statement_id": sid, "scenario_idx": scen,
                "generator": gen, "condition": cond, "score": score,
                "reasoning": args.get("reasoning"),
            })

    with out_path.open("a") as f:
        for r in new_rows:
            f.write(json.dumps(r) + "\n")
    print(f"  appended {len(new_rows)} claude rows to {out_path}")


def alpha_for(rows: list[dict], condition: str, judges: tuple[str, ...]) -> tuple[float | None, int]:
    by_cell = defaultdict(dict)
    for r in rows:
        if r.get("score") is None: continue
        if r.get("condition") != condition: continue
        if r.get("judge") not in judges: continue
        ck = (r["scenario_idx"], r["generator"])
        by_cell[ck][r["judge"]] = r["score"]
    tuples = []
    for ck, scores in by_cell.items():
        valid = tuple(scores.get(j) for j in judges)
        if all(v is not None for v in valid):
            tuples.append(valid)
    if not tuples:
        return None, 0
    return krippendorff_alpha_interval(tuples), len(tuples)


def analyze(round_n: int):
    sid_dir = ITER_DIR / SID
    out_path = sid_dir / f"run5_per_judgment_round_{round_n}.jsonl"
    rows = load_jsonl(out_path)
    print(f"\nLoaded {len(rows)} judgment rows")
    print(f"  by judge: {Counter(r['judge'] for r in rows if 'error' not in r)}")
    print(f"  by cond:  {Counter(r['condition'] for r in rows if 'error' not in r)}")

    # Also load baseline (per_judgment_opposite) for v1 reference
    baseline = [r for r in load_jsonl(DIR / "per_judgment_opposite.jsonl")
                if r.get("statement_id") == SID and r.get("condition") == "rubric_plus_spec"]

    # Also load Run 4 R1 v2 data for comparison
    run4_r1 = [r for r in load_jsonl(ITER_DIR / "per_judgment_iter_round_1.jsonl")
               if r.get("statement_id") == SID and r.get("condition") == "C3"]

    print(f"\n=== α comparison ===")
    print(f"{'state':40s} {'3-judge α':>12s}  {'gpt+gem':>10s}  {'gpt+cla':>10s}  {'gem+cla':>10s}")

    def show(label, r, c):
        a3 = alpha_for(r, c, ("gpt", "gemini", "claude"))
        agg = alpha_for(r, c, ("gpt", "gemini"))
        agc = alpha_for(r, c, ("gpt", "claude"))
        ggc = alpha_for(r, c, ("gemini", "claude"))
        def fmt(x):
            return f"{x[0]:+.3f}(n={x[1]})" if x[0] is not None else f"  n/a (n={x[1]})"
        print(f"{label:40s} {fmt(a3):>12s}  {fmt(agg):>10s}  {fmt(agc):>10s}  {fmt(ggc):>10s}")

    show("v1 baseline (3-judge missing)", baseline, "rubric_plus_spec")
    show("Run 4 R1 v2 (rubric only)", run4_r1, "C3")
    show("Run 5 R1 C_RUBRIC (hierarchical pick)", rows, "C_RUBRIC")
    show("Run 5 R1 C_EXAMPLES (Gem's example, counterfactual)", rows, "C_EXAMPLES")

    # Per-judge mean shift on the new conditions
    print(f"\n=== Per-judge score means (Run 5 R1) ===")
    for cond in ["C_RUBRIC", "C_EXAMPLES"]:
        for j in ["gpt", "gemini", "claude"]:
            scores = [r["score"] for r in rows if r.get("condition") == cond
                      and r.get("judge") == j and r.get("score") is not None]
            if scores:
                print(f"  {cond:14s} {j:8s}: mean={sum(scores)/len(scores):.2f} n={len(scores)} dist={dict(sorted(Counter(scores).items()))}")

    # Summary write
    summary = {
        "round": round_n,
        "alpha_C_RUBRIC_3j": alpha_for(rows, "C_RUBRIC", ("gpt","gemini","claude"))[0],
        "alpha_C_EXAMPLES_3j": alpha_for(rows, "C_EXAMPLES", ("gpt","gemini","claude"))[0],
        "alpha_C_RUBRIC_gpt_cla": alpha_for(rows, "C_RUBRIC", ("gpt","claude"))[0],
        "alpha_C_EXAMPLES_gpt_cla": alpha_for(rows, "C_EXAMPLES", ("gpt","claude"))[0],
    }
    (sid_dir / f"run5_round_{round_n}_analysis_summary.json").write_text(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--skip-fetch", action="store_true")
    args = ap.parse_args()
    if not args.skip_fetch:
        fetch_claude(args.round)
    analyze(args.round)


if __name__ == "__main__":
    main()
