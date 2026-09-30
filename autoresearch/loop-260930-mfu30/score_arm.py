"""Independent arm scorer for mfu30: MFU median over the scored window and loss divergence vs a control.

Usage: WANDB_API_KEY=... uv run --no-sync python autoresearch/loop-260930-mfu30/score_arm.py <run> [<run> ...]
       [--control mhep-ctx4k-s0-20260930] [--lo 180011] [--hi 180059] [--exclude 180021-180023]
"""

import argparse
import statistics as st

import wandb

KEYS = ["_step", "throughput/mfu", "throughput/duration", "train/loss", "memory/peak_gib", "moe/drop_fraction"]


def history(api, run_id):
    run = api.run(f"marin-community/marin_moe/{run_id}")
    rows = {}
    for row in run.scan_history(keys=KEYS):
        rows.setdefault(row["_step"], row)
    return run.state, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--control", default="mhep-ctx4k-s0-20260930")
    ap.add_argument("--lo", type=int, default=180011)
    ap.add_argument("--hi", type=int, default=180059)
    ap.add_argument("--exclude", default="180021-180023")
    args = ap.parse_args()
    ex_lo, ex_hi = (int(x) for x in args.exclude.split("-"))
    api = wandb.Api(timeout=120)
    _, ctl = history(api, args.control)
    for run_id in [args.control, *args.runs]:
        state, rows = history(api, run_id)
        win = [s for s in range(args.lo, args.hi + 1) if s in rows and not ex_lo <= s <= ex_hi]
        mfu = [rows[s]["throughput/mfu"] for s in win]
        dur = [rows[s]["throughput/duration"] for s in win]
        peak = max((r.get("memory/peak_gib") or 0) for r in rows.values()) if rows else float("nan")
        drops = [rows[s].get("moe/drop_fraction") for s in win if rows[s].get("moe/drop_fraction") is not None]
        common = sorted(s for s in rows if s in ctl and rows[s].get("train/loss") is not None)
        d = [rows[s]["train/loss"] - ctl[s]["train/loss"] for s in common]
        first = rows.get(180000, {}).get("train/loss")
        print(
            f"{run_id}: state={state} n={len(win)} mfu_median={st.median(mfu):.3f} dur_median={st.median(dur):.3f} "
            f"peak={peak:.2f} drops_mean={st.mean(drops) if drops else float('nan'):.2e} loss@180000={first}"
            if mfu
            else f"{run_id}: state={state} no scored steps yet (steps {min(rows) if rows else None}..{max(rows) if rows else None})"
        )
        if d and run_id != args.control:
            late = [x for s, x in zip(common, d) if s >= args.lo]
            pos = sum(1 for x in late if x > 0)
            print(
                f"   dloss vs control: max|d|={max(abs(x) for x in d):.2e} mean(late)={st.mean(late) if late else 0:+.2e} "
                f"positive {pos}/{len(late)}; first steps {[f'{x:+.1e}' for x in d[:4]]}"
            )


if __name__ == "__main__":
    main()
