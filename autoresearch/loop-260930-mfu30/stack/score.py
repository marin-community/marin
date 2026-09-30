"""Score mfu30 arms against a matched-window control.

    uv run python autoresearch/loop-260930-mfu30/stack/score.py <run> [<run> ...] [--control mhep-ctx4k-s0-20260930]

For each run: median `throughput/mfu` and `throughput/duration` over the scoring window (steps
180011-180059, excluding the profiled steps 180021-180023 when present), peak memory, drop fraction,
and the pointwise loss difference against the control at matching steps (first step, max |diff|,
mean diff). Needs WANDB_API_KEY.
"""

import argparse
import statistics

import wandb

PROJECT = "marin-community/marin_moe"
WINDOW = range(180011, 180060)
PROFILED = set(range(180021, 180024))
KEYS = ["throughput/mfu", "throughput/duration", "train/loss", "memory/peak_gib", "moe/drop_fraction"]


def history(api, run_id):
    run = api.run(f"{PROJECT}/{run_id}")
    rows = {}
    for row in run.scan_history(keys=["_step", *KEYS]):
        rows[int(row["_step"])] = row
    return run, rows


def summarize(rows, exclude):
    window = [s for s in WINDOW if s in rows and s not in exclude]
    med = lambda key: statistics.median(rows[s][key] for s in window) if window else float("nan")  # noqa: E731
    return dict(
        steps=len(window),
        mfu=med("throughput/mfu"),
        duration=med("throughput/duration"),
        peak_gib=max((rows[s]["memory/peak_gib"] for s in rows if rows[s].get("memory/peak_gib")), default=float("nan")),
        drop_fraction=med("moe/drop_fraction"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--control", default="mhep-ctx4k-s0-20260930")
    args = ap.parse_args()
    api = wandb.Api()
    _, control = history(api, args.control)
    base = summarize(control, set())
    print(
        f"{args.control:34s} mfu {base['mfu']:.3f} dur {base['duration']:.3f} s peak {base['peak_gib']:.2f} GiB "
        f"drop {base['drop_fraction']:.5f} (n={base['steps']})"
    )
    for run_id in args.runs:
        run, rows = history(api, run_id)
        s = summarize(rows, PROFILED)
        common = sorted(set(rows) & set(control) & set(range(180000, 180060)))
        diffs = [rows[t]["train/loss"] - control[t]["train/loss"] for t in common]
        first = diffs[0] if diffs else float("nan")
        print(
            f"{run_id:34s} [{run.state}] mfu {s['mfu']:.3f} ({s['mfu'] - base['mfu']:+.3f}) dur {s['duration']:.3f} s "
            f"({s['duration'] - base['duration']:+.3f}) peak {s['peak_gib']:.2f} GiB drop {s['drop_fraction']:.5f} "
            f"(n={s['steps']}) | loss vs control: first {first:+.2e} max|d| {max(map(abs, diffs), default=float('nan')):.2e} "
            f"mean {statistics.fmean(diffs) if diffs else float('nan'):+.2e} over {len(diffs)} steps"
        )


if __name__ == "__main__":
    main()
