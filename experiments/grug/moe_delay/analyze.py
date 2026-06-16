# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize a delayed-gradient staleness sweep from W&B.

Pulls runs in a W&B group, parses the ``delay-<opt>-d<H>-tau<T>-<corr>-...`` run
names, and prints, per optimizer, the final train loss / Paloma macro and the
*gap to the tau=0 control* — the headline number for "how much does staleness
cost, and does the corrector close it".

    .venv/bin/python -m experiments.grug.moe_delay.analyze --group delay-pp-batch1
"""

import argparse
import re

import wandb

RUN_RE = re.compile(r"delay-(?P<opt>\w+?)-d(?P<h>\d+)-tau(?P<tau>\d+)-(?P<corr>.+?)-s(?P<seed>\d+)-st(?P<steps>\d+)")


def _tail_metric(run, key, n=20):
    """Mean of the last ``n`` logged values of ``key`` (robust to eval cadence)."""
    try:
        hist = run.history(keys=[key], samples=2000)
    except Exception:
        return None
    if hist is None or key not in getattr(hist, "columns", []):
        return run.summary.get(key)
    vals = [v for v in hist[key].tolist() if v is not None]
    if not vals:
        return run.summary.get(key)
    return sum(vals[-n:]) / len(vals[-n:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="delay-pp-batch1")
    ap.add_argument("--project", default="marin-community/marin_moe")
    args = ap.parse_args()

    runs = wandb.Api().runs(args.project, filters={"group": args.group}, order="-created_at")
    rows = []
    for r in runs:
        m = RUN_RE.match(r.name)
        if not m:
            continue
        rows.append(
            {
                "opt": m["opt"],
                "tau": int(m["tau"]),
                "corr": m["corr"],
                "seed": int(m["seed"]),
                "state": r.state,
                "step": r.summary.get("global_step"),
                "loss": _tail_metric(r, "train/loss"),
                "macro": r.summary.get("eval/paloma/macro_loss"),
                "name": r.name,
            }
        )

    if not rows:
        print(f"No parseable runs in group {args.group!r}.")
        return

    by_opt: dict[str, list] = {}
    for row in rows:
        by_opt.setdefault(row["opt"], []).append(row)

    for opt, group_rows in sorted(by_opt.items()):
        group_rows.sort(key=lambda x: (x["corr"], x["tau"]))
        baseline = next((x for x in group_rows if x["tau"] == 0 and x["corr"] == "none"), None)
        ref = baseline["loss"] if baseline and baseline["loss"] is not None else None
        print(f"\n=== {opt} (tau=0 control loss = {ref if ref is None else round(ref, 4)}) ===")
        print(f"{'tau':>4} {'corrector':<16} {'state':<9} {'step':>6} {'loss':>9} {'gap_vs_tau0':>12} {'macro':>8}")
        for x in group_rows:
            gap = (x["loss"] - ref) if (ref is not None and x["loss"] is not None) else None
            loss_s = "-" if x["loss"] is None else f"{x['loss']:.4f}"
            gap_s = "-" if gap is None else f"{gap:+.4f}"
            macro_s = "-" if x["macro"] is None else f"{x['macro']:.3f}"
            step_s = "-" if x["step"] is None else str(x["step"])
            print(f"{x['tau']:>4} {x['corr']:<16} {x['state']:<9} {step_s:>6} {loss_s:>9} {gap_s:>12} {macro_s:>8}")


if __name__ == "__main__":
    main()
