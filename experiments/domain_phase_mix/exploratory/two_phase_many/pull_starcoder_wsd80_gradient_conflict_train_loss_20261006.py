# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["wandb", "pandas", "pyarrow"]
# ///
"""Pull per-step train/loss for the review-v9 P2/P3/P4 and H5 trajectories."""

import csv
import sys
import time
from pathlib import Path

import pandas as pd
import wandb

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
manifest = (
    sys.argv[2]
    if len(sys.argv) > 2
    else str(
        Path(__file__).resolve().parent
        / "reference_outputs/starcoder_wsd80_gradient_conflict_design_20260811_v9/trajectory_manifest.csv"
    )
)
with open(manifest) as handle:
    ids = [r["trajectory_id"] for r in csv.DictReader(handle) if r["arm"] in {"p2", "p3", "p4", "b"}]
api = wandb.Api(timeout=120)
log = (OUT / "pull.log").open("a")


def say(m):
    print(m, flush=True)
    log.write(m + "\n")
    log.flush()


t0 = time.time()
for i, tid in enumerate(ids):
    path = OUT / f"{tid}.train.parquet"
    if path.exists():
        say(f"[{i + 1}/{len(ids)}] skip {tid}")
        continue
    run = None
    for rid in (f"gcfv9_{tid}", f"gcfrecoveryg24_{tid}", tid):
        try:
            run = api.run(f"marin-community/marin/{rid}")
            break
        except Exception:
            continue
    if run is None:
        say(f"[{i + 1}/{len(ids)}] MISSING {tid}")
        continue
    rows = list(run.scan_history(keys=["_step", "train/loss", "train/learning_rate"]))
    df = pd.DataFrame(rows)
    df["wandb_id"] = run.id
    df["state"] = run.state
    df.to_parquet(path)
    el = time.time() - t0
    say(
        f"[{i + 1}/{len(ids)}] {tid} id={run.id} state={run.state} rows={len(rows)} "
        f"elapsed={el / 60:.1f}m eta={(el / (i + 1)) * (len(ids) - i - 1) / 60:.1f}m"
    )
say("DONE")
