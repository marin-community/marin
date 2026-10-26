# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["wandb", "pandas", "pyarrow"]
# ///
"""Pull eval and train histories for the 7.41B confirmation pairs and the long-gate runs."""

import sys
import time
from pathlib import Path

import pandas as pd
import wandb

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
api = wandb.Api(timeout=120)
runs = []
for sup, tied, untied in [("m100", "c109", "c020"), ("m200", "c079", "c011"), ("m400", "c043", "c007")]:
    for seed in range(20260821, 20260826):
        runs.append(f"dsopt_r3d28260_{sup}_tied_{tied}_s{str(seed)[4:]}")
        runs.append(f"dsopt_r3d28260_{sup}_untied_{untied}_s{str(seed)[4:]}")
runs += [
    "gcf_p3_r3d28260_m100a_confirmed-two-phase-winner_s2026082000",
    "gcf_p3_r3d28260_full_confirmed-two-phase-winner_s2026082000",
    "gcf_p1_r3d28260_m100b_common-tied-035_s2026081000",
]
log = (OUT / "pull.log").open("a")


def say(msg):
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()


t0 = time.time()
for i, name in enumerate(runs):
    ev_path = OUT / f"{name}.eval.parquet"
    tr_path = OUT / f"{name}.train.parquet"
    if ev_path.exists() and tr_path.exists():
        say(f"[{i + 1}/{len(runs)}] skip {name}")
        continue
    try:
        run = api.run(f"marin-community/marin/{name}")
    except Exception as e:
        say(f"[{i + 1}/{len(runs)}] MISSING {name}: {e}")
        continue
    keys = [k for k in run.summary.keys() if k.startswith("eval/")]
    keys = [k for k in keys if k.endswith("/bpb") or k.endswith("/loss") or k == "eval/loss"]
    rows = list(run.scan_history(keys=["_step", *keys]))
    pd.DataFrame(rows).to_parquet(ev_path)
    trows = list(run.scan_history(keys=["_step", "train/loss", "train/learning_rate"]))
    pd.DataFrame(trows).to_parquet(tr_path)
    el = time.time() - t0
    say(
        f"[{i + 1}/{len(runs)}] {name}: eval rows={len(rows)} keys={len(keys)} train rows={len(trows)} "
        f"elapsed={el / 60:.1f}m eta={(el / (i + 1)) * (len(runs) - i - 1) / 60:.1f}m"
    )
say("DONE")
