# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pull persisted eval_metrics.jsonl for every review-v9 gradient-conflict trajectory."""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from rigging.filesystem.buckets import filesystem_for

ROOT = "gs://marin-us-central1/checkpoints/pinlin_calvin_xu/data_mixture/starcoder_wsd80_gradient_conflict_review_v9_20260811/trajectories"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
fs, _ = filesystem_for(ROOT)
traj = sorted(p.rstrip("/").split("/")[-1] for p in fs.ls(ROOT, detail=False))
print("trajectories", len(traj), flush=True)


def pull(t):
    url = f"{ROOT}/{t}/2026.08.11.9/checkpoints/eval_metrics.jsonl"
    try:
        with fs.open(url, "rb") as h:
            rows = [json.loads(l) for l in h.read().decode().splitlines() if l.strip()]
    except FileNotFoundError:
        return t, None
    ckpts = sorted(
        int(p.rstrip("/").split("step-")[-1])
        for p in fs.ls(f"{ROOT}/{t}/2026.08.11.9/checkpoints", detail=False)
        if "step-" in p
    )
    df = pd.DataFrame(rows)
    df["trajectory_id"] = t
    df["checkpoint_steps"] = json.dumps(ckpts)
    return t, df


t0 = time.time()
frames = []
missing = []
with ThreadPoolExecutor(16) as ex:
    for i, (t, df) in enumerate(ex.map(pull, traj)):
        if df is None:
            missing.append(t)
        else:
            frames.append(df)
        if (i + 1) % 16 == 0:
            print(f"{i + 1}/{len(traj)} {time.time() - t0:.0f}s missing={len(missing)}", flush=True)
pd.concat(frames).to_parquet(OUT / "gcf_v9_evals.parquet")
with (OUT / "missing.json").open("w") as handle:
    json.dump(missing, handle)
print("DONE", len(frames), "missing", len(missing), flush=True)
