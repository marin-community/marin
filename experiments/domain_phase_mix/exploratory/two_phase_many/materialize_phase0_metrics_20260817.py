# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "google-cloud-storage"]
# ///
"""Phase-boundary readouts for the WSD80 dense panel, alongside the endpoint ones (ATOM-020).

Every panel in this programme regresses the ENDPOINT metric alone. The same `eval_metrics.jsonl` that
supplies it also carries every earlier eval, so a readout from the end of phase 0 costs nothing but a
different line of a file already being downloaded. No new training runs are involved.

What that readout is, exactly. Each (coordinate, rung) is its own run with its own 80/20 schedule, so it
has its own boundary step; this takes the LAST eval at or before that boundary. Evals are on a fixed step
cadence rather than aligned to the boundary, so the readout lands slightly early and the shortfall is
recorded per run as `phase0_fraction` -- the reported step over the boundary step -- rather than assumed
away. Nothing downstream should treat it as exactly the boundary.

Why it is worth having. The phase-1 mixture has not been applied yet, so this quantity is a function of
the PHASE-0 POLICY ALONE. That makes it a direct measurement of the half of the problem the surrogate
currently has to infer, which matters most at 39 buckets where the phase-0 mixture is 38-dimensional and
demonstrably under-identified. Whether it also reduces variance depends on whether it shares run-level
noise with the endpoint, which is an empirical question this file's output is meant to settle rather than
assert: runs sharing a phase-0 policy at one rung differ only in what happens AFTER the boundary, so if
their boundary readouts are identical the quantity is deterministic given the policy and can carry no
shared noise at all.

Usage: ``uv run python ... [--supports full,m400] [--workers 32]``
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for entry in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import pandas as pd  # noqa: E402
import plot_starcoder_wsd80_full_pool_atomic_surface_explorer_20260811 as explorer  # noqa: E402
import starcoder_wsd80_atomic_metrics as atomic  # noqa: E402
from google.cloud import storage  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    atomic_surface_panel_20260811 as panel_module,
)

OUTPUT = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_full_pool_atomic_surface_explorer_20260811"
PHASE0 = OUTPUT / "phase0_atomic_metric_observations.csv"


def boundary_readout(bucket, run_name: str, boundary_step: int) -> dict:
    """The last eval at or before the phase boundary, with the step it actually came from."""
    text = bucket.blob(explorer._metric_blob_name(run_name)).download_as_text()
    events = [json.loads(line) for line in text.splitlines() if line.strip()]
    usable = [
        event
        for event in events
        if event.get("step") is not None and event["step"] <= boundary_step and atomic.METRIC_KEYS[0] in event
    ]
    if not usable:
        return {"run_name": run_name, "phase0_step": None}
    chosen = max(usable, key=lambda event: event["step"])
    return {
        "run_name": run_name,
        "phase0_step": int(chosen["step"]),
        "phase0_fraction": float(chosen["step"]) / float(boundary_step),
        **{key: float(chosen[key]) for key in atomic.METRIC_KEYS if key in chosen},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supports", default="", help="comma-separated support ids; default every one")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    # The merged panel, not the raw coverage file: `metric_run_name` is created by the merge, and the
    # boundary step has to travel with the run it belongs to.
    frames = panel_module.load_all_supports()
    keep = args.supports.split(",") if args.supports else sorted(frames)
    coverage = pd.concat([frames[name] for name in keep], ignore_index=True)
    wanted = coverage.drop_duplicates("metric_run_name")[["metric_run_name", "boundary_step"]]
    done = pd.read_csv(PHASE0) if PHASE0.exists() else pd.DataFrame(columns=["run_name"])
    pending = wanted[~wanted["metric_run_name"].isin(set(done["run_name"]))]
    print(f"{len(wanted)} runs in scope, {len(done)} already cached, {len(pending)} to fetch")

    if len(pending):
        bucket = storage.Client().bucket(explorer.GCS_BUCKET)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(lambda item: boundary_readout(bucket, *item), pending.to_numpy().tolist()))
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        PHASE0.parent.mkdir(parents=True, exist_ok=True)
        done.to_csv(PHASE0, index=False)

    have = done["phase0_step"].notna()
    print(f"phase-0 readout present for {int(have.sum())}/{len(done)} runs")
    if have.any():
        fraction = done.loc[have, "phase0_fraction"]
        print(f"readout step as a fraction of the boundary: median {fraction.median():.4f}, ", end="")
        print(f"10th percentile {fraction.quantile(0.1):.4f}, min {fraction.min():.4f}")


if __name__ == "__main__":
    main()
