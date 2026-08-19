# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "wandb"]
# ///
"""Phase-0 readouts for the 39-bucket delphi 3e18 heldout panel, from W&B history (ATOM-021).

The two-bucket version of this (ATOM-020) was free, because the panel's own metric cache pointed at a
per-run `eval_metrics.jsonl` carrying every eval. The 39-bucket canonical panel carries no run locator at
all, and only 109 of its rows record a GCS metrics path -- but every row records `wandb_run_id` and
`phase_boundary_step`, so the history is reachable.

What the readout is. Evals land every 1000 steps in a 3007-step run and the boundary sits at 2400, so the
last eval at or before the boundary is step 2000 for 1956 of the 1957 coordinate-disjoint rows: **83.3% of
phase 0**, and identically positioned for every row. That uniformity matters more than the shortfall --
the quantity is a pure function of the phase-0 policy either way, since no phase-1 weight has been applied
by step 2000, and being measured at the same point for every row keeps comparisons across policies clean.
It is NOT the state at the boundary and must not be described as such.

Metric identity is checked rather than assumed: `eval/uncheatable_eval/bpb` at the final step reproduces
the panel's own `uncheatable_bpb` to the digit on the rows where both exist.

Usage: ``uv run python ... [--limit N] [--workers 16]``
"""

import argparse
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for entry in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import pandas as pd  # noqa: E402
import wandb  # noqa: E402

DELPHI = SCRIPT_DIR / "reference_outputs" / "delphi_3e18_append_only_heldouts_20260714"
SIXTY = SCRIPT_DIR / "reference_outputs" / "60m_39bucket_checkpoint_audit_20260724"
UNCHEATABLE = "eval/uncheatable_eval/bpb"
# The phase boundary is a fixed fraction of the run in every swarm; where a panel does not record the
# boundary step directly it is recomputed from the run's own final step.
SCALE_ALPHA = {"delphi_3e18": 0.7981376787495837, "60m": 0.80, "300m": 0.80}


def sources(scale: str):
    """(rows with wandb ids, output path). Each row needs `heldout_id`, entity, project and run id."""
    if scale == "delphi_3e18":
        panel = pd.read_csv(DELPHI / "heldout_current.csv")
        panel = panel[panel["fit_panel_overlap"] == "coordinate_disjoint"]
        return panel, DELPHI / "phase0_readouts.csv"
    frames = []
    for name in ("fit_two_phase", "heldout_observations", "all_nonfit_observations"):
        frame = pd.read_csv(SIXTY / f"{name}.csv")
        frame = frame[frame["wandb_run_id"].notna()].copy()
        frame["heldout_id"] = frame["run_name"]
        frame["wandb_entity"] = "marin-community"
        frame["wandb_project"] = "marin"
        frame["phase_boundary_step"] = None
        frames.append(frame[["heldout_id", "wandb_entity", "wandb_project", "wandb_run_id", "phase_boundary_step"]])
    return pd.concat(frames).drop_duplicates("heldout_id").reset_index(drop=True), SIXTY / "phase0_readouts.csv"


def readout(api, row, alpha: float) -> dict:
    """The last eval at or before the phase boundary for one run."""
    identifier = f"{row.wandb_entity}/{row.wandb_project}/{row.wandb_run_id}"
    try:
        run = api.run(identifier)
        history = [
            event
            for event in run.scan_history(keys=["_step", UNCHEATABLE], page_size=2000)
            if event.get(UNCHEATABLE) is not None
        ]
        boundary = row.phase_boundary_step
        if boundary is None or (isinstance(boundary, float) and math.isnan(boundary)):
            # The 60M audit files do not record it, so it is recomputed from the run's own final step.
            boundary = alpha * max(event["_step"] for event in history)
        events = [event for event in history if event["_step"] <= boundary]
    except Exception as error:
        return {"heldout_id": row.heldout_id, "phase0_uncheatable_bpb": None, "error": type(error).__name__}
    if not events:
        return {"heldout_id": row.heldout_id, "phase0_uncheatable_bpb": None, "error": "no eval before boundary"}
    chosen = max(events, key=lambda event: event["_step"])
    return {
        "heldout_id": row.heldout_id,
        "phase0_step": int(chosen["_step"]),
        "phase0_fraction": float(chosen["_step"]) / float(boundary),
        "phase0_uncheatable_bpb": float(chosen[UNCHEATABLE]),
        "error": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", default="delphi_3e18", choices=sorted(SCALE_ALPHA))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    panel, output = sources(args.scale)
    done = pd.read_csv(output) if output.exists() else pd.DataFrame(columns=["heldout_id"])
    pending = panel[~panel["heldout_id"].isin(set(done["heldout_id"]))]
    if args.limit:
        pending = pending.head(args.limit)
    print(f"{len(panel)} rows in scope, {len(done)} cached, {len(pending)} to fetch", flush=True)

    if len(pending):
        api = wandb.Api(timeout=60)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(lambda row: readout(api, row, SCALE_ALPHA[args.scale]), pending.itertuples()))
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(output, index=False)

    have = done["phase0_uncheatable_bpb"].notna()
    print(f"phase-0 readout present for {int(have.sum())}/{len(done)}")
    if not have.all():
        print("failures:", done.loc[~have, "error"].value_counts().to_dict())
    if have.any():
        print(f"readout fraction of the boundary: median {done.loc[have, 'phase0_fraction'].median():.4f}")


if __name__ == "__main__":
    main()
