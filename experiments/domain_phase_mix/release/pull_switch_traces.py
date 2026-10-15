# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-pull from W&B the per-step training loss behind the switch-trace figure.

Panel A of `plot_starcoder_wsd80_switch_traces_20261006.py` (its `ARMS` and `train_paths`) drew the training loss
around an abrupt mixture switch from `wandb/*.train.parquet` and `wandb_gcf/*.train.parquet` files in a scratch folder
that no longer exists. This script rebuilds them as one table, `trajectories.parquet` in the transition folder, with
one row per run and logged step: `run_name, run_label, seed, switch_step, step, train_loss`. The 50 runs are the
figure's five series, labeled as `train_loss_jump.csv` labels them:

- `p3_m100a`: the gradient-conflict v9 P3 trajectories on the m100a support, 0.02 to 0.82 StarCoder at the decay
  onset (8 seeds);
- `conf_m200_untied`, `conf_m400_untied`: the confirmation winners on the 2x and 4x subsets, 0 to 0.50 and 0 to 0.28
  at the decay onset (5 seeds each);
- `h5_boundary_beta_0p60`, `h5_boundary_beta_0p85`: the H5 trajectories that switch at 0.6T and 0.85T (16 seeds each).

The confirmation runs come from their launch design (W&B run id = run name; seed = data seed); the gradient-conflict
trajectories from the v9 trajectory manifest (W&B run id `gcfv9_<trajectory_id>`; seed = training seed). `run_name`
is the training run's name, which the export replaces by an observation id.

`train/loss` is logged at every step from step 2. W&B serves `scan_history(keys=...)` from its sampled-history
endpoint, which returns at most 10,000 rows per request and thins a request whose step window holds more rows than it
samples. `scan_history` pages through windows of `page_size` steps sampled `page_size` times, and a full page's window
includes both of its end steps, so each full page loses one step (step 11,668 of every run at 10,000-row pages). This
script requests 5,000-step windows sampled 10,000 times, drops the step that adjacent windows share after checking
that both copies agree, and checks that every step from the first logged one to the run's last is present. The losses
are float32 values, stored losslessly as float32. The series reproduce `train_loss_jump.csv`, which the analysis
computed from the original pull, to 1e-16.

Run once: `uv run python -m experiments.domain_phase_mix.release.pull_switch_traces`.
"""

import argparse
import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import wandb

from experiments.domain_phase_mix.release import paths, tables

TRANSITION = paths.RO / "starcoder_wsd80_transition_transient_20261006"
ENTITY_PROJECT = "marin-community/marin"
CONFIRMATION_DESIGN = (
    paths.REPO
    / "experiments/domain_phase_mix/starcoder_wsd80_dense_support_empirical_optimum_confirmation_design_20260811.json.gz"
)
GRADIENT_CONFLICT_MANIFEST = paths.RO / "starcoder_wsd80_gradient_conflict_design_20260811_v9/trajectory_manifest.csv"
GRADIENT_CONFLICT_RUN_ID = "gcfv9_{}"
# run_label -> run-name prefix of the five confirmation seeds (the figure's `conf_m200` and `conf_m400`).
CONFIRMATION = {
    "conf_m200_untied": "dsopt_r3d28260_m200_untied_c011_s",
    "conf_m400_untied": "dsopt_r3d28260_m400_untied_c007_s",
}
# run_label -> manifest columns and values selecting the arm's trajectories (the figure's `p3_m100a`, `h5_0.60` and
# `h5_0.85`).
GRADIENT_CONFLICT = {
    "p3_m100a": {"arm": "p3", "support_id": "m100a"},
    "h5_boundary_beta_0p60": {"arm": "b", "policy_role": "boundary_beta_0p60"},
    "h5_boundary_beta_0p85": {"arm": "b", "policy_role": "boundary_beta_0p85"},
}
SEEDS = {
    "conf_m200_untied": 5,
    "conf_m400_untied": 5,
    "p3_m100a": 8,
    "h5_boundary_beta_0p60": 16,
    "h5_boundary_beta_0p85": 16,
}
KEYS = ["_step", "train/loss"]
COLUMNS = ["run_name", "run_label", "seed", "switch_step", "step", "train_loss"]
SAMPLES = 10_000  # the most rows the sampled-history endpoint returns for one request
WINDOW = 5_000  # steps per request, so that no request holds more rows than it samples


def series() -> pd.DataFrame:
    """One row per run the figure reads: `run_name, wandb_id, run_label, seed, switch_step, total_steps`."""
    design = pd.DataFrame(json.loads(gzip.decompress(paths.require(CONFIRMATION_DESIGN).read_bytes()))["runs"])
    parts = []
    for label, prefix in CONFIRMATION.items():
        runs = design[design.run_name.str.startswith(prefix)]
        parts.append(
            pd.DataFrame(
                {
                    "run_name": runs.run_name,
                    "wandb_id": runs.run_name,
                    "run_label": label,
                    "seed": runs.data_seed,
                    "switch_step": runs.boundary_step,
                    "total_steps": runs.total_steps,
                }
            )
        )
    manifest = tables.read_csv(paths.require(GRADIENT_CONFLICT_MANIFEST))
    for label, selector in GRADIENT_CONFLICT.items():
        runs = manifest[np.logical_and.reduce([manifest[column] == value for column, value in selector.items()])]
        parts.append(
            pd.DataFrame(
                {
                    "run_name": runs.trajectory_id,
                    "wandb_id": [GRADIENT_CONFLICT_RUN_ID.format(t) for t in runs.trajectory_id],
                    "run_label": label,
                    "seed": runs.training_seed,
                    "switch_step": runs.boundary_step,
                    "total_steps": runs.total_steps,
                }
            )
        )
    out = pd.concat(parts, ignore_index=True)
    counts = out.groupby("run_label").seed.nunique().to_dict()
    if counts != SEEDS or len(out) != sum(SEEDS.values()):
        raise ValueError(f"expected one run per seed {SEEDS}, found {counts} over {len(out)} runs")
    if out.groupby("run_label").switch_step.nunique().ne(1).any():
        raise ValueError("the runs of a series switch at different steps")
    return out


def history(api: wandb.Api, wandb_id: str, total_steps: int) -> pd.DataFrame:
    """Every logged step (int32) and training loss (float32) of one finished run."""
    run = api.run(f"{ENTITY_PROJECT}/{wandb_id}")
    if run.state != "finished":
        raise ValueError(f"{wandb_id}: W&B state {run.state}")
    windows = [
        pd.DataFrame(list(run.scan_history(keys=KEYS, page_size=SAMPLES, min_step=start, max_step=start + WINDOW)))
        for start in range(0, total_steps, WINDOW)
    ]
    rows = pd.concat(windows, ignore_index=True)
    shared = rows[rows.duplicated("_step", keep=False)]  # adjacent windows both return their boundary step
    if shared.groupby("_step")["train/loss"].nunique().gt(1).any():
        raise ValueError(f"{wandb_id}: a step is returned twice with different losses")
    rows = rows.drop_duplicates("_step").sort_values("_step")
    steps = rows["_step"].to_numpy()
    if steps[-1] != total_steps - 1 or not (np.diff(steps) == 1).all():
        raise ValueError(f"{wandb_id}: logged steps from {steps[0]} miss steps up to {total_steps - 1}")
    loss = rows["train/loss"].to_numpy(dtype=np.float64)
    if not np.isfinite(loss).all() or not np.array_equal(loss.astype(np.float32).astype(np.float64), loss):
        raise ValueError(f"{wandb_id}: a training loss is not a finite float32 value")
    return pd.DataFrame({"step": steps.astype(np.int32), "train_loss": loss.astype(np.float32)})


def write(table: pd.DataFrame, path: Path) -> None:
    """Parquet with delta-coded steps and byte-split float32 losses, which halve the file against the defaults."""
    pq.write_table(
        pa.Table.from_pandas(table, preserve_index=False),
        path,
        compression="zstd",
        use_dictionary=["run_name", "run_label"],
        use_byte_stream_split=["train_loss"],
        column_encoding={"step": "DELTA_BINARY_PACKED"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=TRANSITION / "trajectories.parquet")
    args = parser.parse_args()
    runs = series()
    api = wandb.Api(timeout=120)
    frames = []
    start = time.time()
    for done, run in enumerate(runs.to_dict("records"), start=1):
        logged = history(api, run["wandb_id"], run["total_steps"])
        frame = logged.assign(
            run_name=run["run_name"],
            run_label=run["run_label"],
            seed=run["seed"],
            switch_step=np.int32(run["switch_step"]),
        )
        frames.append(frame[COLUMNS])
        elapsed = time.time() - start
        print(
            f"[{done}/{len(runs)}] {run['run_label']} seed {run['seed']}: {len(frame)} steps, {frame.step.min()} to "
            f"{frame.step.max()}; {elapsed:.0f} s elapsed, about {elapsed / done * (len(runs) - done):.0f} s left",
            flush=True,
        )
    table = pd.concat(frames, ignore_index=True)
    write(table, args.out)
    print(f"wrote {len(table)} rows for {len(runs)} runs to {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
