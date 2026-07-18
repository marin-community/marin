# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backfill the FP8 loss-validation trajectories into W&B (issue #7298, PR #7079).

The two validation arms ran with the ``json_logger`` tracker because no
``WANDB_API_KEY`` was available on the launch box; their full per-step
histories were harvested from the job logs. This script replays those
histories into W&B so the experiment lands in the canonical place with the
canonical metadata, following the ``wandb-reporting`` skill:

- project ``marin_moe`` / entity ``marin-community`` (where grug MoE runs live),
- run name = the run ID used in the logbook / issue / job name,
- shared group ``fp8-loss-val-7079`` for the A/B, shared tags,
- full config logged from the run's ``hparams``,
- raw per-step CSV attached as an artifact.

Run once a key is available, e.g.::

    WANDB_API_KEY=... uv run --package marin-core \
      python experiments/grug/moe/fp8val_wandb_backfill.py --data-dir <loss_series dir>

Idempotency: pass ``--run-suffix`` to avoid colliding with an earlier attempt;
W&B run IDs are derived from the run name so a re-run resumes/overwrites.
"""

import argparse
import json
import os
from pathlib import Path

import wandb

ENTITY = "marin-community"
PROJECT = "marin_moe"
GROUP = "fp8-loss-val-7079"
BASE_TAGS = ["fp8-loss-val", "pr7079", "grug", "moe", "h100", "FP8VAL"]

ARMS = [
    {"arm": "bf16", "run_id": "fp8val-bf16", "source_job": "fp8val-bf16-full6", "raw": "fp8val-bf16-full6.raw"},
    {"arm": "fp8", "run_id": "fp8val-fp8", "source_job": "fp8val-fp8-full4", "raw": "fp8val-fp8-full4.raw"},
]


def _parse_history(raw_path: Path) -> tuple[dict, list[dict]]:
    """Return (hparams, per-step metric rows) from a json_logger raw log."""
    hparams: dict = {}
    per_step: dict[int, dict] = {}
    for line in raw_path.open(errors="ignore"):
        if "fp8val.metrics" not in line:
            continue
        i = line.find('{"tracker"')
        if i < 0:
            continue
        try:
            j = json.loads(line[i:].strip())
        except json.JSONDecodeError:
            continue
        if j.get("event") == "hparams":
            hparams = j.get("hparams", {})
            continue
        step = j.get("step")
        m = j.get("metrics", {})
        if step is None or not m:
            continue
        row = per_step.setdefault(int(step), {})
        for key in ("train/loss", "throughput/tokens_per_second", "throughput/mfu", "train/cross_entropy_loss"):
            if key in m:
                row[key] = m[key]
    rows = [{"step": s, **per_step[s]} for s in sorted(per_step)]
    return hparams, rows


def backfill_arm(spec: dict, data_dir: Path, run_suffix: str) -> str:
    hparams, rows = _parse_history(data_dir / spec["raw"])
    name = spec["run_id"] + run_suffix
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=name,
        id=name,
        group=GROUP,
        tags=BASE_TAGS + [spec["arm"]],
        config={
            **hparams,
            "backfilled_from": "json_logger",
            "arm": spec["arm"],
            "source_job": spec["source_job"],
            "issue": 7298,
            "pr": 7079,
        },
        notes=(
            "Post-hoc backfill of the FP8 loss-curve validation (issue #7298, PR #7079). "
            "Ran with json_logger at train time (no WANDB_API_KEY on the launch box); "
            "per-step history replayed from harvested job logs."
        ),
        reinit=True,
    )
    for row in rows:
        step = row.pop("step")
        run.log(row, step=step)
    # attach raw per-step CSV artifact (<10MB)
    csv_path = data_dir / f"{spec['run_id']}.history.csv"
    with csv_path.open("w") as f:
        f.write("step,train_loss,tokens_per_second,mfu\n")
        for r in rows:
            f.write(f"{r['step']},{r.get('train/loss','')},{r.get('throughput/tokens_per_second','')},{r.get('throughput/mfu','')}\n")
    art = wandb.Artifact(f"{spec['run_id']}-history", type="loss-trajectory")
    art.add_file(str(csv_path))
    run.log_artifact(art)
    url = run.url
    run.finish()
    return url


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path, help="dir with the *.raw harvested logs")
    ap.add_argument("--run-suffix", default="", help="suffix appended to run names/ids (e.g. '-v2')")
    args = ap.parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY not set — export it (or `wandb login`) before running.")
    for spec in ARMS:
        url = backfill_arm(spec, args.data_dir, args.run_suffix)
        print(f"{spec['run_id']}: {url}")


if __name__ == "__main__":
    main()
