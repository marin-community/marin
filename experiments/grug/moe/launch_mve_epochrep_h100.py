# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug-MoE seed-replicated epoch-harm experiment on cw-rno2a H100s (issue #7067).

The DECISIVE kernel-vs-functional-form test. Eighteen runs (2 arms x 3 epochs x
3 seeds) over FIXED-weight two-bucket mixtures whose content histogram h = V.w
is CONSTANT across epochs, so the frozen content kernel predicts a FLAT line by
construction; real epochs come from sub-slicing the epoching bucket's cache via
``max_train_batches`` (the two-bucket factorial's axis-2 mechanism).

- code arm: c01q0 at w=0.2 both phases (partner c05q0 at 0.8, never epochs),
  sliced to e in {4, 16, 32}. A 1:1 replica of the two-bucket factorial points
  ctr / a2_e16 / a2_e32 at seeds != 0, so the factorial's seed-0 runs there pool
  as a 4th seed.
- web arm: c26q1 at w=0.2 both phases (partner c05q0 at 0.8, never epochs),
  sliced to e in {4, 16, 24}. c26q1 is the transect's web bucket.

Predictions per point are committed in ``epochrep_preregistration.json`` (sha
asserted at every start; a byte-identical copy is staged next to this launcher
so it ships in job bundles): the kernel is CONSTANT across epochs at fixed w,
the swoosh harm head RISES. Everything else is IDENTICAL to the seed panel /
transect / twobucket: swarm B200_MODEL shapes + ``gpu_fa4_cute`` (SM90),
``SWARM_OPTIMIZER`` (warmup peak 0.100*N verified at N=4776), H100x8 one-node
runs, cuda_async allocator, CW-mirror data, no in-training eval, post-hoc
``eval_logprob`` readout. Seeds are 1/2/3 (the panel's seed-0 sigma anchors).
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

from levanter.data.text.datasets import LmDataConfig
from levanter.tracker.wandb import WandbConfig
from rigging.filesystem import marin_prefix, prefix_join

from experiments.grug.moe import launch_datakit_moe_mix as datakit_mix
from experiments.grug.moe import launch_mve_seedpanel as tpu_panel
from experiments.grug.moe import launch_mve_seedpanel_b200 as b200_panel
from experiments.grug.moe import launch_mve_seedpanel_h100 as h100_panel
from experiments.grug.moe.launch import GrugMoeLaunchConfig
from experiments.grug.moe.launch_mve_twobucket_h100 import _boundary_step, _verify_schedule_scales
from experiments.grug.moe.train import GrugTrainerConfig
from experiments.marin_tokenizer import marin_tokenizer

logger = logging.getLogger(__name__)

PREREG_PATH = Path(__file__).resolve().parent / "epochrep_preregistration.json"
PREREG_SHA256 = "4a1092f7d7d90dcbafb7947c1d8c5d33a7b44028138b1c295c098a43be6bcfc5"

WANDB_GROUP = "rav_mve_epochrep"
VALID_STEPS = (4_776,)
EPOCHREP_BUCKETS = ("c01q0", "c26q1", "c05q0")


def load_preregistration() -> dict:
    raw = PREREG_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PREREG_SHA256:
        raise AssertionError(f"preregistration sha mismatch: {digest} != {PREREG_SHA256}")
    return json.loads(raw)


def _epochrep_data_config(run: dict) -> LmDataConfig:
    """Fixed-weight two-bucket mixture, real epochs via slicing the epoching bucket's cache."""
    steps = int(run["steps"])
    if steps not in VALID_STEPS:
        raise AssertionError(f"{run['run_id']}: unexpected steps {steps}")
    boundary = _boundary_step(steps)
    if boundary != run["phase_boundary_step"]:
        raise AssertionError(f"{run['run_id']}: boundary {boundary} != prereg {run['phase_boundary_step']}")

    sliced, partner = run["sliced_bucket"], run["partner_bucket"]
    w = float(run["w_target"])
    weights = {sliced: w, partner: 1.0 - w}
    if abs(sum(weights.values()) - 1.0) > 1e-12 or any(v < 0 for v in weights.values()):
        raise AssertionError(f"{run['run_id']}: bad weights {weights}")

    n = int(run["slice_batches"])
    if n < 1:
        raise AssertionError(f"{run['run_id']}: bad slice {n}")
    # exact real epochs: e_phase0 = w * boundary / n, asserted against the prereg
    e0 = w * boundary / n
    if abs(e0 - run["epochs"]["sliced"]["phase0"]) > 1e-9:
        raise AssertionError(f"{run['run_id']}: e0 {e0} != prereg {run['epochs']['sliced']['phase0']}")
    slice_tokens = n * tpu_panel.BATCH_SIZE * tpu_panel.SEQ_LEN
    if run["epochs"]["code_slice_tokens"] != slice_tokens:
        raise AssertionError(f"{run['run_id']}: slice tokens mismatch")

    return LmDataConfig(
        tokenizer=marin_tokenizer,
        cache_dir=None,
        components=b200_panel._abs_datakit_components(),
        train_weights=[
            (0, weights),
            (boundary, dict(weights)),
        ],
        auto_build_caches=False,
        mixture_block_size=tpu_panel.MIXTURE_BLOCK_SIZE,
        target_budget=None,  # budgets OFF -> epochs are real
        experiment_budget=None,
        max_train_batches={sliced: n},
    )


def _epochrep_launch_config(run: dict, *, output_path: str, total_steps: int) -> GrugMoeLaunchConfig:
    return GrugMoeLaunchConfig(
        model=b200_panel.B200_MODEL,  # d512 swarm shapes + gpu_fa4_cute (SM90)
        data=_epochrep_data_config(run),
        output_path=output_path,
        run_id=run["run_id"],
        resources=h100_panel.train_resources(h100_panel.DEFAULT_GPUS_PER_RUN),
        steps=total_steps,
        batch_size=tpu_panel.BATCH_SIZE,
        seed=int(run["seed"]),
        mp="params=float32,compute=bfloat16,output=bfloat16",
        tracker=WandbConfig(
            project=tpu_panel.WANDB_PROJECT,
            tags=["moe", "epochrep", "mve", "d512", "h100", run["arm"], run["point"]],
            group=WANDB_GROUP,
            name=None,  # resolved to run_id at dispatch
        ),
        optimizer=tpu_panel.SWARM_OPTIMIZER,
        grug_trainer=GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1),
        eval=None,
    )


def verify_epochrep_parity(prereg: dict) -> dict:
    """Swarm parity + prereg-consistency asserts; returns a printable summary."""
    summary = tpu_panel.verify_swarm_parity()

    runs = prereg["runs"]
    if len(runs) != 18 or len({r["run_id"] for r in runs}) != 18:
        raise AssertionError(f"expected 18 unique runs, got {len(runs)}")
    seeds = sorted({r["seed"] for r in runs})
    if seeds != [1, 2, 3]:
        raise AssertionError(f"prereg seeds {seeds} != [1, 2, 3]")

    component_names = set(b200_panel._abs_datakit_components().keys())
    if not set(EPOCHREP_BUCKETS) <= component_names:
        raise AssertionError("epochrep buckets missing from the datakit component set")

    points = []
    for run in runs:
        _ = _epochrep_data_config(run)  # runs its own asserts
        points.append(
            {
                "point": run["point"],
                "run_id": run["run_id"],
                "job_name": run["job_name"],
                "arm": run["arm"],
                "sliced_bucket": run["sliced_bucket"],
                "w_target": run["w_target"],
                "e_target": run["e_target"],
                "seed": run["seed"],
                "slice_batches": run["slice_batches"],
                "e0": round(run["epochs"]["sliced"]["phase0"], 4),
                "is_control": run["is_control"],
                "twobucket_twin": run["twobucket_twin"],
                "pred_kernel_humaneval": round(run["pred_kernel_humaneval"], 5),
                "pred_swoosh_humaneval": round(run["pred_swoosh_humaneval"], 5),
            }
        )
    summary["runs"] = points
    summary["seeds"] = seeds
    summary["cluster"] = "cw-rno2a"
    summary["prereg_sha256"] = PREREG_SHA256
    summary["store_root"] = prefix_join(marin_prefix(), datakit_mix._STORE_PREFIX)
    summary["in_training_validation"] = False
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--direct", action="store_true", help="train ONE run in-process")
    parser.add_argument("--point", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None, help="step override (smoke only)")
    args = parser.parse_args()

    prereg = load_preregistration()
    summary = verify_epochrep_parity(prereg)

    if args.dry_run:
        summary["schedule_check"] = _verify_schedule_scales()  # JAX-touching: dry-run only
        print(json.dumps(summary, indent=2))
        print("DRY RUN: parity + preregistration checks passed; nothing launched.")
        return

    if not args.direct or args.point is None:
        raise SystemExit("epochrep jobs are submitted directly (one iris job per --point); use --direct --point <p>.")

    by_point = {r["point"]: r for r in prereg["runs"]}
    if args.point not in by_point:
        raise SystemExit(f"unknown point {args.point}; valid: {sorted(by_point)}")
    run = by_point[args.point]
    total_steps = args.steps or int(run["steps"])
    output_path = prefix_join(marin_prefix(), f"users/rav/grug/{run['run_id']}/dev")
    logger.info("direct mode: %s, %d steps, output %s", run["run_id"], total_steps, output_path)
    config = _epochrep_launch_config(run, output_path=output_path, total_steps=total_steps)
    b200_panel._run_direct(config)


if __name__ == "__main__":
    main()
