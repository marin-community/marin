# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug-MoE MATH-generality FOLLOW-UP: tau-universality + curvature (#7067).

Locks the two caveats the math-generality axis-1 (``launch_mve_mathgen_h100.py``,
seed 0, e in {4,8,16,32}) left open:

1. tau-universality -- axis-1 fit tau_math=10.83 single-seed, HIGHER than code
   tau=8.85. Add SEED REPLICATES at the two ratio points (e16, e32) x seeds {1,2}
   so, pooled with axis-1 seed 0, a 3-seed tau_math CI can test whether it
   EXCLUDES 8.85 (tau PER-BUCKET) or INCLUDES it (tau UNIVERSAL).
2. curvature -- math's e32/e16 ratio R=4.09 admits a PHYSICAL ratio-matching
   quadratic (tau_quad=0.36) the single-seed e8 could not exclude. Add two dense
   points (e6 SUB-threshold, e12 just-above-onset; seed 1) to discriminate the
   LINEAR hinge (harm(e6)~0, harm(e12)~0.106) from mild curvature
   (quadratic harm(e6)~0.061, harm(e12)~0.260).

Six runs total: {e16s1, e16s2, e32s1, e32s2, e6s1, e12s1}. Mechanism / budget /
model / bucket are IDENTICAL to axis-1: bucket ``c02q0`` (math) at FIXED w=0.2 both
phases + ``c05q0`` (web, 876.9B) at 0.8 (never epochs); content h = V.w CONSTANT
across runs (frozen content kernel FLAT by construction); real epochs via
``max_train_batches`` (target_budget OFF); 10B = 4776 steps, boundary 3776; d512
swarm shapes + ``gpu_fa4_cute`` (SM90). Only the epoch grid points and the seeds
differ. n = round(w*boundary/e) = round(755.2/e) -- the SAME rounding rule as the
axis (e4/e8/e16/e32 -> 189/94/47/24), giving e6/e12/e16/e32 -> 126/63/47/24.

Readout = ``logprob_gsm8k_5shot`` bpb (as axis-1), anchored on the axis-1 seed-0
e4 checkpoint. See ``mathgen2_preregistration.json`` (sha asserted at every start;
a byte-identical copy is staged next to this launcher so it ships in job bundles)
for the committed linear-vs-quadratic predictions and the two decision rules.

Submitted at DEFAULT (interactive) priority so these QUEUE BEHIND and never
preempt the running non-preemptible 100B gate (/rav/rav-mve-harm100b-*).
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

PREREG_PATH = Path(__file__).resolve().parent / "mathgen2_preregistration.json"
PREREG_SHA256 = "fd2d09499248da96724fdbdc8afc27fd13887beafba773af6af704f5944f9ff0"

WANDB_GROUP = "rav_mve_mathgen2"
VALID_STEPS = (4_776,)  # 10B budget (== mathgen axis-1 / twobucket-a2 code axis)
MATHGEN_BUCKETS = ("c02q0", "c05q0")
SEEDS = (1, 2)
W_TARGET = 0.2


def load_preregistration() -> dict:
    raw = PREREG_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PREREG_SHA256:
        raise AssertionError(f"preregistration sha mismatch: {digest} != {PREREG_SHA256}")
    return json.loads(raw)


def _mathgen2_data_config(run: dict) -> LmDataConfig:
    """Fixed-weight two-bucket mixture at 10B; real epochs via slicing c02q0's cache."""
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
    # n follows the SAME round-to-nearest rule as the axis: n = round(w*boundary/e_target)
    expected_n = round(w * boundary / float(run["e_target"]))
    if n != expected_n:
        raise AssertionError(f"{run['run_id']}: slice {n} != round(w*boundary/e)={expected_n}")
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
        target_budget=None,  # budgets OFF -> epochs are REAL (same mechanism as mathgen axis-1)
        experiment_budget=None,
        max_train_batches={sliced: n},
    )


def _mathgen2_launch_config(run: dict, *, output_path: str, total_steps: int) -> GrugMoeLaunchConfig:
    return GrugMoeLaunchConfig(
        model=b200_panel.B200_MODEL,  # d512 swarm shapes + gpu_fa4_cute (SM90)
        data=_mathgen2_data_config(run),
        output_path=output_path,
        run_id=run["run_id"],
        resources=h100_panel.train_resources(h100_panel.DEFAULT_GPUS_PER_RUN),
        steps=total_steps,
        batch_size=tpu_panel.BATCH_SIZE,
        seed=int(run["seed"]),
        mp="params=float32,compute=bfloat16,output=bfloat16",
        tracker=WandbConfig(
            project=tpu_panel.WANDB_PROJECT,
            tags=["moe", "mathgen2", "mve", "d512", "h100", run["arm"], run["point"]],
            group=WANDB_GROUP,
            name=None,  # resolved to run_id at dispatch
        ),
        optimizer=tpu_panel.SWARM_OPTIMIZER,
        grug_trainer=GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1),
        eval=None,
    )


def verify_mathgen2_parity(prereg: dict) -> dict:
    """Swarm parity + prereg-consistency asserts; returns a printable summary."""
    summary = tpu_panel.verify_swarm_parity()

    runs = prereg["runs"]
    if len(runs) != 6 or len({r["run_id"] for r in runs}) != 6:
        raise AssertionError(f"expected 6 unique runs, got {len(runs)}")
    seeds = sorted({r["seed"] for r in runs})
    if seeds != list(SEEDS):
        raise AssertionError(f"prereg seeds {seeds} != {list(SEEDS)}")
    if prereg["constants"]["steps"] != VALID_STEPS[0]:
        raise AssertionError("prereg steps != 10B step count")
    if prereg["constants"]["target_budget_tokens"] is not None:
        raise AssertionError("prereg target_budget must be null (real epochs via max_train_batches)")

    # the follow-up grid: e16/e16/e32/e32/e6/e12 -> n = round(755.2/e) = 47/47/24/24/126/63
    expected = {"e16s1": 47, "e16s2": 47, "e32s1": 24, "e32s2": 24, "e6s1": 126, "e12s1": 63}
    got = {r["point"]: r["slice_batches"] for r in runs}
    if got != expected:
        raise AssertionError(f"math follow-up grid mismatch: {got} != {expected}")
    expected_seed = {"e16s1": 1, "e16s2": 2, "e32s1": 1, "e32s2": 2, "e6s1": 1, "e12s1": 1}
    got_seed = {r["point"]: r["seed"] for r in runs}
    if got_seed != expected_seed:
        raise AssertionError(f"per-point seed mismatch: {got_seed} != {expected_seed}")

    component_names = set(b200_panel._abs_datakit_components().keys())
    if not set(MATHGEN_BUCKETS) <= component_names:
        raise AssertionError("mathgen buckets missing from the datakit component set")

    points = []
    for run in runs:
        _ = _mathgen2_data_config(run)  # runs its own asserts
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
                "pred_harm_linear": run["pred_harm_linear"],
                "pred_harm_quadratic_ratio_matched": run["pred_harm_quadratic_ratio_matched"],
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
    summary = verify_mathgen2_parity(prereg)

    if args.dry_run:
        summary["schedule_check"] = _verify_schedule_scales()  # JAX-touching: dry-run only
        print(json.dumps(summary, indent=2))
        print("DRY RUN: parity + preregistration checks passed; nothing launched.")
        return

    if not args.direct or args.point is None:
        raise SystemExit("mathgen2 jobs are submitted directly (one iris job per --point); use --direct --point <p>.")

    by_point = {r["point"]: r for r in prereg["runs"]}
    if args.point not in by_point:
        raise SystemExit(f"unknown point {args.point}; valid: {sorted(by_point)}")
    run = by_point[args.point]
    total_steps = args.steps or int(run["steps"])
    output_path = prefix_join(marin_prefix(), f"users/rav/grug/{run['run_id']}/dev")
    logger.info("direct mode: %s, %d steps, output %s", run["run_id"], total_steps, output_path)
    config = _mathgen2_launch_config(run, output_path=output_path, total_steps=total_steps)
    b200_panel._run_direct(config)


if __name__ == "__main__":
    main()
