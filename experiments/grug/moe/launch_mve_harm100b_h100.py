# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug-MoE DECISIVE budget-transfer-of-harm experiment on cw-rno2a H100s (#7067).

Settles whether the epoch-harm term is needed at PRODUCTION scale. The 10B
fixed-weight two-bucket experiments (twobucket-a2, E2/epochrep) that epoch the
code bucket via REAL subset-repeat (``max_train_batches``) show SEVERE harm
(code e32 +1.267 bpb / 162 sigma). The 100B transect that epochs via SIMULATED
epoching (``target_budget``) shows NO harm. Those differ in BOTH budget AND
mechanism -- confounded. THIS experiment runs the SAME mechanism as
twobucket-a2/a3/E2 (``max_train_batches`` real subset-repeat of ``c01q0`` at
fixed w=0.2, partner ``c05q0`` at 0.8 never epochs) at 100B budget, so BUDGET is
the only change from the 10B harm.

Three single-seed (seed 1) runs, code arm only, e in {4, 16, 32} (e4 = the
below-threshold baseline). Content histogram h = V.w is CONSTANT across epochs
(the frozen content kernel is FLAT by construction). Two committed hypotheses in
``harm100b_preregistration.json`` (sha asserted at every start; a byte-identical
copy is staged next to this launcher so it ships in job bundles):

  (a) a3 power law HOLDS  -> harm PERSISTS: e4->e16 ~ +0.071, e4->e32 ~ +0.245 bpb;
  (b) transect generalizes -> harm VANISHES: rise ~ 0 (< 2*sqrt(2)*sigma = 0.016 bpb).

Everything else is IDENTICAL to the seed panel / transect / twobucket / epochrep:
100B swarm budget (47759 steps, boundary 38144), swarm ``B200_MODEL`` shapes +
``gpu_fa4_cute`` (SM90), ``SWARM_OPTIMIZER`` (warmup peak 0.100*N verified at
N=47759 in --dry-run), H100x8 one-node runs, cuda_async allocator, CW-mirror
data, no in-training eval, post-hoc ``eval_logprob`` readout.
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

PREREG_PATH = Path(__file__).resolve().parent / "harm100b_preregistration.json"
PREREG_SHA256 = "465dace264b2aa5e3fe4ee26e9077bbd7a59b5b79f4c9160eec057a27b0d8362"

WANDB_GROUP = "rav_mve_harm100b"
VALID_STEPS = (47_759,)  # 100B production budget (== seed panel / transect)
HARM100B_BUCKETS = ("c01q0", "c05q0")
SEED = 1


def load_preregistration() -> dict:
    raw = PREREG_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PREREG_SHA256:
        raise AssertionError(f"preregistration sha mismatch: {digest} != {PREREG_SHA256}")
    return json.loads(raw)


def _harm100b_data_config(run: dict) -> LmDataConfig:
    """Fixed-weight two-bucket mixture at 100B; real epochs via slicing c01q0's cache."""
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
        target_budget=None,  # budgets OFF -> epochs are REAL (same mechanism as twobucket-a2/E2)
        experiment_budget=None,
        max_train_batches={sliced: n},
    )


def _harm100b_launch_config(run: dict, *, output_path: str, total_steps: int) -> GrugMoeLaunchConfig:
    return GrugMoeLaunchConfig(
        model=b200_panel.B200_MODEL,  # d512 swarm shapes + gpu_fa4_cute (SM90)
        data=_harm100b_data_config(run),
        output_path=output_path,
        run_id=run["run_id"],
        resources=h100_panel.train_resources(h100_panel.DEFAULT_GPUS_PER_RUN),
        steps=total_steps,
        batch_size=tpu_panel.BATCH_SIZE,
        seed=int(run["seed"]),
        mp="params=float32,compute=bfloat16,output=bfloat16",
        tracker=WandbConfig(
            project=tpu_panel.WANDB_PROJECT,
            tags=["moe", "harm100b", "mve", "d512", "h100", run["arm"], run["point"]],
            group=WANDB_GROUP,
            name=None,  # resolved to run_id at dispatch
        ),
        optimizer=tpu_panel.SWARM_OPTIMIZER,
        grug_trainer=GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1),
        eval=None,
    )


def verify_harm100b_parity(prereg: dict) -> dict:
    """Swarm parity + prereg-consistency asserts; returns a printable summary."""
    summary = tpu_panel.verify_swarm_parity()

    runs = prereg["runs"]
    if len(runs) != 3 or len({r["run_id"] for r in runs}) != 3:
        raise AssertionError(f"expected 3 unique runs, got {len(runs)}")
    seeds = sorted({r["seed"] for r in runs})
    if seeds != [SEED]:
        raise AssertionError(f"prereg seeds {seeds} != [{SEED}]")
    if prereg["constants"]["steps"] != tpu_panel.STEPS:
        raise AssertionError("prereg steps != seed-panel STEPS (100B)")
    if prereg["constants"]["phase_boundary_step"] != tpu_panel.PHASE_1_START_STEP:
        raise AssertionError("prereg boundary != seed-panel boundary")
    if prereg["constants"]["target_budget_tokens"] is not None:
        raise AssertionError("prereg target_budget must be null (real epochs via max_train_batches)")

    component_names = set(b200_panel._abs_datakit_components().keys())
    if not set(HARM100B_BUCKETS) <= component_names:
        raise AssertionError("harm100b buckets missing from the datakit component set")

    points = []
    for run in runs:
        _ = _harm100b_data_config(run)  # runs its own asserts
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
                "pred_a3_powerlaw_rise_vs_e4": run["pred_a3_powerlaw_rise_vs_e4"],
                "pred_transect_vanishes_rise_vs_e4": run["pred_transect_vanishes_rise_vs_e4"],
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
    summary = verify_harm100b_parity(prereg)

    if args.dry_run:
        summary["schedule_check"] = _verify_schedule_scales()  # JAX-touching: dry-run only
        print(json.dumps(summary, indent=2))
        print("DRY RUN: parity + preregistration checks passed; nothing launched.")
        return

    if not args.direct or args.point is None:
        raise SystemExit("harm100b jobs are submitted directly (one iris job per --point); use --direct --point <p>.")

    by_point = {r["point"]: r for r in prereg["runs"]}
    if args.point not in by_point:
        raise SystemExit(f"unknown point {args.point}; valid: {sorted(by_point)}")
    run = by_point[args.point]
    total_steps = args.steps or int(run["steps"])
    output_path = prefix_join(marin_prefix(), f"users/rav/grug/{run['run_id']}/dev")
    logger.info("direct mode: %s, %d steps, output %s", run["run_id"], total_steps, output_path)
    config = _harm100b_launch_config(run, output_path=output_path, total_steps=total_steps)
    b200_panel._run_direct(config)


if __name__ == "__main__":
    main()
