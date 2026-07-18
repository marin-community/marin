# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug-MoE repetition-harm transect on cw-rno2a H100s (issue #7067, swoosh PART 2).

Eight single-seed runs along two per-bucket dose transects, per the frozen
pre-registration ``transect_preregistration.json`` (sha ``90e5a5eb…``; a
byte-identical copy of ``scratch/mixture_features/grug/transect_preregistration.json``
staged next to this launcher so it ships in job bundles — predictions were
committed before launch and the file is never modified):

- web: ``c26q1`` phase-0 share set for target epochs e = {2, 4, 8, 16, 24}
  (points ``e2..e24``), phase-1 at anchor, all other buckets anchor-renormalized;
- code contrast: ``c01q0`` at e = {4, 16, 24} (points ``c4, c16, c24``).

Everything else is IDENTICAL to the seed panel (``launch_mve_seedpanel_h100``):
swarm constants, B200_MODEL shapes + ``gpu_fa4_cute`` (SM90 path),
``SWARM_OPTIMIZER``, H100x8 one-node runs, cuda_async allocator, CW-mirror
data, no in-training eval, post-hoc ``eval_logprob`` readout. Seed is FIXED at
0 for all 8 runs (the swarm's seed; the concurrent 10-seed panel provides the
anchor sigma).
"""

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path

from levanter.data.text.datasets import LmDataConfig
from levanter.tracker.wandb import WandbConfig
from rigging.filesystem import marin_prefix, prefix_join

from experiments.grug.moe import launch_datakit_moe_mix as datakit_mix
from experiments.grug.moe import launch_mve_seedpanel as tpu_panel
from experiments.grug.moe import launch_mve_seedpanel_b200 as b200_panel
from experiments.grug.moe import launch_mve_seedpanel_h100 as h100_panel
from experiments.grug.moe.launch import GrugMoeLaunchConfig
from experiments.grug.moe.train import GrugTrainerConfig
from experiments.marin_tokenizer import marin_tokenizer

logger = logging.getLogger(__name__)

PREREG_PATH = Path(__file__).resolve().parent / "transect_preregistration.json"
PREREG_SHA256 = "90e5a5eb6738bc2ee444644141d239368392e6029b827d66ce3275edf20c3c45"

POINTS = ("e2", "e4", "e8", "e16", "e24", "c4", "c16", "c24")
RUN_ID_TEMPLATE = "rav_mve_transect_{point}"
WANDB_GROUP = "rav_mve_transect"
TRANSECT_SEED = 0  # fixed for all runs (pre-registered; the seed panel provides sigma)


def load_preregistration() -> dict:
    raw = PREREG_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PREREG_SHA256:
        raise AssertionError(f"preregistration sha mismatch: {digest} != {PREREG_SHA256}")
    return json.loads(raw)


def _transect_data_config(prereg: dict, point: str) -> LmDataConfig:
    """The seed panel's data config with the pre-registered phase-0 weights for ``point``.

    Phase-1 stays at the anchor (asserted against the launcher's own table).
    """
    run_id = RUN_ID_TEMPLATE.format(point=point)
    mixture = prereg["mixtures"][run_id]
    phase0 = dict(mixture["phase0"])

    if len(phase0) != 168:
        raise AssertionError(f"{run_id}: phase0 has {len(phase0)} buckets, expected 168")
    if not math.isclose(sum(phase0.values()), 1.0, abs_tol=1e-9):
        raise AssertionError(f"{run_id}: phase0 sums to {sum(phase0.values())}")
    # The prereg records phase1 as the directive string "anchor (…unchanged)".
    if not str(mixture["phase1"]).startswith("anchor"):
        raise AssertionError(f"{run_id}: unexpected phase1 spec {mixture['phase1']!r}")
    phase1 = datakit_mix._phase_weights(1)

    return LmDataConfig(
        tokenizer=marin_tokenizer,
        cache_dir=None,
        components=b200_panel._abs_datakit_components(),
        train_weights=[
            (0, phase0),
            (tpu_panel.PHASE_1_START_STEP, phase1),
        ],
        auto_build_caches=False,
        mixture_block_size=tpu_panel.MIXTURE_BLOCK_SIZE,
        target_budget=tpu_panel.TARGET_BUDGET_TOKENS,
        experiment_budget=tpu_panel.EXPERIMENT_BUDGET_TOKENS,
    )


def _transect_launch_config(prereg: dict, *, point: str, output_path: str, total_steps: int) -> GrugMoeLaunchConfig:
    run_id = RUN_ID_TEMPLATE.format(point=point)
    return GrugMoeLaunchConfig(
        model=b200_panel.B200_MODEL,  # swarm shapes + gpu_fa4_cute (SM90-supported)
        data=_transect_data_config(prereg, point),
        output_path=output_path,
        run_id=run_id,
        resources=h100_panel.train_resources(h100_panel.DEFAULT_GPUS_PER_RUN),
        steps=total_steps,
        batch_size=tpu_panel.BATCH_SIZE,
        seed=TRANSECT_SEED,
        mp="params=float32,compute=bfloat16,output=bfloat16",
        tracker=WandbConfig(
            project=tpu_panel.WANDB_PROJECT,
            tags=["moe", "transect", "mve", "d512", "h100", point],
            group=WANDB_GROUP,
            name=None,  # resolved to run_id at dispatch
        ),
        optimizer=tpu_panel.SWARM_OPTIMIZER,
        grug_trainer=GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1),
        eval=None,
    )


def verify_transect_parity(prereg: dict) -> dict:
    """Swarm parity + prereg-consistency asserts; returns a printable summary."""
    summary = tpu_panel.verify_swarm_parity()

    constants = prereg["constants"]
    if constants["total_steps"] != tpu_panel.STEPS:
        raise AssertionError("prereg total_steps != launcher STEPS")
    if constants["phase_boundary_step"] != tpu_panel.PHASE_1_START_STEP:
        raise AssertionError("prereg boundary != launcher boundary")
    if constants["target_budget_tokens"] != tpu_panel.TARGET_BUDGET_TOKENS:
        raise AssertionError("prereg target_budget != launcher target_budget")
    if constants["seed"] != TRANSECT_SEED:
        raise AssertionError("prereg seed != launcher seed")

    runs_by_name = {r["run_name"]: r for r in prereg["runs"]}
    points = []
    for point in POINTS:
        run_id = RUN_ID_TEMPLATE.format(point=point)
        run = runs_by_name[run_id]
        phase0 = prereg["mixtures"][run_id]["phase0"]
        share = phase0[run["bucket"]]
        if not math.isclose(share, run["phase0_share"], rel_tol=1e-12):
            raise AssertionError(f"{run_id}: mixture share {share} != runs table {run['phase0_share']}")
        _ = _transect_data_config(prereg, point)  # runs its own asserts
        points.append(
            {
                "point": point,
                "run_id": run_id,
                "bucket": run["bucket"],
                "target_epochs_phase0": run["target_epochs_phase0"],
                "phase0_share": round(run["phase0_share"], 6),
                "step_name": f"users/rav/grug/{run_id}",
            }
        )
    summary["runs"] = points
    summary["seed"] = TRANSECT_SEED
    summary["cluster"] = "cw-rno2a"
    summary["prereg_sha256"] = PREREG_SHA256
    summary["store_root"] = prefix_join(marin_prefix(), datakit_mix._STORE_PREFIX)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--direct", action="store_true", help="train ONE transect point in-process")
    parser.add_argument("--point", type=str, choices=POINTS)
    parser.add_argument("--steps", type=int, default=None, help="step override (smoke only)")
    args = parser.parse_args()

    prereg = load_preregistration()
    summary = verify_transect_parity(prereg)

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        print("DRY RUN: parity + preregistration checks passed; nothing launched.")
        return

    if not args.direct or args.point is None:
        raise SystemExit("transect jobs are submitted directly (one iris job per --point); use --direct --point <p>.")

    total_steps = args.steps or tpu_panel.STEPS
    run_id = RUN_ID_TEMPLATE.format(point=args.point)
    output_path = prefix_join(marin_prefix(), f"users/rav/grug/{run_id}/dev")
    logger.info("direct mode: %s, %d steps, output %s", run_id, total_steps, output_path)
    config = _transect_launch_config(prereg, point=args.point, output_path=output_path, total_steps=total_steps)
    b200_panel._run_direct(config)


if __name__ == "__main__":
    main()
