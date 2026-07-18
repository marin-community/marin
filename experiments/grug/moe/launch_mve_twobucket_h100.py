# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug-MoE two-bucket factorial on cw-rno2a H100s (issue #7067, rav directive 2026-07-18).

Twenty-five single-seed (seed 0) runs over mixtures of exactly TWO buckets —
code ``c01q0`` (152.6B tok) and web ``c05q0`` (876.9B tok, the largest
web_text bucket) — per the frozen pre-registration
``twobucket_preregistration.json`` (built by
``experiments/datakit/mixture_features/twobucket_design.py``; a byte-identical
copy is staged next to this launcher so it ships in job bundles; the sha is
asserted at every start):

- NATURAL arm (8): w_code in {0,.05,.1,.2,.35,.5,.75,1}, both phases,
  simulated epoching ON (target budget unchanged) — the #2846-replica sweep.
- FACTORIAL arms (17): budgets OFF (real epochs); the code stream is
  sub-sliced via ``max_train_batches={'c01q0': n}`` (slice-after-shuffle, the
  same subset mechanism as simulated epoching; content-fair at io-block
  granularity) so epochs and weight decouple:
  axis 1 w at fixed e~4; axis 2 e in {1..32} at w=.2 (kernel predicts NO
  change along it — the pure harm-term test); axis 3 budget {2.5B,40B} x
  e {4,16}; axis 4 d256 x e {2,8,32}.

Training constants otherwise IDENTICAL to the seed panel / transect
(``launch_mve_seedpanel_h100``): swarm shapes + ``gpu_fa4_cute``,
``SWARM_OPTIMIZER`` (fractional schedule — warmup peak verified at 0.100*N for
every step count used here), H100x8 one-node runs, cuda_async allocator,
CW-mirror data, no in-training eval, post-hoc ``eval_logprob`` readout.

The d256 model (axis 4) is built from ``MoeHeuristic().build_model_config(256,
seq_len=4096)`` plus the SAME four swarm-family replacements that reproduce the
d512 swarm model exactly from ``build_model_config(512)`` — asserted at every
start (config-parity in lieu of a smoke; the build path is unchanged).
"""

import argparse
import dataclasses
import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np
from levanter.data.text.datasets import LmDataConfig
from levanter.tracker.wandb import WandbConfig
from rigging.filesystem import marin_prefix, prefix_join

from experiments.grug.moe import launch_datakit_moe_mix as datakit_mix
from experiments.grug.moe import launch_mve_seedpanel as tpu_panel
from experiments.grug.moe import launch_mve_seedpanel_b200 as b200_panel
from experiments.grug.moe import launch_mve_seedpanel_h100 as h100_panel
from experiments.grug.moe.heuristic import MoeHeuristic
from experiments.grug.moe.launch import GrugMoeLaunchConfig
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugTrainerConfig
from experiments.marin_tokenizer import marin_tokenizer

logger = logging.getLogger(__name__)

PREREG_PATH = Path(__file__).resolve().parent / "twobucket_preregistration.json"
PREREG_SHA256 = "96dfba307182529ed88fc14300dc28b61fe6e432117d9e4a2c3598e37ef81083"

CODE_BUCKET = "c01q0"
WEB_BUCKET = "c05q0"
RUN_ID_TEMPLATE = "rav_mve_twobucket_{point}"
WANDB_GROUP = "rav_mve_twobucket"
TWOBUCKET_SEED = 0
VALID_STEPS = (1_194, 4_776, 19_104)


def load_preregistration() -> dict:
    raw = PREREG_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != PREREG_SHA256:
        raise AssertionError(f"preregistration sha mismatch: {digest} != {PREREG_SHA256}")
    return json.loads(raw)


def _boundary_step(total_steps: int) -> int:
    """The program's block-quantized 0.8-fraction phase boundary."""
    step_multiple = tpu_panel.MIXTURE_BLOCK_SIZE // math.gcd(tpu_panel.MIXTURE_BLOCK_SIZE, tpu_panel.BATCH_SIZE)
    requested = max(1, int(total_steps * tpu_panel.PHASE_1_START_FRACTION))
    return max(step_multiple, (requested // step_multiple) * step_multiple)


# The four fields on which the swarm/panel model differs from the heuristic's
# architecture build at the same hidden size (verified: applying these to
# build_model_config(512) reproduces B200_MODEL exactly).
_SWARM_FAMILY_REPLACEMENTS = dict(
    num_experts=64,
    sliding_window=4096,
    router_z_loss_coef=0.001,
    disable_long_rope=False,
    attention_implementation="gpu_fa4_cute",
)


def _family_model(hidden_dim: int) -> GrugModelConfig:
    base = MoeHeuristic().build_model_config(hidden_dim, seq_len=tpu_panel.SEQ_LEN)
    return dataclasses.replace(base, **_SWARM_FAMILY_REPLACEMENTS)


def _assert_d256_parity() -> None:
    """Config-parity note for the d256 arm: the same build path reproduces d512 exactly."""
    if _family_model(512) != b200_panel.B200_MODEL:
        d1 = dataclasses.asdict(_family_model(512))
        d2 = dataclasses.asdict(b200_panel.B200_MODEL)
        diff = {k: (d1[k], d2[k]) for k in d1 if d1[k] != d2[k]}
        raise AssertionError(f"family build at d512 does not reproduce the swarm model: {diff}")


def _model_for(name: str) -> GrugModelConfig:
    if name == "d512":
        return b200_panel.B200_MODEL
    if name == "d256":
        return _family_model(256)
    raise ValueError(f"unknown model {name}")


def _two_bucket_weights(w_code: float) -> dict[str, float]:
    return {CODE_BUCKET: w_code, WEB_BUCKET: 1.0 - w_code}


def _twobucket_data_config(run: dict) -> LmDataConfig:
    steps = int(run["steps"])
    if steps not in VALID_STEPS:
        raise AssertionError(f"{run['run_id']}: unexpected steps {steps}")
    boundary = _boundary_step(steps)
    if boundary != run["phase_boundary_step"]:
        raise AssertionError(f"{run['run_id']}: boundary {boundary} != prereg {run['phase_boundary_step']}")
    weights = _two_bucket_weights(float(run["w_code"]))
    if abs(sum(weights.values()) - 1.0) > 1e-12 or any(v < 0 for v in weights.values()):
        raise AssertionError(f"{run['run_id']}: bad weights {weights}")

    if run["simulated_epoching"]:
        target_budget = int(run["target_budget_tokens"])
        experiment_budget = steps * tpu_panel.BATCH_SIZE * tpu_panel.SEQ_LEN
        if target_budget != tpu_panel.TARGET_BUDGET_TOKENS:
            raise AssertionError(f"{run['run_id']}: target budget {target_budget}")
        if experiment_budget != run["experiment_budget_tokens"]:
            raise AssertionError(f"{run['run_id']}: experiment budget mismatch")
        max_train_batches = None
    else:
        target_budget = None
        experiment_budget = None
        n = int(run["code_slice_batches"])
        if n < 1:
            raise AssertionError(f"{run['run_id']}: bad slice {n}")
        # exact real epochs: e0_code = w * boundary / n, asserted against the prereg
        e0 = float(run["w_code"]) * boundary / n
        if abs(e0 - run["epochs"]["code"]["phase0"]) > 1e-9:
            raise AssertionError(f"{run['run_id']}: e0 {e0} != prereg {run['epochs']['code']['phase0']}")
        if run["code_slice_tokens"] != n * tpu_panel.BATCH_SIZE * tpu_panel.SEQ_LEN:
            raise AssertionError(f"{run['run_id']}: slice tokens mismatch")
        max_train_batches = {CODE_BUCKET: n}

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
        target_budget=target_budget,
        experiment_budget=experiment_budget,
        max_train_batches=max_train_batches,
    )


def _twobucket_launch_config(run: dict, *, output_path: str, total_steps: int) -> GrugMoeLaunchConfig:
    return GrugMoeLaunchConfig(
        model=_model_for(run["model"]),
        data=_twobucket_data_config(run),
        output_path=output_path,
        run_id=run["run_id"],
        resources=h100_panel.train_resources(h100_panel.DEFAULT_GPUS_PER_RUN),
        steps=total_steps,
        batch_size=tpu_panel.BATCH_SIZE,
        seed=TWOBUCKET_SEED,
        mp="params=float32,compute=bfloat16,output=bfloat16",
        tracker=WandbConfig(
            project=tpu_panel.WANDB_PROJECT,
            tags=["moe", "twobucket", "mve", "h100", run["model"], run["arm"], run["point"]],
            group=WANDB_GROUP,
            name=None,  # resolved to run_id at dispatch
        ),
        optimizer=tpu_panel.SWARM_OPTIMIZER,
        grug_trainer=GrugTrainerConfig(z_loss_weight=1e-4, ema_beta=None, log_every=1),
        eval=None,
    )


def _verify_schedule_scales() -> dict:
    """Warmup peak lands at 0.100*N for every step count used in this experiment.

    Evaluating optax schedules runs JAX computations, which would initialise the
    XLA backend BEFORE ``jax.distributed.initialize()`` inside a GPU job — so
    this check runs in ``--dry-run`` only (it is deterministic in the step
    counts; the dry-run result is recorded in the pre-registration).
    """
    out = {}
    for n in (*VALID_STEPS, tpu_panel.STEPS):
        sched = tpu_panel.SWARM_OPTIMIZER.lr_scheduler(n)
        lrs = np.array([float(sched(i)) for i in range(n)])
        peak = int(np.argmax(lrs))
        frac = peak / n
        if abs(frac - 0.1) > 0.001:
            raise AssertionError(f"warmup peak fraction {frac} != 0.1 at N={n}")
        out[str(n)] = {"peak_step": peak, "peak_frac": frac, "peak_lr": float(lrs.max())}
    return out


def verify_twobucket_parity(prereg: dict) -> dict:
    """Swarm parity + d256 parity + prereg-consistency asserts; returns a printable summary."""
    summary = tpu_panel.verify_swarm_parity()
    _assert_d256_parity()

    runs = prereg["runs"]
    if len(runs) != 25 or len({r["run_id"] for r in runs}) != 25:
        raise AssertionError(f"expected 25 unique runs, got {len(runs)}")
    if prereg["constants"]["seed"] != TWOBUCKET_SEED:
        raise AssertionError("prereg seed != launcher seed")
    if prereg["buckets"]["code"]["bucket"] != CODE_BUCKET or prereg["buckets"]["web"]["bucket"] != WEB_BUCKET:
        raise AssertionError("prereg buckets != launcher buckets")

    component_names = set(b200_panel._abs_datakit_components().keys())
    if not {CODE_BUCKET, WEB_BUCKET} <= component_names:
        raise AssertionError("two-bucket components missing from the datakit component set")

    points = []
    for run in runs:
        _ = _twobucket_data_config(run)  # runs its own asserts
        points.append(
            {
                "point": run["point"],
                "run_id": run["run_id"],
                "arm": run["arm"],
                "model": run["model"],
                "w_code": run["w_code"],
                "steps": run["steps"],
                "code_slice_batches": run["code_slice_batches"],
                "e0_code": round(run["epochs"]["code"]["phase0"], 4),
                "e0_web": round(run["epochs"]["web"]["phase0"], 4),
                "simulated_epoching": run["simulated_epoching"],
            }
        )
    summary["runs"] = points
    summary["seed"] = TWOBUCKET_SEED
    summary["cluster"] = "cw-rno2a"
    summary["prereg_sha256"] = PREREG_SHA256
    summary["store_root"] = prefix_join(marin_prefix(), datakit_mix._STORE_PREFIX)
    summary["in_training_validation"] = False
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--direct", action="store_true", help="train ONE grid point in-process")
    parser.add_argument("--point", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None, help="step override (smoke only)")
    args = parser.parse_args()

    prereg = load_preregistration()
    summary = verify_twobucket_parity(prereg)

    if args.dry_run:
        summary["schedule_check"] = _verify_schedule_scales()  # JAX-touching: dry-run only
        print(json.dumps(summary, indent=2))
        print("DRY RUN: parity + preregistration checks passed; nothing launched.")
        return

    if not args.direct or args.point is None:
        raise SystemExit("twobucket jobs are submitted directly (one iris job per --point); use --direct --point <p>.")

    by_point = {r["point"]: r for r in prereg["runs"]}
    if args.point not in by_point:
        raise SystemExit(f"unknown point {args.point}; valid: {sorted(by_point)}")
    run = by_point[args.point]
    total_steps = args.steps or int(run["steps"])
    output_path = prefix_join(marin_prefix(), f"users/rav/grug/{run['run_id']}/dev")
    logger.info("direct mode: %s, %d steps, output %s", run["run_id"], total_steps, output_path)
    config = _twobucket_launch_config(run, output_path=output_path, total_steps=total_steps)
    b200_panel._run_direct(config)


if __name__ == "__main__":
    main()
