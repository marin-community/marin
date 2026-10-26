# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Evaluate the review-v9 switch-window checkpoints on programming-languages and C4 BPB.

The gradient-conflict panel kept full checkpoints 256 and 64 steps before, at, and 64 and 256
steps after the decay-onset switch of P2 (tied 0.70), P3 (0.02 to 0.82), and P4 (tied 0.18), and
64 steps before, at, and 64 steps after the constant-learning-rate switches of the H5 0.6T and
0.7T arms. Evaluating them offline fills the window that the 1,000-step training-time eval
cadence misses. Final-step checkpoints are evaluated too, as a control against the persisted
training-time evals. The final checkpoints of the August 4x confirmation runs and of the
transition-ramp runs are evaluated with the same code, so endpoint comparisons across launches
do not depend on evaluation-code drift.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import equinox as eqx
import haliax as hax
import jax
import jmp
from fray.types import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import load_checkpoint
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.eval import LossFnOutput, TaggedEvaluator, eval_model
from levanter.models.lm_model import LmExample, LmHeadModel
from levanter.tracker.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.mesh import MeshConfig
from levanter.utils.tree_utils import inference_mode
from marin.execution.remote import remote
from rigging.filesystem.buckets import filesystem_for

from experiments.datasets.paloma import paloma_datasets
from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base
from experiments.llama import llama3_tokenizer
from experiments.scaling_law_sweeps.completed_adamh import CompletedAdamHHeuristic

logger = logging.getLogger(__name__)

PREFIX = "gs://marin-us-central1"
TRAJECTORY_ROOT = (
    f"{PREFIX}/checkpoints/pinlin_calvin_xu/data_mixture/"
    "starcoder_wsd80_gradient_conflict_review_v9_20260811/trajectories"
)
TRAJECTORY_VERSION = "2026.08.11.9"
OUTPUT_ROOT = f"{PREFIX}/checkpoints/pinlin_calvin_xu/data_mixture/starcoder_wsd80_switch_window_evals_20261006"
VERSION = "2026.10.06"
EVAL_SETS = ("dolma_100_programing_languages", "c4_en")
HIDDEN_SIZE = 640
DECAY_WINDOW_STEPS = (22352, 22544, 22608, 22672, 22864, 28259)
P_ARMS = {
    "p2": "confirmed-tied-comparator",
    "p3": "confirmed-two-phase-winner",
    "p4": "aggregate-nearest-tied-018",
}
P_SUPPORTS = ("m100a", "full")
P_SEEDS = tuple(range(2026082000, 2026082008))
H5_ARMS = {
    "boundary_beta_0p60": (16832, 16896, 16960, 28159),
    "boundary_beta_0p70": (19648, 19712, 19776, 28159),
}
H5_SEEDS = tuple(range(2026084000, 2026084016))
CONFIRMATION_ROOT = (
    f"{PREFIX}/checkpoints/pinlin_calvin_xu/data_mixture/"
    "starcoder_wsd80_dense_support_empirical_optimum_confirmation_20260811"
)
RAMP_ROOT = f"{PREFIX}/checkpoints/pinlin_calvin_xu/data_mixture/starcoder_wsd80_transition_ramp_20261006"
TRAINING_VERSION = "2026.07.11"
FINAL_STEP = 28259
CONFIRMATION_SEEDS = ("0821", "0822", "0823", "0824", "0825")
CONFIRMATION_RUNS = {
    "august_abrupt": "dsopt_r3d28260_m400_untied_c007_s{seed}",
    "august_tied": "dsopt_r3d28260_m400_tied_c043_s{seed}",
}
RAMP_ARMS = ("abrupt", "ramp0128", "ramp0256", "ramp0512", "ramp2048", "ramp5632")
TPU = ResourceConfig.with_tpu("v5p-8", cpu=8, ram="32g", disk="32g", regions=("us-central1",), zone="us-central1-a")
WORKERS = 4


def checkpoint_requests() -> list[dict[str, Any]]:
    """Every (trajectory, step) pair to evaluate, with its checkpoint and result locations."""
    requests: list[dict[str, Any]] = []

    def add(trajectory_id: str, arm: str, support_id: str, seed: int, step: int) -> None:
        requests.append(
            {
                "trajectory_id": trajectory_id,
                "arm": arm,
                "support_id": support_id,
                "training_seed": seed,
                "step": step,
                "checkpoint": f"{TRAJECTORY_ROOT}/{trajectory_id}/{TRAJECTORY_VERSION}/checkpoints/step-{step}",
                "result": f"{OUTPUT_ROOT}/{VERSION}/{trajectory_id}/step-{step}.json",
            }
        )

    for arm, role in P_ARMS.items():
        for support in P_SUPPORTS:
            for seed in P_SEEDS:
                for step in DECAY_WINDOW_STEPS:
                    add(f"gcf_{arm}_r3d28260_{support}_{role}_s{seed}", arm, support, seed, step)
    for role, steps in H5_ARMS.items():
        for seed in H5_SEEDS:
            for step in steps:
                add(f"gcf_b_h5d28160_m100a_{role}_s{seed}", role, "m100a", seed, step)

    def add_final(root: str, run_name: str, arm: str, seed: str) -> None:
        checkpoint = f"{root}/{run_name}/{TRAINING_VERSION}/checkpoints/step-{FINAL_STEP}"
        if not _exists(f"{checkpoint}/metadata.json"):
            return
        requests.append(
            {
                "trajectory_id": run_name,
                "arm": arm,
                "support_id": "m400",
                "training_seed": int("2026" + seed),
                "step": FINAL_STEP,
                "checkpoint": checkpoint,
                "result": f"{OUTPUT_ROOT}/{VERSION}/{run_name}/step-{FINAL_STEP}.json",
            }
        )

    for arm, template in CONFIRMATION_RUNS.items():
        for seed in CONFIRMATION_SEEDS:
            add_final(CONFIRMATION_ROOT, template.format(seed=seed), arm, seed)
    for arm in RAMP_ARMS:
        for seed in CONFIRMATION_SEEDS:
            add_final(RAMP_ROOT, f"tramp_r3d28260_m400_{arm}_s{seed}", arm, seed)
    return requests


def eval_set_paths() -> dict[str, str]:
    handles = paloma_datasets(tokenizer=llama3_tokenizer)
    return {handles[name].name: handles[name].path(PREFIX) for name in EVAL_SETS}


def _exists(url: str) -> bool:
    fs, path = filesystem_for(url)
    return fs.exists(path)


def _read_json(url: str) -> dict[str, Any]:
    fs, path = filesystem_for(url)
    with fs.open(path, "rb") as handle:
        return json.loads(handle.read().decode())


def _write_json(url: str, payload: dict[str, Any]) -> None:
    fs, path = filesystem_for(url)
    with fs.open(path, "wb") as handle:
        handle.write(json.dumps(payload, indent=1).encode())


def eval_loss(model: LmHeadModel, batch: LmExample) -> LossFnOutput:
    model = jmp.get_policy("p=f32,c=bfloat16").cast_to_compute(inference_mode(model, True))
    losses = model.compute_next_token_loss(batch, reduction=None, reduction_axis=()).array
    return losses, batch.loss_weight.array, jax.numpy.roll(batch.tokens.array, -1, axis=-1)


def evaluate_checkpoints(requests: list[dict[str, Any]], paths: dict[str, str]) -> None:
    """Reuse one compiled evaluator for every checkpoint of the shard; skip results that exist."""
    pending = [request for request in requests if not _exists(request["result"])]
    logger.info("Evaluating %d of %d checkpoints", len(pending), len(requests))
    if not pending:
        return
    trainer = TrainerConfig(
        tracker=NoopConfig(),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=base.BATCH_SIZE,
        per_device_parallelism=-1,
        per_device_eval_parallelism=-1,
        allow_nondivisible_batch_size=True,
        max_eval_batches=None,
        log_jaxprs=False,
        log_xla_hlo=False,
        mesh=MeshConfig(
            axes={"data": -1, "replica": 1, "model": 1},
            compute_mapping={
                "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
            },
        ),
    )
    trainer.initialize()
    data = LmDataConfig(
        tokenizer=llama3_tokenizer,
        cache_dir=None,
        auto_build_caches=False,
        components={name: DatasetComponent(cache_dir=path) for name, path in paths.items()},
        train_weights={name: 0.0 for name in paths},
    )
    tokenizer = data.the_tokenizer
    model_config = CompletedAdamHHeuristic()._build_model_config(HIDDEN_SIZE, seq_len=base.SEQ_LEN)
    with trainer.use_device_mesh():
        evaluator = TaggedEvaluator(
            EvalBatch=trainer.EvalBatch,
            tagged_eval_sets=data.tagged_eval_sets(model_config.max_Pos.resize(base.SEQ_LEN)),
            loss_fn=eval_loss,
            tokenizer=tokenizer,
            axis_mapping=trainer.compute_axis_mapping,
        )
        for index, request in enumerate(pending):
            with use_cpu_device():
                model = eqx.filter_eval_shape(
                    model_config.build, hax.Axis("vocab", len(tokenizer)), key=jax.random.PRNGKey(0)
                )
                model = load_checkpoint(model, request["checkpoint"], subpath="model")
            model = hax.shard_with_axis_mapping(model, trainer.parameter_axis_mapping)
            metrics = {key: float(value) for key, value in eval_model(evaluator, model, prefix="eval").items()}
            if jax.process_index() == 0:
                _write_json(request["result"], {**request, "metrics": metrics})
                logger.info(
                    "[%d/%d] %s step %d done", index + 1, len(pending), request["trajectory_id"], request["step"]
                )
            del model


async def submit() -> None:
    requests = [request for request in checkpoint_requests() if not _exists(request["result"])]
    logger.info("%d checkpoints pending", len(requests))
    if not requests:
        return
    paths = eval_set_paths()
    shards: list[list[dict[str, Any]]] = [[] for _ in range(WORKERS)]
    trajectories = sorted({request["trajectory_id"] for request in requests})
    for index, trajectory_id in enumerate(trajectories):
        shards[index % WORKERS].extend(r for r in requests if r["trajectory_id"] == trajectory_id)
    calls = []
    for index, shard in enumerate(shards):
        if not shard:
            continue
        worker = remote(
            evaluate_checkpoints, name=f"sw-window-eval-{index}", resources=TPU, env_vars={"MARIN_PREFIX": PREFIX}
        )
        calls.append(asyncio.to_thread(worker, shard, paths))
    await asyncio.gather(*calls)


def collect(output: Path) -> None:
    """Gather every result into one CSV with the persisted training-time evals as a control."""
    requests = checkpoint_requests()
    with ThreadPoolExecutor(16) as executor:
        done = list(executor.map(lambda r: _exists(r["result"]), requests))
    rows = []
    for request, finished in zip(requests, done, strict=True):
        if not finished:
            continue
        result = _read_json(request["result"])
        row = {key: request[key] for key in ("trajectory_id", "arm", "support_id", "training_seed", "step")}
        for name in EVAL_SETS:
            row[f"{name}_bpb"] = result["metrics"][f"eval/paloma/{name}-llama3/bpb"]
        rows.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Collected %d of %d results into %s", len(rows), len(requests), output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--collect", type=Path, help="Write the gathered results to this CSV")
    parser.add_argument("--list", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    if os.getenv("CI") is not None:
        return
    if args.list:
        requests = checkpoint_requests()
        print(f"{len(requests)} checkpoints across {len({r['trajectory_id'] for r in requests})} trajectories")
        return
    if args.submit:
        asyncio.run(submit())
    if args.collect is not None:
        collect(args.collect)


if __name__ == "__main__":
    main()
