# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""C1: the longest single sequence the Megatron policy can score on one node.

Boots the SkyRL Megatron policy worker on a Snowball checkpoint with the production
scoring geometry — tensor-parallel 1, pipeline-parallel 1, expert-parallel 8, no
context parallelism (Snowball's sliding-window layers and five KV heads block both
context-parallel backends) — then runs the reference score pass (top-256 per
position, ``probe_score_forward``) on synthetic token ids at increasing lengths,
recording memory, wall time, and the largest length that succeeds.

    python -m experiments.post_training.mismatch_probe.long_context \
        --model-path s3://marin-rno2a/.../models/b-step92 --lengths 32768 65536 --long-rows 8

An out-of-memory at one length is a recorded outcome: the run stops there and the
report names the largest length that scored. Run inside a SkyRL GPU job, whose
runtime environment provides megatron and ray.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from contextlib import contextmanager

import fsspec
import numpy as np
import ray
import skyrl_train.models.grug_megatron_bridge  # noqa: F401  # registers the Grug bridge
import torch
from hydra import compose, initialize_config_dir
from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils import initialize_ray
from skyrl_train.workers.megatron.megatron_worker import PolicyWorker

from experiments.post_training.mismatch_probe.scoring_pass import TOP_K


def scoring_config(model_path: str, *, num_gpus: int) -> object:
    """The policy-worker config for one-node forward-only scoring."""
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="ppo_base_config")
    cfg.trainer.policy.model.path = model_path
    cfg.trainer.critic.model.path = None
    cfg.trainer.strategy = "megatron"
    cfg.trainer.flash_attn = True
    cfg.trainer.bf16 = True
    cfg.trainer.gradient_checkpointing = True
    cfg.trainer.use_sample_packing = False
    cfg.trainer.train_batch_size = 1
    cfg.trainer.policy_mini_batch_size = 1
    cfg.trainer.eval_batch_size = 1
    cfg.generator.n_samples_per_prompt = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.micro_forward_batch_size_per_gpu = 1
    cfg.trainer.logger = "console"
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.policy_num_gpus_per_node = num_gpus
    megatron = cfg.trainer.policy.megatron_config
    megatron.tensor_model_parallel_size = 1
    megatron.pipeline_model_parallel_size = 1
    megatron.context_parallel_size = 1
    megatron.expert_model_parallel_size = num_gpus
    megatron.expert_tensor_parallel_size = 1
    return cfg


SHORT_ROW_TOKENS = 1_024


def synthetic_batch(length: int, vocabulary_size: int, seed: int, *, rows: int, long_rows: int) -> TrainingInputBatch:
    """``rows`` synthetic sequences of in-vocabulary token ids, the first ``long_rows`` of ``length``.

    The mesh dispatch splits the batch over the data-parallel ranks, one row per
    rank. The remaining rows hold ``SHORT_ROW_TOKENS`` real tokens, left-padded:
    every rank's tokens meet in the expert all-to-all, so the number of long rows
    per forward, not only their length, sets each rank's expert load.
    """
    generator = np.random.default_rng(seed)
    token_ids = generator.integers(10, vocabulary_size, size=(rows, length))
    attention_mask = torch.ones(rows, length, dtype=torch.int64)
    attention_mask[long_rows:, : length - SHORT_ROW_TOKENS] = 0
    batch = TrainingInputBatch({"sequences": torch.from_numpy(token_ids), "attention_mask": attention_mask})
    batch.metadata = {"response_length": length}
    return batch


def score_length(policy, length: int, vocabulary_size: int, seed: int, *, num_gpus: int, long_rows: int) -> dict:
    """One reference score pass of one ``length``-token sequence per GPU; returns the timing and memory record."""
    batch = synthetic_batch(length, vocabulary_size, seed, rows=num_gpus, long_rows=long_rows)
    batch.metadata["probe_micro_batch_size"] = 1
    batch.metadata["probe_score_top_k"] = TOP_K
    started = time.monotonic()
    ray.get(policy.async_run_ray_method("mesh", "probe_score_forward", data=batch))
    duration = time.monotonic() - started
    memory = ray.get(policy.async_run_ray_method("pass_through", "get_cuda_memory"))
    return {"length": length, "long_rows": long_rows, "seconds": duration, "memory": memory}


def vocabulary_size_of(model_path: str) -> int:
    filesystem, path = fsspec.core.url_to_fs(model_path)
    with filesystem.open(f"{path.rstrip('/')}/config.json") as handle:
        return int(json.load(handle)["vocab_size"])


@contextmanager
def scoring_worker(model_path: str, *, num_gpus: int, worker_type=PolicyWorker) -> Iterator[object]:
    """The policy worker with the checkpoint loaded for forward-only scoring.

    ``init_model_for_export`` builds the model without DDP gradient buffers or an
    optimizer; the training init's fp32 buffers alone overflow an H100 at EP8. On
    exit the actors and their placement group are released, so a second worker
    can claim the node's GPUs.
    """
    from ray.util.placement_group import placement_group, remove_placement_group  # noqa: PLC0415  # ray only at runtime
    from skyrl_train.utils import get_ray_pg_ready_with_timeout  # noqa: PLC0415  # skyrl_train only in a SkyRL runtime
    from skyrl_train.workers.worker import PPORayActorGroup  # noqa: PLC0415

    cfg = scoring_config(model_path, num_gpus=num_gpus)
    if not ray.is_initialized():  # the exact pass boots a second worker in the same driver
        initialize_ray(cfg)
    pg = placement_group([{"GPU": num_gpus, "CPU": num_gpus}], strategy="PACK")
    get_ray_pg_ready_with_timeout(pg, timeout=600)
    policy = PPORayActorGroup(
        cfg,
        num_nodes=1,
        num_gpus_per_node=num_gpus,
        ray_actor_type=worker_type,
        pg=pg,
        num_gpus_per_actor=0.75,
        colocate_all=False,
        sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
        record_memory=cfg.trainer.policy.record_memory,
    )
    try:
        ray.get(policy.async_run_ray_method("pass_through", "init_model_for_export", model_path))
        yield policy
    finally:
        policy.kill_actors()
        remove_placement_group(pg)


def scan_lengths(policy, args) -> dict:
    """Score increasing lengths until one fails; the report names the largest that scored."""
    vocabulary_size = vocabulary_size_of(args.model_path)
    records = []
    largest = None
    for length in sorted(args.lengths):
        try:
            record = score_length(
                policy, length, vocabulary_size, args.seed, num_gpus=args.num_gpus, long_rows=args.long_rows
            )
        except Exception as error:
            # The worker's own message (e.g. CUDA's "Tried to allocate") ends the Ray error text.
            records.append({"length": length, "failed": type(error).__name__, "detail": str(error)[-3000:]})
            print(f"length {length}: FAILED {type(error).__name__}: {str(error)[-3000:]}")
            break
        largest = length
        records.append(record)
        peak = max(rank["peak_allocated"] for rank in record["memory"])
        print(f"length {length}: {record['seconds']:.0f} s, max peak allocated {peak / 2**30:.1f} GiB over ranks")
    return {"model_path": args.model_path, "long_rows": args.long_rows, "largest_scored": largest, "lengths": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--long-rows", type=int, required=True, help="rows (one per GPU) at full length")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    try:
        with scoring_worker(args.model_path, num_gpus=args.num_gpus) as policy:
            report = scan_lengths(policy, args)
        with open(args.report, "w") as handle:
            json.dump(report, handle, indent=2)
        print(f"LARGEST SCORED LENGTH: {report['largest_scored']}")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
