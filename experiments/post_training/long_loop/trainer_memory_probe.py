# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify chunked PPO output on a retained batch and the real trainer-only Ray geometry.

Run with ten eight-H100 Iris tasks for Exp5's PP2/CP5/EP8/DP8 policy. The source
is a SHA-pinned resolved launch document. No inference engine, sandbox, policy
publication or training checkpoint is created. Run-owned outputs stay separate.
"""

import argparse
import hashlib
import json
import math
import pickle
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import ray
import torch
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join
from skyrl_train.io import io
from skyrl_train.projected_softmax import projected_token_statistics
from skyrl_train.stable_softmax import token_entropy, token_logprobs
from skyrl_train.utils.utils import initialize_ray
from skyrl_train.workers.worker import PPORayActorGroup

from experiments.post_training.long_loop.trainer_memory_probe_config import compiled_config

MODULE = "experiments.post_training.long_loop.trainer_memory_probe"
CUDA_ALLOCATION_HISTORY_ENTRIES = 256


@contextmanager
def capture_cuda_failure(output_prefix, local_directory, *, rank, phase):
    """Persist a rank-local failure without requiring any peer to enter a collective."""
    torch.cuda.memory._record_memory_history(
        enabled="all", context="alloc", stacks="python", max_entries=CUDA_ALLOCATION_HISTORY_ENTRIES, clear_history=True
    )
    try:
        yield
    except Exception as error:
        directory = Path(local_directory) / "trainer-failures"
        directory.mkdir(parents=True, exist_ok=True)
        stem = f"{phase}-rank{rank}"

        def persist(name, raw):
            io.write_bytes_atomic(str(directory / name), raw)
            io.write_bytes_atomic(prefix_join(output_prefix, f"failures/{name}"), raw)

        try:
            persist(
                stem + ".json", json.dumps({"rank": rank, "phase": phase, "traceback": traceback.format_exc()}).encode()
            )
            if isinstance(error, torch.OutOfMemoryError):
                persist(stem + ".memory.txt", torch.cuda.memory_summary().encode())
                snapshot = directory / (stem + ".pickle")
                torch.cuda.memory._dump_snapshot(str(snapshot))
                io.write_bytes_atomic(prefix_join(output_prefix, f"failures/{snapshot.name}"), snapshot.read_bytes())
        except Exception as capture_error:
            error.add_note(f"Failure evidence capture also failed: {capture_error}")
            raise error from capture_error
        raise
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)


def compare_projection(hidden, weight, targets, project, *, temperature, fused):
    """Compare values and native fp32 weight gradients on actual decoder states."""
    hidden = hidden.detach()[:128].clone()
    targets = targets[: hidden.shape[0]]
    coefficients = torch.linspace(-1, 1, hidden.shape[0], device=hidden.device)
    reference_weight = torch.nn.Parameter(weight.detach().clone())
    reference_weight.main_grad = torch.zeros_like(reference_weight, dtype=torch.float32)
    reference_weight.grad_added_to_main_grad = False
    reference_hidden = hidden.detach().clone().requires_grad_()
    with torch.enable_grad():
        logits = project(reference_hidden, reference_weight).div_(temperature)
        lp = token_logprobs(logits.unsqueeze(0), targets.unsqueeze(0), 0, weight.shape[0], None).squeeze(0)
        entropy = token_entropy(logits.unsqueeze(0), None).squeeze(0)
        expected_values = torch.stack((lp, entropy), -1).detach().cpu()
        loss = ((lp + 0.25 * entropy) * coefficients).mean()
        dh, dw = torch.autograd.grad(loss, (reference_hidden, reference_weight))
    expected_hidden = dh.cpu()
    expected_weight = (reference_weight.main_grad if fused else dw).cpu()
    del reference_hidden, reference_weight, logits, lp, entropy, loss, dh, dw

    test_weight = torch.nn.Parameter(weight.detach().clone())
    test_weight.main_grad = torch.zeros_like(test_weight, dtype=torch.float32)
    test_weight.grad_added_to_main_grad = False
    test_hidden = hidden.detach().clone().requires_grad_()
    with torch.enable_grad():
        actual = projected_token_statistics(
            test_hidden,
            test_weight,
            targets,
            project=project,
            chunk_size=31,
            temperature=temperature,
            entropy_gradient=True,
            fused_weight_gradient=fused,
        )
        dh, dw = torch.autograd.grad(
            ((actual[:, 0] + 0.25 * actual[:, 1]) * coefficients).mean(), (test_hidden, test_weight)
        )
    actual_weight = (test_weight.main_grad if fused else dw).cpu()
    actual_hidden = dh.cpu()
    tolerance = torch.finfo(torch.bfloat16).eps if hidden.dtype == torch.bfloat16 else 3e-6
    torch.testing.assert_close(actual.detach().cpu(), expected_values, rtol=tolerance, atol=tolerance)
    metrics = {"value_max_abs": (actual.detach().cpu() - expected_values).abs().max().item()}
    for name, observed, expected in (
        ("hidden_gradient", actual_hidden, expected_hidden),
        ("weight_gradient", actual_weight, expected_weight),
    ):
        relative = (observed.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-20)
        if relative.item() > tolerance:
            raise ValueError(f"Projection {name} relative L2 {relative.item()} exceeds {tolerance}")
        metrics[name + "_relative_l2"] = relative.item()
        metrics[name + "_max_abs"] = (observed.float() - expected.float()).abs().max().item()
    return {"passed": True, "tokens": hidden.shape[0], "tolerance": tolerance, "fused_main_gradient": fused, **metrics}


def applied_update_metrics(outputs):
    """Reject skipped or non-finite native updates before publishing qualification."""
    metrics = []
    for rank, output in enumerate(outputs):
        status = output.metadata["train_status"]
        values = {
            key: float(status[key]) for key in ("policy_update_steps", "skipped_steps", "policy_loss", "raw_grad_norm")
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"Non-finite native update on output {rank}: {values}")
        if values["policy_update_steps"] <= 0 or values["skipped_steps"] != 0 or values["raw_grad_norm"] <= 0:
            raise ValueError(f"Native optimizer did not apply every update on output {rank}: {values}")
        metrics.append(values)
    if not metrics:
        raise ValueError("Native optimizer returned no update receipts")
    return metrics


def driver(args, worker_type):
    cfg = compiled_config(args)
    initialize_ray(cfg)
    placement = cfg.trainer.placement
    try:
        actors = PPORayActorGroup(
            cfg,
            placement.policy_num_nodes,
            placement.policy_num_gpus_per_node,
            ray.remote(num_gpus=1)(worker_type),
            num_gpus_per_actor=1,
            colocate_all=False,
            sequence_parallel_size=1,
            actor_env_vars={"NCCL_BUFFSIZE": str(cfg.trainer.policy.nccl_buffer_size_bytes)},
        )
        identities = ray.get(actors.async_init_model(cfg.trainer.policy.model.path, num_training_steps=10000))
        if len({r["gpu_uuid"] for r in identities}) != len(identities):
            raise RuntimeError("Trainer qualification actors share a GPU")
        # The immutable batch is produced by the owned production run; deserialize only in-region.
        fs, path = filesystem_for(args.batch_uri)
        raw = fs.cat(path)
        batch_sha256 = hashlib.sha256(raw).hexdigest()
        batch = pickle.loads(raw)
        del raw
        batch.metadata = {**batch.metadata, "global_step": 0}
        numerical = ray.get(actors.async_run_ray_method("mesh", "qualify_real_batch", batch))
        io.write_bytes_atomic(
            prefix_join(args.output_prefix, "real-batch-numerics.json"),
            json.dumps({"batch_uri": args.batch_uri, "batch_sha256": batch_sha256, "ranks": numerical}).encode(),
        )
        timings = []
        for index in range(2):
            batch.metadata = {**batch.metadata, "global_step": index + 1}
            started = time.monotonic()
            outputs = ray.get(actors.async_run_ray_method("mesh", "ppo_train", batch))
            timings.append(
                {
                    "warm_optimizer": index > 0,
                    "seconds": time.monotonic() - started,
                    "rank_outputs": len(outputs),
                    "updates": applied_update_metrics(outputs),
                    "rank_measurements": [output.metadata["qualification"] for output in outputs],
                }
            )
            io.write_bytes_atomic(
                prefix_join(args.output_prefix, "matched-batch-timing.json"),
                json.dumps({"variant": args.variant, "batch_sha256": batch_sha256, "timings": timings}).encode(),
            )
        memory = (
            ray.get(actors.async_run_ray_method("pass_through", "profile_train_memory"))
            if args.variant == "chunked"
            else []
        )
        result = {
            "completed": True,
            "qualification_passed": True,
            "variant": args.variant,
            "policy_ranks": len(identities),
            "identities": identities,
            "numerical": numerical,
            "timings": timings,
            "memory": memory,
        }
        io.write_bytes_atomic(prefix_join(args.output_prefix, "receipt.json"), json.dumps(result).encode())
        print(json.dumps({"qualification_passed": True, "policy_ranks": len(identities)}), flush=True)
    except Exception:
        io.write_bytes_atomic(
            prefix_join(args.output_prefix, "driver-failure.json"),
            json.dumps({"variant": args.variant, "traceback": traceback.format_exc()}).encode(),
        )
        raise
    finally:
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", action="store_true")
    parser.add_argument("--config-uri", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--batch-uri", required=True)
    parser.add_argument("--max-sequence-length", type=int, required=True)
    parser.add_argument("--variant", required=True, choices=("chunked", "offload"))
    args = parser.parse_args()
    if args.driver:
        # Megatron's CUDA extensions are optional for CPU driver admission tests.
        from experiments.post_training.long_loop.trainer_memory_probe_worker import QualifyingWorker  # noqa: PLC0415

        driver(args, QualifyingWorker)
        return
    from experiments.post_training.long_loop.snowball_ray_replica_probe import (  # noqa: PLC0415 - GPU launcher
        run_ray_probe_task,
    )

    cfg = compiled_config(args)
    placement = cfg.trainer.placement
    run_ray_probe_task(
        args,
        cfg,
        num_tasks=placement.policy_num_nodes,
        gpus_per_task=placement.policy_num_gpus_per_node,
        driver_module=MODULE,
        driver_arguments=(
            "--batch-uri",
            args.batch_uri,
            "--variant",
            args.variant,
            "--max-sequence-length",
            str(args.max_sequence_length),
        ),
    )


if __name__ == "__main__":
    main()
