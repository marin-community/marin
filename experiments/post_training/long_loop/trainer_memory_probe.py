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
import pickle
import time

import megatron.core.parallel_state as mpu
import ray
import torch
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join
from skyrl_train.projected_softmax import projected_token_statistics
from skyrl_train.stable_softmax import token_entropy, token_logprobs
from skyrl_train.utils.utils import initialize_ray
from skyrl_train.workers.megatron.megatron_model_wrapper import MegatronModelWrapper
from skyrl_train.workers.megatron.megatron_worker import MegatronPolicyWorkerBase
from skyrl_train.workers.worker import PPORayActorGroup

from experiments.post_training.long_loop.snowball_ray_replica_probe import run_ray_probe_task
from experiments.post_training.long_loop.snowball_replica_probe import write_receipt
from experiments.post_training.long_loop.trainer_memory_probe_config import compiled_config

MODULE = "experiments.post_training.long_loop.trainer_memory_probe"


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


class QualifyingWrapper(MegatronModelWrapper):
    qualification = None

    def _output_statistics(self, **kwargs):
        if self.qualification is None:
            hidden = kwargs["hidden_states"]
            sequences, mask, packed, temperature = kwargs["context"]
            targets = self._projection_targets(sequences, mask, packed, kwargs["config"])
            layer = kwargs["output_layer"]
            weight = layer.weight if kwargs["output_weight"] is None else kwargs["output_weight"]

            def project(chunk, parameter):
                logits, _ = layer(chunk.unsqueeze(1), weight=parameter, runtime_gather_output=False)
                return kwargs["scale_logits"](logits).squeeze(1)

            self.qualification = compare_projection(
                hidden.reshape(-1, hidden.shape[-1]),
                weight,
                targets,
                project,
                temperature=temperature,
                fused=kwargs["config"].gradient_accumulation_fusion,
            )
        return super()._output_statistics(**kwargs)


class QualifyingWorker(MegatronPolicyWorkerBase):
    def init_model(self, *args, **kwargs):
        super().init_model(*args, **kwargs)
        replay = self.model.router_replay
        self.model = QualifyingWrapper(
            self.cfg,
            self.actor_module,
            self.optimizer,
            self.policy_loss_fn,
            vocabulary_size=self.strategy.hf_config.vocab_size,
            output_projection_chunk_size=self.cfg.trainer.policy.megatron_config.output_projection_chunk_size,
        )
        self.model.router_replay = replay
        return {"rank": self._rank, "gpu_uuid": str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)}

    def qualify_real_batch(self, batch):
        self.forward(batch)
        report = {
            "rank": self._rank,
            "pipeline_last": mpu.is_pipeline_last_stage(ignore_virtual=True),
            "projection": self.model.qualification,
        }
        if (
            report["pipeline_last"]
            and self.model._output_projection_chunk_size is not None
            and report["projection"] is None
        ):
            raise RuntimeError("Last pipeline stage did not qualify the real-batch output projection")
        return report


def driver(args):
    cfg = compiled_config(args)
    initialize_ray(cfg)
    placement = cfg.trainer.placement
    try:
        actors = PPORayActorGroup(
            cfg,
            placement.policy_num_nodes,
            placement.policy_num_gpus_per_node,
            ray.remote(num_gpus=1)(QualifyingWorker),
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
        write_receipt(
            prefix_join(args.output_prefix, "real-batch-numerics.json"),
            {"batch_uri": args.batch_uri, "batch_sha256": batch_sha256, "ranks": numerical},
        )
        timings = []
        for index in range(2):
            batch.metadata = {**batch.metadata, "global_step": index + 1}
            started = time.monotonic()
            outputs = ray.get(actors.async_run_ray_method("mesh", "ppo_train", batch))
            timings.append(
                {"warm_optimizer": index > 0, "seconds": time.monotonic() - started, "rank_outputs": len(outputs)}
            )
            write_receipt(
                prefix_join(args.output_prefix, "matched-batch-timing.json"),
                {"variant": args.variant, "batch_sha256": batch_sha256, "timings": timings},
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
        write_receipt(prefix_join(args.output_prefix, "receipt.json"), result)
        print(json.dumps({"qualification_passed": True, "policy_ranks": len(identities)}), flush=True)
    finally:
        ray.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", action="store_true")
    parser.add_argument("--config-uri", required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--batch-uri", required=True)
    parser.add_argument("--variant", required=True, choices=("chunked", "offload"))
    args = parser.parse_args()
    if args.driver:
        driver(args)
        return
    cfg = compiled_config(args)
    placement = cfg.trainer.placement
    run_ray_probe_task(
        args,
        cfg,
        num_tasks=placement.policy_num_nodes,
        gpus_per_task=placement.policy_num_gpus_per_node,
        driver_module=MODULE,
        driver_arguments=("--batch-uri", args.batch_uri, "--variant", args.variant),
    )


if __name__ == "__main__":
    main()
