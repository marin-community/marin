# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GPU workers for the trainer-only memory qualification."""

import math
import time

import megatron.core.parallel_state as mpu
import torch
from megatron.core.optimizer import ChainedOptimizer
from skyrl_train.distributed.megatron.megatron_utils import dp_weight_checksum_mismatch
from skyrl_train.workers.megatron.megatron_model_wrapper import MegatronModelWrapper
from skyrl_train.workers.megatron.megatron_worker import MegatronPolicyWorkerBase

from experiments.post_training.long_loop.trainer_memory_probe import compare_projection


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
        self.clip_observations = []
        children = (
            self.optimizer.chained_optimizers if isinstance(self.optimizer, ChainedOptimizer) else [self.optimizer]
        )
        thresholds = [child.config.clip_grad for child in children if child.get_parameters()]
        if not thresholds or any(value != 0.5 for value in thresholds):
            raise ValueError(f"Qualification requires native clipping at 0.5, got {thresholds}")
        native_step = self.optimizer.step_with_ready_grads

        def observed_step():
            # Every policy rank executes this extra norm collective, after native
            # clipping and before the optimizer consumes the gradients.
            started = time.monotonic()
            norm = float(self.optimizer.get_grad_norm())
            observation = {"post_clip_norm": norm, "norm_check_seconds": time.monotonic() - started}
            self.clip_observations.append(observation)
            if not math.isfinite(norm) or norm > 0.5 * 1.001:
                raise ValueError(f"Native clipping did not bound the consumed gradient: {observation}")
            return native_step()

        self.optimizer.step_with_ready_grads = observed_step
        groups = [
            {
                "route": group.get("optimizer"),
                "lr": float(group["lr"]),
                "parameters": len(group["params"]),
                "elements": sum(parameter.numel() for parameter in group["params"]),
                "dtypes": sorted({str(parameter.dtype) for parameter in group["params"]}),
            }
            for group in self.optimizer.param_groups
        ]
        return {
            "rank": self._rank,
            "gpu_uuid": str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
            "optimizer_groups": groups,
            "optimizer_types": [type(child.optimizer).__name__ for child in children],
            "clip_thresholds": thresholds,
        }

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

    def ppo_train(self, batch):
        self.clip_observations = []
        started = time.monotonic()
        output = super().ppo_train(batch)
        learner_seconds = time.monotonic() - started
        started = time.monotonic()
        mismatch = dp_weight_checksum_mismatch(self.actor_module)
        output.metadata["qualification"] = {
            "rank": self._rank,
            "learner_seconds": learner_seconds,
            "checksum_seconds": time.monotonic() - started,
            "dp_weight_checksum_mismatch": mismatch,
            "clipping": self.clip_observations,
        }
        if mismatch != 0 or not self.clip_observations:
            raise ValueError(f"Native update qualification failed: {output.metadata['qualification']}")
        return output
