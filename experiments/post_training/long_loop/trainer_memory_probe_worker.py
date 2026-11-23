# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GPU workers for the trainer-only memory qualification."""

import megatron.core.parallel_state as mpu
import torch
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
