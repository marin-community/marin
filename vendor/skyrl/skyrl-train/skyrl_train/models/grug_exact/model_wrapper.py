"""The Megatron model wrapper of the exact Grug trainer."""

from functools import partial
from typing import Any

import torch

from skyrl_train.distributed.megatron.megatron_utils import compact_left_padded_tokens, scatter_token_values
from skyrl_train.models.grug_exact.context import ExactForward, check_engine_dp_ranks
from skyrl_train.models.grug_exact.kernel_values import kernel_value
from skyrl_train.models.grug_exact.vllm_kernels import token_logprobs
from skyrl_train.workers.megatron.megatron_model_wrapper import (
    MegatronForwardMicroBatch,
    MegatronModelWrapper,
    MegatronPolicyMicroBatch,
)


class ExactMegatronModelWrapper(MegatronModelWrapper):
    """The Megatron model wrapper of ``GrugExactGPTModel``. Each model call receives the engine data-parallel rank of
    every sequence. The response log-probabilities are the values of the engine's log-probability kernel, with the
    gradient of the trainer's own log-probabilities."""

    def _model_kwargs(self, micro_batch: MegatronForwardMicroBatch | MegatronPolicyMicroBatch) -> dict[str, Any]:
        ranks, loss_mask = micro_batch.rollout_engine_dp_ranks, micro_batch.loss_mask
        if ranks is None or loss_mask is None:
            raise ValueError("exact numerics need each sequence's engine data-parallel rank and loss mask")
        ranks = ranks.to(device=loss_mask.device, dtype=torch.long)
        generator = self.cfg.generator
        # A trajectory that failed before its model call is a row of placeholder tokens, no loss tokens and no
        # engine rank; no loss reads its values.
        check_engine_dp_ranks(ranks, (loss_mask != 0).any(dim=1), int(generator.inference_engine_data_parallel_size))
        return {"exact": ExactForward(ranks, int(generator.inference_engine_expert_parallel_size))}

    def _token_logprobs(
        self, logits: torch.Tensor, sequences: torch.Tensor, attention_mask: torch.Tensor, packed_seq_params
    ) -> torch.Tensor:
        batch, seq_len, vocab = logits.shape
        targets = compact_left_padded_tokens(sequences, attention_mask).roll(shifts=-1, dims=-1)
        with torch.no_grad():
            values = token_logprobs(logits.reshape(batch * seq_len, vocab), targets.reshape(-1))
        engine = scatter_token_values(values.view(batch, seq_len)[:, :-1], attention_mask, drop_last=True)
        trainer = partial(super()._token_logprobs, logits, sequences, attention_mask, packed_seq_params)
        return kernel_value(engine, trainer)
