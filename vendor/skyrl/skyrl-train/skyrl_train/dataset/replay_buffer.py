"""Experience tensors consumed by the training batch iterator and workers."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from jaxtyping import Float, Integer

import torch

from skyrl_train.ftpo import FTPOTargets
from skyrl_train.distillation import TopKEvidence


BasicType = Union[int, float, str, bool]


def to(tensor: Union[torch.Tensor, List[torch.Tensor], BasicType], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    elif isinstance(tensor, torch.Tensor):
        return tensor.to(device)
    else:
        return tensor


def pin_memory(tensor: Union[torch.Tensor, List[torch.Tensor], BasicType]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    elif isinstance(tensor, torch.Tensor):
        return tensor.pin_memory()
    else:
        return tensor


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    base_action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advatanges: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions/ response length.
    """

    sequences: Integer[torch.Tensor, "batch seq_len"]
    action_log_probs: Float[torch.Tensor, "batch response_len"]
    base_action_log_probs: Optional[Float[torch.Tensor, "batch response_len"]]
    values: Optional[Float[torch.Tensor, "batch response_len"]]
    returns: Optional[Float[torch.Tensor, "batch response_len"]]
    advantages: Optional[Float[torch.Tensor, "batch response_len"]]
    attention_mask: Optional[Integer[torch.LongTensor, "batch seq_len"]]
    loss_mask: Optional[Integer[torch.LongTensor, "batch response_len"]]
    action_mask: Optional[Integer[torch.Tensor, "batch response_len"]]
    rollout_logprobs: Optional[Float[torch.Tensor, "batch response_len"]]
    num_actions: int
    info: Optional[dict]
    kl: Optional[Float[torch.Tensor, "batch response_len"]] = None
    metadata: Optional[Dict[str, Any]] = None
    # MoE router-replay (R3) field — present only when moe_router_replay is on.
    rollout_routed_experts: Optional[Integer[torch.Tensor, "batch response_len L K"]] = None
    # Stage D (F7) per-token span tags (SPAN_THINK==1) — present only when the
    # token-reward channel is on; used to down-weight <think> tokens in the loss.
    response_span_tags: Optional[Integer[torch.Tensor, "batch response_len"]] = None
    distillation: Optional[TopKEvidence] = None
    ftpo: FTPOTargets | None = None
    correction_weights: Optional[Float[torch.Tensor, "batch response_len"]] = None
    # DPO pairing evidence: +1 chosen / -1 rejected per row; present only for
    # static preference-pair batches.
    pair_roles: Optional[Float[torch.Tensor, "batch"]] = None  # noqa: F821
    score_topk_indices: Optional[Integer[torch.Tensor, "batch response_len top_k"]] = None
    score_old_logprobs: Optional[Float[torch.Tensor, "batch response_len top_k"]] = None
    score_behavior_logprobs: Optional[Float[torch.Tensor, "batch response_len top_k"]] = None

    @torch.no_grad()
    def to_device(self, device: torch.device) -> None:
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        if self.base_action_log_probs is not None:
            self.base_action_log_probs = to(self.base_action_log_probs, device)
        if self.values is not None:
            self.values = to(self.values, device)
        if self.returns is not None:
            self.returns = to(self.returns, device)
        if self.advantages is not None:
            self.advantages = to(self.advantages, device)
        if self.attention_mask is not None:
            self.attention_mask = to(self.attention_mask, device)
        if self.loss_mask is not None:
            self.loss_mask = to(self.loss_mask, device)
        if self.action_mask is not None:
            self.action_mask = to(self.action_mask, device)
        if self.rollout_logprobs is not None:
            self.rollout_logprobs = to(self.rollout_logprobs, device)
        if self.correction_weights is not None:
            self.correction_weights = to(self.correction_weights, device)
        if self.pair_roles is not None:
            self.pair_roles = to(self.pair_roles, device)
        if self.score_topk_indices is not None:
            self.score_topk_indices = to(self.score_topk_indices, device)
        if self.score_old_logprobs is not None:
            self.score_old_logprobs = to(self.score_old_logprobs, device)
        if self.score_behavior_logprobs is not None:
            self.score_behavior_logprobs = to(self.score_behavior_logprobs, device)
        if self.rollout_routed_experts is not None:
            self.rollout_routed_experts = to(self.rollout_routed_experts, device)
        if self.response_span_tags is not None:
            self.response_span_tags = to(self.response_span_tags, device)
        if self.ftpo is not None:
            self.ftpo = self.ftpo.to(device)
        if self.distillation is not None:
            self.distillation = self.distillation.to(device)

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        if self.base_action_log_probs is not None:
            self.base_action_log_probs = pin_memory(self.base_action_log_probs)
        if self.values is not None:
            self.values = pin_memory(self.values)
        if self.returns is not None:
            self.returns = pin_memory(self.returns)
        if self.advantages is not None:
            self.advantages = pin_memory(self.advantages)
        if self.attention_mask is not None:
            self.attention_mask = self.attention_mask.pin_memory()
        if self.loss_mask is not None:
            self.loss_mask = self.loss_mask.pin_memory()
        if self.action_mask is not None:
            self.action_mask = self.action_mask.pin_memory()
        if self.rollout_logprobs is not None:
            self.rollout_logprobs = self.rollout_logprobs.pin_memory()
        if self.correction_weights is not None:
            self.correction_weights = self.correction_weights.pin_memory()
        if self.pair_roles is not None:
            self.pair_roles = self.pair_roles.pin_memory()
        if self.score_topk_indices is not None:
            self.score_topk_indices = self.score_topk_indices.pin_memory()
        if self.score_old_logprobs is not None:
            self.score_old_logprobs = self.score_old_logprobs.pin_memory()
        if self.score_behavior_logprobs is not None:
            self.score_behavior_logprobs = self.score_behavior_logprobs.pin_memory()
        if self.rollout_routed_experts is not None:
            self.rollout_routed_experts = self.rollout_routed_experts.pin_memory()
        if self.response_span_tags is not None:
            self.response_span_tags = self.response_span_tags.pin_memory()
        if self.ftpo is not None:
            self.ftpo = self.ftpo.pin_memory()
        if self.distillation is not None:
            self.distillation = self.distillation.pin_memory()
        return self
