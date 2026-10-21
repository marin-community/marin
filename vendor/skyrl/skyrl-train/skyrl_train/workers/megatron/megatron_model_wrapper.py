from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, List, Optional

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from megatron.core.pipeline_parallel import get_forward_backward_func
import megatron.core.parallel_state as mpu
from megatron.core.distributed import DistributedDataParallel as MegatronDDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.transformer.module import Float16Module

from skyrl_train.distributed.megatron.model_utils import (
    from_parallel_logits_to_logprobs,
    from_parallel_logits_to_logprobs_packed_sequences,
    vocab_parallel_entropy,
)
from skyrl_train.distributed.megatron.megatron_utils import get_model_config
from skyrl_train.ftpo import FTPOTargets, FTPOInputs, boundary_values, compact_boundary_logits, ftpo_counts
from skyrl_train.distillation import TopKEvidence, student_topk_logprobs, INVALID_TOPK_INDEX
from skyrl_train.models.megatron_router_replay import MegatronRouterReplay
from skyrl_train.config.objective_spec import topk_loss_params, score_centering_tis_cap
from skyrl_train.objective.score_centering import ppo_tis_score_centering_correction, score_centering_diagnostics
from skyrl_train.config.ftpo import ftpo_config
from skyrl_train.objective.objective import (
    FTPOAuxiliaryBatch,
    TopKTeacherBatch,
    build_objective_micro_batch,
    compute_policy_objective,
    megatron_loss_scale,
)
from skyrl_train.objective.reduction import policy_data_weights, step_counts
from skyrl_train.utils.profiler import Profiler
from skyrl_train.timing_observability import PhaseBreakdown
from skyrl_train.utils.importance_ratio_diagnostics import LogRatioMonitor, gather_ratio_tensor

from skyrl_train.distributed.megatron.megatron_utils import (
    compact_left_padded_tokens,
    make_batch_generator,
    pack_padded_tokens,
    preprocess_packed_seqs,
    remove_left_padding,
    scatter_token_values,
    unpack_packed_token_values,
)
from skyrl_train.models.megatron_router_replay import (
    sequence_major_flatten,
    slice_sequence_parallel,
    validate_replay_geometry,
)
from skyrl_train.models.megatron_router_replay import dense_replay_targets


# Sentinel: distinguishes "caller did not pass logprob_chunk_size" (=> fall back to
# the policy config key, preserving prior behavior) from an explicit None (=> chunking
# disabled). A plain None default could not tell these apart.
_UNSET = object()


@dataclass(frozen=True)
class MegatronForwardMicroBatch:
    """Typed forward-only payload consumed by the Megatron pipeline scheduler."""

    sequences: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    num_actions: int
    rollout_routed_experts: Optional[torch.Tensor] = None
    ftpo_chosen_mask: torch.Tensor | None = None
    probe_row_indices: Optional[torch.Tensor] = None
    score_topk_indices: Optional[torch.Tensor] = None
    score_probe_positions: torch.Tensor | None = None


@dataclass(frozen=True)
class MegatronForwardResult:
    scores: torch.Tensor
    selected_logprobs: torch.Tensor | None = None


@dataclass(frozen=True)
class RouterReplayTargets:
    """Per-layer router targets and aligned row masks for one forward."""

    per_layer: dict[int, torch.Tensor]
    mask: torch.Tensor
    prediction_mask: torch.Tensor
    probe_positions: Optional[torch.Tensor]


@dataclass(frozen=True)
class MegatronPolicyMicroBatch:
    """Typed policy payload consumed by the Megatron pipeline scheduler."""

    sequences: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    num_actions: int
    old_action_log_probs: torch.Tensor
    base_action_log_probs: Optional[torch.Tensor]
    advantages: torch.Tensor
    loss_mask: torch.Tensor
    rollout_action_logprobs: Optional[torch.Tensor]
    response_span_tags: Optional[torch.Tensor]
    distillation: Optional[TopKEvidence] = None
    ftpo: FTPOTargets | None = None
    correction_weights: Optional[torch.Tensor] = None
    rollout_routed_experts: Optional[torch.Tensor] = None
    pair_roles: Optional[torch.Tensor] = None
    score_topk_indices: Optional[torch.Tensor] = None
    score_old_logprobs: Optional[torch.Tensor] = None
    score_behavior_logprobs: Optional[torch.Tensor] = None


class MegatronModelWrapper:
    # MoE router replay (R3); set as an instance attribute by the worker after
    # install. Class-level None keeps flag-off behavior for every wrapper.
    router_replay: Optional[MegatronRouterReplay] = None

    def __init__(
        self,
        config,
        actor_module: List[nn.Module],
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
        policy_loss_fn: Optional[Callable] = None,
        logprob_chunk_size: Any = _UNSET,
        vocabulary_size: int | None = None,
    ):
        self.cfg = config
        self.vocabulary_size = vocabulary_size
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.policy_loss_fn = policy_loss_fn
        self.use_sample_packing = self.cfg.trainer.use_sample_packing
        # Bound legacy FP32 temporaries and native TE scoring copies.
        # TE training uses one full model-dtype gradient buffer.
        #
        # Callers pass this EXPLICITLY (the policy worker its own
        # trainer.policy.megatron_config.logprob_chunk_size, the ref worker its own
        # trainer.ref.megatron_config.logprob_chunk_size) so each model honors its
        # own config key. If left unset we fall back to reading the policy key, so
        # any external caller that doesn't pass it keeps the prior behavior.
        if logprob_chunk_size is _UNSET:
            logprob_chunk_size = OmegaConf.select(
                self.cfg, "trainer.policy.megatron_config.logprob_chunk_size", default=None
            )
        self._logprob_chunk_size = logprob_chunk_size

        config = get_model_config(self.actor_module[0])
        # This is set to None by default: https://github.com/NVIDIA/Megatron-LM/blob/07b22a05136a3cb08ece05f7de38cf6aeeb165fb/megatron/core/model_parallel_config.py#L95
        # use the build in finalize_model_grads function to all reduce gradients across parallelism dimensions
        config.finalize_model_grads_func = finalize_model_grads

    def train(self):
        [module.train() for module in self.actor_module]

    def eval(self):
        [module.eval() for module in self.actor_module]

    def _token_logprobs(
        self,
        logits: torch.Tensor,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        packed_seq_params,
    ) -> torch.Tensor:
        """Compute logprobs before reconstructing only scalar token values."""
        tp_group = mpu.get_tensor_model_parallel_group()
        tp_rank = mpu.get_tensor_model_parallel_rank()
        if self.use_sample_packing:
            if packed_seq_params is None:
                raise ValueError("Packed sequence parameters are required when sample packing is enabled.")
            packed_sequences = pack_padded_tokens(sequences, attention_mask, packed_seq_params)
            return from_parallel_logits_to_logprobs_packed_sequences(
                logits,
                packed_sequences,
                packed_seq_params.cu_seqlens_q_padded,
                attention_mask,
                vocab_start_index=tp_rank * logits.shape[-1],
                vocab_end_index=(tp_rank + 1) * logits.shape[-1],
                group=tp_group,
                inference_only=not self.actor_module[0].training,
                cp_group=mpu.get_context_parallel_group(),
                chunk_size=self._logprob_chunk_size,
            )

        compact_sequences = compact_left_padded_tokens(sequences, attention_mask)
        compact_logprobs = from_parallel_logits_to_logprobs(
            logits,
            compact_sequences,
            vocab_start_index=tp_rank * logits.shape[-1],
            vocab_end_index=(tp_rank + 1) * logits.shape[-1],
            tp_group=tp_group,
            inference_only=not self.actor_module[0].training,
            cp_group=None,
            chunk_size=self._logprob_chunk_size,
        )
        return scatter_token_values(compact_logprobs, attention_mask, drop_last=True)

    def _response_logprobs(
        self,
        logits: torch.Tensor,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        num_actions: int,
        packed_seq_params,
        candidate_ids: torch.Tensor | None = None,
        candidate_positions: torch.Tensor | None = None,
    ) -> MegatronForwardResult:
        """Score response actions and candidates with the same differentiable normalizer.

        Logits already include the scoring temperature. Returned scores use the
        original response suffix, despite Megatron's compacted model positions.
        """
        chosen = self._token_logprobs(logits, sequences, attention_mask.to(bool), packed_seq_params)[:, -num_actions:]
        selected = None
        if candidate_ids is not None:
            if candidate_ids.ndim != 3 or candidate_ids.shape[0] != chosen.shape[0]:
                raise ValueError("selected response IDs must align with chosen logprobs")
            if candidate_positions is None:
                if candidate_ids.shape[:2] != chosen.shape:
                    raise ValueError("selected response IDs must align with chosen logprobs")
                response_positions = torch.arange(num_actions, device=logits.device).expand(chosen.shape[0], -1)
            else:
                if candidate_positions.shape != candidate_ids.shape[:2]:
                    raise ValueError("compact selected response positions must align with candidate IDs")
                response_positions = candidate_positions.long()
                if torch.any(response_positions < 0) or torch.any(response_positions >= num_actions):
                    raise ValueError("compact selected positions are outside the response")
            query_start = attention_mask.shape[1] - num_actions - 1
            if query_start < 0:
                raise ValueError("selected response IDs require at least one prompt token")
            queries = query_start + response_positions
            positions = attention_mask.long().cumsum(1).gather(1, queries) - 1
            valid = candidate_ids != INVALID_TOPK_INDEX
            valid_rows = valid.any(-1)
            if torch.any(
                valid_rows & ~(attention_mask.gather(1, queries).bool() & attention_mask.gather(1, queries + 1).bool())
            ):
                raise ValueError("selected response IDs refer to padded prompt or response positions")
            if torch.any(candidate_ids < INVALID_TOPK_INDEX) or torch.any(candidate_ids[valid] >= logits.shape[-1]):
                raise ValueError("selected token IDs are outside the student vocabulary")
            positions = positions.masked_fill(~valid_rows, 0)
            rows = torch.arange(logits.shape[0], device=logits.device)[:, None]
            head_logits = logits[rows[:, :, None], positions[:, :, None], candidate_ids.masked_fill(~valid, 0).long()]
            chosen_logits = logits[rows, positions, sequences[:, -num_actions:].gather(1, response_positions)]
            # Reuse the differentiable chosen normalizer; save no second FP32 vocabulary tensor.
            selected = (
                chosen.gather(1, response_positions).unsqueeze(-1)
                + head_logits.float()
                - chosen_logits.float().unsqueeze(-1)
            ).masked_fill(~valid, torch.nan)
        return MegatronForwardResult(chosen, selected)

    def _token_entropies(self, logits: torch.Tensor, attention_mask: torch.Tensor, packed_seq_params) -> torch.Tensor:
        """Compute entropy before reconstructing only scalar token values."""
        token_entropies = vocab_parallel_entropy(logits, chunk_size=self._logprob_chunk_size)
        if self.use_sample_packing:
            if packed_seq_params is None:
                raise ValueError("Packed sequence parameters are required when sample packing is enabled.")
            return unpack_packed_token_values(token_entropies, packed_seq_params, attention_mask)
        return scatter_token_values(token_entropies, attention_mask, drop_last=False)

    def _build_router_replay_targets(
        self,
        sequences: torch.Tensor,
        attention_mask: torch.Tensor,
        rollout_routed_experts: torch.Tensor,
        num_actions: int,
        layer_indices: tuple[int, ...],
        probe_row_indices: Optional[torch.Tensor] = None,
    ):
        """Build per-layer router targets in the exact token order the routers see.

        Pushes the dense ``[B, S, L, K]`` targets and the replay / prediction
        masks through the SAME sequence transform the model input takes
        (packing or left-pad removal, with the CP chunk split), flattens
        sequence-major (``s*B + b``, mirroring the router view), and slices to
        this TP rank's contiguous sequence chunk under sequence parallelism.
        The result contains per-layer targets, replay and prediction masks, and
        optional probe positions aligned to the model chunk's router rows.
        """
        controller = self.router_replay
        assert controller is not None
        config = get_model_config(self.actor_module[0])
        batch_size, seq_len = sequences.shape
        _, response_len, _, _ = rollout_routed_experts.shape
        validate_replay_geometry(
            num_layers_captured=rollout_routed_experts.shape[2],
            expected_moe_layers=controller.num_moe_layers_total,
            topk_captured=rollout_routed_experts.shape[3],
            expected_topk=controller.topk,
            num_experts=config.num_moe_experts,
            targets=rollout_routed_experts,
            response_len=response_len,
            num_actions=num_actions,
        )

        device = sequences.device
        dense, mask_BS = dense_replay_targets(rollout_routed_experts, batch_size, seq_len, num_actions)
        prediction_BS = torch.zeros_like(mask_BS)
        prediction_window = slice(seq_len - num_actions - 1, seq_len - 1)
        prediction_BS[:, prediction_window] = True
        probe_positions = None
        if probe_row_indices is not None:
            if probe_row_indices.shape != (batch_size,):
                raise ValueError("probe row indices must have one entry per sequence")
            # Zero encodes padding through the shared sequence transforms.
            probe_positions = torch.zeros((batch_size, seq_len, 2), dtype=torch.long, device=device)
            probe_positions[:, prediction_window, 0] = probe_row_indices[:, None] + 1
            probe_positions[:, prediction_window, 1] = torch.arange(1, num_actions + 1, device=device)

        if self.use_sample_packing:
            # The routes tensor is ours, not the pipeline's input: always run the
            # packing arithmetic locally (pre_process=True) so every PP stage
            # derives its own layer targets from the replicated batch.
            dense, _ = preprocess_packed_seqs(dense, attention_mask, pre_process=True)
            mask_BS, _ = preprocess_packed_seqs(mask_BS, attention_mask, pre_process=True)
            prediction_BS, _ = preprocess_packed_seqs(prediction_BS, attention_mask, pre_process=True)
            if probe_positions is not None:
                probe_positions, _ = preprocess_packed_seqs(probe_positions, attention_mask, pre_process=True)
        else:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
            dense, _, _ = remove_left_padding(dense, attention_mask, position_ids, pre_process=True)
            mask_BS, _, _ = remove_left_padding(mask_BS, attention_mask, position_ids, pre_process=True)
            prediction_BS, _, _ = remove_left_padding(prediction_BS, attention_mask, position_ids, pre_process=True)
            if probe_positions is not None:
                probe_positions, _, _ = remove_left_padding(
                    probe_positions, attention_mask, position_ids, pre_process=True
                )

        flat = sequence_major_flatten(dense)
        mask = sequence_major_flatten(mask_BS)
        prediction_mask = sequence_major_flatten(prediction_BS)
        if probe_positions is not None:
            probe_positions = sequence_major_flatten(probe_positions) - 1
        tp_size = mpu.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            # Under TP sequence parallelism the router sees this rank's
            # contiguous chunk of the sequence dim.
            transform_batch, transform_seq = dense.shape[0], dense.shape[1]
            slice_kwargs = dict(
                seq_len=transform_seq,
                batch_size=transform_batch,
                tp_rank=mpu.get_tensor_model_parallel_rank(),
                tp_size=tp_size,
            )
            flat = slice_sequence_parallel(flat, **slice_kwargs)
            mask = slice_sequence_parallel(mask, **slice_kwargs)
            prediction_mask = slice_sequence_parallel(prediction_mask, **slice_kwargs)
            if probe_positions is not None:
                probe_positions = slice_sequence_parallel(probe_positions, **slice_kwargs)
        per_layer = {idx: flat[:, idx, :].to(device) for idx in layer_indices}
        return RouterReplayTargets(per_layer, mask.to(device), prediction_mask.to(device), probe_positions)

    def _forward_micro_batch(
        self,
        model,
        sequences,
        attention_mask,
        position_ids,
        rollout_routed_experts: Optional[torch.Tensor] = None,
        probe_row_indices: Optional[torch.Tensor] = None,
        num_actions: Optional[int] = None,
        record_recompute: bool = False,
    ):
        """Run the shared packed or left-unpadded Megatron model boundary.

        When router replay is installed and routes are present, brackets the
        model call with ``begin_forward`` / ``end_forward`` (never falling back
        to native routing); with replay installed but no routes, fails fast
        before the model runs.
        """
        attention_mask = attention_mask.to(bool)
        armed = False
        if self.router_replay is not None:
            if rollout_routed_experts is None:
                raise ValueError("moe_router_replay is on but the micro-batch carries no rollout_routed_experts")
            # With virtual pipelining each chunk fires a disjoint layer set
            # inside its own bracket; single-chunk models arm every local layer.
            layer_indices = self.router_replay.local_indices_for_module.get(
                id(model), self.router_replay.local_layer_indices
            )
            targets = self._build_router_replay_targets(
                sequences, attention_mask, rollout_routed_experts, num_actions, layer_indices, probe_row_indices
            )
            self.router_replay.begin_forward(
                targets.per_layer,
                targets.mask,
                targets.prediction_mask,
                record_recompute=record_recompute,
                probe_positions=targets.probe_positions,
            )
            armed = True
        try:
            if self.use_sample_packing:
                model_sequences, packed_seq_params = preprocess_packed_seqs(
                    sequences,
                    attention_mask,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                )
                model_attention_mask = None
                model_position_ids = None
            else:
                model_sequences, model_attention_mask, model_position_ids = remove_left_padding(
                    sequences,
                    attention_mask,
                    position_ids,
                    pre_process=mpu.is_pipeline_first_stage(ignore_virtual=True),
                )
                packed_seq_params = None

            # Only the mixed-precision wrapper consumes the precision kwarg; a raw fp32
            # module (equivalence experiments) already returns module-precision outputs.
            inner = model.module if isinstance(model, MegatronDDP) else model
            precision_kwargs = {"fp32_output": False} if isinstance(inner, Float16Module) else {}
            outputs = model(
                model_sequences,
                model_position_ids,
                model_attention_mask,
                packed_seq_params=packed_seq_params,
                **precision_kwargs,
            )
            if armed:
                self.router_replay.end_forward()
        except BaseException:
            if armed:
                # A strict end_forward would raise its own "layer never fired"
                # error and mask the original failure; reset and re-raise.
                self.router_replay.abort_forward()
            raise
        return outputs, packed_seq_params

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self,
        micro_batches: List[MegatronForwardMicroBatch],
        seq_len: int,
        micro_batch_size: int,
        temperature: float = 1.0,
    ) -> MegatronForwardResult:
        """
        Score response tokens, or return raw vocabulary logits at FTPO boundaries.

        Args:
            micro_batches: Typed forward micro-batches.
            seq_len: Padded sequence length per sample.
            micro_batch_size: Per-micro-batch size.
            temperature: Optional temperature scaling for logits.

        Returns:
            Response scores [B,T], or FTPO boundary logits [B,V], and optional
            aligned selected log probabilities. Valid on the pipeline last stage only.
        """
        forward_backward_func = get_forward_backward_func()

        def collection_func(logits, data, packed_seq_params):
            sequences = data.sequences
            if data.ftpo_chosen_mask is not None:
                boundary_logits = compact_boundary_logits(
                    logits[..., : self.vocabulary_size], data.attention_mask, data.ftpo_chosen_mask
                )
                return boundary_logits.new_zeros(()), {"scores": boundary_logits}

            if temperature != 1.0:
                logits.div_(temperature)

            response = self._response_logprobs(
                logits,
                sequences,
                data.attention_mask,
                data.num_actions,
                packed_seq_params,
                data.score_topk_indices,
                data.score_probe_positions,
            )
            result = {"scores": response.scores}
            if response.selected_logprobs is not None:
                result["selected_logprobs"] = response.selected_logprobs
            return response.scores.new_zeros(()), result

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            outputs, packed_seq_params = self._forward_micro_batch(
                model,
                batch.sequences,
                batch.attention_mask,
                batch.position_ids,
                rollout_routed_experts=batch.rollout_routed_experts,
                probe_row_indices=batch.probe_row_indices,
                num_actions=batch.num_actions,
            )

            return outputs, partial(collection_func, data=batch, packed_seq_params=packed_seq_params)

        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        output = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            num_microbatches=len(micro_batches),
            seq_length=seq_len,
            micro_batch_size=micro_batch_size,
            forward_only=True,
        )

        if self.router_replay is not None:
            # No backward ran, so nothing may be left to recompute; discard the
            # pass's metrics so they do not bleed into the next train step.
            self.router_replay.assert_drained()
            self.router_replay.pop_metrics()

        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            scores = [o["scores"] for o in output]
            scores = torch.cat(scores, dim=0)
            selected_logprobs = (
                torch.cat([o["selected_logprobs"] for o in output], dim=0)
                if micro_batches[0].score_topk_indices is not None
                else None
            )
        else:
            # return dummy tensor for non-last pp stages
            device = micro_batches[0].sequences.device
            scores = torch.zeros(size=(1, 1), dtype=torch.bfloat16, device=device)
            selected_logprobs = None
        return MegatronForwardResult(scores, selected_logprobs)

    def _distillation_student_logprobs(
        self,
        logits: torch.Tensor,
        data: MegatronPolicyMicroBatch,
    ) -> Optional[torch.Tensor]:
        if data.distillation is None:
            return None
        token_ids = data.distillation.student_token_ids()
        if token_ids is None:
            return None
        if self.use_sample_packing or mpu.get_context_parallel_world_size() != 1:
            raise ValueError(
                "selected-ID distillation on Megatron does not yet support sample packing or context parallelism"
            )
        if mpu.get_tensor_model_parallel_world_size() != 1:
            raise ValueError("selected-ID distillation on Megatron requires a tensor-parallel top-K gather")
        response_logits = logits[:, -data.num_actions - 1 : -1]
        return student_topk_logprobs(response_logits, token_ids)

    def forward_backward_mini_batch(
        self,
        micro_batches: List[MegatronPolicyMicroBatch],
        seq_len: int,
        micro_batch_size: int,
        temperature: float = 1.0,
        timings: PhaseBreakdown | None = None,
        profiler: Profiler | None = None,
    ) -> List[dict]:
        """
        Run forward-backward over a full mini-batch consisting of multiple micro-batches.

        Args:
            micro_batches: Typed policy micro-batches containing model inputs,
                policy targets, response span tags, and teacher evidence.
            seq_len: Sequence length (tokens) per sample (assumed same across micros after padding).
            micro_batch_size: Micro-batch size per forward pass.
            temperature: Optional temperature for logits scaling.
            timings: Optional recorder for the forward-backward scheduler and pipeline metric broadcast.
            profiler: Optional profiler for the first forward micro-batch.

        Returns:
            List[dict]: one metrics dict per micro-batch in order.
        """
        forward_backward_func = get_forward_backward_func()
        log_ratio_monitor = None
        completed_microbatches = 0

        def sum_data_parallel(value):
            torch.distributed.all_reduce(value, group=mpu.get_data_parallel_group(with_context_parallel=False))
            return value

        counts = step_counts(
            [
                policy_data_weights(
                    micro.loss_mask, micro.response_span_tags, self.cfg.trainer.algorithm.think_token_weight
                )
                for micro in micro_batches
            ],
            [micro.loss_mask for micro in micro_batches],
            [
                micro.loss_mask * micro.distillation.valid_mask
                for micro in micro_batches
                if micro.distillation is not None
            ],
            [micro.advantages for micro in micro_batches],
            self.cfg.trainer.algorithm.max_seq_len,
            sum_data_parallel,
        )
        ftpo_normalization = None
        if micro_batches[0].ftpo is not None:
            ftpo_normalization = ftpo_counts(
                [micro.ftpo for micro in micro_batches], [micro.loss_mask for micro in micro_batches], sum_data_parallel
            )
        scale = megatron_loss_scale(
            len(micro_batches),
            torch.distributed.get_world_size(mpu.get_data_parallel_group(with_context_parallel=False)),
        )

        def loss_func(logits, data, packed_seq_params):
            nonlocal completed_microbatches, log_ratio_monitor
            sequences = data.sequences
            num_actions = data.num_actions
            old_action_log_probs = data.old_action_log_probs
            base_action_log_probs = data.base_action_log_probs
            advantages = data.advantages
            loss_mask = data.loss_mask
            rollout_action_logprobs = data.rollout_action_logprobs
            response_span_tags = data.response_span_tags

            ftpo_inputs = None
            if data.ftpo is not None:
                assert ftpo_normalization is not None
                ftpo_inputs = FTPOInputs(
                    compact_boundary_logits(
                        logits[..., : self.vocabulary_size], data.attention_mask, data.ftpo.chosen_mask
                    ),
                    data.ftpo,
                    boundary_values(sequences[:, -num_actions:], data.ftpo.chosen_mask),
                    ftpo_normalization,
                )
            # FTPO's logit margin and reference tether use unscaled model logits.
            if data.ftpo is None and temperature != 1.0:
                logits.div_(temperature)

            centering_width = self.cfg.trainer.algorithm.get("score_centering_topk", 0)
            response = self._response_logprobs(
                logits,
                sequences,
                data.attention_mask,
                num_actions,
                packed_seq_params,
                data.score_topk_indices if centering_width else None,
            )
            action_log_probs = response.scores
            score_centering = None
            if centering_width:
                if any(
                    value is None
                    for value in (data.score_topk_indices, data.score_old_logprobs, data.score_behavior_logprobs)
                ):
                    raise ValueError("score centering requires behavior IDs and old and behavior log probabilities")
                if self.cfg.trainer.algorithm.get("score_centering_enabled", False):
                    score_centering = ppo_tis_score_centering_correction(
                        response.selected_logprobs,
                        data.score_old_logprobs,
                        data.score_behavior_logprobs,
                        advantages,
                        loss_mask,
                        tis_cap=score_centering_tis_cap(self.cfg.trainer.algorithm),
                        eps_clip_low=self.cfg.trainer.algorithm.eps_clip_low,
                        eps_clip_high=self.cfg.trainer.algorithm.eps_clip_high,
                    )

            structural = None
            if data.ftpo is not None and data.ftpo.weights is not None:
                params = ftpo_config(self.cfg.trainer.algorithm)
                if params is None or params.auxiliary_weight is None or ftpo_inputs is None:
                    raise ValueError("Separate structural weights require configured auxiliary FTPO")
                structural = FTPOAuxiliaryBatch(
                    ftpo_inputs, data.ftpo.weights, action_log_probs, params, params.auxiliary_weight
                )
                ftpo_inputs = None

            sparse_student_logprobs = self._distillation_student_logprobs(logits, data)

            # Without an entropy loss the entropy is a metric only. Computing it under no_grad
            # avoids saving two vocab-sized copies of the logits for backward on the last stage.
            with torch.set_grad_enabled(self.cfg.trainer.algorithm.use_entropy_loss):
                token_entropies = self._token_entropies(logits, data.attention_mask.to(bool), packed_seq_params)
            teacher = None
            if data.distillation is not None:
                assert sparse_student_logprobs is not None
                teacher = TopKTeacherBatch(
                    data.distillation,
                    sparse_student_logprobs,
                    topk_loss_params(self.cfg.trainer.algorithm),
                    logits.shape[-1],
                )
            batch = build_objective_micro_batch(
                action_log_probs=action_log_probs,
                old_action_log_probs=old_action_log_probs,
                base_action_log_probs=base_action_log_probs,
                advantages=advantages,
                loss_mask=loss_mask,
                rollout_logprobs=rollout_action_logprobs,
                correction_weights=data.correction_weights,
                response_span_tags=response_span_tags,
                token_entropy=token_entropies[:, -num_actions - 1 : -1],
                think_token_weight=self.cfg.trainer.algorithm.think_token_weight,
                teacher=teacher,
                ftpo=ftpo_inputs,
                structural=structural,
                pair_roles=data.pair_roles,
                score_centering=score_centering,
            )
            objective = compute_policy_objective(
                batch,
                loss=self.policy_loss_fn,
                counts=counts,
                config=self.cfg.trainer.algorithm,
                loss_scale=scale,
                report_scale=scale,
            )
            if log_ratio_monitor is None:
                log_ratio_monitor = LogRatioMonitor(action_log_probs.device)
            log_ratio_monitor.add(action_log_probs, old_action_log_probs, loss_mask)
            completed_microbatches += 1

            metrics = {
                "final_loss": objective.optimization_loss.detach().item(),
                "policy_loss": objective.rows.policy.detach().item(),
                "policy_entropy": objective.rows.entropy.detach().item(),
                "policy_kl": objective.rows.kl.detach().item(),
            }
            metrics.update(objective.metrics)
            diagnostic_tokens = self.cfg.trainer.algorithm.get("score_centering_diagnostic_tokens", 0)
            if centering_width and diagnostic_tokens:
                metrics.update(
                    score_centering_diagnostics(
                        response.selected_logprobs,
                        data.score_old_logprobs,
                        data.score_behavior_logprobs,
                        action_log_probs,
                        rollout_action_logprobs,
                        score_centering,
                        loss_mask,
                        diagnostic_tokens,
                    )
                )
            if completed_microbatches == len(micro_batches):
                # Token logprobs are already reconstructed across TP/CP, so pool across data-parallel ranks only.
                group = (
                    mpu.get_data_parallel_group(with_context_parallel=False)
                    if torch.distributed.is_initialized()
                    else None
                )
                metrics.update(log_ratio_monitor.metrics(gather_fn=partial(gather_ratio_tensor, group=group)))
            return objective.optimization_loss, metrics

        def forward_step(batch_iter, model):
            batch = next(batch_iter)

            with profiler.capture_forward() if profiler is not None else nullcontext():
                outputs, packed_seq_params = self._forward_micro_batch(
                    model,
                    batch.sequences,
                    batch.attention_mask,
                    batch.position_ids,
                    rollout_routed_experts=batch.rollout_routed_experts,
                    num_actions=batch.num_actions,
                    record_recompute=True,
                )

            return outputs, partial(loss_func, data=batch, packed_seq_params=packed_seq_params)

        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        timing = timings or PhaseBreakdown("ppo_train", enabled=False)
        with timing.span("megatron_forward_backward_scheduler"):
            metrics_list = forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=batch_generator,
                model=self.actor_module,
                num_microbatches=len(micro_batches),
                seq_length=seq_len,
                micro_batch_size=micro_batch_size,
                forward_only=False,
            )

        if self.router_replay is not None:
            # Fail before the optimizer step: a non-empty FIFO or a masked row
            # that was not replayed means a layout bug, not a metric.
            self.router_replay.assert_drained()
            replay_metrics = self.router_replay.pop_metrics()
            if replay_metrics["hit_fraction"] != 1.0:
                raise ValueError(
                    f"router replay: hit_fraction {replay_metrics['hit_fraction']} != 1.0; "
                    "masked rows fell through to native routing"
                )
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                metrics_list[-1]["router_replay/hit_fraction"] = replay_metrics["hit_fraction"]
                metrics_list[-1]["router_replay/sentinel_fraction"] = replay_metrics["sentinel_fraction"]

        # broadcast metrics to all pp ranks
        if not mpu.is_pipeline_last_stage(ignore_virtual=True):
            metrics_list = [None] * len(micro_batches)
        with timing.span("megatron_pipeline_metric_broadcast"), torch.no_grad():
            torch.distributed.broadcast_object_list(
                metrics_list,
                src=mpu.get_pipeline_model_parallel_last_rank(),
                group=mpu.get_pipeline_model_parallel_group(),
            )

        return metrics_list
