"""Focused CPU checks of score centering through real Megatron scoring."""

from dataclasses import fields, replace
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from skyrl_train.objective.losses import ppo_policy_loss
from skyrl_train.objective.score_centering import ppo_tis_score_centering_correction
from skyrl_train.distributed.dispatch import ActorInfo, MeshRank, concatenate_outputs_after_mesh_dispatch
from skyrl_train.training_batch import TrainingOutputBatch
from tests.cpu.util import stub_megatron_modules

stub_megatron_modules()

from skyrl_train.workers.megatron import megatron_model_wrapper as mmw  # noqa: E402


def _packed_cp_reference(rank, rendezvous):
    """Use real Gloo collectives; only the optional Megatron mesh is substituted."""
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=5, timeout=timedelta(seconds=45)
    )
    singleton = None
    for owner in range(5):
        group = dist.new_group([owner])
        if owner == rank:
            singleton = group
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(mmw.mpu, "get_context_parallel_world_size", lambda: 5, raising=False)
            patch.setattr(mmw.mpu, "get_context_parallel_rank", lambda: rank, raising=False)
            patch.setattr(mmw.mpu, "get_context_parallel_group", lambda: dist.group.WORLD, raising=False)
            patch.setattr(mmw.mpu, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
            patch.setattr(mmw.mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
            patch.setattr(mmw.mpu, "get_tensor_model_parallel_group", lambda: singleton, raising=False)
            torch.manual_seed(31)
            parameters = torch.randn(7, 7, requires_grad=True)
            reference = parameters.detach().clone().requires_grad_()
            sequences = torch.arange(3 * 27).reshape(3, 27) % 7
            attention = torch.ones_like(sequences, dtype=torch.bool)
            attention[0, 4] = False
            attention[1, :3] = False
            attention[1, -2:] = False
            attention[2, :8] = False
            attention[2, -5:] = False
            actions = 12
            mask = attention[:, -actions:].float()
            mask[0, 4] = 0  # An interior loss hole must not shift later evidence.
            mask[2, 1] = 0
            advantages = torch.tensor([1.7, -2.3, 0.0] * 4).expand(3, -1).clone()
            advantages[~mask.bool()] = torch.nan
            candidates = torch.arange(3).expand(3, actions, 3).clone()
            candidates[~mask.bool()] = -1
            old_mass = torch.tensor([0.23, 0.21, 0.17, 0.13, 0.11, 0.09, 0.06])
            q_mass = torch.tensor([0.31, 0.20, 0.13, 0.11, 0.10, 0.08, 0.07])
            old_head = old_mass[:3].log().expand_as(candidates)
            q_head = q_mass[:3].log().expand_as(candidates)
            local_tokens, packed = mmw.preprocess_packed_seqs(sequences, attention)
            wrapper = object.__new__(mmw.MegatronModelWrapper)
            wrapper.actor_module = [SimpleNamespace(training=True)]
            wrapper.use_sample_packing = True
            wrapper._logprob_chunk_size = 3
            logits = parameters[local_tokens]
            response = wrapper._response_logprobs(logits, sequences, attention, actions, packed, candidates)

            # Dense logits are indexed at the original causal prefix, independently
            # of packing, CP chunk order and the production score reconstruction.
            dense = reference[sequences[:, -actions - 1 : -1]].log_softmax(-1)
            expected_chosen = dense.gather(-1, sequences[:, -actions:, None]).squeeze(-1)
            expected_head = dense[..., :3]
            valid = mask.bool()
            torch.testing.assert_close(response.scores[valid], expected_chosen[valid], atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(response.selected_logprobs[valid], expected_head[valid], atol=1e-5, rtol=1e-5)
            assert torch.isnan(response.selected_logprobs[~valid]).all()

            positions = torch.tensor([[0, 5, 11], [0, 4, 9], [0, 1, 2]])
            compact_ids = candidates.gather(1, positions[..., None].expand(-1, -1, 3))
            with torch.no_grad():
                compact = wrapper._response_logprobs(
                    logits, sequences, attention, actions, packed, compact_ids, positions
                )
            compact_valid = compact_ids >= 0
            compact_expected = expected_head.gather(1, positions[..., None].expand_as(compact_ids))
            torch.testing.assert_close(
                compact.selected_logprobs[compact_valid], compact_expected[compact_valid], atol=1e-5, rtol=1e-5
            )
            assert torch.isnan(compact.selected_logprobs[~compact_valid]).all()

            # Compare the actual SC term and its gradient with an independent dense
            # head-plus-proportional-tail expression, including all advantage signs.
            p = dense.exp()
            p_tail = p[..., 3:].sum(-1)
            o_tail, q_tail = old_mass[3:].sum(), q_mass[3:].sum()
            modeled_old = torch.cat(
                (old_mass[:3].expand(3, actions, -1), p[..., 3:].detach() * (o_tail / p_tail.detach()).unsqueeze(-1)),
                -1,
            )
            modeled_q = torch.cat(
                (q_mass[:3].expand(3, actions, -1), p[..., 3:].detach() * (q_tail / p_tail.detach()).unsqueeze(-1)), -1
            )
            safe_advantages = advantages.nan_to_num()
            ratio = p / modeled_old
            sampled_loss = -torch.minimum(
                ratio * safe_advantages[..., None], ratio.clamp(0.8, 1.2) * safe_advantages[..., None]
            )
            expected_ppo = (modeled_q * (modeled_old / modeled_q).clamp(max=1.3) * sampled_loss).sum(-1)
            coefficient = -torch.autograd.grad(expected_ppo.sum(), dense, retain_graph=True)[0]
            tail_coefficient = coefficient[..., 3] / p[..., 3].detach()
            expected_correction = (
                (coefficient[..., :3] - tail_coefficient[..., None] * p[..., :3]).detach() * dense[..., :3]
            ).sum(-1)
            correction = ppo_tis_score_centering_correction(
                response.selected_logprobs,
                old_head,
                q_head,
                advantages,
                mask,
                tis_cap=1.3,
                eps_clip_low=0.2,
                eps_clip_high=0.2,
            )
            torch.testing.assert_close(correction[valid], expected_correction[valid], atol=1e-5, rtol=1e-5)
            weights = mask * torch.tensor([1.0, 0.5, 1.0]).repeat(4)
            selected_loss = (torch.where(valid, correction, 0) * weights).sum() / weights.sum()
            expected_loss = (expected_correction * weights).sum() / weights.sum()
            selected_loss.backward()
            expected_loss.backward()
            # The replicated loss has shard-local activation gradients. Megatron's
            # CP parameter reduction must sum them exactly once for the whole step.
            dist.all_reduce(parameters.grad)
            torch.testing.assert_close(parameters.grad, reference.grad, atol=1e-6, rtol=1e-6)
            assert parameters.grad.abs().sum() > 0
    finally:
        dist.destroy_process_group()


def test_packed_cp5_selected_scores_and_centering_gradients_match_dense(tmp_path):
    mp.spawn(_packed_cp_reference, args=(str(tmp_path / "cp-store"),), nprocs=5, join=True)


@pytest.fixture
def megatron_wrapper(monkeypatch, single_rank_group):
    # Fake only the external model scheduler and mesh; scoring and loss math are real.
    def schedule(forward_step_func, data_iterator, model, num_microbatches, forward_only, **kwargs):
        if not mmw.mpu.is_pipeline_last_stage(ignore_virtual=True):
            return []
        outputs = []
        for _ in range(num_microbatches):
            logits, closure = forward_step_func(data_iterator, model[0])
            loss, metrics = closure(logits)
            if not forward_only:
                (loss / num_microbatches).backward()
            outputs.append(metrics)
        return outputs

    monkeypatch.setattr(mmw, "get_forward_backward_func", lambda: schedule)
    monkeypatch.setattr(mmw.mpu, "get_tensor_model_parallel_group", lambda: single_rank_group, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_context_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_context_parallel_group", lambda: single_rank_group, raising=False)
    monkeypatch.setattr(mmw.mpu, "is_pipeline_first_stage", lambda **kwargs: True, raising=False)
    monkeypatch.setattr(mmw.mpu, "is_pipeline_last_stage", lambda **kwargs: True, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_data_parallel_group", lambda **kwargs: single_rank_group, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_pipeline_model_parallel_group", lambda: single_rank_group, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_pipeline_model_parallel_last_rank", lambda: 0, raising=False)
    wrapper = object.__new__(mmw.MegatronModelWrapper)
    wrapper.actor_module = [SimpleNamespace(training=True)]
    wrapper.use_sample_packing = False
    wrapper._logprob_chunk_size = 2
    return wrapper


def test_pipeline_two_row_outputs_construct_before_collection(monkeypatch, megatron_wrapper):
    # The reported bug had a two-row selected tensor beside a one-row dummy
    # on non-final stages. Construct the real container before mesh filtering.
    sequences = torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]])
    ids = torch.tensor([[[0, 1], [2, 3]], [[4, 5], [1, 6]]])
    torch.manual_seed(92)
    logits = torch.randn(2, 4, 7)

    def model(sequences, *args, **kwargs):
        return logits.clone()

    model.training = True
    megatron_wrapper.actor_module = [model]
    micro = mmw.MegatronForwardMicroBatch(
        sequences=sequences,
        attention_mask=torch.ones_like(sequences),
        position_ids=torch.arange(4).expand(2, 4),
        num_actions=2,
        score_topk_indices=ids,
    )
    actors, outputs = [], []
    for pp in (0, 1):
        monkeypatch.setattr(mmw.mpu, "is_pipeline_last_stage", lambda pp_rank=pp, **kwargs: pp_rank == 1)
        result = megatron_wrapper.forward([micro], seq_len=4, micro_batch_size=2)
        output = TrainingOutputBatch({"output": result.scores})
        if result.selected_logprobs is not None:
            output["score_old_logprobs"] = result.selected_logprobs
        actors.append(ActorInfo(handle=None, rank=MeshRank(0, 0, 0, pp, 2, 1, 2)))
        outputs.append(output)
    collected = concatenate_outputs_after_mesh_dispatch(actors, outputs)
    normalized = logits.log_softmax(-1)[:, 1:3]
    torch.testing.assert_close(collected["output"], normalized.gather(-1, sequences[:, -2:].unsqueeze(-1)).squeeze(-1))
    torch.testing.assert_close(collected["score_old_logprobs"], normalized.gather(-1, ids))


@pytest.mark.parametrize("dtype,index_dtype", [(torch.float32, torch.int64), (torch.bfloat16, torch.int32)])
@pytest.mark.parametrize("packing", [False, True])
def test_selected_response_reuses_sampled_normalizer_without_full_float_logits(
    dtype, index_dtype, packing, megatron_wrapper
):
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 0, 0]])
    sequences = torch.tensor([[0, 0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 2, 0, 0]])
    selected_ids = torch.tensor([[[1, 3], [2, 4], [0, 5]], [[6, 1], [-1, -1], [-1, -1]]])
    sampled_ids = torch.tensor([[3, 4, 5], [2, 0, 0]])
    selected_ids = selected_ids.to(index_dtype)
    torch.manual_seed(91)
    base_logits = torch.randn((2, 6, 7), dtype=dtype, requires_grad=True)
    dense_logits = base_logits / 0.7
    normalized = dense_logits.float().log_softmax(dim=-1)
    compact_positions = torch.tensor([[2, 3, 4], [1, 0, 0]])
    batch_positions = torch.arange(2)[:, None]
    expected_chosen = normalized[batch_positions, compact_positions, sampled_ids]

    megatron_wrapper.use_sample_packing = packing
    packed_params = None
    logits = dense_logits
    if packing:
        # Model logits are already compact; independently concatenate each row
        # to keep the oracle separate from the production sequence transform.
        logits = torch.cat([dense_logits[i, : int(mask.sum())] for i, mask in enumerate(attention_mask)]).unsqueeze(0)
        _, packed_params = mmw.preprocess_packed_seqs(sequences, attention_mask.bool())

    saved_tensors = []

    def save(tensor):
        saved_tensors.append((tensor.shape, tensor.dtype))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
        response = megatron_wrapper._response_logprobs(
            logits, sequences, attention_mask, 3, packed_params, selected_ids
        )
    selected = response.selected_logprobs

    valid = selected_ids >= 0
    score_positions = compact_positions
    expected = normalized[torch.arange(2)[:, None, None], score_positions[:, :, None], selected_ids.clamp_min(0)]
    torch.testing.assert_close(selected[valid], expected[valid], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        response.scores[attention_mask[:, -3:].bool()],
        expected_chosen[attention_mask[:, -3:].bool()],
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.isnan(selected[~valid]).all()
    if dtype == torch.bfloat16:
        assert (logits.shape, torch.float32) not in saved_tensors

    selected_gradient = torch.autograd.grad(selected[valid].sum(), base_logits, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected[valid].sum(), base_logits)[0]
    tolerance = 5e-3 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(selected_gradient, expected_gradient, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("centering_enabled", [False, True])
@pytest.mark.parametrize("packing", [False, True])
def test_training_centering_matches_dense_value_and_gradient(megatron_wrapper, centering_enabled, packing):
    # q and o have tails proportional to p on IDs 2:, so a two-token head
    # must center the full behavior expectation for both advantage signs.
    p = torch.tensor([0.45, 0.30, 0.10, 0.09, 0.06])
    old = torch.tensor([0.33, 0.27, 0.16, 0.144, 0.096])
    q = torch.tensor([0.60, 0.16, 0.096, 0.0864, 0.0576])
    expected_value = 0
    expected_gradient = torch.zeros_like(p)
    for advantage, weight in ((1.7, 1.0), (-2.3, 0.5), (0.0, 1.0)):
        # Differentiate the dense sampled PPO/TIS formula to get its score
        # coefficients. Eliminate the proportional tail with sum p*score = 0.
        lp = p.log().requires_grad_()
        ratio = (lp - old.log()).exp()
        sampled = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
        dense_loss = (q * (old / q).clamp(max=1.3) * sampled).sum()
        coefficient = -torch.autograd.grad(dense_loss, lp)[0]
        tail_coefficient = coefficient[2] / p[2]
        centering = ((coefficient[:2] - tail_coefficient * p[:2]) * lp[:2]).sum()
        expected_value += (dense_loss + centering * centering_enabled).detach() * weight / 2.5
        if not centering_enabled:
            expected_gradient += (-coefficient + p * coefficient.sum()) * weight / (2.5 * 0.7)

    sequences = torch.tensor([[1, 2, action, action, action] for action in range(5)] + [[1, 2, 0, 0, 0]])
    mask = torch.cat((q[:, None].expand(5, 3), torch.zeros(1, 3)))
    advantages = torch.tensor([1.7, -2.3, 0.0]).expand(6, 3).clone()
    advantages[-1] = torch.nan
    candidates = torch.arange(2).expand(6, 3, 2).clone()
    candidates[-1] = -1
    old_head, behavior_head = old[:2].log().expand(6, 3, 2).clone(), q[:2].log().expand(6, 3, 2).clone()
    old_head[-1], behavior_head[-1] = torch.nan, torch.nan
    old_chosen, behavior_chosen = old.log()[sequences[:, -3:]], q.log()[sequences[:, -3:]]
    parameters = (p.log() * 0.7).requires_grad_()

    def model(sequences, *args, **kwargs):
        return parameters.expand(*sequences.shape, 5).clone()

    model.training = True
    megatron_wrapper.actor_module = [model]
    megatron_wrapper.use_sample_packing = packing
    megatron_wrapper.policy_loss_fn = ppo_policy_loss
    megatron_wrapper.cfg = OmegaConf.create(
        {
            "trainer": {
                "algorithm": {
                    "score_centering_enabled": centering_enabled,
                    "max_seq_len": 5,
                    "think_token_weight": 0.5,
                    "off_policy_correction": "custom",
                    "off_policy_correction_rules": [{"kind": "token", "action": "truncate", "high": 1.3}],
                    "eps_clip_low": 0.2,
                    "eps_clip_high": 0.2,
                    "loss_reduction": "token_mean",
                    "use_entropy_loss": False,
                    "use_kl_loss": False,
                    "kl_loss_coef": 0,
                }
            }
        }
    )
    micro = mmw.MegatronPolicyMicroBatch(
        sequences=sequences,
        attention_mask=torch.ones_like(sequences),
        position_ids=torch.arange(5).expand(6, 5),
        num_actions=3,
        old_action_log_probs=old_chosen,
        base_action_log_probs=None,
        advantages=advantages,
        loss_mask=mask,
        rollout_action_logprobs=behavior_chosen,
        response_span_tags=torch.tensor([[0, 1, 0]]).expand(6, 3),
        correction_weights=(old_chosen - behavior_chosen).exp().clamp(max=1.3),
        score_topk_indices=candidates,
        score_old_logprobs=old_head,
        score_behavior_logprobs=behavior_head,
    )
    # Unequal response weights must use the whole step's denominator even when
    # the scheduler divides accumulated gradients by the microbatch count.
    for partitions in ((slice(None),), (slice(0, 2), slice(2, 4), slice(4, 6))):
        parameters.grad = None
        micros = [
            replace(
                micro,
                **{
                    field.name: value[part]
                    for field in fields(micro)
                    if isinstance(value := getattr(micro, field.name), torch.Tensor)
                },
            )
            for part in partitions
        ]
        with torch.compiler.set_stance("force_eager"):
            metrics = megatron_wrapper.forward_backward_mini_batch(
                micros, seq_len=5, micro_batch_size=6 // len(micros), temperature=0.7
            )
        assert sum(row["policy_loss"] for row in metrics) / len(micros) == pytest.approx(
            expected_value.item(), abs=1e-7, rel=1e-6
        )
        torch.testing.assert_close(parameters.grad, expected_gradient, atol=1e-7, rtol=0)
