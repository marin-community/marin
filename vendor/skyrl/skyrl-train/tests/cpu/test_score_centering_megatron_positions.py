"""Focused CPU checks of score centering through real Megatron scoring."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from skyrl_train.objective.losses import ppo_policy_loss
from skyrl_train.distributed.dispatch import ActorInfo, MeshRank, concatenate_outputs_after_mesh_dispatch
from skyrl_train.training_batch import TrainingOutputBatch
from tests.cpu.util import stub_megatron_modules

stub_megatron_modules()

from skyrl_train.workers.megatron import megatron_model_wrapper as mmw  # noqa: E402


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
                loss.backward()
            outputs.append(metrics)
        return outputs

    monkeypatch.setattr(mmw, "get_forward_backward_func", lambda: schedule)
    monkeypatch.setattr(mmw.mpu, "get_tensor_model_parallel_group", lambda: single_rank_group, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_selected_response_reuses_sampled_normalizer_without_full_float_logits(
    dtype, compact, index_dtype, megatron_wrapper
):
    attention_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 0, 0]])
    sequences = torch.tensor([[0, 0, 1, 2, 3, 4, 5], [0, 0, 0, 1, 2, 0, 0]])
    selected_ids = torch.tensor([[[1, 3], [2, 4], [0, 5]], [[6, 1], [-1, -1], [-1, -1]]])
    sampled_ids = torch.tensor([[3, 4, 5], [2, 0, 0]])
    candidate_positions = torch.tensor([[0, 2], [0, 1]]) if compact else None
    if compact:
        selected_ids = selected_ids.gather(1, candidate_positions.unsqueeze(-1).expand(-1, -1, 2))
    selected_ids = selected_ids.to(index_dtype)
    torch.manual_seed(91)
    base_logits = torch.randn((2, 6, 7), dtype=dtype, requires_grad=True)
    logits = base_logits / 0.7
    normalized = logits.float().log_softmax(dim=-1)
    compact_positions = torch.tensor([[2, 3, 4], [1, 0, 0]])
    batch_positions = torch.arange(2)[:, None]
    expected_chosen = normalized[batch_positions, compact_positions, sampled_ids]

    saved_tensors = []

    def save(tensor):
        saved_tensors.append((tensor.shape, tensor.dtype))
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
        response = megatron_wrapper._response_logprobs(
            logits, sequences, attention_mask, 3, None, selected_ids, candidate_positions
        )
    selected = response.selected_logprobs

    valid = selected_ids >= 0
    score_positions = compact_positions.gather(1, candidate_positions) if compact else compact_positions
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


@pytest.mark.parametrize("capture_width,centering_enabled", [(0, False), (2, False), (2, True)])
def test_training_centering_and_capture_control_match_dense_value_and_gradient(
    megatron_wrapper, capture_width, centering_enabled
):
    # q and o have tails proportional to p on IDs 2:, so a two-token head
    # must center the full behavior expectation for both advantage signs.
    p = torch.tensor([0.45, 0.30, 0.10, 0.09, 0.06])
    old = torch.tensor([0.33, 0.27, 0.16, 0.144, 0.096])
    q = torch.tensor([0.60, 0.16, 0.096, 0.0864, 0.0576])
    expected_value = 0
    expected_gradient = torch.zeros_like(p)
    for advantage in (1.7, -2.3):
        # Differentiate the dense sampled PPO/TIS formula to get its score
        # coefficients. Eliminate the proportional tail with sum p*score = 0.
        lp = p.log().requires_grad_()
        ratio = (lp - old.log()).exp()
        sampled = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)
        dense_loss = (q * (old / q).clamp(max=1.3) * sampled).sum()
        coefficient = -torch.autograd.grad(dense_loss, lp)[0]
        tail_coefficient = coefficient[2] / p[2]
        centering = ((coefficient[:2] - tail_coefficient * p[:2]) * lp[:2]).sum()
        expected_value += (dense_loss + centering * centering_enabled).detach() / 2
        if not centering_enabled:
            expected_gradient += (-coefficient + p * coefficient.sum()) / (2 * 0.7)

    sequences = torch.tensor([[1, 2, action, action] for action in range(5)] + [[1, 2, 0, 0]])
    mask = torch.cat((q[:, None].expand(5, 2), torch.zeros(1, 2)))
    advantages = torch.tensor([1.7, -2.3]).expand(6, 2).clone()
    advantages[-1] = torch.nan
    candidates = torch.arange(2).expand(6, 2, 2).clone()
    candidates[-1] = -1
    old_head, behavior_head = old[:2].log().expand(6, 2, 2).clone(), q[:2].log().expand(6, 2, 2).clone()
    old_head[-1], behavior_head[-1] = torch.nan, torch.nan
    old_chosen, behavior_chosen = old.log()[sequences[:, -2:]], q.log()[sequences[:, -2:]]
    parameters = (p.log() * 0.7).requires_grad_()

    def model(sequences, *args, **kwargs):
        return parameters.expand(len(sequences), 4, 5).clone()

    model.training = True
    megatron_wrapper.actor_module = [model]
    megatron_wrapper.policy_loss_fn = ppo_policy_loss
    megatron_wrapper.cfg = OmegaConf.create(
        {
            "trainer": {
                "algorithm": {
                    "score_centering_topk": capture_width,
                    "score_centering_enabled": centering_enabled,
                    "max_seq_len": 4,
                    "think_token_weight": 1,
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
        position_ids=torch.arange(4).expand(6, 4),
        num_actions=2,
        old_action_log_probs=old_chosen,
        base_action_log_probs=None,
        advantages=advantages,
        loss_mask=mask,
        rollout_action_logprobs=behavior_chosen,
        response_span_tags=None,
        correction_weights=(old_chosen - behavior_chosen).exp().clamp(max=1.3),
        score_topk_indices=candidates,
        score_old_logprobs=old_head,
        score_behavior_logprobs=behavior_head,
    )
    # Run CPU math without a native compiler for entropy telemetry.
    with torch.compiler.set_stance("force_eager"):
        metrics = megatron_wrapper.forward_backward_mini_batch([micro], seq_len=4, micro_batch_size=6, temperature=0.7)
    assert metrics[0]["policy_loss"] == pytest.approx(expected_value.item(), abs=1e-7, rel=1e-6)
    torch.testing.assert_close(parameters.grad, expected_gradient, atol=1e-7, rtol=0)
