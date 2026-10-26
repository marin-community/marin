from functools import partial
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from skyrl_train.distributed.megatron import stable_softmax
from skyrl_train.distillation import student_topk_logprobs
from skyrl_train.tensor_math import LOG_PROB_DELTA_CLIP
from skyrl_train.utils.policy_math import differentiable_approx_kl
from tests.softmax_reference import assert_softmax_numerics, decimal_reference, numerical_rows


def test_token_softmax_confident_values_and_gradients() -> None:
    for dtype in (torch.bfloat16, torch.float32):
        logits, targets = numerical_rows(33, dtype)
        for chunk in (None, 17):
            logprobs = partial(stable_softmax.token_logprobs, vocab_start=0, vocab_end=33, group=None, chunk_size=chunk)
            entropy = partial(stable_softmax.token_entropy, group=None, chunk_size=chunk)
            assert_softmax_numerics(logits, targets, logprobs, entropy)
        scoring = stable_softmax.token_logprobs(logits.requires_grad_(), targets, 0, 33, None, inference_only=True)
        assert not scoring.requires_grad
        expected_lp = decimal_reference(logits, targets)[0]
        torch.testing.assert_close(scoring.double(), expected_lp, rtol=5e-6, atol=0)


def _distributed_softmax(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        full, targets = numerical_rows(8, torch.bfloat16)
        expected_lp, expected_h, expected_glp, expected_gh = decimal_reference(full, targets)
        shard = slice(rank * 4, (rank + 1) * 4)
        logits = full[..., shard].clone().requires_grad_()
        lp = stable_softmax.token_logprobs(logits, targets, rank * 4, (rank + 1) * 4, dist.group.WORLD, chunk_size=7)
        h = stable_softmax.token_entropy(logits, dist.group.WORLD, chunk_size=7)
        torch.testing.assert_close(lp.double(), expected_lp, rtol=5e-6, atol=0)
        torch.testing.assert_close(h.double(), expected_h, rtol=5e-6, atol=0)
        weights = torch.linspace(-0.7, 1.3, targets.numel()).reshape(targets.shape)
        actual = torch.autograd.grad(((0.7 * lp - 0.3 * h) * weights).sum(), logits)[0].double()
        lp_gradient = (0.7 * expected_glp * weights.double().unsqueeze(-1)).to(logits.dtype)
        h_gradient = (-0.3 * expected_gh * weights.double().unsqueeze(-1)).to(logits.dtype)
        expected = (lp_gradient + h_gradient).double()[..., shard]
        relative = (actual - expected).norm(dim=-1) / expected.norm(dim=-1).clamp_min(1e-300)
        assert relative.max() < 5e-3, (
            rank,
            relative.max().item(),
            relative.argmax().item(),
            actual[0, relative.argmax()].tolist(),
            expected[0, relative.argmax()].tolist(),
        )
    finally:
        dist.destroy_process_group()


def test_token_softmax_remote_targets_and_cross_rank_ties(tmp_path: Path) -> None:
    mp.spawn(_distributed_softmax, args=(f"file://{tmp_path / 'gloo'}",), nprocs=2)


def test_selected_logprobs_confident_target_and_invalid_ids() -> None:
    data = torch.tensor([[[16.0, 1.0, -2.0], [40.0, 35.0, 40.0]]])
    indices = torch.tensor([[[0, 1, 0, -1], [2, 0, 1, -1]]])
    logits = data.clone().requires_grad_()
    actual = student_topk_logprobs(logits, indices)
    assert actual[..., -1].isnan().all()
    for column in range(3):
        lp, _, gradient, _ = decimal_reference(data, indices[..., column])
        torch.testing.assert_close(actual[..., column].double(), lp, rtol=5e-6, atol=0)
        actual_gradient = torch.autograd.grad(actual[..., column].sum(), logits, retain_graph=True)[0].double()
        torch.testing.assert_close(actual_gradient, gradient, rtol=5e-5, atol=0)


def test_k3_small_ratios_clipping_mask_and_gradient_conventions() -> None:
    data = torch.tensor([[-1.0, -1.0, -1.0, -1.0, -30.0, 30.0, -1.0]])
    base = data + torch.tensor([[1e-5, -1e-5, 0.2, 0.0, 60.0, -60.0, 0.2]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]])
    for estimator in ("k3", "k3_unbiased_gradient"):
        logprobs = data.clone().requires_grad_()
        actual = differentiable_approx_kl(logprobs, base, mask, kl_estimator_type=estimator)
        ref = data.double().requires_grad_()
        ratio = (base.double() - ref).clamp(-LOG_PROB_DELTA_CLIP, LOG_PROB_DELTA_CLIP)
        k3 = (ratio.expm1() - ratio).clamp(-10, 10)
        k2 = 0.5 * ratio.square()
        expected = (k3.detach() + k2 - k2.detach() if estimator == "k3_unbiased_gradient" else k3) * mask
        # Near |ratio|=1e-5, fp32 subtraction costs about 1% relative accuracy in k3 and its gradient.
        torch.testing.assert_close(actual.double(), expected.detach(), rtol=0.02, atol=0)
        expected_gradient = torch.autograd.grad(expected.sum(), ref)[0]
        actual_gradient = torch.autograd.grad(actual.sum(), logprobs)[0].double()
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0.01, atol=0)
