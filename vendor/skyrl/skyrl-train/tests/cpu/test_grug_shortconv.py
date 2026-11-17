"""Packed ShortConv values and gradients against independent per-document convolution."""

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from skyrl_train.models.grug_log_scaling import thd_document_positions
from skyrl_train.models.grug_shortconv import causal_short_conv


def _reference(x, weight, lengths):
    outputs = []
    for document in x.split(lengths):
        padded = F.pad(document.permute(1, 2, 0).float(), (weight.shape[0] - 1, 0))
        outputs.append(F.conv1d(padded, weight.flip(0).T[:, None].float(), groups=x.shape[-1]))
    return torch.cat(outputs, dim=-1).permute(2, 0, 1).to(x.dtype)


@pytest.mark.parametrize("lengths", [(1, 2, 7), (8, 16, 24)])
@pytest.mark.parametrize("taps", [1, 4, 9])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_packed_shortconv_matches_document_values_and_gradients(lengths, taps, dtype):
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(sum(lengths), 2, 3, generator=generator, dtype=dtype, requires_grad=True)
    weight = torch.randn(taps, 3, generator=generator, requires_grad=True)
    cumulative = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()])
    actual = causal_short_conv(x, weight, cumulative)
    expected = _reference(x, weight, lengths)
    upstream = torch.randn(actual.shape, generator=generator, dtype=dtype)
    actual_grads = torch.autograd.grad(actual, (x, weight), upstream)
    expected_grads = torch.autograd.grad(expected, (x, weight), upstream)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # FP32 weight gradients sum the same products in a different order.
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-5)


def _check_context_rank(rank, size, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=size)
    try:
        lengths = (2 * size, 6 * size, 10 * size)
        generator = torch.Generator().manual_seed(24)
        full = torch.randn(sum(lengths), 1, 3, generator=generator, requires_grad=True)
        weight = torch.randn(5, 3, generator=generator, requires_grad=True)
        upstream = torch.randn(full.shape, generator=generator)
        cumulative = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()])
        indices = []
        offset = 0
        for length in lengths:
            chunks = torch.arange(offset, offset + length).chunk(2 * size)
            indices.extend((chunks[rank], chunks[2 * size - 1 - rank]))
            offset += length
        index = torch.cat(indices)
        local = full.detach()[index].requires_grad_()
        actual = causal_short_conv(local, weight, cumulative, dist.group.WORLD)
        expected = _reference(full, weight, lengths)
        actual_input_grad, actual_weight_grad = torch.autograd.grad(actual, (local, weight), upstream[index])
        dist.all_reduce(actual_weight_grad)
        expected_input_grad, expected_weight_grad = torch.autograd.grad(expected, (full, weight), upstream)
        torch.testing.assert_close(actual, expected[index], rtol=0, atol=0)
        torch.testing.assert_close(actual_input_grad, expected_input_grad[index], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(actual_weight_grad, expected_weight_grad, rtol=1e-5, atol=1e-5)
        document, position = thd_document_positions(cumulative, len(index), size, rank)
        assert torch.equal(cumulative[document] + position, index)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("size", [2, 4])
def test_shortconv_zigzag_context_parallel_values_and_gradients(tmp_path, size):
    mp.spawn(_check_context_rank, args=(size, f"file://{tmp_path}/rendezvous"), nprocs=size, join=True)
