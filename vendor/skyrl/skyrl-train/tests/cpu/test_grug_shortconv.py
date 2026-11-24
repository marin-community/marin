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
@pytest.mark.parametrize(
    ("dtype", "weight_dtype"),
    [(torch.float32, torch.float32), (torch.bfloat16, torch.float32), (torch.bfloat16, torch.bfloat16)],
)
@pytest.mark.parametrize("packed", [True, False])
def test_packed_shortconv_matches_document_values_and_gradients(lengths, taps, dtype, weight_dtype, packed):
    generator = torch.Generator().manual_seed(42)
    if not packed:
        lengths = (sum(lengths),)
    if weight_dtype == torch.bfloat16:
        # Bounded integers keep BF16 reductions exact without loosening the FP32 tolerances.
        x = torch.randint(-1, 2, (sum(lengths), 2, 3), generator=generator).to(dtype).requires_grad_()
        weight = torch.randint(-1, 2, (taps, 3), generator=generator).to(weight_dtype).requires_grad_()
        upstream = torch.randint(-1, 2, x.shape, generator=generator).to(dtype)
    else:
        x = torch.randn(sum(lengths), 2, 3, generator=generator, dtype=dtype, requires_grad=True)
        weight = torch.randn(taps, 3, generator=generator, requires_grad=True)
        upstream = torch.randn(x.shape, generator=generator, dtype=dtype)
    cumulative = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    actual = causal_short_conv(x, weight, cumulative if packed else None)
    expected = _reference(x, weight, lengths)
    actual_grads = torch.autograd.grad(actual, (x, weight), upstream)
    expected_grads = torch.autograd.grad(expected, (x, weight), upstream)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # FP32 weight gradients sum the same products in a different order.
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-5)


def _context_indices(lengths, size, rank):
    indices = []
    offset = 0
    for length in lengths:
        chunks = torch.arange(offset, offset + length).chunk(2 * size)
        indices.extend((chunks[rank], chunks[2 * size - 1 - rank]))
        offset += length
    return torch.cat(indices)


def _check_context_rank(rank, size, rendezvous, dtype, packed):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=size)
    try:
        lengths = (2 * size, 6 * size, 10 * size) if packed else (6 * size,)
        generator = torch.Generator().manual_seed(24)
        if dtype == torch.bfloat16:
            # Every per-rank and reduced sum fits exactly in BF16 for these bounded inputs.
            full = torch.randint(-1, 2, (sum(lengths), 1, 3), generator=generator).to(dtype).requires_grad_()
            weight = torch.randint(-1, 2, (5, 3), generator=generator).to(dtype).requires_grad_()
            upstream = torch.randint(-1, 2, full.shape, generator=generator).to(dtype)
        else:
            full = torch.randn(sum(lengths), 1, 3, generator=generator, requires_grad=True)
            weight = torch.randn(5, 3, generator=generator, requires_grad=True)
            upstream = torch.randn(full.shape, generator=generator)
        cumulative = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
        index = _context_indices(lengths, size, rank)
        local = full.detach()[index].requires_grad_()
        actual = causal_short_conv(local, weight, cumulative if packed else None, dist.group.WORLD)
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
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("packed", [True, False])
def test_shortconv_zigzag_context_parallel_values_and_gradients(tmp_path, size, dtype, packed):
    mp.spawn(_check_context_rank, args=(size, f"file://{tmp_path}/rendezvous", dtype, packed), nprocs=size, join=True)


def _check_nonfinite_context_rank(rank, rendezvous, nonfinite_source):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        lengths = (4, 4)
        full = torch.ones(8, 1, 2)
        upstream = torch.zeros_like(full)
        if nonfinite_source == "input":
            full[0] = float("inf")
            upstream.fill_(1)
        else:
            upstream[0] = float("inf")
        full.requires_grad_()
        weight = torch.ones(4, 2)
        cumulative = torch.tensor([0, 4, 8], dtype=torch.int32)
        index = _context_indices(lengths, 2, rank)
        local = full.detach()[index].requires_grad_()
        actual = causal_short_conv(local, weight, cumulative, dist.group.WORLD)
        expected = _reference(full, weight, lengths)
        (actual_grad,) = torch.autograd.grad(actual, (local,), upstream[index])
        (expected_grad,) = torch.autograd.grad(expected, (full,), upstream)
        torch.testing.assert_close(actual, expected[index], rtol=0, atol=0)
        torch.testing.assert_close(actual_grad, expected_grad[index], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("nonfinite_source", ["input", "upstream"])
def test_shortconv_context_padding_does_not_propagate_nonfinite_values(tmp_path, nonfinite_source):
    mp.spawn(_check_nonfinite_context_rank, args=(f"file://{tmp_path}/rendezvous", nonfinite_source), nprocs=2, join=True)
