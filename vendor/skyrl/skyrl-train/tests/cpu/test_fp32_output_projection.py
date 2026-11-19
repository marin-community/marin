import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F

from skyrl_train import stable_softmax
from skyrl_train.workers.megatron.output_projection_layout import (
    install_fp32_output_projection,
    use_fp32_output_projection,
)
from tests.cpu.util import stub_megatron_modules


class _OutputProjection(nn.Module):
    def __init__(self, weight: torch.Tensor | None):
        super().__init__()
        self.weight = None if weight is None else nn.Parameter(weight)
        self.gradient_accumulation_fusion = True
        self.seen_operand_dtypes: list[tuple[torch.dtype, torch.dtype]] = []

    def forward(
        self,
        hidden: torch.Tensor,
        weight: torch.Tensor | None = None,
        runtime_gather_output: bool | None = None,
    ):
        del runtime_gather_output
        active_weight = self.weight if weight is None else weight
        assert active_weight is not None
        self.seen_operand_dtypes.append((hidden.dtype, active_weight.dtype))
        return F.linear(hidden, active_weight), None


def _float64_head_reference(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ref_hidden = hidden.detach().double().requires_grad_()
    ref_weight = weight.detach().double().requires_grad_()
    ref_logits = F.linear(ref_hidden, ref_weight) / temperature
    ref_logits.retain_grad()
    log_probs_all = F.log_softmax(ref_logits, dim=-1)
    ref_logprobs = log_probs_all.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    ref_entropy = -(log_probs_all.exp() * log_probs_all).sum(dim=-1)
    (0.7 * ref_logprobs - 0.3 * ref_entropy).sum().backward()
    assert ref_logits.grad is not None and ref_hidden.grad is not None and ref_weight.grad is not None
    return ref_logits, ref_logprobs, ref_entropy, ref_logits.grad, ref_hidden.grad, ref_weight.grad


@pytest.mark.parametrize("weight_mode", ["untied", "tied_kwarg", "tied_positional"])
def test_fp32_output_projection_matches_float64_reference_through_logprobs_and_entropy(weight_mode: str):
    generator = torch.Generator().manual_seed(17)
    batch_size, seq_len, hidden_size, vocab_size = 2, 6, 256, 48
    temperature = 0.8
    hidden = (torch.randn(batch_size, seq_len, hidden_size, generator=generator) * 0.5).to(torch.bfloat16)
    weight = (torch.randn(vocab_size, hidden_size, generator=generator) * 0.25).to(torch.bfloat16)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), generator=generator)

    input_under_test = hidden.clone().requires_grad_()
    if weight_mode == "untied":
        projection = _OutputProjection(weight.clone())
        weight_param = projection.weight
        assert weight_param is not None
        use_fp32_output_projection(projection)
        # Match GPTModel._postprocess: [S, B, H] input and keyword weight=None.
        logits_sbh, _ = projection(input_under_test.transpose(0, 1), weight=None, runtime_gather_output=None)
    else:
        weight_param = nn.Parameter(weight.clone())
        projection = _OutputProjection(None)
        use_fp32_output_projection(projection)
        if weight_mode == "tied_kwarg":
            logits_sbh, _ = projection(
                input_under_test.transpose(0, 1), weight=weight_param, runtime_gather_output=None
            )
        else:
            logits_sbh, _ = projection(input_under_test.transpose(0, 1), weight_param)

    assert projection.gradient_accumulation_fusion is False
    assert projection.seen_operand_dtypes == [(torch.float32, torch.float32)]

    logits = logits_sbh.transpose(0, 1).contiguous()
    assert logits.dtype == torch.float32
    logits.div_(temperature)
    assert logits.dtype == torch.float32

    seen_grad_logits: list[torch.Tensor] = []
    logits.register_hook(lambda grad: seen_grad_logits.append(grad.detach().clone()))

    logprobs = stable_softmax.token_logprobs(logits, targets, 0, vocab_size, None, chunk_size=3)
    entropy = stable_softmax.token_entropy(logits, None, chunk_size=3)
    assert logprobs.dtype == torch.float32
    assert entropy.dtype == torch.float32

    with torch.no_grad():
        scoring_logprobs = stable_softmax.token_logprobs(
            logits.detach(), targets, 0, vocab_size, None, inference_only=True, chunk_size=3
        )
    assert scoring_logprobs.dtype == torch.float32
    torch.testing.assert_close(scoring_logprobs, logprobs.detach(), rtol=0, atol=0)

    (0.7 * logprobs - 0.3 * entropy).sum().backward()
    assert len(seen_grad_logits) == 1
    assert seen_grad_logits[0].dtype == torch.float32
    assert input_under_test.grad is not None and input_under_test.grad.dtype == torch.bfloat16
    assert weight_param.grad is not None and weight_param.grad.dtype == torch.bfloat16

    ref_logits, ref_logprobs, ref_entropy, ref_dlogits, ref_dhidden, ref_dweight = _float64_head_reference(
        hidden, weight, targets, temperature
    )
    torch.testing.assert_close(logits.detach().double(), ref_logits.detach(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(logprobs.detach().double(), ref_logprobs.detach(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(entropy.detach().double(), ref_entropy.detach(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(seen_grad_logits[0].double(), ref_dlogits, rtol=1e-5, atol=1e-5)
    # Gradients flow back to BF16 parameters/activations after the FP32 head backward (BF16 precision: 1 ULP ~= 7.8e-3).
    torch.testing.assert_close(input_under_test.grad.double(), ref_dhidden, rtol=1e-2, atol=1e-5)
    torch.testing.assert_close(weight_param.grad.double(), ref_dweight, rtol=1e-2, atol=1e-5)

    # Confirm the BF16 head baseline has materially larger rounding error vs float64 than the FP32 head.
    bf16_logits = (F.linear(hidden, weight) / temperature).double()
    bf16_logprobs = F.log_softmax(bf16_logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    fp32_logit_err = (logits.detach().double() - ref_logits.detach()).abs().max().item()
    bf16_logit_err = (bf16_logits - ref_logits.detach()).abs().max().item()
    fp32_lp_err = (logprobs.detach().double() - ref_logprobs.detach()).abs().max().item()
    bf16_lp_err = (bf16_logprobs - ref_logprobs.detach()).abs().max().item()
    assert fp32_logit_err < bf16_logit_err / 100
    assert fp32_lp_err < bf16_lp_err / 100


def test_install_fp32_output_projection_targets_pp_last_stage_and_validates_layout(monkeypatch):
    stub_megatron_modules()

    class _FakeColumnParallelLinear(_OutputProjection):
        def __init__(
            self,
            weight: torch.Tensor,
            *,
            output_size: int | None = None,
            output_size_per_partition: int | None = None,
            sequence_parallel: bool = False,
            defer_embedding_wgrad_compute: bool = False,
            bias: torch.Tensor | None = None,
        ):
            super().__init__(weight)
            vocab = weight.shape[0]
            self.output_size = vocab if output_size is None else output_size
            self.output_size_per_partition = vocab if output_size_per_partition is None else output_size_per_partition
            self.sequence_parallel = sequence_parallel
            self.config = SimpleNamespace(defer_embedding_wgrad_compute=defer_embedding_wgrad_compute)
            self.bias = bias

    def _get_attr_wrapped_model(model, attr, allow_none=True):
        while not hasattr(model, attr):
            if not hasattr(model, "module"):
                raise RuntimeError(f"couldn't find attribute {attr}")
            model = model.module
        return getattr(model, attr)

    tp_layers = types.ModuleType("megatron.core.tensor_parallel.layers")
    tp_layers.ColumnParallelLinear = _FakeColumnParallelLinear  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "megatron.core.tensor_parallel.layers", tp_layers)
    monkeypatch.setattr(sys.modules["megatron.core.utils"], "get_attr_wrapped_model", _get_attr_wrapped_model)

    def _wrap(model):
        # Mirror MegatronDDP(Float16Module(GrugGPTModel)).
        return SimpleNamespace(module=SimpleNamespace(module=model))

    stage0 = _wrap(SimpleNamespace(post_process=False))
    head = _FakeColumnParallelLinear(torch.ones((16, 8), dtype=torch.bfloat16))
    stage1 = _wrap(SimpleNamespace(post_process=True, output_layer=head))

    install_fp32_output_projection([stage0, stage1])
    assert head.gradient_accumulation_fusion is False
    logits, _ = head(torch.ones((2, 1, 8), dtype=torch.bfloat16))
    head(torch.ones((2, 1, 8), dtype=torch.bfloat16))
    assert logits.dtype == torch.float32
    assert head.seen_operand_dtypes == [(torch.float32, torch.float32), (torch.float32, torch.float32)]

    unsupported_projections = [
        _FakeColumnParallelLinear(torch.ones((16, 8), dtype=torch.bfloat16), output_size=32, output_size_per_partition=16),
        _FakeColumnParallelLinear(torch.ones((16, 8), dtype=torch.bfloat16), sequence_parallel=True),
        _FakeColumnParallelLinear(torch.ones((16, 8), dtype=torch.bfloat16), defer_embedding_wgrad_compute=True),
        _FakeColumnParallelLinear(
            torch.ones((16, 8), dtype=torch.bfloat16), bias=torch.zeros(16, dtype=torch.bfloat16)
        ),
        nn.Linear(8, 16, bias=False),
    ]
    for bad_head in unsupported_projections:
        bad_stage = _wrap(SimpleNamespace(post_process=True, output_layer=bad_head))
        with pytest.raises(ValueError, match="fp32_lm_head"):
            install_fp32_output_projection([stage0, bad_stage])


def _cp5_packed_worker(rank: int, rendezvous: str, cp_size: int) -> None:
    torch.set_num_threads(1)
    stub_megatron_modules()
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=cp_size)
    try:
        from skyrl_train.distributed.megatron import megatron_utils, model_utils

        for module in (megatron_utils, model_utils):
            module.mpu.get_tensor_model_parallel_world_size = lambda: 1
            module.mpu.get_tensor_model_parallel_rank = lambda: 0
            module.mpu.get_tensor_model_parallel_group = lambda: None
            module.mpu.get_context_parallel_world_size = lambda: cp_size
            module.mpu.get_context_parallel_rank = lambda: rank
            module.mpu.get_context_parallel_group = lambda: dist.group.WORLD

        generator = torch.Generator().manual_seed(29)
        batch_size, seq_len, hidden_size, vocab_size = 2, 16, 128, 32
        temperature = 0.8
        lengths = (12, 9)
        attention_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
        for row, length in enumerate(lengths):
            attention_mask[row, seq_len - length :] = True
        sequences = torch.randint(0, vocab_size, (batch_size, seq_len), generator=generator)
        hidden = (torch.randn(batch_size, seq_len, hidden_size, generator=generator) * 0.5).to(torch.bfloat16)
        weight = (torch.randn(vocab_size, hidden_size, generator=generator) * 0.25).to(torch.bfloat16)

        _, packed_seq_params = megatron_utils.preprocess_packed_seqs(sequences, attention_mask, pre_process=True)
        packed_hidden, _ = megatron_utils.preprocess_packed_seqs(hidden, attention_mask, pre_process=True)
        packed_hidden = packed_hidden.detach().clone().requires_grad_()

        projection = _OutputProjection(weight.clone())
        use_fp32_output_projection(projection)
        # Match Snowball untied TP1 head: [T_local, 1, H] -> [1, T_local, V] FP32.
        logits = projection(packed_hidden.transpose(0, 1), weight=None, runtime_gather_output=None)[0]
        logits = logits.transpose(0, 1).contiguous()
        assert projection.seen_operand_dtypes == [(torch.float32, torch.float32)]
        assert logits.dtype == torch.float32
        logits.div_(temperature)

        packed_targets = megatron_utils.pack_padded_tokens(sequences, attention_mask, packed_seq_params)
        logprobs = model_utils.from_parallel_logits_to_logprobs_packed_sequences(
            logits,
            packed_targets,
            packed_seq_params.cu_seqlens_q_padded,
            attention_mask,
            vocab_start_index=0,
            vocab_end_index=vocab_size,
            group=None,
            inference_only=False,
            cp_group=dist.group.WORLD,
            chunk_size=4,
        )
        packed_entropies = model_utils.vocab_parallel_entropy(logits, chunk_size=4)
        entropies = megatron_utils.unpack_packed_token_values(packed_entropies, packed_seq_params, attention_mask)

        assert logprobs.dtype == torch.float32
        assert entropies.dtype == torch.float32

        num_actions = 5
        loss = (0.7 * logprobs[:, -num_actions:] - 0.3 * entropies[:, -num_actions - 1 : -1]).sum()
        loss.backward()
        assert projection.weight is not None and projection.weight.grad is not None
        total_wgrad = projection.weight.grad.float().clone()
        dist.all_reduce(total_wgrad, op=dist.ReduceOp.SUM, group=dist.group.WORLD)

        # Unsharded float64 reference on the exact same sequences.
        ref_hidden = hidden.double().requires_grad_()
        ref_weight = weight.double().requires_grad_()
        ref_logits = F.linear(ref_hidden, ref_weight) / temperature
        ref_log_probs_all = F.log_softmax(ref_logits, dim=-1)
        ref_next_logprobs = ref_log_probs_all[:, :-1, :].gather(-1, sequences[:, 1:].unsqueeze(-1)).squeeze(-1)
        ref_entropy = -(ref_log_probs_all.exp() * ref_log_probs_all).sum(dim=-1)
        valid_pred = attention_mask[:, :-1] & attention_mask[:, 1:]
        ref_logprobs_masked = torch.where(valid_pred, ref_next_logprobs, 0.0)
        ref_entropy_masked = torch.where(attention_mask, ref_entropy, 0.0)
        (0.7 * ref_logprobs_masked[:, -num_actions:] - 0.3 * ref_entropy_masked[:, -num_actions - 1 : -1]).sum().backward()
        assert ref_weight.grad is not None

        torch.testing.assert_close(logprobs.detach().double(), ref_logprobs_masked.detach(), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(entropies.detach().double(), ref_entropy_masked.detach(), rtol=1e-5, atol=1e-5)
        # Each CP rank rounds its local wgrad slice to BF16 before summing across 5 ranks.
        torch.testing.assert_close(total_wgrad.double(), ref_weight.grad, rtol=2e-2, atol=1e-3)
    finally:
        dist.destroy_process_group()


def test_fp32_output_projection_cp5_packed_logprobs_and_entropy_match_float64_reference(tmp_path: Path):
    mp.spawn(_cp5_packed_worker, args=(f"file://{tmp_path / 'cp5'}", 5), nprocs=5)
