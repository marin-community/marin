# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare CP4 packed sliding attention against CP1 at recorded trajectory lengths."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformer_engine.pytorch import DotProductAttention
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import get_thd_partitioned_indices


def probe(inputs: Path, output: Path) -> None:
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    try:
        lengths = [r["length"] for r in json.loads(inputs.read_text())["rows"]]
        size = dist.get_world_size()
        if size != 4 or sum(lengths) < 98000 or max(lengths) < 50000:
            raise ValueError("Probe requires CP4 and a recorded natural-long packed batch")
        padded = [(n + 2 * size - 1) // (2 * size) * (2 * size) for n in lengths]
        cu = torch.tensor([0, *lengths], dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
        cup = torch.tensor([0, *padded], dtype=torch.int32, device="cuda").cumsum(0, dtype=torch.int32)
        valid = torch.cat([torch.arange(int(cup[i]), int(cup[i]) + n, device="cuda") for i, n in enumerate(lengths)])
        indices = get_thd_partitioned_indices(cup, sum(padded), size, rank).long()
        valid_local = torch.isin(indices, valid)
        global_to_valid = torch.full((sum(padded),), -1, device="cuda", dtype=torch.long)
        global_to_valid[valid] = torch.arange(sum(lengths), device="cuda")
        reference_indices = global_to_valid[indices[valid_local]]
        generator = torch.Generator(device="cuda").manual_seed(20261111)
        full = [
            torch.randn(sum(padded), h, 128, generator=generator, device="cuda", dtype=torch.bfloat16)
            for h in (20, 5, 5)
        ]
        reference = [x[valid].detach().requires_grad_() for x in full]
        local = [x[indices].detach().requires_grad_() for x in full]
        shared = dict(
            num_attention_heads=20,
            num_gqa_groups=5,
            kv_channels=128,
            attention_dropout=0.0,
            qkv_format="thd",
            attn_mask_type="padding_causal",
            window_size=(2047, 0),
        )
        baseline = DotProductAttention(**shared).cuda()
        parallel = DotProductAttention(
            **shared,
            cp_group=dist.group.WORLD,
            cp_global_ranks=list(range(size)),
            cp_stream=torch.cuda.Stream(),
            cp_comm_type="all_gather",
        ).cuda()
        expected = baseline(
            *reference, cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=max(lengths), max_seqlen_kv=max(lengths)
        )
        expected.backward(torch.ones_like(expected))
        actual = parallel(
            *local,
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            cu_seqlens_q_padded=cup,
            cu_seqlens_kv_padded=cup,
            max_seqlen_q=max(padded),
            max_seqlen_kv=max(padded),
        )
        gradient = torch.zeros_like(actual)
        gradient[valid_local] = 1
        actual.backward(gradient)
        comparisons = [("output", actual[valid_local], expected[reference_indices])]
        comparisons.extend(
            (name, x.grad[valid_local], y.grad[reference_indices])
            for name, x, y in zip(("dq", "dk", "dv"), local, reference, strict=True)
        )
        results = {}
        for name, actual_value, expected_value in comparisons:
            difference = actual_value.float() - expected_value.float()
            relative_l2 = difference.norm() / expected_value.float().norm().clamp_min(1e-12)
            metrics = torch.stack([difference.abs().max(), relative_l2])
            dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
            results[name] = {"max_abs": metrics[0].item(), "relative_l2": metrics[1].item()}
            # BF16 kernels sum in different CP orders; require bounded aggregate
            # error and a pointwise bound, not bitwise equality.
            if not torch.isfinite(metrics).all() or metrics[0] > 0.125 or metrics[1] > 0.03:
                raise AssertionError(f"CP parity failed for {name}: {results[name]}")
        if rank == 0:
            output.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "lengths": lengths,
                        "padded_lengths": padded,
                        "comparisons": results,
                        "scope": "TE kernel parity, synthetic QKV at recorded lengths",
                    },
                    indent=2,
                )
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probe(args.inputs, args.output)
