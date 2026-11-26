"""NCCL parity for the interleaved scalar channels produced by vocabulary projection."""

from pathlib import Path

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from skyrl_train.distributed.megatron.megatron_utils import unpack_packed_token_values

from tests.gpu.grug_gpu_gates import require_hoppers


def _check_projected_channels(rank: int, rendezvous: str) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2)
    try:
        parallel_state.initialize_model_parallel(context_parallel_size=2)
        # Mirrored context parallelism assigns a first-half and last-half block to each rank.
        positions = torch.tensor([rank, 3 - rank], dtype=torch.float32, device=device)
        stats = torch.stack((positions, positions + 10), dim=-1).unsqueeze(0).requires_grad_()
        offsets = torch.tensor([0, 4], dtype=torch.int32, device=device)
        packed = PackedSeqParams(qkv_format="thd", cu_seqlens_q_padded=offsets)
        mask = torch.ones((1, 4), dtype=torch.bool, device=device)
        logprobs, entropies = [unpack_packed_token_values(v, packed, mask) for v in stats.unbind(-1)]
        reference = torch.arange(4, dtype=torch.float32, device=device).unsqueeze(0)
        torch.testing.assert_close(logprobs, reference)
        torch.testing.assert_close(entropies, reference + 10)
        (logprobs * reference + entropies * (reference + 1)).sum().backward()
        torch.testing.assert_close(stats.grad, torch.stack((positions, positions + 1), dim=-1).unsqueeze(0))
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def test_packed_projection_scalar_channels_across_two_nccl_ranks(tmp_path: Path) -> None:
    require_hoppers(2)
    torch.multiprocessing.spawn(_check_projected_channels, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2)
