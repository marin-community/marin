import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

from skyrl_train.inference_engines.vllm.decode_invariant.engine import DECODE_INVARIANT_ENGINE_ENVIRONMENT
from skyrl_train.models.grug_exact.ep_combine import ep_combine
from skyrl_train.models.grug_moe import GrugMoeConfig
from tests.gpu.grug_gpu_gates import require_hoppers

SNOWBALL = GrugMoeConfig()
HIDDEN = SNOWBALL.hidden_size
EXPERTS = SNOWBALL.num_local_experts
TOP_K = SNOWBALL.num_experts_per_tok
# Snowball's decode-invariant engines: eight data- and expert-parallel ranks on one node.
RANKS = 8
EXPERTS_PER_RANK = EXPERTS // RANKS
# Tokens per engine rank of each combine. A CUDA-graph step pads every rank to one count and vLLM's reduce_scatterv
# adds the partials with ncclReduceScatter; an eager step keeps each rank's count and, when the counts differ, adds
# them with one ncclReduce per rank.
GRAPH_SIZES = ((2,) * RANKS, (64,) * RANKS, (512,) * RANKS)
EAGER_SIZES = ((2,) * RANKS, (512,) * RANKS, (2048,) * RANKS, (8192,) * RANKS, (1, 512, 7, 2048, 100, 3, 33, 4096))


def _slots(tokens: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Each token's bf16 ``[TOP_K, HIDDEN]`` weighted expert outputs and ``[TOP_K]`` experts, in slot order; every
    rank draws the same values."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    selected = torch.rand(tokens, EXPERTS, generator=generator, device="cuda").argsort(dim=1)[:, :TOP_K]
    # Token 0's slots share rank 0; token 1's sit on four ranks.
    selected[0] = torch.tensor([3, 0, 2, 1])
    selected[1] = torch.arange(TOP_K) * EXPERTS_PER_RANK + 5
    # Magnitudes from 1e-6 to 1e6 make both the fp32 rank sums and the bf16 additions across ranks round.
    magnitudes = 10.0 ** torch.empty(tokens, TOP_K, 1, device="cuda").uniform_(-6.0, 6.0, generator=generator)
    slots = torch.randn(tokens, TOP_K, HIDDEN, generator=generator, device="cuda") * magnitudes
    slots[:, :, :8] = 0.0
    slots[:, :, 8:16] = -0.0
    slots[:, :, 16:24] = torch.randn(tokens, TOP_K, 8, generator=generator, device="cuda") * 1e-39
    # In slot order (2^24 + 1) - 2^24 rounds to 0 in fp32, where expert order would give 1.
    slots[0, :, 24:] = torch.tensor([0.0, 2.0**24, 1.0, -(2.0**24)], device="cuda")[:, None]
    return slots.to(torch.bfloat16), selected


def _rank_partials(slots: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    """``[RANKS, tokens, HIDDEN]``: each engine rank's partial, the fp32 sum of the slots it owns in slot order,
    rounded once."""
    owner = selected // EXPERTS_PER_RANK
    partials = []
    for rank in range(RANKS):
        total = torch.zeros(slots.shape[0], HIDDEN, dtype=torch.float32, device=slots.device)
        for slot in range(TOP_K):
            total = total + torch.where((owner[:, slot] == rank)[:, None], slots[:, slot].float(), 0.0)
        partials.append(total.to(torch.bfloat16))
    return torch.stack(partials)


def _megatron_permuted(slots: torch.Tensor, selected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The slots in Megatron's permuted order (by expert, then token) and the ``[tokens, EXPERTS]`` routing map."""
    tokens = slots.shape[0]
    routing_map = torch.zeros(tokens, EXPERTS, dtype=torch.bool, device=slots.device).scatter(1, selected, True)
    slot_of_expert = torch.zeros(tokens, EXPERTS, dtype=torch.long, device=slots.device)
    slot_of_expert.scatter_(1, selected, torch.arange(TOP_K, device=slots.device).expand(tokens, TOP_K))
    expert, token = routing_map.t().nonzero().unbind(1)
    return slots[token, slot_of_expert[token, expert]], routing_map


def _ring_sum(partials: torch.Tensor, home: int, direction: int) -> torch.Tensor:
    """The rank partials added in bf16 around the ring in ``direction`` from the rank after ``home``, ``home`` last."""
    order = [(home + direction * step) % RANKS for step in range(1, RANKS + 1)]
    total = partials[order[0]]
    for rank in order[1:]:
        total = (total.float() + partials[rank].float()).to(torch.bfloat16)
    return total


def _combine_on_rank(rank: int, rendezvous: str) -> None:
    os.environ.update(DECODE_INVARIANT_ENGINE_ENVIRONMENT)
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=RANKS, timeout=timedelta(minutes=5))
    communicator = PyNcclCommunicator(dist.group.WORLD, device=rank)
    cases = [(sizes, True) for sizes in GRAPH_SIZES] + [(sizes, False) for sizes in EAGER_SIZES]
    for seed, (sizes, graphed) in enumerate(cases):
        slots, selected = _slots(sum(sizes), seed)
        first = sum(sizes[:rank])
        mine = slice(first, first + sizes[rank])
        partial = _rank_partials(slots, selected)[rank].contiguous()
        combined = torch.empty(sizes[rank], HIDDEN, dtype=torch.bfloat16, device="cuda")
        uniform = len(set(sizes)) == 1
        if graphed:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                communicator.reduce_scatter(combined, partial, stream=torch.cuda.current_stream())
            graph.replay()
        elif uniform:
            communicator.reduce_scatter(combined, partial, stream=torch.cuda.current_stream())
        else:
            communicator.reduce_scatterv(combined, partial, list(sizes), stream=torch.cuda.current_stream())
        permuted, routing_map = _megatron_permuted(slots[mine], selected[mine])
        home = torch.full((sizes[rank],), rank, dtype=torch.long, device="cuda")
        expected = ep_combine(permuted, routing_map, selected[mine], home, RANKS)
        reversed_ring = _ring_sum(_rank_partials(slots[mine], selected[mine]), rank, -1)
        counts = torch.tensor(
            [
                int((combined.view(torch.int16) != expected.view(torch.int16)).sum()),
                int((combined.view(torch.int16) != reversed_ring.view(torch.int16)).sum()),
                combined.numel(),
            ]
        )
        dist.all_reduce(counts)
        differing, order_sensitive, values = counts.tolist()
        if rank == 0:
            print(
                f"sizes {sizes} {'graph' if graphed else 'eager'}: {differing} of {values} values differ from "
                f"ep_combine, {order_sensitive} from the reversed ring",
                flush=True,
            )
        assert differing == 0, (sizes, graphed, differing)
        # The values tell the ring's direction apart, so a combine in another order fails the check above.
        assert order_sensitive > 0, (sizes, graphed)
        if rank == 0:
            assert torch.equal(combined[0, 24:], torch.zeros(HIDDEN - 24, dtype=torch.bfloat16, device="cuda"))
    dist.destroy_process_group()


def test_ep_combine_equals_the_engines_nccl_combine_on_eight_ranks(tmp_path):
    """Eight ranks add their expert partials with vLLM's NCCL communicator in a decode-invariant engine's NCCL
    environment, as the engine combines them in CUDA graphs and eagerly, with equal and unequal token counts; every
    rank's tokens equal ep_combine of the same expert outputs bit for bit."""
    require_hoppers(RANKS)
    mp.spawn(_combine_on_rank, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=RANKS, join=True)
