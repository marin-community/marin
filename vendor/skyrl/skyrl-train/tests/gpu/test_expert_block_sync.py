"""Real NCCL shard transfers, online FP8 reloads, and replay verification."""

from dataclasses import asdict
from datetime import timedelta
import json
from pathlib import Path
from uuid import uuid4

import pytest
import torch
import torch.distributed as dist
from torch import nn
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.model import ModelConfig
from vllm.distributed import cleanup_dist_env_and_memory, init_distributed_environment, initialize_model_parallel
from vllm.forward_context import get_forward_context, set_forward_context
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.reload import record_metadata_for_reloading
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.grugmoe import (
    GrugMoeDenseMLP,
    GrugMoeMLP,
    GrugMoeRuntimeConfig,
)
from vllm.transformers_utils.configs.grugmoe import GrugMoeConfig
from vllm.utils.network_utils import get_open_port
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.workspace import init_workspace_manager

from skyrl_train.weight_sync.expert_block.groups import Rendezvous, destroy_groups
from skyrl_train.weight_sync.expert_block.receiver import ExpertBlockReceiver
from skyrl_train.weight_sync.expert_block.schedule import (
    DenseSlice,
    ExpertEntry,
    ReceiverRank,
    TrainerRank,
    build_schedule,
)
from skyrl_train.weight_sync.expert_block.source_views import ExpertSource, dtype_name
from skyrl_train.weight_sync.expert_block.stream import Stream, bind

HIDDEN = 128
LATENT = 64
INTERMEDIATE = 64
EXPERTS = 4
VOCAB = 66
TOKENS = (3, 5)
SYNC_TIMEOUT = 180


class TinyExpertModel(nn.Module):
    """Hero's latent MoE and shared experts, plus a transposed FP8 dense weight."""

    def __init__(self, config, quant_config, prefix):
        super().__init__()
        self.tie_word_embeddings = False
        self.model = nn.Module()
        self.model.embed_tokens = VocabParallelEmbedding(VOCAB, HIDDEN, params_dtype=torch.bfloat16)
        # Checkpoint loading leaves vocabulary padding untouched.
        with torch.no_grad():
            self.model.embed_tokens.weight.zero_()
        layer = nn.Module()
        layer.proj = ReplicatedLinear(
            HIDDEN,
            HIDDEN,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=quant_config,
            disable_tp=True,
            prefix=f"{prefix}.proj",
        )
        layer.mlp = GrugMoeMLP(config, quant_config=quant_config, prefix=f"{prefix}.mlp")
        layer.shared_experts = nn.ModuleList(
            GrugMoeDenseMLP(
                HIDDEN,
                INTERMEDIATE,
                torch.bfloat16,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts.{index}",
            )
            for index in range(2)
        )
        self.model.layers = nn.ModuleList([layer])
        record_metadata_for_reloading(self)

    def forward(self, x):
        layer = self.model.layers[0]
        projected, _ = layer.proj(x)
        result = layer.mlp(projected)
        for shared in layer.shared_experts:
            result = result + shared(projected)
        return result


def checkpoint(version):
    generator = torch.Generator().manual_seed(42 + version)
    shapes = {
        "model.embed_tokens.weight": (VOCAB, HIDDEN),
        "model.layers.0.proj.weight": (HIDDEN, HIDDEN),
        "model.layers.0.mlp.router.weight": (EXPERTS, HIDDEN),
        "model.layers.0.mlp.router.bias": (EXPERTS,),
        "model.layers.0.mlp.latent_down_proj.weight": (LATENT, HIDDEN),
        "model.layers.0.mlp.latent_norm.weight": (LATENT,),
        "model.layers.0.mlp.latent_up_proj.weight": (HIDDEN, LATENT),
    }
    for index in range(2):
        for projection in ("gate", "up", "down"):
            shapes[f"model.layers.0.shared_experts.{index}.{projection}_proj.weight"] = (
                (HIDDEN, INTERMEDIATE) if projection == "down" else (INTERMEDIATE, HIDDEN)
            )
    dense = {
        name: (
            torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32) * (0.05 + 0.1 * version)
        ).bfloat16()
        for name, shape in shapes.items()
    }
    dense["model.layers.0.mlp.latent_norm.weight"].add_(1)
    dense["model.layers.0.mlp.router.bias"] = torch.tensor([-0.3, 0.5, 0.1, -0.2], device="cpu", dtype=torch.float32)
    experts = {}
    for expert in range(EXPERTS):
        for projection, shape in (("fc1", (2 * INTERMEDIATE, LATENT)), ("fc2", (LATENT, INTERMEDIATE))):
            name = f"model.layers.0.mlp.experts.{projection}.expert{expert}"
            experts[name] = (
                torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32)
                * (expert + 1)
                * (0.06 + 0.1 * version)
            ).bfloat16()
    return dense, experts


def load_checkpoint(model, version, vllm_config, device):
    dense, experts = checkpoint(version)
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, weight in dense.items():
            parameter = parameters[name]
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            loader(parameter, weight.to(device))
        for expert in range(EXPERTS):
            prefix = "model.layers.0.mlp.experts"
            w13 = experts[f"{prefix}.fc1.expert{expert}"].to(device)
            w2 = experts[f"{prefix}.fc2.expert{expert}"].to(device)
            # The model's checkpoint loader accepts fused global expert tensors.
            # Its expert map discards the other receiver's experts.
            for projection, weight in (("gate", w13[:INTERMEDIATE]), ("up", w13[INTERMEDIATE:]), ("down", w2)):
                parameter_name = f"{prefix}.{projection}_proj.weight"
                routed = model.model.layers[0].mlp.experts.routed_experts
                destination = routed.w2_weight if projection == "down" else routed.w13_weight
                loader = destination.weight_loader
                loader(
                    destination,
                    weight,
                    parameter_name,
                    shard_id={"gate": "w1", "up": "w3", "down": "w2"}[projection],
                    expert_id=expert,
                )
    process_weights_after_loading(model, vllm_config.model_config, device)


def serving_output(model, config, x, ep_rank):
    with set_forward_context(
        None, config, num_tokens=TOKENS[ep_rank], num_tokens_across_dp=torch.tensor(TOKENS, device="cpu")
    ):
        with get_forward_context().dp_metadata.sp_local_sizes(1):
            return model(x)


def receiver_worker(rank, store, directory, quantization, dp_port):
    device = torch.device("cuda", rank)
    ep_rank = rank - 1
    config = VllmConfig()
    config.parallel_config.data_parallel_size = 2
    config.parallel_config.data_parallel_rank = ep_rank
    config.parallel_config.data_parallel_master_port = dp_port
    config.parallel_config.enable_expert_parallel = True
    config.parallel_config.all2all_backend = "allgather_reducescatter"
    config.kernel_config.moe_backend = "triton"
    hf = GrugMoeConfig(
        vocab_size=VOCAB,
        hidden_dim=HIDDEN,
        latent_dim=LATENT,
        intermediate_dim=INTERMEDIATE,
        shared_expert_intermediate_dim=INTERMEDIATE,
        num_shared_experts=2,
        num_experts=EXPERTS,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=4,
        num_kv_heads=1,
        head_dim=32,
        grugmoe_artifact_schema_version=2,
    )
    hf.architectures = ["GrugMoeForCausalLM"]
    model_directory = Path(directory, f"model-{rank}")
    hf.save_pretrained(model_directory)
    config.model_config = ModelConfig(
        model=str(model_directory),
        dtype="bfloat16",
        quantization=quantization,
        max_model_len=128,
    )
    config.quant_config = Fp8Config() if quantization else None
    cfg = GrugMoeRuntimeConfig.from_hf_config(hf)
    with set_current_vllm_config(config), torch.device(device), set_default_torch_dtype(torch.bfloat16):
        init_distributed_environment(world_size=1, rank=0, local_rank=rank, backend="nccl")
        initialize_model_parallel(1, 1)
        init_workspace_manager(device)
        model = TinyExpertModel(cfg, config.quant_config, "installed")
        load_checkpoint(model, 0, config, device)
        receiver = ExpertBlockReceiver(
            config,
            device,
            model,
            ep_rank=ep_rank,
            ep_size=2,
            pp_rank=0,
            pp_size=1,
            gpu_uuid=str(torch.cuda.get_device_properties(device).uuid),
        )
        inventory = receiver.inventory()
        store.set(f"inventory-{ep_rank}", json.dumps(inventory))
        receiver.init_transfer_engine(json.loads(store.get("plan")))
        x = torch.randn(
            TOKENS[ep_rank], HIDDEN, generator=torch.Generator(device=device).manual_seed(27), device=device
        )
        # Capture a dense projection: its FP8 weight and scale addresses must
        # remain valid after both updates and read-only replay verification.
        graph = torch.cuda.CUDAGraph()
        projection = model.model.layers[0].proj
        projection(x)
        torch.cuda.synchronize(device)
        with torch.cuda.graph(graph):
            captured = projection(x)[0]
        reports = []
        for version in range(2):
            receiver.receive_weights({"version": version})
            reference = TinyExpertModel(cfg, config.quant_config, f"reference{version}")
            load_checkpoint(reference, version, config, device)
            for name, actual in model.named_parameters():
                expected = dict(reference.named_parameters())[name]
                assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)), name
            output = serving_output(model, config, x, ep_rank)
            expected_output = serving_output(reference, config, x, ep_rank)
            torch.testing.assert_close(output, expected_output, atol=0, rtol=0)
            graph.replay()
            torch.testing.assert_close(captured, reference.model.layers[0].proj(x)[0], atol=0, rtol=0)
            report = receiver.verify({"version": version})
            assert report["mismatched_bytes"] == 0
            assert report["compared_bytes"] == report["parameter_bytes"]
            if quantization:
                assert report["quantized_bytes"] > 0
            reports.append(report)
            after_verify = serving_output(model, config, x, ep_rank)
            torch.testing.assert_close(after_verify, expected_output, atol=0, rtol=0)
        if quantization:
            scale = model.model.layers[0].mlp.experts.routed_experts.w13_weight_scale
            if ep_rank == 0:
                scale.mul_(2)
            report = receiver.verify({"version": 1})
            assert (report["mismatched_bytes"] > 0) == (ep_rank == 0)
            receiver.receive_weights({"version": 1})
            assert receiver.verify({"version": 1})["mismatched_bytes"] == 0
            torch.testing.assert_close(serving_output(model, config, x, ep_rank), expected_output, atol=0, rtol=0)
        Path(directory, f"receiver-{ep_rank}.json").write_text(json.dumps(reports))
        receiver.shutdown()
        cleanup_dist_env_and_memory()


def transfer_worker(rank, rendezvous_values, directory, quantization, dp_port):
    torch.cuda.set_device(rank)
    torch.set_default_device("cpu")
    torch.set_num_threads(1)
    rendezvous = Rendezvous(**rendezvous_values)
    store = dist.TCPStore(
        rendezvous.address, rendezvous.port, world_size=None, is_master=False, timeout=timedelta(seconds=SYNC_TIMEOUT)
    )
    if rank:
        receiver_worker(rank, store, directory, quantization, dp_port)
        return
    inventories = [json.loads(store.get(f"inventory-{index}")) for index in range(2)]
    assert inventories[0]["dense"] == inventories[1]["dense"]
    dense, experts = checkpoint(0)
    sources = {name: tensor.cuda(rank) for name, tensor in {**dense, **experts}.items()}
    expert_sources = {
        name: ExpertSource(
            ExpertEntry(name, 0, 0, int(name.rsplit("expert", 1)[1]), name.split(".")[-2], value.numel() * 2),
            name,
            tuple(value.shape),
        )
        for name, value in experts.items()
    }
    dense_inventory = [
        DenseSlice(name, 0, tensor.numel(), dtype_name(tensor.dtype), name, 0, 0) for name, tensor in dense.items()
    ]
    schedule = build_schedule(
        [TrainerRank(0, 0, 0, 0)],
        [ReceiverRank(0, 0, 0), ReceiverRank(1, 0, 1)],
        {0: [item.entry for item in expert_sources.values()]},
        {0: dense_inventory},
        receiver_ep=2,
        num_experts=EXPERTS,
        receiver_layers_by_pp=[[0]],
        dense_holders={name: [0] for name in dense},
        dense_numel={name: int(torch.tensor(row[0]).prod()) for name, row in inventories[0]["dense"].items()},
    )
    plan = {
        "schedule": asdict(schedule),
        "participants": {row["gpu_uuid"]: 1 + index for index, row in enumerate(inventories)},
        "rendezvous": rendezvous_values,
    }
    store.set("plan", json.dumps(plan))
    bound = bind(
        0,
        schedule,
        rendezvous,
        torch.device("cuda", rank),
        make_stream=lambda groups: Stream(
            0, schedule, groups, device=torch.device("cuda", rank), sources=sources, expert_sources=expert_sources
        ),
    )
    stream = bound.stream
    for version in range(2):
        dense, experts = checkpoint(version)
        for name, value in {**dense, **experts}.items():
            sources[name].copy_(value)
        stream.run(version)
        stream.verify(version)
    if quantization:
        stream.verify(1)
        stream.run(1)
        stream.verify(1)
    destroy_groups(bound.groups)


@pytest.mark.vllm
@pytest.mark.skipif(torch.cuda.device_count() < 3, reason="requires three CUDA GPUs")
@pytest.mark.parametrize("quantization", [None, "fp8"], ids=["bf16", "online-fp8"])
def test_expert_block_repeated_updates_match_fresh_model(tmp_path, quantization):
    store = dist.TCPStore(
        "127.0.0.1", 0, world_size=None, is_master=True, timeout=timedelta(seconds=SYNC_TIMEOUT), wait_for_workers=False
    )
    rendezvous = Rendezvous("127.0.0.1", store.port, uuid4().hex, SYNC_TIMEOUT)
    torch.multiprocessing.spawn(
        transfer_worker, args=(asdict(rendezvous), str(tmp_path), quantization, get_open_port()), nprocs=3, join=True
    )
    reports = [json.loads((tmp_path / f"receiver-{index}.json").read_text()) for index in range(2)]
    assert all(row["mismatched_bytes"] == 0 for receiver in reports for row in receiver)
