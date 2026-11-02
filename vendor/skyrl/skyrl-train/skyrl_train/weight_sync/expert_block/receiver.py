"""Receiver side of expert-block sync. Each vLLM worker holds one.

The constructor and method names follow vLLM's ``WeightTransferEngine``:
``(vllm_config, device, model)``, ``init_transfer_engine``, ``receive_weights``, ``shutdown``.
The trainer calls it through one worker RPC.

Expert weights are written in place, so the engine's ``w13_weight`` must keep the trainer's
``[gate;up]`` order. The MoE backend sets that order, so ``inventory`` accepts only the
backend that was tested.
"""

from dataclasses import asdict

import torch

from skyrl_train.weight_sync.expert_block.groups import Rendezvous, destroy_groups
from skyrl_train.weight_sync.expert_block.schedule import Schedule, from_wire
from skyrl_train.weight_sync.expert_block.source_views import LAYER_PREFIX, ROUTED_EXPERTS, dtype_name
from skyrl_train.weight_sync.expert_block.stream import Stream, bind, storage_identity
from skyrl_train.weight_sync.expert_block.verify_weights import replay

SUPPORTED_MODEL_TYPE = "grug_moe"
# The only backend tested. TRITON keeps the trainer's [gate;up] order in w13_weight. FlashInfer
# CUTLASS swaps it to [up;gate] at load time. BATCHED_TRITON keeps the order but is untested.
SUPPORTED_MOE_BACKEND = "TRITON"


def installable_parameters(model) -> dict[str, torch.Tensor]:
    """The model's parameters, with vocabulary tensors narrowed to their HF rows.

    vLLM pads the rows of the embedding and the LM head and zeroes the padding. The trainer
    sends only the HF rows.
    """
    rows = {}
    for name, module in model.named_modules():
        if not hasattr(module, "org_vocab_size") or not hasattr(module, "num_embeddings_padded"):
            continue
        if module.tp_size != 1 or module.shard_indices.org_vocab_start_index != 0:
            raise ValueError(f"Vocabulary tensor {name} is tensor-parallel; the receiver must run TP=1")
        rows[f"{name}.weight"] = int(module.org_vocab_size)
    return {name: value.narrow(0, 0, rows[name]) if name in rows else value for name, value in model.named_parameters()}


class ExpertBlockReceiver:
    def __init__(
        self,
        vllm_config,
        device,
        model,
        *,
        ep_rank: int,
        ep_size: int,
        pp_rank: int,
        pp_size: int,
        gpu_uuid: str,
    ):
        self.vllm_config = vllm_config
        self.device = device
        self.model = model
        self.ep_rank = ep_rank
        self.ep_size = ep_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.gpu_uuid = gpu_uuid
        self.participant: int | None = None
        self.parameters: dict[str, torch.Tensor] = {}
        self.identity: dict[str, tuple] = {}
        self.expert_maps: dict[str, tuple[int, ...]] = {}
        self.groups = {}
        self.stream: Stream | None = None

    def inventory(self) -> dict:
        """Check that this transport can write into the model, and report what this worker holds."""
        hf, parallel = self.vllm_config.model_config.hf_config, self.vllm_config.parallel_config
        if hf.model_type != SUPPORTED_MODEL_TYPE:
            raise ValueError(f"Expert-block sync supports {SUPPORTED_MODEL_TYPE}, not {hf.model_type}")
        if self.vllm_config.model_config.quantization is not None:
            raise ValueError("Expert-block sync requires unquantised weights")
        if parallel.tensor_parallel_size != 1:
            raise ValueError("Expert-block sync requires TP=1 inference engines")
        if parallel.enable_eplb:
            raise ValueError("Expert-block sync requires a static expert placement (no EPLB)")
        expert_hidden_size = hf.latent_dim if hf.latent_dim is not None else hf.hidden_size
        # With pipeline parallelism this worker holds only its stage's layers.
        maps = {}
        for name, module in self.model.named_modules():
            if not name.endswith(ROUTED_EXPERTS):
                continue
            backend = getattr(getattr(module.quant_method, "unquantized_backend", None), "name", None)
            if backend != SUPPORTED_MOE_BACKEND:
                raise ValueError(
                    f"Expert-block sync requires the {SUPPORTED_MOE_BACKEND} MoE backend, whose w13 layout is "
                    f"[gate;up]; the engine selected {backend}"
                )
            maps[name] = tuple(
                int(module._map_global_expert_id_to_local_expert_id(expert)) for expert in range(hf.num_experts)
            )
        layers = sorted(int(LAYER_PREFIX.match(name)[1]) for name in maps)
        if self.pp_size == 1 and len(layers) != hf.num_hidden_layers:
            raise ValueError(f"Found {len(layers)} routed-expert layers, expected {hf.num_hidden_layers}")
        self.parameters = installable_parameters(self.model)
        self.identity = storage_identity(dict(self.model.named_parameters()))
        self.expert_maps = maps
        dense = {
            name: (tuple(value.shape), dtype_name(value.dtype))
            for name, value in self.parameters.items()
            if not name.endswith((".w13_weight", ".w2_weight"))
        }
        return {
            "gpu_uuid": self.gpu_uuid,
            "ep_rank": self.ep_rank,
            "expert_parallel_size": self.ep_size,
            "pp_rank": self.pp_rank,
            "pp_size": self.pp_size,
            "layers": layers,
            "model": {
                "num_experts": hf.num_experts,
                "hidden_size": hf.hidden_size,
                "expert_hidden_size": expert_hidden_size,
                "intermediate_size": hf.moe_intermediate_size,
                "num_hidden_layers": hf.num_hidden_layers,
            },
            "dense": {name: [list(shape), dtype] for name, (shape, dtype) in sorted(dense.items())},
        }

    def init_transfer_engine(self, init_info: dict) -> dict:
        """Create this worker's groups. Returns the warm-up seconds per group."""
        if self.stream is not None:
            raise RuntimeError("Expert-block receiver is already initialised")
        participants = init_info["participants"]
        if self.gpu_uuid not in participants:
            raise RuntimeError(f"GPU {self.gpu_uuid} is not a receiver in the expert-block schedule")
        plan = from_wire(Schedule, init_info["schedule"])
        participant = self.participant = participants[self.gpu_uuid]
        self.groups, self.stream, warm = bind(
            participant,
            plan,
            Rendezvous(**init_info["rendezvous"]),
            self.device,
            parameters=self.parameters,
            expert_maps=self.expert_maps,
        )
        return {"participant": participant, "warmup_seconds": warm}

    def receive_weights(self, update_info: dict) -> dict:
        """Receive this sync's weights into the live parameters. Returns the install report."""
        if self.stream is None:
            raise RuntimeError("Expert-block receiver is not initialised")
        # If a reload reallocated the parameters, the broadcasts would write into freed buffers.
        if storage_identity(dict(self.model.named_parameters())) != self.identity:
            raise RuntimeError("Model parameter storage changed since the expert-block receiver was initialised")
        return asdict(self.stream.run(update_info["version"]))

    def verify(self, update_info: dict) -> dict:
        """Replay the sync and count the bytes that differ from the installed weights."""
        if self.stream is None:
            raise RuntimeError("Expert-block receiver is not initialised")
        return asdict(replay(self.stream, update_info["version"]))

    def shutdown(self) -> None:
        destroy_groups(self.groups)
        self.stream = None
