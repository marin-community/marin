"""Trainer side of expert-block sync. Each Megatron policy worker holds one.

Method names follow vLLM's ``TrainerWeightTransferEngine``: ``trainer_init``, ``send_weights``,
``shutdown``. vLLM's class sends from rank 0 only and this transport sends from every rank, so
this is a separate class.

Once per run the worker reports what it holds and creates its groups. Before each send it
checks that its parameters have not been reallocated and that the update being synced is the
one it just finished.
"""

from dataclasses import asdict

import torch

from skyrl_train.weight_sync.expert_block.groups import Rendezvous, destroy_groups
from skyrl_train.weight_sync.expert_block.schedule import Schedule, TrainerRank, from_wire, to_wire
from skyrl_train.weight_sync.expert_block.megatron_source import local_source_slices
from skyrl_train.weight_sync.expert_block.source_views import local_expert_sources
from skyrl_train.weight_sync.expert_block.stream import Stream, bind, storage_identity
from skyrl_train.weight_sync.expert_block.verify_weights import compare_replicas, replay


class ExpertBlockSender:
    def __init__(self, worker, parallel_state):
        self.worker = worker
        self.parallel_state = parallel_state
        self.trainer: TrainerRank | None = None
        self.sources: dict[str, torch.Tensor] = {}
        self.identity: dict[str, tuple] = {}
        self.expert_sources = {}
        self.groups = {}
        self.stream: Stream | None = None

    def inventory(self) -> dict:
        """This rank's coordinates, expert matrices and dense slices."""
        state = self.parallel_state
        self.trainer = TrainerRank(
            torch.distributed.get_rank(),
            state.get_expert_data_parallel_rank(),
            state.get_pipeline_model_parallel_rank(),
            state.get_expert_model_parallel_rank(),
        )
        provider = self.worker.provider
        local = local_source_slices(
            self.worker.bridge.get_conversion_tasks(self.worker.actor_module), provider, pp=self.trainer.pp
        )
        sources = local.sources
        expert_hidden_size = provider.moe_latent_size if provider.moe_latent_size is not None else provider.hidden_size
        experts = local_expert_sources(
            local.experts,
            sources,
            self.trainer,
            num_experts=provider.num_moe_experts,
            expert_parallel_size=state.get_expert_model_parallel_world_size(),
            expert_hidden_size=expert_hidden_size,
            intermediate_size=provider.moe_ffn_hidden_size,
        )
        self.sources = sources
        self.identity = storage_identity(sources)
        self.expert_sources = {item.entry.name: item for item in experts}
        return {
            "trainer": to_wire(self.trainer),
            "expert_parallel_size": state.get_expert_model_parallel_world_size(),
            "layers": sorted({item.entry.layer for item in experts}),
            "model": {
                "num_experts": provider.num_moe_experts,
                "hidden_size": provider.hidden_size,
                "expert_hidden_size": expert_hidden_size,
                "intermediate_size": provider.moe_ffn_hidden_size,
            },
            "experts": [to_wire(item.entry) for item in experts],
            "dense": [to_wire(item) for item in local.dense],
        }

    def trainer_init(self, init_info: dict) -> dict:
        """Create this rank's groups and resolve the tensors the schedule needs. Returns the warm-up seconds per group."""
        if self.stream is not None:
            raise RuntimeError("Expert-block sender is already initialised")
        plan = from_wire(Schedule, init_info["schedule"])
        device = next(iter(self.sources.values())).device
        self.groups, self.stream, warm = bind(
            self.trainer.rank,
            plan,
            Rendezvous(**init_info["rendezvous"]),
            device,
            sources=self.sources,
            expert_sources=self.expert_sources,
        )
        return {"participant": self.trainer.rank, "warmup_seconds": warm}

    def send_weights(self, update_info: dict) -> dict:
        """Send this rank's weights for one update. Refuses if the rank has not finished that update."""
        if self.stream is None:
            raise RuntimeError("Expert-block sender is not initialised")
        version = update_info["version"]
        # Before the first update the weights are the loaded checkpoint. After that, the sync must
        # name the update this rank just finished.
        completed = self.worker._model_version_step
        if completed is not None and completed != version:
            raise RuntimeError(f"Sync names update {version} but this rank last completed {completed}")
        # ``sources`` holds the parameters themselves, so a reassigned ``param.data`` shows up here.
        if storage_identity(self.sources) != self.identity:
            raise RuntimeError("Policy parameter storage changed since the sync was prepared")
        return asdict(self.stream.run(version))

    def verify(self, update_info: dict) -> dict:
        """Send this rank's weights again, and check that its data-parallel peers hold the same bytes."""
        if self.stream is None:
            raise RuntimeError("Expert-block sender is not initialised")
        version = update_info["version"]
        state = self.parallel_state
        expert_keys = {item.source_key for item in self.expert_sources.values()}
        # Megatron's DDP reduces gradients over these groups, so their members must hold the same
        # bytes: expert weights over expert data parallelism, dense weights over data and context
        # parallelism.
        groups = {
            name: state.get_expert_data_parallel_group()
            if name in expert_keys
            else state.get_data_parallel_group(with_context_parallel=True)
            for name in self.sources
        }
        return {
            "replay": asdict(replay(self.stream, version)),
            "replicas": asdict(compare_replicas(self.sources, groups, self.trainer.rank, version)),
        }

    def shutdown(self) -> None:
        destroy_groups(self.groups)
        self.stream = None
