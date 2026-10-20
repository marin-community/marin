import asyncio
import copy
import os
from collections import defaultdict
from datetime import timedelta
from enum import StrEnum
from typing import Any

import megatron.core.parallel_state as mpu
import ray
import skyrl_train.models.grug_megatron_bridge  # noqa: F401  # registers the Grug bridge with Megatron-Bridge
import torch
import torch.distributed
from huggingface_hub import snapshot_download
from loguru import logger
from megatron.bridge import AutoBridge
from megatron.core.optimizer import ChainedOptimizer, DistributedOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
from omegaconf import OmegaConf
from skyrl_train.distributed.dispatch import MeshRank
from skyrl_train.distributed.megatron.grug_muonh import MegatronGrugMuonH
from skyrl_train.distributed.megatron.megatron_strategy import MegatronStrategy
from skyrl_train.distributed.megatron.megatron_utils import (
    get_model_config,
    materialize_megatron_params,
    dp_weight_checksum_mismatch,
    print_model_size,
)
from skyrl_train.distributed.megatron.optimizer import (
    get_megatron_optimizer,
    get_megatron_optimizer_param_scheduler,
)
from skyrl_train.distributed.megatron.remote_model import install_remote_hf_state, make_prefetch_hook
from skyrl_train.config.remote_read import RemoteReadMode
from skyrl_train.distributed.utils import init_worker_process_group_with_device
from skyrl_train.mismatch_probe.modes import TRAINER_MODES
from skyrl_train.models.grug_moe import GRUG_MOE_MODEL_TYPE, validate_grug_training_strategy
from skyrl_train.timing_observability import PhaseBreakdown
from marinskyrl.runtime_options import PolicyLossType
from skyrl_train.training_batch import (
    TrainingBatchIterator,
    TrainingOutputBatch,
    gradient_accumulation_steps,
)
from skyrl_train.utils.metrics import policy_progress_metrics, policy_training_metrics
from skyrl_train.utils.profiler import Profiler
from skyrl_train.utils.progress import tqdm
from skyrl_train.utils.utils import (
    get_physical_gpu_id,
    moe_router_replay_requested,
    str_to_torch_dtype,
    update_model_config,
)
from skyrl_train.weight_sync.expert_block.sender import ExpertBlockSender
from skyrl_train.weight_sync.weight_extractor import validate_weight_sync_mode
from skyrl_train.workers.grug_validation import GrugValidationSnapshot
from skyrl_train.workers.megatron.megatron_model_wrapper import (
    MegatronForwardMicroBatch,
    MegatronModelWrapper,
    MegatronPolicyMicroBatch,
)
from skyrl_train.workers.megatron.bridge_provider_config import configure_bridge_provider
from skyrl_train.workers.megatron.router_replay_install import install_megatron_router_replay
from skyrl_train.workers.megatron.output_projection_layout import install_dpo_batch_major_output_projection
from skyrl_train.workers.megatron.weight_extractor import BucketedMegatronWeightExtractor, mapping_hf_names
from skyrl_train.workers.worker import (
    CriticWorkerBase,
    PolicyWorkerBase,
    RefWorkerBase,
    log_r3_resident_set,
)
from torch import nn
from transformers import AutoConfig, AutoTokenizer

from marinskyrl.hugging_face_retry import load_hugging_face_with_retry
from marinskyrl.runtime_options import WeightSyncTransport


class _MegatronInitMode(StrEnum):
    TRAINING = "training"
    CHECKPOINT_EXPORT = "checkpoint-export"


class MegatronWorker:
    @property
    def scoring_temperature(self) -> float:
        if self.cfg.trainer.algorithm.policy_loss_type == PolicyLossType.FTPO:
            return 1.0
        return self.cfg.generator.sampling_params.temperature

    def _download_hf_snapshot_if_needed(self, model_path: str, model_config) -> None:
        """Populate the local Hub cache only for non-streamed remote model IDs."""
        if model_config.get("source_uri") or self._local_rank != 0 or os.path.exists(model_path):
            return
        retry = self.cfg.trainer.model_load_retry
        revision = model_config.get("revision")
        load_hugging_face_with_retry(
            lambda: snapshot_download(model_path, revision=revision),
            resource_id=model_path,
            resource_kind="model snapshot",
            max_retries=int(retry.max_retries),
            backoff_base=float(retry.backoff_base_seconds),
            backoff_cap=float(retry.backoff_cap_seconds),
        )

    def init_configs(
        self,
        model_path,
        megatron_config,
        model_config_kwargs,
        transformer_config_kwargs,
        tokenizer_path: str,
        tokenizer_revision: str | None,
        remote_read_mode: RemoteReadMode,
        bf16=True,
        flash_attn=False,
        model_revision: str | None = None,
        model_source_uri: str | None = None,
    ):
        """
        Initialize the Megatron-Bridge bridge and provider objects + hf_config and tokenizer
        """
        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, revision=model_revision)
        validate_grug_training_strategy(getattr(hf_config, "model_type", None), "megatron")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            revision=tokenizer_revision,
        )

        override_config_kwargs = {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        override_config_kwargs.update(model_config_kwargs.get("model_config", {}))
        update_model_config(hf_config, override_config_kwargs=override_config_kwargs)

        # if flash_attn is enabled, we use flash attention backend, otherwise fall back to fused attention backend
        transformer_config_kwargs = OmegaConf.to_container(transformer_config_kwargs, resolve=True)
        transformer_config_kwargs["attention_backend"] = "flash" if flash_attn else "fused"

        if not self.cfg.trainer.gradient_checkpointing:
            for key in ("recompute_granularity", "recompute_method", "recompute_num_layers"):
                transformer_config_kwargs[key] = None

        bridge = AutoBridge.from_hf_pretrained(model_path, trust_remote_code=True, revision=model_revision)
        self.remote_hf_state = None
        if model_source_uri:
            self.remote_hf_state = install_remote_hf_state(bridge, model_source_uri, model_path)
        provider = bridge.to_megatron_provider()
        configure_bridge_provider(
            provider,
            tensor_model_parallel_size=megatron_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=megatron_config.pipeline_model_parallel_size,
            context_parallel_size=megatron_config.context_parallel_size,
            expert_model_parallel_size=megatron_config.expert_model_parallel_size,
            expert_tensor_parallel_size=megatron_config.expert_tensor_parallel_size,
            bf16=bf16,
            attention_backend="flash" if flash_attn else "fused",
        )
        if self.remote_hf_state is not None and remote_read_mode == RemoteReadMode.PREFETCH:
            provider.register_pre_wrap_hook(make_prefetch_hook(bridge, self.remote_hf_state.store), prepend=True)

        for k, v in transformer_config_kwargs.items():
            setattr(provider, k, v)
        provider.finalize()

        self.provider = provider
        self.bridge = bridge

        self.strategy.hf_config = hf_config
        self.tokenizer = tokenizer

    def make_megatron_module(
        self,
        wrap_with_ddp: bool = True,
        ddp_config: dict[str, Any] | None = None,
        bf16: bool = True,
    ) -> list[nn.Module]:
        """
        Creates a megatron GPTModel (optionally DDP wrapped) using the bridge.
        """
        from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig

        default_ddp_config = DistributedDataParallelConfig()
        if wrap_with_ddp:
            default_ddp_config.use_distributed_optimizer = True
        if ddp_config is not None:
            for k, v in ddp_config.items():
                setattr(default_ddp_config, k, v)
        model = self.provider.provide_distributed_model(
            ddp_config=default_ddp_config, wrap_with_ddp=wrap_with_ddp, bf16=bf16
        )
        install_dpo_batch_major_output_projection(model, self.provider, self.cfg.trainer.algorithm.policy_loss_type)
        return model

    def forward(self, data, *, probe_micro_batch_size: int | None = None):
        """
        Override `Worker.forward` to support passing the full mini batch to the MegatronModelWrapper.forward method.
        """
        log_r3_resident_set(self._rank, data)
        # Run in micro batches grouped into a single mini-batch
        micro_bsz = probe_micro_batch_size or self.cfg.trainer.micro_forward_batch_size_per_gpu
        micro_batches = data.chunk(micro_bsz)

        # Build typed micro-batches expected by MegatronModelWrapper.forward
        micro_payloads = []
        device = torch.cuda.current_device()
        for micro in micro_batches:
            micro.to(device)
            sequences = micro["sequences"]
            attention_mask = micro["attention_mask"]
            num_actions = micro.metadata["response_length"]
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
            micro_payloads.append(
                MegatronForwardMicroBatch(
                    sequences=sequences,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    num_actions=num_actions,
                    ftpo_chosen_mask=micro.get("ftpo_chosen_mask"),
                    rollout_routed_experts=micro.routed_experts_tensor(),
                    probe_row_indices=micro.get("probe_row_indices"),
                )
            )

        self.model.eval()
        seq_len = micro_payloads[0].sequences.shape[1]
        mbs = micro_payloads[0].sequences.shape[0]
        with torch.no_grad():
            log_probs = self.model.forward(
                micro_batches=micro_payloads,
                seq_len=seq_len,
                micro_batch_size=mbs,
                temperature=self.scoring_temperature,
            )
        if self.cfg.trainer.policy.megatron_config.check_train_eval_parity:
            self._log_forward_fingerprint("forward", micro_payloads)
            with torch.no_grad():
                repeated = self.model.forward(
                    micro_batches=micro_payloads,
                    seq_len=seq_len,
                    micro_batch_size=mbs,
                    temperature=self.scoring_temperature,
                )
            if mpu.is_pipeline_last_stage(ignore_virtual=True):
                diff = (repeated.float() - log_probs.float()).abs()
                logger.info(
                    f"parity probe forward() repeat dp_rank={mpu.get_data_parallel_rank()}: mean abs "
                    f"{diff.mean().item():.6f}, max abs {diff.max().item():.6f}"
                )

        log_probs = log_probs.to("cpu")
        output = TrainingOutputBatch({"output": log_probs})
        output.metadata = data.metadata
        return output

    def probe_forward(self, data):
        """Use the normal Megatron policy forward with a scoped probe routing mode."""
        mode = data.metadata["probe_mode"]
        micro_batch_size = data.metadata["probe_micro_batch_size"]
        controller = self.model.router_replay
        if mode not in TRAINER_MODES:
            raise ValueError(f"unsupported probe mode: {mode}")
        module_modes = [(module, module.training) for chunk in self.actor_module for module in chunk.modules()]
        rng_state = MegatronStrategy.get_rng_state()
        rng_tracker = get_cuda_rng_tracker()
        tracker_states = copy.deepcopy(rng_tracker.get_states())
        scope = TRAINER_MODES[mode].context(self, data.metadata)
        try:
            with scope:
                if controller is not None:
                    controller.take_probe_observations()
                output = self.forward(data, probe_micro_batch_size=micro_batch_size)
                output.metadata = dict(output.metadata)
                output.metadata["probe_routes"] = controller.take_probe_observations() if controller is not None else []
                return output
        finally:
            for module, was_training in module_modes:
                module.training = was_training
            MegatronStrategy.load_rng_state(rng_state)
            rng_tracker.set_states(tracker_states)

    def _log_forward_fingerprint(self, call: str, micro_payloads: list[MegatronForwardMicroBatch]) -> None:
        """Log checksums of this rank's inputs and parameters so two calls can be compared."""
        token_sum = sum(int(micro.sequences.long().sum().item()) for micro in micro_payloads)
        mask_sum = sum(int(micro.attention_mask.long().sum().item()) for micro in micro_payloads)
        position_sum = sum(int(micro.position_ids.long().sum().item()) for micro in micro_payloads)
        shapes = [tuple(micro.sequences.shape) for micro in micro_payloads[:3]]
        with torch.no_grad():
            param_sum = 0.0
            param_count = 0
            for chunk in self.actor_module:
                for param in chunk.parameters():
                    param_sum += param.double().sum().item()
                    param_count += param.numel()
        logger.info(
            f"parity probe {call} fingerprint rank={torch.distributed.get_rank()} "
            f"dp_rank={mpu.get_data_parallel_rank()} pp_rank={mpu.get_pipeline_model_parallel_rank()} "
            f"micros={len(micro_payloads)} shapes={shapes} tokens={token_sum} mask={mask_sum} "
            f"positions={position_sum} num_actions={micro_payloads[0].num_actions} "
            f"params={param_count} param_sum={param_sum:.6f}"
        )

    def _log_train_eval_parity_probe(self, micro_buffer: list[MegatronPolicyMicroBatch]) -> None:
        """Re-run forward-only passes on the training micro-batches and compare them with the old log-probs.

        The eval-mode pass measures whether the log-prob forward is repeatable at all; the
        train-mode pass isolates module train/eval behaviour from the backward pass. The
        training metrics then report the remaining forward-backward drift.
        """
        micro_payloads = [
            MegatronForwardMicroBatch(
                sequences=micro.sequences,
                attention_mask=micro.attention_mask,
                position_ids=micro.position_ids,
                num_actions=micro.num_actions,
                rollout_routed_experts=micro.rollout_routed_experts,
            )
            for micro in micro_buffer
        ]
        self._log_forward_fingerprint("ppo_train", micro_payloads)
        seq_len = micro_buffer[0].sequences.shape[1]
        micro_bsz = micro_buffer[0].sequences.shape[0]
        old = torch.cat([micro.old_action_log_probs for micro in micro_buffer]).float()
        mask = torch.cat([micro.loss_mask for micro in micro_buffer]).bool()
        for mode in ("eval", "train"):
            self.model.eval() if mode == "eval" else self.model.train()
            with torch.no_grad():
                repeated = self.model.forward(
                    micro_batches=micro_payloads,
                    seq_len=seq_len,
                    micro_batch_size=micro_bsz,
                    temperature=self.scoring_temperature,
                )
            if not mpu.is_pipeline_last_stage(ignore_virtual=True):
                continue
            diff = (repeated.float() - old.to(repeated.device)).abs()[mask.to(repeated.device)]
            logger.info(
                f"train/eval parity probe dp_rank={mpu.get_data_parallel_rank()} {mode}-mode forward vs old "
                f"log-probs: mean abs {diff.mean().item():.6f}, max abs {diff.max().item():.6f}, "
                f"exact fraction {(diff == 0).float().mean().item():.4f}"
            )

    def _maybe_install_router_replay(self, role: str) -> None:
        """Install the MoE router replay controller when the role requests it.

        Called after ``self.model`` and ``self.actor_module`` exist. The knob
        lives on ``trainer.<role>.megatron_config.moe_router_replay``; for
        ``strategy=megatron`` the top-level config guard admits it only once
        the replay plumbing is complete, so tests enable it after
        ``validate_cfg``.
        """
        if not moe_router_replay_requested(self.cfg, role=role):
            return
        model_config = get_model_config(self.actor_module[0])
        recompute_router = model_config.recompute_granularity == "full" or (
            model_config.recompute_granularity == "selective" and "moe" in model_config.recompute_modules
        )
        self.model.router_replay = install_megatron_router_replay(
            self.actor_module,
            recompute_enabled=recompute_router,
        )

    def save_hf_model(self, export_dir: str, tokenizer):
        # Save model in HuggingFace safetensors format
        self.strategy.save_hf_model(
            self.bridge,
            self.model,
            export_dir,
            tokenizer=tokenizer,
        )


class MegatronPolicyWorkerBase(MegatronWorker, PolicyWorkerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: MegatronModelWrapper = None
        self.actor_module: list[nn.Module] = None
        self.scheduler: OptimizerParamScheduler = None
        self.optimizer: DistributedOptimizer = None
        self.profiler: Profiler | None = None
        self._warned_exact_unit_policy_ratio = False
        self._consecutive_nonfinite_steps = 0

    def forward(self, data, *, probe_micro_batch_size: int | None = None):
        with self._memory.span("forward", step=data.metadata.get("global_step")):
            return super().forward(data, probe_micro_batch_size=probe_micro_batch_size)

    def offload_to_cpu(self, pin_memory=True, non_blocking=True, offload_optimizer=True, offload_model=True):
        self.strategy.offload_to_cpu(
            self.actor_module, self.optimizer, pin_memory, non_blocking, offload_optimizer, offload_model
        )

    def backload_to_gpu(self, non_blocking=True, backload_optimizer=True, backload_model=True):
        self.strategy.backload_to_gpu(
            self.actor_module, self.optimizer, non_blocking, backload_optimizer, backload_model
        )

    def init_worker_process_group(self, master_addr: str, master_port: int):
        """
        Override DistributedTorchRayActor.init_worker_process_group to use megatron distributed setup to create the mesh.
        """
        # Device-pinned NCCL PG init via the shared helper — pins set_device(LOCAL_RANK) and
        # passes device_id so ProcessGroupNCCL never "guesses device ID based on global rank".
        # The guess deadlocks the first collective (weight-init barrier) on unmasked-CVD clusters
        # where every actor sees all GPUs (cw-rno2a); see init_worker_process_group_with_device.
        init_worker_process_group_with_device(
            master_addr=master_addr,
            master_port=master_port,
            timeout_seconds=int(self.cfg.trainer.distributed.worker_collective_timeout_seconds),
        )

        # Explicitly wrap torch.distributed.broadcast in torch.no_grad() to avoid a warning in Megatron training where the
        # autograd engine tries to track gradients through the default Torch kernel. This fixes a deprecated behaviour in
        # PyTorch, preventing potential silent errors in future versions.

        if not getattr(torch.distributed, "_skyrl_broadcast_no_grad_patched", False):
            _orig_broadcast = torch.distributed.broadcast

            def _broadcast_no_grad(*args, **kwargs):
                with torch.no_grad():
                    return _orig_broadcast(*args, **kwargs)

            torch.distributed.broadcast = _broadcast_no_grad
            torch.distributed._skyrl_broadcast_no_grad_patched = True

        self.strategy = MegatronStrategy(
            megatron_config=self.cfg.trainer.policy.megatron_config,
            optimizer_config=self.cfg.trainer.policy.optimizer_config,
            seed=self.cfg.trainer.seed,
        )
        self.strategy.setup_distributed(
            timeout=timedelta(seconds=self.cfg.trainer.distributed.worker_collective_timeout_seconds)
        )

        self.mesh_rank = MeshRank(
            dp=mpu.get_data_parallel_rank(),
            sp=mpu.get_context_parallel_rank(),
            tp=mpu.get_tensor_model_parallel_rank(),
            pp=mpu.get_pipeline_model_parallel_rank(),
            world_size=self._world_size,
            dp_size=mpu.get_data_parallel_world_size(),
            pp_size=mpu.get_pipeline_model_parallel_world_size(),
        )

    def _initialize_policy_modules(self, model_path: str, *, mode: _MegatronInitMode) -> None:
        """Construct the shared Megatron model graph at the checkpoint geometry."""
        for_training = mode is _MegatronInitMode.TRAINING
        self.init_configs(
            model_path,
            self.cfg.trainer.policy.megatron_config,
            self.cfg.trainer.policy.megatron_config.model_config_kwargs,
            self.cfg.trainer.policy.megatron_config.transformer_config_kwargs,
            bf16=self.cfg.trainer.bf16,
            flash_attn=self.cfg.trainer.flash_attn,
            model_revision=self.cfg.trainer.policy.model.get("revision"),
            model_source_uri=self.cfg.trainer.policy.model.get("source_uri"),
            remote_read_mode=RemoteReadMode(self.cfg.trainer.policy.model.remote_read_mode),
            tokenizer_path=self.cfg.trainer.policy.model.get("tokenizer_path"),
            tokenizer_revision=self.cfg.trainer.policy.model.get("tokenizer_revision"),
        )

        ddp_config = self.cfg.trainer.policy.megatron_config.ddp_config if for_training else None
        if for_training and str(self.cfg.trainer.policy.optimizer_config.optimizer).lower() == "muonh":
            ddp_config = dict(ddp_config)
            if ddp_config.get("use_distributed_optimizer", False):
                raise ValueError("Grug MuonH needs unsharded Megatron DDP gradient buffers")
            ddp_config["use_distributed_optimizer"] = False
            ddp_config["overlap_param_gather"] = False

        self.actor_module = self.make_megatron_module(
            wrap_with_ddp=for_training,
            ddp_config=ddp_config,
            bf16=self.cfg.trainer.bf16,
        )

        self._download_hf_snapshot_if_needed(model_path, self.cfg.trainer.policy.model)
        torch.distributed.barrier()

        if self.remote_hf_state is not None:
            self.remote_hf_state.store.close()
            stats = self.remote_hf_state.store.read_stats
            logger.info(
                "Loaded Megatron policy weights directly from {} "
                "(rank range bytes read: {}; opens={} reads={} read_seconds={:.2f} MB_per_s={:.2f}; "
                "gets={} prefetch_misses={} prefetch_unused={})",
                self.cfg.trainer.policy.model.source_uri,
                stats.bytes_read,
                stats.opens,
                stats.reads,
                stats.read_seconds,
                stats.bytes_read / 1_000_000 / stats.read_seconds if stats.read_seconds else 0.0,
                stats.gets,
                stats.prefetch_misses,
                stats.prefetch_unused,
            )

        if self._rank == 0:
            print_model_size(self.actor_module[0])

    def init_model(self, model_path, num_training_steps: int = 1e9):
        """Initialize the model, optimizer, and scheduler for the policy worker."""
        self._initialize_policy_modules(model_path, mode=_MegatronInitMode.TRAINING)

        # create profiler
        if self.cfg.trainer.policy.megatron_config.torch_profiler_config.enable:
            self.profiler = Profiler(self.cfg.trainer.policy.megatron_config.torch_profiler_config)

        # create optimizer
        self.optimizer = get_megatron_optimizer(
            self.actor_module,
            self.cfg.trainer.policy.optimizer_config,
            self.cfg.trainer.policy.megatron_config.optimizer_config_kwargs,
        )

        self._normalize_mini_batch_size()

        # create scheduler
        self.scheduler = get_megatron_optimizer_param_scheduler(
            optimizer=self.optimizer,
            config=self.cfg.trainer.policy.optimizer_config,
            num_training_steps=num_training_steps,
        )

        # create worker model
        self.model = MegatronModelWrapper(
            config=self.cfg,
            vocabulary_size=self.strategy.hf_config.vocab_size,
            actor_module=self.actor_module,
            actor_optimizer=self.optimizer,
            policy_loss_fn=self.policy_loss_fn,
            logprob_chunk_size=OmegaConf.select(
                self.cfg, "trainer.policy.megatron_config.logprob_chunk_size", default=None
            ),
        )
        self._maybe_install_router_replay("policy")

        self._expert_block_sender = (
            ExpertBlockSender(self, mpu)
            if self.cfg.generator.weight_sync_transport == WeightSyncTransport.EXPERT_BLOCK
            else None
        )

        # Initialize weight extractor
        self.use_cuda_ipc = self.cfg.generator.weight_sync_backend == "nccl" and self.cfg.trainer.placement.colocate_all
        model_type = self.strategy.hf_config.model_type
        validate_weight_sync_mode(model_type, fuse_weights=bool(self.cfg.generator.fuse_weights))
        self.weight_extractor = BucketedMegatronWeightExtractor(
            bridge=self.bridge,
            actor_module=self.actor_module,
            model_type=model_type,
            bucket_size_threshold_GB=self.cfg.generator.weight_transfer_threshold_cuda_ipc_GB,
        )

        self.empty_cuda_cache = self.cfg.trainer.policy.megatron_config.empty_cuda_cache

    def init_model_for_export(self, model_path: str) -> None:
        """Initialize Megatron model structure without optimizer or training state."""
        self._initialize_policy_modules(model_path, mode=_MegatronInitMode.CHECKPOINT_EXPORT)
        self.model = MegatronModelWrapper(
            config=self.cfg,
            vocabulary_size=self.strategy.hf_config.vocab_size,
            actor_module=self.actor_module,
            logprob_chunk_size=OmegaConf.select(
                self.cfg, "trainer.policy.megatron_config.logprob_chunk_size", default=None
            ),
        )

    def _ppo_train_impl(self, train_data, timing: PhaseBreakdown) -> "TrainingOutputBatch":
        """Train through Megatron Core's pipeline scheduler."""
        self._drain_r3_decentral_stagger(train_data)
        if self.model.router_replay is not None and train_data.routed_experts is None:
            raise ValueError("moe_router_replay is on but the batch carries no rollout_routed_experts")
        dataloader = TrainingBatchIterator(train_data, self.cfg.trainer.micro_train_batch_size_per_gpu)

        micro_batches_per_mini_batch = gradient_accumulation_steps(
            self.policy_mini_batch_size_per_gpu,
            self.cfg.trainer.micro_train_batch_size_per_gpu,
        )

        status_list = []
        all_metrics = defaultdict(list)
        policy_update_steps = 0

        if self.profiler is not None:
            self.profiler.begin_update()

        for epoch in range(self.cfg.trainer.update_epochs_per_batch):
            self.optimizer.zero_grad()
            pbar = tqdm(
                dataloader,
                desc=f"Policy Train epoch [{epoch + 1}/{self.cfg.trainer.update_epochs_per_batch}]",
                disable=not self.strategy.is_rank_0(),
            )

            micro_buffer = []
            for local_step, experience in enumerate(pbar):
                experience.to_device(torch.cuda.current_device())
                sequences = experience.sequences
                attention_mask = experience.attention_mask
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)

                micro_buffer.append(
                    MegatronPolicyMicroBatch(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        num_actions=experience.num_actions,
                        old_action_log_probs=experience.action_log_probs,
                        base_action_log_probs=experience.base_action_log_probs,
                        advantages=experience.advantages,
                        loss_mask=experience.loss_mask,
                        rollout_action_logprobs=experience.rollout_logprobs,
                        correction_weights=experience.correction_weights,
                        response_span_tags=experience.response_span_tags,
                        distillation=experience.distillation,
                        ftpo=experience.ftpo,
                        rollout_routed_experts=experience.rollout_routed_experts,
                        pair_roles=experience.pair_roles,
                    )
                )

                if len(micro_buffer) == micro_batches_per_mini_batch:
                    if experience.ftpo is not None:
                        active = torch.stack([micro.loss_mask.sum() for micro in micro_buffer]).sum()
                        for micro in micro_buffer:
                            if micro.ftpo.weights is not None:
                                active = active + micro.ftpo.weights.sum()
                        torch.distributed.all_reduce(
                            active, group=mpu.get_data_parallel_group(with_context_parallel=False)
                        )
                        if active.item() == 0:
                            # Adam momentum/weight decay must not update an empty FTPO mini-batch.
                            micro_buffer = []
                            continue
                    if self.cfg.trainer.policy.megatron_config.check_train_eval_parity:
                        self._log_train_eval_parity_probe(micro_buffer)
                    # run mini-batch forward-backward and then one optimizer step
                    self.model.train()
                    for chunk in self.actor_module:
                        # if use distributed optimizer, zero grad buffer will be handled by optimizer
                        chunk.zero_grad_buffer()
                    seq_len = micro_buffer[0].sequences.shape[1]
                    micro_bsz = micro_buffer[0].sequences.shape[0]
                    profiler = self.profiler.for_mini_batch(policy_update_steps) if self.profiler is not None else None

                    metrics_list = self.model.forward_backward_mini_batch(
                        micro_batches=micro_buffer,
                        seq_len=seq_len,
                        micro_batch_size=micro_bsz,
                        temperature=self.scoring_temperature,
                        timings=timing,
                        profiler=profiler,
                    )

                    if self.empty_cuda_cache:
                        torch.cuda.empty_cache()

                    with timing.span("megatron_optimizer_step"):
                        step_result = self.strategy.optimizer_step(
                            self.optimizer,
                            self.model,
                            self.scheduler,
                            name="actor",
                            consecutive_nonfinite_steps=self._consecutive_nonfinite_steps,
                            max_consecutive_nonfinite_steps=self.cfg.trainer.policy.max_consecutive_nonfinite_steps,
                        )
                    checksum_mismatch = None
                    if self.cfg.trainer.policy.megatron_config.check_dp_weight_consistency:
                        with timing.span("megatron_dp_weight_checksum"):
                            checksum_mismatch = dp_weight_checksum_mismatch(self.actor_module)
                    if step_result.applied:
                        self._consecutive_nonfinite_steps = 0
                    elif step_result.grad_norm is None:
                        self._consecutive_nonfinite_steps += 1

                    # within a DP group, metrics are already the same across all workers - we then just all reduce across
                    # the whole world size to get the metrics for the global micro batch
                    for i, metrics in enumerate(metrics_list):
                        status = metrics.copy()
                        status["policy_lr"] = self.optimizer.param_groups[0]["lr"]
                        if not self.cfg.trainer.algorithm.use_kl_loss:
                            status.pop("policy_kl")

                        # Attach grad norm only for the last micro in the mini-batch
                        if i == len(metrics_list) - 1:
                            status["skipped_steps"] = float(not step_result.applied)
                            if checksum_mismatch is not None:
                                status["dp_weight_checksum_mismatch"] = checksum_mismatch
                            if step_result.grad_norm is not None:
                                status["raw_grad_norm"] = step_result.grad_norm

                        # attach response_length
                        status["response_length"] = micro_buffer[i].num_actions

                        with timing.span("megatron_world_metric_reduction"):
                            status = self.strategy.all_reduce(status)
                        status_list.append(status)
                        for k, v in status.items():
                            all_metrics[k].append(v)

                    pbar.set_postfix(policy_progress_metrics(status_list[-1]))

                    policy_update_steps += int(step_result.applied)
                    micro_buffer = []

            # drop any trailing micros that don't fill a mini-batch (keep behavior consistent)
            micro_buffer = []

        with timing.span("megatron_final_barrier"):
            torch.distributed.barrier()
        if self.profiler is not None:
            self.profiler.save()

        status_mean = policy_training_metrics(all_metrics, policy_update_steps)
        if "ftpo/active_examples" in all_metrics:
            examples = sum(all_metrics["ftpo/active_examples"])
            status_mean["ftpo/chosen_win"] = sum(all_metrics["ftpo/chosen_win_sum"]) / max(examples, 1e-12)
        if status_mean.get("ppo_ratio_exact_unit_fraction") == 1.0 and not self._warned_exact_unit_policy_ratio:
            logger.warning(
                "Megatron's recomputed old log probabilities exactly match the training forward for every policy "
                "token. PPO clip bounds cannot activate until the mini-batch contains a forward after an optimizer "
                "update; clip-bound sweeps are inert with the current update geometry."
            )
            self._warned_exact_unit_policy_ratio = True

        output = TrainingOutputBatch()
        output.metadata = {"train_status": status_mean}
        return output

    async def expert_block_rpc(self, method: str, *args):
        """Call a method of this rank's expert-block sender."""
        return getattr(self._expert_block_sender, method)(*args)

    async def _broadcast_to_inference_engines(self, inference_engine_client):
        from torch.multiprocessing.reductions import reduce_tensor

        use_prefix_cache = self.cfg.generator.enable_prefix_caching
        materialize_megatron_params(self.actor_module)
        generator_dtype = str_to_torch_dtype(self.cfg.generator.model_dtype)
        cache_reset_task = None
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            cache_reset_task = inference_engine_client.reset_prefix_cache()

        torch.cuda.empty_cache()

        # Bracket
        # the WHOLE multi-chunk sync with vLLM's layerwise reload so model.load_weights defers
        # processing and a single finalize re-runs process_weights_after_loading (re-applying
        # swap_w13_to_w31) EXACTLY once. This is required for both NCCL broadcast and colocated
        # CUDA IPC: transport changes how tensors arrive, not vLLM's kernel-layout contract.
        # Rank 0 drives the engine RPC (same global-rank-0 semantics as the update loop below).
        _w13_bracket = not bool(self.cfg.generator.fuse_weights)

        await self._begin_vllm_layerwise_weight_reload(inference_engine_client, enabled=_w13_bracket)

        # Extract weights using the initialized extractor
        if not self.use_cuda_ipc:
            # Broadcast one bucket per engine RPC and training-rank barrier. The
            # receiver still broadcasts each tensor in its original order.
            for chunk in self.weight_extractor.extract_weights(generator_dtype):
                if torch.distributed.get_rank() == 0:
                    update_weight_task = asyncio.create_task(
                        inference_engine_client.update_named_weights(
                            {
                                "names": chunk.names,
                                "dtypes": chunk.dtypes,
                                "shapes": chunk.shapes,
                            }
                        )
                    )

                # Broadcast weights from trainer rank 0 to serving ranks via
                # the update group, in the same order as the receiver request.
                if torch.distributed.get_rank() == 0:

                    def broadcast_bucket():
                        for tensor in chunk.tensors:
                            torch.distributed.broadcast(tensor.data, 0, group=self._model_update_group)

                    await asyncio.to_thread(broadcast_bucket)
                if torch.distributed.get_rank() == 0:
                    await update_weight_task
                torch.distributed.barrier()

        else:
            # CUDA IPC path: one chunk per bucket (for packing)
            device = torch.cuda.current_device()
            weights_update_request = {"names": [], "dtypes": [], "shapes": [], "sizes": [], "extras": []}

            for chunk in self.weight_extractor.extract_weights(generator_dtype):
                # Each chunk contains all parameters in one bucket
                # Calculate total size for packing (in number of elements)
                total_numel = sum(t.numel() for t in chunk.tensors)
                chunk_dtypes = {t.dtype for t in chunk.tensors}
                assert len(chunk_dtypes) == 1, f"packed weight chunk mixes dtypes: {chunk_dtypes}"
                packed_tensor = torch.empty(
                    total_numel,
                    device=device,
                    dtype=chunk_dtypes.pop(),
                    requires_grad=False,
                )

                offset = 0
                # Copy tensors into consolidated buffers
                for name, tensor, shape, dtype_name in zip(chunk.names, chunk.tensors, chunk.shapes, chunk.dtypes):
                    size = tensor.numel()
                    packed_tensor[offset : offset + size].copy_(tensor.detach().view(-1))
                    offset += size
                    weights_update_request["names"].append(name)
                    weights_update_request["dtypes"].append(dtype_name)
                    weights_update_request["shapes"].append(shape)
                    weights_update_request["sizes"].append(size)

                ipc_handle = reduce_tensor(packed_tensor)
                ipc_handle = {get_physical_gpu_id(): ipc_handle}
                ipc_handle_list = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

                ipc_handles = {}
                for d in ipc_handle_list:
                    ipc_handles.update(d)

                weights_update_request["extras"].append({"ipc_handles": ipc_handles})
                weights_update_request["packed"] = True

                if torch.distributed.get_rank() == 0:
                    await inference_engine_client.update_named_weights(weights_update_request)
                    weights_update_request = {"names": [], "dtypes": [], "shapes": [], "sizes": [], "extras": []}

                # force collect any sent tensors if possible to be memory efficient
                torch.cuda.ipc_collect()

        # Finalize after every transport chunk so vLLM materializes and processes each layer
        # exactly once. In particular, this restores FlashInfer-CUTLASS's w13 kernel layout.
        await self._finish_vllm_layerwise_weight_reload(inference_engine_client, enabled=_w13_bracket)

        torch.distributed.barrier()
        torch.cuda.synchronize()

        if cache_reset_task is not None:
            await cache_reset_task
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    def grug_optimizer_route_snapshot(self):
        """Return this rank and the optimizer route for each named parameter."""
        if str(self.cfg.trainer.policy.optimizer_config.optimizer).lower() != "muonh":
            raise ValueError("Optimizer route snapshot requires Grug MuonH")
        names = {id(parameter): name for chunk in self.actor_module for name, parameter in chunk.named_parameters()}
        optimizers = (
            self.optimizer.chained_optimizers if isinstance(self.optimizer, ChainedOptimizer) else [self.optimizer]
        )
        routes = {}
        for wrapped in optimizers:
            default_route = "grug_muonh" if isinstance(wrapped.optimizer, MegatronGrugMuonH) else "adam"
            for index, group in enumerate(wrapped.optimizer.param_groups):
                route = group.get("optimizer", default_route)
                for parameter in (*wrapped.float16_groups[index], *wrapped.fp32_from_fp32_groups[index]):
                    routes[names[id(parameter)]] = route
        return {"rank": torch.distributed.get_rank(), "routes": routes}

    def grug_validation_snapshot(self, names=()):
        """Return the calling rank and requested Grug weights in HF layout, gathered on rank 0.

        Every rank must call this with the same names because the bridge export
        is collective. The weights mapping is empty on nonzero ranks.
        """
        if self.strategy.hf_config.model_type != GRUG_MOE_MODEL_TYPE:
            raise ValueError("grug_validation_snapshot is only valid for Grug models")
        materialize_megatron_params(self.actor_module)
        wanted = set(names)
        is_rank0 = torch.distributed.get_rank() == 0
        weights = {}
        tasks = self.bridge.get_conversion_tasks(self.actor_module)
        task_names = [mapping_hf_names(task.mapping) for task in tasks]
        known_names = set().union(*task_names)
        requested_names = sorted(wanted)
        # Bridge orders peer tasks by local expert index. If any EP rank needs
        # a task, every peer must run it to enter the same gathers.
        # Append name matches so unknown requests fail on every peer before export.
        selection = torch.tensor(
            [bool(wanted.intersection(names)) for names in task_names]
            + [name in known_names for name in requested_names],
            dtype=torch.uint8,
            device=torch.cuda.current_device(),
        )
        if mpu.get_expert_model_parallel_world_size() > 1:
            torch.distributed.all_reduce(
                selection, op=torch.distributed.ReduceOp.MAX, group=mpu.get_expert_model_parallel_group()
            )
        selected, found = selection.split([len(tasks), len(requested_names)])
        missing = [name for name, matches in zip(requested_names, found.tolist(), strict=True) if not matches]
        if missing:
            raise KeyError(f"missing Grug state entries: {missing}")
        tasks = [task for task, keep in zip(tasks, selected.tolist(), strict=True) if keep]
        for name, tensor in self.bridge.export_hf_weights(
            self.actor_module, show_progress=False, conversion_tasks=tasks
        ):
            if is_rank0 and name in wanted:
                weights[name] = tensor.detach().to("cpu", dtype=torch.float32).contiguous()
        missing = wanted.difference(weights) if is_rank0 else set()
        if missing:
            raise KeyError(f"missing Grug state entries: {sorted(missing)}")
        return GrugValidationSnapshot(
            rank=torch.distributed.get_rank(),
            attention_backend=str(self.provider.attention_backend),
            weights=weights,
        )

    def get_weight_statistics(self):
        """Compute lightweight statistics for model weights"""
        raise NotImplementedError()

    def _set_pad_token_id(self, pad_token_id):
        # this already gets set in the init_model method
        pass


class MegatronRefWorkerBase(MegatronWorker, RefWorkerBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model: MegatronModelWrapper = None
        self.actor_module: list[nn.Module] = None

    def offload_to_cpu(self, pin_memory=True, non_blocking=True, **kwargs):
        self.strategy.offload_to_cpu(self.actor_module, None, pin_memory, non_blocking)

    def backload_to_gpu(self, non_blocking=True, **kwargs):
        self.strategy.backload_to_gpu(self.actor_module, None, non_blocking)

    def init_worker_process_group(self, master_addr: str, master_port: int):
        """
        Override DistributedTorchRayActor.init_worker_process_group to use megatron distributed setup to create the mesh.
        """
        # Device-pinned NCCL PG init via the shared helper (see init_worker_process_group_with_device) —
        # avoids the ProcessGroupNCCL device-guess collective deadlock on unmasked-CVD clusters (cw-rno2a).
        init_worker_process_group_with_device(
            master_addr=master_addr,
            master_port=master_port,
            timeout_seconds=int(self.cfg.trainer.distributed.worker_collective_timeout_seconds),
        )

        self.strategy = MegatronStrategy(
            megatron_config=self.cfg.trainer.ref.megatron_config,
            optimizer_config=None,
            seed=self.cfg.trainer.seed,
        )
        self.strategy.setup_distributed(
            timeout=timedelta(seconds=self.cfg.trainer.distributed.worker_collective_timeout_seconds)
        )

        self.mesh_rank = MeshRank(
            dp=mpu.get_data_parallel_rank(),
            sp=mpu.get_context_parallel_rank(),
            tp=mpu.get_tensor_model_parallel_rank(),
            pp=mpu.get_pipeline_model_parallel_rank(),
            world_size=self._world_size,
            dp_size=mpu.get_data_parallel_world_size(),
            pp_size=mpu.get_pipeline_model_parallel_world_size(),
        )

    def init_model(self, model_path, num_training_steps: int = 1e9):
        """
        Initialize the model for the ref worker.
        """
        # initialize the bridge and provider objects
        self.init_configs(
            model_path,
            self.cfg.trainer.ref.megatron_config,
            self.cfg.trainer.ref.megatron_config.model_config_kwargs,
            self.cfg.trainer.ref.megatron_config.transformer_config_kwargs,
            bf16=self.cfg.trainer.bf16,
            flash_attn=self.cfg.trainer.flash_attn,
            model_revision=self.cfg.trainer.ref.model.get("revision"),
            model_source_uri=self.cfg.trainer.ref.model.get("source_uri"),
            remote_read_mode=RemoteReadMode(self.cfg.trainer.ref.model.remote_read_mode),
            tokenizer_path=self.cfg.trainer.ref.model.get("tokenizer_path"),
            tokenizer_revision=self.cfg.trainer.ref.model.get("tokenizer_revision"),
        )

        self.actor_module = self.make_megatron_module(
            wrap_with_ddp=False,
            ddp_config=None,
            bf16=self.cfg.trainer.bf16,
        )
        if self.remote_hf_state is not None:
            self.remote_hf_state.store.close()

        self._download_hf_snapshot_if_needed(model_path, self.cfg.trainer.ref.model)
        torch.distributed.barrier()

        # load weights
        if self._rank == 0:
            print_model_size(self.actor_module[0])

        # create worker model — ref honors its OWN logprob_chunk_size key
        self.model = MegatronModelWrapper(
            config=self.cfg,
            vocabulary_size=self.strategy.hf_config.vocab_size,
            actor_module=self.actor_module,
            logprob_chunk_size=OmegaConf.select(
                self.cfg, "trainer.ref.megatron_config.logprob_chunk_size", default=None
            ),
        )
        self._maybe_install_router_replay("ref")

    def get_weight_statistics(self):
        """Compute lightweight statistics for model weights"""
        raise NotImplementedError()

    def _set_pad_token_id(self, pad_token_id):
        # this already gets set in the init_model method
        pass


class MegatronCriticWorkerBase(MegatronWorker, CriticWorkerBase):
    def __init__(self, **kwargs):
        raise NotImplementedError()


PolicyWorker = ray.remote(num_gpus=1)(MegatronPolicyWorkerBase)
RefWorker = ray.remote(num_gpus=1)(MegatronRefWorkerBase)
CriticWorker = ray.remote(num_gpus=1)(MegatronCriticWorkerBase)
