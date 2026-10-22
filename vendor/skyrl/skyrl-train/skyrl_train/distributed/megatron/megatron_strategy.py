import os
import math
import random
import tempfile
from pathlib import Path
from contextlib import nullcontext
from datetime import timedelta
from typing import List, Union, Optional
from loguru import logger

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch import distributed as dist

from skyrl_train.distributed.strategy import DistributedStrategy
from skyrl_train.distributed.utils import ModelOrModelOptimPair
from skyrl_train.io import io
from skyrl_train.workers.megatron.megatron_model_wrapper import MegatronModelWrapper
import megatron.core.parallel_state as mpu
from skyrl_train.distributed.megatron.megatron_utils import (
    offload_megatron_model_to_cpu,
    load_megatron_model_to_gpu,
    offload_megatron_optimizer,
    load_megatron_optimizer,
    offload_megatron_grads_to_cpu,
    load_megatron_grads_to_gpu,
    materialize_megatron_params,
    restore_offloaded_optimizer_state,
)
from skyrl_train.distributed.megatron.direct_checkpoint import (
    DirectS3TorchDistLoadShardedStrategy,
    DirectS3TorchDistSaveShardedStrategy,
)
from skyrl_train.distributed.megatron.checkpoint_metadata import remote_checkpoint_metadata
from skyrl_train.distributed.megatron.checkpoint_restore import (
    defer_model_weights_to_optimizer,
    restore_model_weights_from_optimizer,
)
from marinskyrl.remote_io import abort_multipart_uploads

from megatron.core.dist_checkpointing.strategies import base as ckpt_base
from megatron.core.dist_checkpointing.strategies.async_utils import AsyncCallsQueue
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.serialization import (
    get_default_save_sharded_strategy,
)
from megatron.core.dist_checkpointing.strategies.fully_parallel import (
    FullyParallelLoadStrategyWrapper,
    FullyParallelSaveStrategyWrapper,
)
from transformers import PreTrainedTokenizer
from megatron.core.optimizer import ChainedOptimizer, DistributedOptimizer
from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32
from skyrl_train.distributed.step_policy import (
    NonfiniteStepPolicy,
    OptimizerStepResult,
    nonfinite_step_policy,
)
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

from skyrl_train import hf_model_io
from skyrl_train.hf_release_export import ReleaseExportOptions, finalize_grug_release
from marin_hf_release import ReleasePolicy, canonical_grug_config
from skyrl_train.distributed.megatron.grug_muonh import MegatronGrugMuonH


# Both formats make non-flattened ShardedTensors accepted by torch_dist. The default
# fully_reshardable format gathers optimizer state onto DP rank zero's CPU; dp_reshardable
# writes DP-local optimizer shards without a gather but cannot change non-DP geometry on load.
_OPTIMIZER_CHECKPOINT_SHARDING_TYPES = {"fully_reshardable", "dp_reshardable"}
_PARAM_STATE_SHARDING_TYPE_KEY = "param_state_sharding_type"


def _optimizer_checkpoint_metadata(sharding_type: str) -> dict:
    if sharding_type not in _OPTIMIZER_CHECKPOINT_SHARDING_TYPES:
        raise ValueError(f"Unsupported Megatron optimizer checkpoint sharding type: {sharding_type}")
    metadata = {"distrib_optim_sharding_type": sharding_type}
    if sharding_type == "fully_reshardable":
        metadata["distrib_optim_fully_reshardable_mem_efficient"] = True
    return metadata


def _saved_optimizer_sharding_type(common_state: dict) -> str:
    optimizer_state = common_state["optimizer"]
    if _PARAM_STATE_SHARDING_TYPE_KEY in optimizer_state:
        saved_types = {optimizer_state[_PARAM_STATE_SHARDING_TYPE_KEY]}
    else:
        saved_types = {state[_PARAM_STATE_SHARDING_TYPE_KEY] for state in optimizer_state.values()}
    if len(saved_types) != 1:
        raise ValueError(f"Checkpoint contains mixed optimizer sharding types: {saved_types}")
    sharding_type = saved_types.pop()
    _optimizer_checkpoint_metadata(sharding_type)
    return sharding_type


_MUONH_CHECKPOINT_RECIPE = "MuonH"


def _muonh_fp32_substates(optimizer_state: dict) -> list[dict]:
    return (
        [optimizer_state]
        if "fp32_from_fp16_params" in optimizer_state
        else [value for value in optimizer_state.values() if isinstance(value, dict)]
    )


def _muonh_fp32_groups_as_mapping(optimizer_state: dict) -> None:
    """Avoid mixed empty/nonempty list merges across PP and EP ranks."""
    for state in _muonh_fp32_substates(optimizer_state):
        groups = state.get("fp32_from_fp16_params")
        if groups is not None:
            state["fp32_from_fp16_params"] = {
                group_index: {param_index: param for param_index, param in enumerate(group)}
                for group_index, group in enumerate(groups)
            }


def _muonh_fp32_groups_as_lists(optimizer_state: dict) -> None:
    """Restore the nested lists expected by Megatron's Float16 wrapper."""
    for state in _muonh_fp32_substates(optimizer_state):
        groups = state.get("fp32_from_fp16_params")
        if groups is not None:
            state["fp32_from_fp16_params"] = [
                [params[index] for index in sorted(params)] for _, params in sorted(groups.items())
            ]


def _muonh_adamh_states(optimizer):
    optimizers = optimizer.chained_optimizers if isinstance(optimizer, ChainedOptimizer) else [optimizer]
    for wrapped in optimizers:
        if not isinstance(wrapped.optimizer, MegatronGrugMuonH):
            continue
        for group in wrapped.optimizer.param_groups:
            if group.get("optimizer") == "grug_adamh":
                for parameter in group["params"]:
                    yield wrapped.optimizer.state[parameter], parameter.device


def _muonh_adamh_step(optimizer) -> int:
    """Return the shared AdamH step, including ranks without an output head."""
    local_steps = {
        int(step.item()) for state, _ in _muonh_adamh_states(optimizer) if (step := state.get("step")) is not None
    }
    if len(local_steps) > 1:
        raise ValueError(f"Grug AdamH parameters have different steps: {local_steps}")
    shared_step = torch.tensor(max(local_steps, default=0), device=torch.cuda.current_device())
    dist.all_reduce(shared_step, op=dist.ReduceOp.MAX)
    return int(shared_step.item())


def _restore_muonh_adamh_step(optimizer, step: int) -> None:
    for state, device in _muonh_adamh_states(optimizer):
        state["step"] = torch.tensor(step, dtype=torch.int64, device=device)


class MegatronStrategy(DistributedStrategy):
    """
    The strategy for training with Megatron.
    """

    def __init__(
        self,
        megatron_config,
        optimizer_config=None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.megatron_config = megatron_config
        self.optimizer_config = optimizer_config
        self.seed = seed
        self.hf_config = None  # Set by the megatron worker once configs are initialized.
        if optimizer_config is not None:
            _optimizer_checkpoint_metadata(megatron_config.optimizer_checkpoint_sharding_type)

        # NOTE: Set Megatron dist checkpoint async backend to persistent to avoid `os.fork()`-ing
        # short-lived background workers, which does not work well with Ray.
        ckpt_base.async_calls = AsyncCallsQueue(persistent=True)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.cuda.device_count() > 0:
            from megatron.core import tensor_parallel

            tensor_parallel.model_parallel_cuda_manual_seed(seed)

    def setup_distributed(self, timeout=timedelta(minutes=30)) -> None:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != -1:
            torch.cuda.set_device(local_rank)

        mpu.initialize_model_parallel(
            tensor_model_parallel_size=self.megatron_config.tensor_model_parallel_size,
            pipeline_model_parallel_size=self.megatron_config.pipeline_model_parallel_size,
            expert_model_parallel_size=self.megatron_config.expert_model_parallel_size,
            expert_tensor_parallel_size=self.megatron_config.expert_tensor_parallel_size,
            use_sharp=False,
            context_parallel_size=self.megatron_config.context_parallel_size,
            nccl_communicator_config_path=None,
            distributed_timeout_minutes=timeout.total_seconds() / 60,
        )
        self.set_seed(self.seed)
        self.world_size = dist.get_world_size()

    def offload_to_cpu(
        self, model, optimizer, pin_memory=True, non_blocking=True, offload_optimizer=True, offload_model=True
    ):
        """
        Offload model weights and optimizer to CPU memory.
        """
        if offload_model:
            offload_megatron_model_to_cpu(model)
        if optimizer and offload_optimizer:
            offload_megatron_grads_to_cpu(model)
            offload_megatron_optimizer(optimizer)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def backload_to_gpu(self, model, optimizer, non_blocking=True, backload_optimizer=True, backload_model=True):
        """Reload model weights back to GPU."""
        if backload_model:
            load_megatron_model_to_gpu(model)
        if optimizer and backload_optimizer:
            load_megatron_grads_to_gpu(model)
            load_megatron_optimizer(optimizer)
        torch.cuda.synchronize()

    def backward(self, loss: torch.Tensor, model, optimizer: optim.Optimizer, **kwargs) -> None:
        raise NotImplementedError()

    def _all_ranks_finite_grad_norm(
        self,
        optimizer: optim.Optimizer,
        consecutive_nonfinite_steps: int,
        max_consecutive_nonfinite_steps: int | None,
    ) -> float | None:
        """Return the finite norm, clear gradients on a skip, or raise when the skip allowance is exhausted."""
        found_inf = bool(optimizer.prepare_grads())
        flag = torch.tensor(int(found_inf), device=self.collective_device())
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        found_inf = bool(flag.item())
        grad_norm = None if found_inf else float(optimizer.get_grad_norm())
        flag.fill_(int(found_inf or not math.isfinite(grad_norm)))
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item():
            action = nonfinite_step_policy(consecutive_nonfinite_steps, max_consecutive_nonfinite_steps)
            optimizer.zero_grad()
            if action is NonfiniteStepPolicy.FAIL:
                raise RuntimeError(
                    f"nonfinite policy gradients after {consecutive_nonfinite_steps} consecutive skipped steps; "
                    f"max_consecutive_nonfinite_steps={max_consecutive_nonfinite_steps}"
                )
            return None
        return grad_norm

    @torch.no_grad()
    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model,
        scheduler,
        name="model",
        consecutive_nonfinite_steps: int = 0,
        max_consecutive_nonfinite_steps: int | None = None,
        **kwargs,
    ) -> OptimizerStepResult:
        """Apply prepared gradients only when all training ranks can take a finite step."""
        grad_norm = self._all_ranks_finite_grad_norm(
            optimizer, consecutive_nonfinite_steps, max_consecutive_nonfinite_steps
        )
        if grad_norm is None:
            return OptimizerStepResult(grad_norm=None, applied=False)

        should_skip_update = False
        optimizers = optimizer.chained_optimizers if isinstance(optimizer, ChainedOptimizer) else [optimizer]
        for child in optimizers:
            if getattr(child, "is_stub_optimizer", False):
                continue
            parameters = child.get_parameters()
            if not parameters:
                continue
            if child.config.clip_grad > 0:
                clip_grad_by_total_norm_fp32(
                    parameters,
                    max_norm=child.config.clip_grad,
                    total_norm=grad_norm,
                    use_decoupled_grad=(
                        child.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
                        or (
                            child.config.use_precision_aware_optimizer
                            and getattr(parameters[0], "__fsdp_param__", False)
                        )
                    ),
                )
            should_skip_update |= grad_norm > child.config.grad_norm_skip_threshold
        flag = torch.tensor(int(should_skip_update), device=self.collective_device())
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        applied = False if flag.item() else bool(optimizer.step_with_ready_grads())
        if applied:
            scheduler.step(1)
        optimizer.zero_grad()
        return OptimizerStepResult(grad_norm=grad_norm, applied=applied)

    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        raise NotImplementedError()

    def save_checkpoint(
        self,
        model: MegatronModelWrapper,
        ckpt_dir: str,
        node_local_rank: int,
        optimizer: Optional[DistributedOptimizer] = None,
        scheduler: Optional[OptimizerParamScheduler] = None,
        client_state: dict = {},
        tag=None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        # Extract base model.
        model: List[nn.Module] = model.actor_module
        materialize_megatron_params(model)
        assert len(model) == 1, "Megatron virtual pipeline parallel is not yet supported"
        model = model[0]
        if hasattr(model, "module"):
            model = model.module

        # Create checkpoint directory if it doesn't exist.
        if node_local_rank == 0:
            io.makedirs(ckpt_dir, exist_ok=True)

        # All ranks wait for the checkpoint directory to be created before saving.
        dist.barrier()

        # Collect the sharded state dicts for model and optimizer, and full state dict for the scheduler.
        sharded_state_dict = {}
        model_sharded_state_dict = model.sharded_state_dict()
        sharded_state_dict["model"] = model_sharded_state_dict
        if optimizer:
            # An empty training selection can reach checkpointing before the first
            # update. Megatron's serializer still requires the lazy optimizer state.
            optimizers = optimizer.chained_optimizers if isinstance(optimizer, ChainedOptimizer) else [optimizer]
            for child in optimizers:
                if child.optimizer is not None and child.init_state_fn is not None:
                    child.init_state_fn(child.optimizer, child.config)
            sharded_state_dict["optimizer"] = optimizer.sharded_state_dict(
                model_sharded_state_dict,
                metadata=_optimizer_checkpoint_metadata(self.megatron_config.optimizer_checkpoint_sharding_type),
            )
            if self.optimizer_config is not None and str(self.optimizer_config.optimizer).lower() == "muonh":
                _muonh_fp32_groups_as_mapping(sharded_state_dict["optimizer"])
                sharded_state_dict["optimizer_recipe"] = _MUONH_CHECKPOINT_RECIPE
                sharded_state_dict["optimizer_recipe_step"] = _muonh_adamh_step(optimizer)
        if scheduler:
            sharded_state_dict["lr_scheduler"] = scheduler.state_dict()

        # Save RNG state.
        sharded_state_dict["rng"] = self.get_rng_state()

        # Save the checkpoint across ranks in parallel. Each S3 rank shard is one
        # multipart object; the local work directory contains only small control files.
        if ckpt_dir.startswith("s3://"):
            if self.is_rank_0():
                try:
                    abort_multipart_uploads(ckpt_dir)
                except ValueError:
                    raise
                except Exception:
                    logger.opt(exception=True).warning(
                        "Could not remove stale multipart uploads below {}; relying on bucket lifecycle cleanup",
                        ckpt_dir,
                    )
            dist.barrier()
            save_strategy = DirectS3TorchDistSaveShardedStrategy(ckpt_dir)
        else:
            save_strategy = get_default_save_sharded_strategy("torch_dist")
        save_strategy = FullyParallelSaveStrategyWrapper(
            save_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
        )

        with io.local_work_dir(ckpt_dir) as work_dir:
            # TODO(tgriggs): Support configurable async saves.
            async_save_request = dist_checkpointing.save(
                sharded_state_dict=sharded_state_dict,
                checkpoint_dir=work_dir,
                sharded_strategy=save_strategy,
                async_sharded_save=False,
                validate_access_integrity=True,
            )
            assert async_save_request is None, "Async save is not yet supported for Megatron"

            # Only global rank 0 saves the Huggingface config and tokenizer.
            if self.is_rank_0():
                hf_dir = os.path.join(work_dir, "huggingface")
                self.save_hf_configs(self.hf_config, hf_dir, tokenizer)

                # Replicated client state is written by rank zero and read by every rank.
                extra_state_path = os.path.join(work_dir, "extra_state.pt")
                with io.open_file(extra_state_path, "wb") as f:
                    torch.save({"client_state": client_state, "tag": tag}, f)

        dist.barrier()
        ckpt_base.async_calls.close()
        ckpt_base.async_calls = AsyncCallsQueue(persistent=True)
        self.log(f"Checkpoint successfully saved to {ckpt_dir}")

    def load_checkpoint(
        self,
        model: MegatronModelWrapper,
        ckpt_dir: str,
        optimizer: Optional[DistributedOptimizer] = None,
        scheduler: Optional[OptimizerParamScheduler] = None,
        load_module_strict: bool = True,
        load_training_state: bool = True,
    ):
        load_strategy = DirectS3TorchDistLoadShardedStrategy(ckpt_dir)
        if not io.exists(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

        # Extract base model.
        model: List[nn.Module] = model.actor_module
        materialize_megatron_params(model)
        assert len(model) == 1, "Megatron virtual pipeline parallel is not yet supported"
        unwrapped_model = model[0]
        if hasattr(unwrapped_model, "module"):
            unwrapped_model = unwrapped_model.module

        # Extract sharded state dicts.
        sharded_state_dict = {}
        model_sharded_state_dict = unwrapped_model.sharded_state_dict()
        sharded_state_dict["model"] = model_sharded_state_dict
        if scheduler and load_training_state:
            sharded_state_dict["lr_scheduler"] = scheduler.state_dict()

        muonh = self.optimizer_config is not None and str(self.optimizer_config.optimizer).lower() == "muonh"
        model_weights_from_optimizer = False
        with remote_checkpoint_metadata(ckpt_dir) as read_dir:
            if optimizer and load_training_state:
                common_state = dist_checkpointing.load_common_state_dict(read_dir)
                if muonh and common_state.get("optimizer_recipe") != _MUONH_CHECKPOINT_RECIPE:
                    raise ValueError("Checkpoint does not contain Grug MuonH optimizer state")
                if muonh and "optimizer_recipe_step" not in common_state:
                    raise ValueError("Checkpoint does not contain Grug AdamH step state")
                saved_type = (
                    self.megatron_config.optimizer_checkpoint_sharding_type
                    if muonh
                    else _saved_optimizer_sharding_type(common_state)
                )
                # Gradients are not checkpointed. Free their GPU buffers now: building the
                # optimizer's sharded state dict for loading allocates a full set of moments
                # before optimizer.load_state_dict allocates the checkpointed ones, and with the
                # gradients resident that second copy is what OOMs a policy that trains fine.
                # The empty buffers come back after the optimizer state is restored below.
                offload_megatron_grads_to_cpu(model)
                sharded_state_dict["optimizer"] = optimizer.sharded_state_dict(
                    model_sharded_state_dict,
                    is_loading=True,
                    metadata=_optimizer_checkpoint_metadata(saved_type),
                )
                if muonh:
                    _muonh_fp32_groups_as_mapping(sharded_state_dict["optimizer"])
                model_weights_from_optimizer = defer_model_weights_to_optimizer(
                    model_sharded_state_dict, sharded_state_dict["optimizer"]
                )
            # Load the checkpoint in parallel.
            if not muonh:
                load_strategy = FullyParallelLoadStrategyWrapper(
                    load_strategy, mpu.get_data_parallel_group(with_context_parallel=True)
                )
            state_dict = dist_checkpointing.load(
                sharded_state_dict=sharded_state_dict, checkpoint_dir=read_dir, sharded_strategy=load_strategy
            )

        # Load the model, optimizer, and scheduler state dicts.
        assert "model" in state_dict, (
            f"Model state dict not found in checkpoint loaded from {ckpt_dir}. Available keys: {state_dict.keys()}"
        )
        model[0].load_state_dict(state_dict.pop("model"), strict=load_module_strict)
        self.log("Loaded model state dict.")

        if optimizer and load_training_state:
            assert "optimizer" in state_dict, (
                f"Optimizer state dict not found in checkpoint loaded from {ckpt_dir}. Available keys: {state_dict.keys()}"
            )
            if muonh:
                _muonh_fp32_groups_as_lists(state_dict["optimizer"])
            optimizer.load_state_dict(state_dict.pop("optimizer"))
            if muonh:
                _restore_muonh_adamh_step(optimizer, common_state["optimizer_recipe_step"])
            restore_offloaded_optimizer_state(optimizer)
            if model_weights_from_optimizer:
                restore_model_weights_from_optimizer(optimizer, model)
            load_megatron_grads_to_gpu(model)
            self.log("Loaded optimizer state dict.")

        if scheduler and load_training_state:
            assert "lr_scheduler" in state_dict, (
                f"LR scheduler state dict not found in checkpoint loaded from {ckpt_dir}. Available keys: {state_dict.keys()}"
            )
            scheduler.load_state_dict(state_dict["lr_scheduler"])
            self.log("Loaded LR scheduler state dict.")

        # Load RNG state, if present.
        if load_training_state and "rng" in state_dict:
            self.load_rng_state(state_dict["rng"])

        states = {}
        extra_state_path = os.path.join(ckpt_dir, "extra_state.pt")
        if load_training_state and io.exists(extra_state_path):
            with io.open_file(extra_state_path, "rb") as f:
                extra_state = torch.load(f, weights_only=False)
            states = extra_state.get("client_state", {}) or {}
            self.log("Loaded client state from checkpoint.")

        return ckpt_dir, states

    def save_hf_model(
        self, bridge, model: MegatronModelWrapper, output_dir: str, tokenizer=None,
        release_options: ReleaseExportOptions | None = None, **kwargs
    ) -> None:
        if release_options is not None:
            ReleasePolicy(release_options.precision)
            canonical_grug_config(self.hf_config.to_dict())
        materialize_megatron_params(model.actor_module)
        # Create checkpoint directory if it doesn't exist.
        rank_zero = self.is_rank_0()
        if rank_zero:
            io.makedirs(output_dir, exist_ok=True)
        dist.barrier()

        # Every rank exhausts Bridge's collective conversion. Rank zero owns artifact finalization and cloud
        # publication; local nonzero ranks write directly to the shared directory without publishing a manifest.
        if rank_zero:
            model_dir = hf_model_io.local_hf_model_dir(output_dir)
        elif io.is_cloud_path(output_dir):
            model_dir = tempfile.TemporaryDirectory()
        else:
            model_dir = nullcontext(output_dir)
        with model_dir as work_dir:
            bridge.save_hf_weights(model.actor_module, work_dir)
            self.log(f"Successfully saved HF safetensors model to {output_dir}")

            # Only rank 0 saves the Huggingface config and tokenizer.
            if rank_zero:
                if release_options is None:
                    self.save_hf_configs(self.hf_config, work_dir, tokenizer)
                else:
                    finalize_grug_release(Path(work_dir), self.hf_config.to_dict(), tokenizer, release_options)
                self.log(f"Successfully saved HF config and tokenizer to {output_dir}")

        # The Ray caller waits for every rank result; no collective needs to span artifact publication.
