"""Restore BF16 model weights from optimizer masters without reading them twice."""

from megatron.core.dist_checkpointing.dict_utils import dict_list_map_inplace, nested_values
from megatron.core.dist_checkpointing.mapping import (
    LocalNonpersistentObject,
    ShardedStateDict,
    ShardedTensor,
    ShardedTensorFactory,
)
from megatron.core.distributed import DistributedDataParallel
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.optimizer.optimizer import MegatronOptimizer, MixedPrecisionOptimizer
import torch
from torch import nn

from skyrl_train.distributed.megatron.checkpoint_writer import model_key_for_master


def defer_model_weights_to_optimizer(model_state: ShardedStateDict, optimizer_state: ShardedStateDict) -> bool:
    """Keep master-backed BF16 parameters local until the optimizer is restored."""
    masters = {
        (model_key_for_master(value.key), tuple(value.data.shape))
        for value in nested_values(optimizer_state)
        if isinstance(value, (ShardedTensor, ShardedTensorFactory))
        and value.data is not None
        and value.data.dtype == torch.float32
    }
    deferred = False

    def defer(value):
        nonlocal deferred
        if (
            isinstance(value, (ShardedTensor, ShardedTensorFactory))
            and value.data is not None
            and value.data.dtype == torch.bfloat16
            and (value.key, tuple(value.data.shape)) in masters
        ):
            deferred = True
            # MCore puts this tensor back into the loaded state dict without
            # scheduling a storage read, including for factory-backed weights.
            return LocalNonpersistentObject(value.data)
        return value

    dict_list_map_inplace(defer, model_state)
    return deferred


@torch.no_grad()
def restore_model_weights_from_optimizer(optimizer: MegatronOptimizer, model: list[nn.Module]) -> None:
    """Cast restored masters into model buffers and gather distributed shards."""
    optimizers = optimizer.chained_optimizers if isinstance(optimizer, ChainedOptimizer) else [optimizer]
    for child in optimizers:
        if isinstance(child, MixedPrecisionOptimizer):
            # MCore's pinned mixed-precision adapter owns the master/model
            # mapping, including DP-local ranges and chained optimizers.
            child._copy_main_params_to_model_params()
    for chunk in model:
        if isinstance(chunk, DistributedDataParallel) and chunk.ddp_config.use_distributed_optimizer:
            chunk.start_param_sync(force_sync=True)
