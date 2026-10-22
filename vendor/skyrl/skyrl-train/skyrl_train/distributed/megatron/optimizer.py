# Utils ported from Verl
# https://github.com/volcengine/verl/blob/e1603dc97f3c20c58feed1f5be34acd5c72a830c/verl/utils/megatron/optimizer.py#L4
# The original copyright is reproduced below:

# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping

import torch
from megatron.core.optimizer import OptimizerConfig, ParamKey, ParamWithNamePredicate, get_standard_config_overrides
from megatron.core.optimizer import get_megatron_optimizer as get_megatron_optimizer_native
from megatron.core.optimizer.emerging_optimizers import EmergingOptimizerEntry, _EMERGING_OPTIMIZERS
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.utils import get_model_config

from skyrl_train.distributed.megatron.grug_muonh import MegatronGrugMuonH, megatron_grug_route


_GRUG_MUONH_NAME = "grug_muonh"
_GRUG_EMERGING_ROUTES = ("grug_muonh_qkv", "grug_muonh_gate_up", "grug_adamh")


def _grug_muonh_extra(optim_config: Mapping) -> dict:
    raw = optim_config.get("optimizer_kwargs", {})
    if not isinstance(raw, Mapping):
        raise TypeError("MuonH optimizer_kwargs must be a mapping")
    extra = dict(raw)
    known = {"adam_lr", "momentum", "nesterov", "backend_steps", "epsilon", "muon_epsilon", "offload_momentum"}
    unknown = sorted(set(extra) - known)
    if unknown:
        raise ValueError(f"Unknown MuonH optimizer_kwargs: {unknown}")
    if "offload_momentum" in extra and not isinstance(extra["offload_momentum"], bool):
        raise TypeError("MuonH offload_momentum must be a bool")
    if float(optim_config.get("weight_decay", 0.0)) != 0.0:
        raise ValueError("MuonH requires weight_decay=0")
    return extra


def _register_grug_muonh(optim_config: Mapping) -> None:
    """Register the Grug recipe with the optimizer factory pinned at MCore 0.18."""
    extra = _grug_muonh_extra(optim_config)
    adam_lr = float(extra.get("adam_lr", optim_config["lr"]))
    muon_eps = float(extra.get("muon_epsilon", 1e-8))
    overrides = {}
    for route in (*_GRUG_EMERGING_ROUTES, "adam"):
        predicate = ParamWithNamePredicate(
            name=f"grug_{route}",
            fn=lambda parameter, name, expected=route: megatron_grug_route(name, parameter) == expected,
        )
        override = {"optimizer": route}
        if route == "adam":
            override["max_lr"] = adam_lr
        overrides[ParamKey(with_name_predicate=predicate)] = override

    def config_to_kwargs(config, model_chunks, pg_collection):
        model_config = get_model_config(model_chunks[0])
        if model_config.tensor_model_parallel_size != 1:
            raise ValueError("Grug MuonH fused-matrix updates currently require tensor parallel size 1")
        expert_tp_size = model_config.expert_tensor_parallel_size
        if expert_tp_size is None:
            expert_tp_size = model_config.tensor_model_parallel_size
        if expert_tp_size != 1:
            raise ValueError("Grug MuonH whole-matrix updates require expert tensor parallel size 1")
        kv_rows = model_config.kv_channels
        query_rows = model_config.num_attention_heads // model_config.num_query_groups * kv_rows
        return {
            "lr": config.lr,
            "betas": (config.adam_beta1, config.adam_beta2),
            "momentum": config.muon_momentum,
            "nesterov": config.muon_nesterov,
            "ns_steps": config.muon_num_ns_steps,
            "eps": config.adam_eps,
            "muon_eps": muon_eps,
            "qkv_split_shapes": (query_rows, kv_rows, kv_rows),
            "offload_momentum": extra.get("offload_momentum", False),
        }

    entry = EmergingOptimizerEntry(
        optimizer_cls=MegatronGrugMuonH,
        init_state_fn=lambda optimizer, config=None: optimizer.initialize_state(),
        config_to_kwargs=config_to_kwargs,
        default_param_overrides=overrides,
    )
    for route in (_GRUG_MUONH_NAME, *_GRUG_EMERGING_ROUTES):
        _EMERGING_OPTIMIZERS[route] = entry


class _MegatronParamScheduler(OptimizerParamScheduler):
    def load_state_dict(self, state_dict: dict) -> None:
        self.num_steps = 0
        super().load_state_dict(state_dict)


def _init_megatron_optim_config(optim_config: Mapping, optimizer_config_kwargs: Mapping) -> OptimizerConfig:
    # megatron-core only recognizes 'adam' / 'sgd' as standard optimizers (anything
    # else routes to `_get_megatron_emerging_optimizer`, which raises
    # `ValueError: Unsupported emerging optimizer: AdamW`). megatron's 'adam' IS
    # AdamW (decoupled weight decay via weight_decay), so normalize the common
    # HF-style names to 'adam'.
    _optim_name = str(optim_config.get("optimizer", "adam")).lower()
    if _optim_name in (_GRUG_MUONH_NAME, *_GRUG_EMERGING_ROUTES):
        raise ValueError("Use optimizer='MuonH'; internal Megatron route names are not policy recipes")
    if _optim_name == "adamw":
        _optim_name = "adam"
    if _optim_name == "muonh":
        _optim_name = _GRUG_MUONH_NAME
    optim_args = {
        "optimizer": _optim_name,
        "lr": optim_config.get("lr"),
        "min_lr": optim_config.get("min_lr", 0.0),
        "clip_grad": optim_config.get("max_grad_norm", 1.0),
        "weight_decay": optim_config.get("weight_decay", 0.01),
        "bf16": True,
        "params_dtype": torch.bfloat16,
        "use_distributed_optimizer": True,
    }

    if _optim_name == "adam":
        beta1, beta2 = optim_config.get("adam_betas", (0.9, 0.999))
        optim_args.update(
            adam_beta1=beta1,
            adam_beta2=beta2,
            adam_eps=optim_config.get("optimizer_kwargs", {}).get("eps", 1e-8),
        )
    optim_args.update(optimizer_config_kwargs)

    if _optim_name == _GRUG_MUONH_NAME:
        extra = _grug_muonh_extra(optim_config)
        if float(optim_args["weight_decay"]) != 0.0:
            raise ValueError("MuonH requires weight_decay=0, including Megatron overrides")
        if optimizer_config_kwargs.get("use_distributed_optimizer", False):
            raise ValueError("Grug MuonH uses Megatron's full-matrix optimizer path")
        if optim_args.get("optimizer_cpu_offload", False) or optim_args.get("optimizer_offload_fraction", 0.0):
            raise ValueError("Grug MuonH does not support Megatron's AdamW CPU optimizer offload")
        if optim_args.get("use_precision_aware_optimizer", False):
            raise ValueError("Grug MuonH requires FP32 master parameters")
        betas = tuple(float(beta) for beta in optim_config.get("adam_betas", (0.9, 0.95)))
        if len(betas) != 2:
            raise ValueError("MuonH adam_betas must contain two values")
        optim_args.update(
            adam_beta1=betas[0],
            adam_beta2=betas[1],
            adam_eps=float(extra.get("epsilon", 1e-8)),
            muon_momentum=float(extra.get("momentum", 0.95)),
            muon_nesterov=bool(extra.get("nesterov", True)),
            muon_num_ns_steps=int(extra.get("backend_steps", 5)),
            decoupled_weight_decay=False,
            use_distributed_optimizer=False,
        )

    config = OptimizerConfig(**optim_args)
    return config


def get_megatron_optimizer(
    model,
    optim_config: Mapping,
    optimizer_config_kwargs: Mapping,
):
    """Build the native optimizer from one policy recipe and its Megatron overrides.

    Grug routing and state options stay paired with the normalized
    configuration. Megatron owns parameter groups, FP32 masters and checkpoints.
    """
    config = _init_megatron_optim_config(optim_config, optimizer_config_kwargs)
    if config.optimizer == _GRUG_MUONH_NAME:
        _register_grug_muonh(optim_config)
        return get_megatron_optimizer_native(config=config, model_chunks=model)

    config_overrides = get_standard_config_overrides(config)
    if config.optimizer == "adam":
        # SkyRL's AdamW recipe applies weight decay to every trainable parameter.
        config_overrides = {key: override for key, override in config_overrides.items() if "wd_mult" not in override}
    return get_megatron_optimizer_native(
        config=config,
        model_chunks=model,
        config_overrides=config_overrides,
    )


def get_megatron_optimizer_param_scheduler(
    optimizer,
    config,
    num_training_steps: int = 1e9,  # default to a large number for constant lr/wd
):
    """
    Get the optimizer parameter scheduler for Megatron.
    """
    # Each supported scheduler maps to Megatron's native lr_decay_style.
    scheduler = config.get("scheduler", "constant_with_warmup")
    lr_decay_styles = {
        "constant_with_warmup": "constant",
        "cosine": "cosine",
    }
    if scheduler not in lr_decay_styles:
        raise ValueError(f"Unsupported scheduler {scheduler!r} for Megatron; choose one of {sorted(lr_decay_styles)}")

    lr_warmup_steps = config.num_warmup_steps
    if config.get("lr_decay_steps", None) is None:
        lr_decay_steps = num_training_steps
    if config.get("lr_warmup_steps_ratio", None) is not None and (
        config.get("lr_warmup_steps", None) is None or config.lr_warmup_steps <= 0
    ):
        lr_warmup_steps = int(config.lr_warmup_steps_ratio * lr_decay_steps)

    opt_param_scheduler = _MegatronParamScheduler(
        optimizer,
        init_lr=config.get("lr_warmup_init", 0.0),
        max_lr=config.lr,
        min_lr=config.get("min_lr", 0.0),
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=lr_decay_styles[scheduler],
        start_wd=config.weight_decay,
        end_wd=config.weight_decay,
        wd_incr_steps=num_training_steps,
        wd_incr_style="constant",
        use_checkpoint_opt_param_scheduler=False,
        override_opt_param_scheduler=True,
        wsd_decay_steps=None,
        lr_wsd_decay_style="exponential",
    )

    return opt_param_scheduler


def get_megatron_last_lr(optimizer):
    """
    Get the last learning rate from the optimizer parameter scheduler.
    """
    return optimizer.param_groups[0]["lr"]
