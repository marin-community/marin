# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Speedruns using the PrismH optimizer for various Qwen model sizes (Chinchilla optimal steps)
configs mirroring those in marin/experiments/speedrun/muonh_llama_scaling/muonh_sweep.py
"""

import dataclasses
import logging
import os
import jax
import jax.numpy as jnp
import optax
import haliax
from dataclasses import dataclass
from typing import NamedTuple
from optax import tree_utils as otu
from haliax.nn import Linear
from levanter.optim.util import map_flattened_linear_layers
from levanter.utils.jax_utils import leaf_key_paths
from levanter.optim.adamh import scale_by_adamh
from levanter.optim.config import OptimizerConfig
from levanter.models.qwen import Qwen3Config
from levanter.models.llama import LlamaConfig

from experiments.llama import llama_1_4b, llama_150m, llama_300m, llama_600m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from fray.cluster import ResourceConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

AUTHOR = Author(
    name="Franz Louis Cesista",
    affiliation="",
    url="https://leloykun.github.io",
)

logger = logging.getLogger("ray")


def get_num_train_steps(param_count, batch_size, seq_len):
    """Compute the number of steps for Chinchilla optimal training (20x params tokens)."""
    total_tokens = param_count * 20
    tokens_per_step = batch_size * seq_len
    return total_tokens // tokens_per_step


def _to_qwen3_from_llama(llama_cfg: LlamaConfig, *, seq_len_override=None) -> Qwen3Config:
    """
    Build a Qwen3Config with identical sizes to a given LLaMA config.
    """
    qwen = Qwen3Config(
        max_seq_len=seq_len_override if seq_len_override is not None else llama_cfg.max_seq_len,
        hidden_dim=llama_cfg.hidden_dim,
        intermediate_dim=llama_cfg.intermediate_dim,
        num_layers=llama_cfg.num_layers,
        num_heads=llama_cfg.num_heads,
        num_kv_heads=llama_cfg.num_kv_heads,
        head_dim=getattr(llama_cfg, "head_dim", None),
        use_bias=getattr(llama_cfg, "use_bias", False),
        rope=llama_cfg.rope,
        activation_function=llama_cfg.activation_function,
        initializer_range=llama_cfg.initializer_range,
        layer_norm_epsilon=llama_cfg.layer_norm_epsilon,
        tie_word_embeddings=llama_cfg.tie_word_embeddings,
        upcast_attn=llama_cfg.upcast_attn,
        attn_backend=llama_cfg.attn_backend,
        flash_attention_block_size=llama_cfg.flash_attention_block_size,
        scan_layers=getattr(llama_cfg, "scan_layers", False),
        gradient_checkpointing=getattr(llama_cfg, "gradient_checkpointing", False),
        hybrid_norm=True,
    )
    return qwen


@OptimizerConfig.register_subclass("PrismH")
@dataclass(frozen=True)
class PrismHConfig(OptimizerConfig):
    """
    This is a variant of the Muon optimizer configuration: Momentum Orthogonalized by Newton-Schulz (https://github.com/KellerJordan/modded-nanogpt).

    We ensure that the linear weights stay exactly constant norm as initialization by applying the following update rule:

    p_new_intermediate = p - learning_rate * u * norm(p) / norm(u)
    p_new = p_new_intermediate / norm(p_new_intermediate) * norm(p)

    where p is the parameter, u is the update and norm is the Frobenius norm of a matrix.
    """

    adam_lr: float = 6e-4  # Adam LR
    momentum: float = 0.95
    nesterov: bool = True
    backend_steps: int = 5  # Number of steps for Newton-Schulz orthogonalization
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    muon_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    gamma_L: float = 0.0
    gamma_R: float = 0.0
    bidirectional: bool = False

    def build(self, num_train_steps):
        """
        Creates the optimizer.
        """
        if self.gamma_R <= 0.0:
            raise ValueError(f"PrismHConfig.gamma_R must be > 0 for PrismH; got {self.gamma_R}.")
        if self.bidirectional and self.gamma_L <= 0.0:
            raise ValueError(f"PrismHConfig.gamma_L must be > 0 for bidirectional PrismH; got {self.gamma_L}.")
        learning_rate_schedule = self.lr_scheduler(num_train_steps)
        adam_lr_schedule = self.lr_scheduler(num_train_steps, override_lr=self.adam_lr)

        def optimizer(learning_rate, adam_lr):
            def prismh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(
                    scale_with_prismh(
                        self.momentum,
                        self.nesterov,
                        self.backend_steps,
                        self.muon_epsilon,
                        learning_rate,
                        gamma_L=self.gamma_L,
                        gamma_R=self.gamma_R,
                        bidirectional=self.bidirectional,
                    )
                )
                optimizer = optax.chain(*components)
                return optimizer

            def adamh_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_adamh(self.beta1, self.beta2, self.epsilon, learning_rate))
                optimizer = optax.chain(*components)
                return optimizer

            def adam_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                components.append(optax.scale(-adam_lr))
                optimizer = optax.chain(*components)
                return optimizer

            transformations = {
                "prismh": prismh_transform(),
                "adamh": adamh_transform(),
                "adam": adam_transform(),
            }

            return optax.multi_transform(transformations, self.create_mask)

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule, adam_lr=adam_lr_schedule)

    def create_mask(self, params):
        """
        Creates a mask that labels parameters as 'prismh' 'adamh' 'adam' based on their
        dimensionality and module path, using Adam for Embedding and vector parameters.
        using AdamH for lm_head parameters.
        using PrismH for other parameters.
        """
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str:
                return "adam"
            elif "lm_head" in path_str:
                return "adamh"
            elif isinstance(param, Linear):
                # prismh for linear layers
                return dataclasses.replace(param, weight="prismh", bias="adam" if param.bias is not None else None)
            else:
                return "adam"

        return haliax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, Linear))


class ScaleByPrismHState(NamedTuple):
    """State for the Prism algorithm."""

    momentum_buffer: optax.Updates


def scale_with_prismh(
    momentum=0.95,
    nesterov=True,
    steps=5,
    muon_eps=1e-8,
    learning_rate=0.02,
    coefficient_scale=1.001,
    gamma_L=0.0,
    gamma_R=0.0,
    bidirectional=False,
):
    # Convert steps to concrete int at function definition time
    steps = int(steps)

    def init_fn(params):
        momentum_buffer = otu.tree_zeros_like(params)
        return ScaleByPrismHState(momentum_buffer=momentum_buffer)

    def update_fn(updates, state, params=None):
        buf = state.momentum_buffer
        buf = jax.tree.map(
            lambda m, g: None if g is None else momentum * m + g,
            buf,
            updates,
            is_leaf=lambda x: x is None,
        )
        if nesterov:
            updates_with_momentum = jax.tree.map(
                lambda m, g: None if g is None else momentum * m + g,
                buf,
                updates,
                is_leaf=lambda x: x is None,
            )
        else:
            updates_with_momentum = buf

        def transform_linear_layer(layer: haliax.nn.Linear, layer_orig: haliax.nn.Linear):
            assert layer.weight.ndim == 2, "PrismH is only designed for linear layers"
            # steps is now a concrete int
            M = layer.weight.array
            G = layer_orig.weight.array
            D = G - M
            if bidirectional:
                H_L = M @ M.mT + gamma_L**2 * D @ D.mT + muon_eps * jnp.eye(M.shape[0], dtype=M.dtype)
                H_R = M.mT @ M + gamma_R**2 * D.mT @ D + muon_eps * jnp.eye(M.shape[1], dtype=M.dtype)
                updated_weight_array = double_sided_matmul_invroot(
                    H_L,
                    M,
                    H_R,
                    r=4,
                    steps=steps,
                    eps=muon_eps,
                    scale=coefficient_scale,
                )
            else:
                H_R = M.mT @ M + gamma_R**2 * D.mT @ D + muon_eps * jnp.eye(M.shape[1], dtype=M.dtype)
                updated_weight_array = matmul_invroot(M, H_R, r=2, steps=steps, eps=muon_eps, scale=coefficient_scale)

            updated_weight = dataclasses.replace(layer.weight, array=updated_weight_array)

            return dataclasses.replace(layer, weight=updated_weight)  # type: ignore

        muon_updates = map_flattened_linear_layers(transform_linear_layer, updates_with_momentum, updates)

        # projected training for linear weight
        def scale_invariant_update(p, u):
            if p is None:
                return None
            if p.ndim == 2:
                # this is the case for no layer stacking
                new_p = p - learning_rate * u * jnp.linalg.norm(p) / jnp.maximum(jnp.linalg.norm(u), 1e-10)
                return new_p / jnp.linalg.norm(new_p) * jnp.linalg.norm(p) - p
            else:
                axes = tuple(range(1, p.ndim))
                p_norm = jnp.sqrt(jnp.sum(jnp.square(p), axis=axes, keepdims=True))
                u_norm = jnp.sqrt(jnp.sum(jnp.square(u), axis=axes, keepdims=True))
                new_p = p - learning_rate * u * p_norm / jnp.maximum(u_norm, 1e-10)
                new_p_norm = jnp.sqrt(jnp.sum(jnp.square(new_p), axis=axes, keepdims=True))
                return new_p / jnp.maximum(new_p_norm, 1e-10) * p_norm - p

        muonh_updates = jax.tree_util.tree_map(
            scale_invariant_update,
            params,
            muon_updates,
            is_leaf=lambda x: x is None,
        )

        return muonh_updates, ScaleByPrismHState(momentum_buffer=buf)

    return optax.GradientTransformation(init_fn, update_fn)


coefs = [
    None,  # r = 0
    None,  # r = 1, omitted
    [  # r = 2
        (7.42487, -18.3958, 12.8967),
        (3.48773, -2.33004, 0.440469),
        (2.77661, -2.07064, 0.463023),
        (1.99131, -1.37394, 0.387593),
        (15 / 8, -5 / 4, 3 / 8),
    ],
    None,  # r = 3, omitted
    [  # r = 4
        (3.85003, -10.8539, 8.61893),
        (1.80992, -0.587778, 0.0647852),
        (1.50394, -0.594516, 0.121161),
        (45 / 32, -9 / 16, 5 / 32),
    ],
]


def abc(r=1, steps=None, scale=1.0):
    w, steps = coefs[r], steps or len(coefs[r])
    for a, b, c in w[:steps] + w[-1:] * max(steps - len(w), 0):
        yield a / scale, b / scale ** (r + 1), c / scale ** (2 * r + 1)


def _sym(M: jax.Array) -> jax.Array:
    return 0.5 * (M + M.mT)


def matmul_invroot(G: jax.Array, P: jax.Array, r: int, s=1, steps=None, eps=1e-5, scale: float = 1.001):
    # Computes G @ P^(-s/r)
    I_n = jnp.eye(P.shape[0], dtype=P.dtype)
    P = P / (t := (P * P.mT).sum() ** 0.5) + eps * I_n
    for a, b, c in abc(r, steps, scale=scale):
        W = a * I_n + b * P + c * P @ P
        W1, W2 = jnp.linalg.matrix_power(W, s), jnp.linalg.matrix_power(W, r)
        G, P = G @ W1, _sym(P @ W2)
    return G * t ** (-s / r)


def double_sided_matmul_invroot(
    Q: jax.Array,
    G: jax.Array,
    P: jax.Array,
    *,
    r: int,
    s=1,
    steps=None,
    eps: float = 1e-5,
    scale: float = 1.001,
):
    # Computes Q^(-s/r) @ G @ P^(-s/r)
    I_m, I_n = jnp.eye(G.shape[0], dtype=Q.dtype), jnp.eye(G.shape[1], dtype=P.dtype)
    Q = Q / (tQ := jnp.sum(Q * Q.mT) ** 0.5) + eps * I_m
    P = P / (tP := jnp.sum(P * P.mT) ** 0.5) + eps * I_n
    for a, b, c in abc(r, steps, scale=scale):
        WQ = a * I_m + b * Q + c * Q @ Q
        WP = a * I_n + b * P + c * P @ P
        WQ1, WQ2 = jnp.linalg.matrix_power(WQ, s), jnp.linalg.matrix_power(WQ, r)
        WP1, WP2 = jnp.linalg.matrix_power(WP, s), jnp.linalg.matrix_power(WP, r)
        Q, G, P = _sym(Q @ WQ2), WQ1 @ G @ WP1, _sym(P @ WP2)
    G = G * tQ ** (-s / r) * tP ** (-s / r)
    return G


def build_config(size: str, gamma_L, gamma_R, bidirectional) -> tuple[str, SpeedrunConfig]:
    param_counts = {
        "130m": 130_000_000,
        "300m": 300_000_000,
        "520m": 520_000_000,
        "1_2b": 1_200_000_000,
    }

    llama_model_cfgs = {
        "130m": llama_150m,
        "300m": llama_300m,
        "520m": llama_600m,
        "1_2b": llama_1_4b,
    }

    batch_sizes = {
        "130m": 128,
        "300m": 128,
        "520m": 256,
        "1_2b": 256,
    }

    resource_cfgs = {
        "130m": ResourceConfig.with_tpu("v5litepod-64"),
        "300m": ResourceConfig.with_tpu("v5litepod-64"),
        "520m": ResourceConfig.with_tpu("v5litepod-64"),
        "1_2b": ResourceConfig.with_tpu("v5litepod-64"),
    }

    # Optimizer configs for each size
    muon_configs = {
        "130m": PrismHConfig(
            learning_rate=0.02,
            adam_lr=0.008,
            min_lr_ratio=0,
            momentum=0.95,
            beta1=0.9,
            beta2=0.98,
            epsilon=1e-15,
            muon_epsilon=1e-5,
            max_grad_norm=1,
            warmup=1000,
            gamma_L=gamma_L,
            gamma_R=gamma_R,
            bidirectional=bidirectional,
        ),
        "300m": PrismHConfig(
            learning_rate=0.01,
            adam_lr=0.002,
            min_lr_ratio=0,
            momentum=0.98,
            beta1=0.9,
            beta2=0.98,
            epsilon=1e-15,
            muon_epsilon=1e-5,
            max_grad_norm=1,
            warmup=1000,
            gamma_L=gamma_L,
            gamma_R=gamma_R,
            bidirectional=bidirectional,
        ),
        "520m": PrismHConfig(
            learning_rate=0.01,
            adam_lr=0.002,
            min_lr_ratio=0,
            momentum=0.98,
            beta1=0.9,
            beta2=0.98,
            epsilon=1e-15,
            muon_epsilon=1e-5,
            max_grad_norm=1,
            warmup=1000,
            gamma_L=gamma_L,
            gamma_R=gamma_R,
            bidirectional=bidirectional,
        ),
        "1_2b": PrismHConfig(
            learning_rate=0.01,
            adam_lr=0.0015,
            min_lr_ratio=0,
            momentum=0.98,
            beta1=0.9,
            beta2=0.98,
            epsilon=1e-15,
            muon_epsilon=1e-5,
            max_grad_norm=2,
            warmup=1000,
            gamma_L=gamma_L,
            gamma_R=gamma_R,
            bidirectional=bidirectional,
        ),
    }

    descriptions = {
        "130m": "Qwen3 ~130M (LLaMA-geometry-matched) with PrismH.",
        "300m": "Qwen3 ~300M (LLaMA-geometry-matched) with PrismH.",
        "520m": "Qwen3 ~520M (LLaMA-geometry-matched) with PrismH.",
        "1_2b": "Qwen3 ~1.2B (LLaMA-geometry-matched) with PrismH.",
    }

    run_names = {
        "130m": "qwen3_130m_muonh_4096_lr_0.02_adam_lr_0.008",
        "300m": "qwen3_300m_muonh_4096_lr_0.01",
        "520m": "qwen3_520m_muonh_4096_lr_0.01",
        "1_2b": "qwen3_1_2b_muonh_4096_low_lr",
    }

    if size not in param_counts:
        raise ValueError(f"Unknown size: {size}")

    llama_cfg = llama_model_cfgs[size]
    batch_size = batch_sizes[size]
    resource_config = resource_cfgs[size]
    muon = muon_configs[size]
    description = descriptions[size]
    run_name = run_names[size]

    # Convert to Qwen3Config and set seq_len=4096 for the sweep
    model_config = _to_qwen3_from_llama(llama_cfg, seq_len_override=4096)
    seq_len = model_config.max_seq_len

    num_train_steps = get_num_train_steps(param_counts[size], batch_size, seq_len)

    train = SimpleTrainConfig(
        resource_config,
        train_batch_size=batch_size,
        num_train_steps=num_train_steps,
        learning_rate=muon.learning_rate,
        optimizer_config=muon,
    )

    cfg = SpeedrunConfig(
        author=AUTHOR,
        description=description,
        model_config=model_config,
        train_config=train,
    )
    return run_name, cfg


def main():
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment, needs HF access.")
        return

    runs = [build_config("130m", 0.0, gamma_r, False) for gamma_r in [0.1, 1.0, 2.0]] + [
        build_config("130m", gamma, gamma, True) for gamma in [0.1, 1.0, 2.0]
    ]

    steps = []
    for name, cfg in runs:
        cfg.print_run_info()
        steps.extend(default_speedrun(name, cfg))

    executor_main(steps=steps, description="Qwen3 PrismH speedruns (Chinchilla-optimal tokens, w/ QK-Norm)")


if __name__ == "__main__":
    main()
