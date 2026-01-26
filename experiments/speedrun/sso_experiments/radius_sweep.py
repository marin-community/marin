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
Sweep over radius_scaler (R) values for SSO and MuonSphere optimizers on 130m model with 1x Chinchilla
Testing radius_scaler values: 0.5, 1.0, 2.0, 4.0
Testing LR multipliers: 0.5, 1.0, 2.0
Total: 4 radius_scalers × 3 LR_multipliers × 2 optimizers = 24 runs
"""

import logging
import os
import dataclasses

from levanter.models.qwen import Qwen3Config
from levanter.models.llama import LlamaConfig
from levanter.optim import SSOConfig, MuonSphereConfig

from experiments.llama import llama_150m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from fray.cluster import ResourceConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

AUTHOR = Author(
    name="Kaiyue Wen",
    affiliation="",
    url="https://whenwen.github.io",
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


def build_config(
    radius_scaler: float, lr_multiplier: float, use_sso: bool
) -> tuple[str, SpeedrunConfig]:
    size = "130m"
    param_count = 130_000_000
    batch_size = 128
    seq_len = 4096

    llama_cfg = llama_150m
    resource_config = ResourceConfig.with_tpu("v5litepod-64")

    # Base learning rates for 130m (matching SSO paper defaults)
    base_learning_rate = 0.01
    base_adam_lr = 0.005

    # Create SSO or MuonSphere config
    if use_sso:
        optimizer_config = SSOConfig(
            learning_rate=base_learning_rate * lr_multiplier,
            adam_lr=base_adam_lr * lr_multiplier,
            min_lr_ratio=0,
            momentum=0.9,
            nesterov=True,
            msign_steps=8,
            solver_tol=1e-8,
            solver_max_iter=20,
            power_iter_steps=20,
            radius_scaler=radius_scaler,
            eps=1e-12,
            beta1=0.9,
            beta2=0.95,
            epsilon=1e-8,
            max_grad_norm=1.0,
            warmup=1000,
        )
    else:
        optimizer_config = MuonSphereConfig(
            learning_rate=base_learning_rate * lr_multiplier,
            adam_lr=base_adam_lr * lr_multiplier,
            min_lr_ratio=0,
            momentum=0.9,
            nesterov=True,
            msign_steps=8,
            power_iter_steps=20,
            radius_scaler=radius_scaler,
            eps=1e-12,
            beta1=0.9,
            beta2=0.95,
            epsilon=1e-8,
            max_grad_norm=1.0,
            warmup=1000,
        )

    # Convert to Qwen3Config and set seq_len=4096 for the sweep
    model_config = _to_qwen3_from_llama(llama_cfg, seq_len_override=seq_len)

    num_train_steps = get_num_train_steps(param_count, batch_size, seq_len)

    optimizer_name = "sso" if use_sso else "muon_sphere"
    run_name = f"qwen3_130m_{optimizer_name}_4096_1x_radius_{radius_scaler:.2f}_lr_{lr_multiplier:.2f}x"
    description = f"Qwen3 ~130M with {'SSO' if use_sso else 'MuonSphere'}, 1x Chinchilla, radius_scaler={radius_scaler}, LR={lr_multiplier}x"

    train = SimpleTrainConfig(
        resource_config,
        train_batch_size=batch_size,
        num_train_steps=num_train_steps,
        learning_rate=optimizer_config.learning_rate,
        optimizer_config=optimizer_config,
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

    # Radius scalers to test (R = radius_scaler * sqrt(d_out / d_in))
    radius_scalers = [0.5, 1.0, 2.0, 4.0]

    # Learning rate multipliers to test
    lr_multipliers = [0.5, 1.0, 2.0]

    # Test both SSO (True) and MuonSphere (False)
    use_sso_options = [True, False]

    runs = []
    for radius_scaler in radius_scalers:
        for lr_mult in lr_multipliers:
            for use_sso in use_sso_options:
                runs.append(build_config(radius_scaler, lr_mult, use_sso))

    steps = []
    for name, cfg in runs:
        cfg.print_run_info()
        steps.extend(default_speedrun(name, cfg))

    executor_main(steps=steps, description="Qwen3 SSO/MuonSphere radius_scaler sweep (130m, 1x Chinchilla)")


if __name__ == "__main__":
    main()
