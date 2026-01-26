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
GPT-OSS-style Grug speedrun (FLOP-matched to the grugformer starter).

This mirrors experiments/speedrun/grugformer_starter/grugformer_speedrun.py but
enables GPT-OSS-ish knobs in the Grug core (clipped gated activation and
attention sinks by default).

How to run:
  python marin/run/ray_run.py -- \
    python -m experiments.speedrun.gpt_oss_starter.gpt_oss_speedrun
"""

# nodryrun

import dataclasses
import logging
import os
from dataclasses import dataclass, replace as dataclass_replace

from fray.cluster import ResourceConfig
from haliax import Axis
from jaxtyping import PRNGKeyArray

from levanter.grug.model import GrugModelConfig
from levanter.models.grug_wrapper import GrugWrapper
from levanter.models.lm_model import LmConfig
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.mesh import MeshConfig
from marin.execution.executor import executor_main
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun
from marin.training.training import TrainLmOnPodConfig

from experiments.llama import llama3_tokenizer_vocab_size
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger("ray")


def _get_num_train_steps(param_count: int, batch_size: int, max_seq_len: int, tpp: int = 20) -> int:
    total_tokens = param_count * tpp
    return max(1, total_tokens // (batch_size * max_seq_len))


def _size_presets() -> dict[str, "GptOssGrugConfig"]:
    base = dict(max_seq_len=2048, head_dim=None)
    return {
        "130m": GptOssGrugConfig(
            hidden_dim=512, intermediate_dim=1792, num_layers=6, num_heads=8, num_kv_heads=8, **base
        ),
        "300m": GptOssGrugConfig(
            hidden_dim=768, intermediate_dim=2688, num_layers=12, num_heads=12, num_kv_heads=12, **base
        ),
        "520m": GptOssGrugConfig(
            hidden_dim=1024, intermediate_dim=3584, num_layers=24, num_heads=16, num_kv_heads=16, **base
        ),
        "1_2b": GptOssGrugConfig(
            hidden_dim=2048, intermediate_dim=7168, num_layers=16, num_heads=16, num_kv_heads=16, **base
        ),
    }


def _resource_presets(use_tpu: bool = False):
    if use_tpu:
        return {
            "130m": ResourceConfig.with_tpu("v5p-8"),
            "300m": ResourceConfig.with_tpu("v5p-8"),
            "520m": ResourceConfig.with_tpu("v5p-8"),
            "1_2b": ResourceConfig.with_tpu("v5p-8"),
        }
    return {
        "130m": ResourceConfig.with_gpu("A100-80G", count=1),
        "300m": ResourceConfig.with_gpu("A100-80G", count=1),
        "520m": ResourceConfig.with_gpu("A100-80G", count=2),
        "1_2b": ResourceConfig.with_gpu("A100-80G", count=4),
    }


def _batch_sizes() -> dict[str, int]:
    return {"130m": 128, "300m": 128, "520m": 128, "1_2b": 256}


def _apply_expert_mesh(step, *, experts_axis_size: int = 8):
    if not hasattr(step, "config") or not isinstance(step.config, TrainLmOnPodConfig):
        return step
    trainer = step.config.train_config.trainer
    base_mesh = trainer.mesh or MeshConfig()
    expert_mesh = dataclasses.replace(
        base_mesh,
        axes={"data": 1, "model": 1, "experts": experts_axis_size, "replica": 1},
        dcn_axes={"replica_dcn": -1},
    )
    new_trainer = dataclasses.replace(trainer, mesh=expert_mesh, use_explicit_mesh_axes=True)
    new_train_cfg = dataclasses.replace(step.config.train_config, trainer=new_trainer)
    new_config = dataclasses.replace(step.config, train_config=new_train_cfg)
    return dataclasses.replace(step, config=new_config)


def _speedrun_with_expert_mesh(name: str, cfg: SpeedrunConfig, *, experts_axis_size: int = 8):
    steps = list(default_speedrun(name, cfg))
    if steps:
        steps[0] = _apply_expert_mesh(steps[0], experts_axis_size=experts_axis_size)
    return steps


@LmConfig.register_subclass("gpt_oss_grug")
@dataclass(frozen=True)
class GptOssGrugConfig(LmConfig[GrugWrapper]):
    """LmConfig wrapper around GPT-OSS-style Grug hyperparameters."""

    # LmConfig field
    max_seq_len: int = 2048

    # Grug core hyperparams (FLOP-matched to grugformer starter sizes)
    hidden_dim: int = 1024
    intermediate_dim: int = 2752
    num_layers: int = 12
    num_heads: int = 16
    num_kv_heads: int = 16
    head_dim: int | None = None

    # GPT-OSS-ish toggles
    use_clipped_gated_activation: bool = True
    clipped_gated_alpha: float = 1.702
    use_attention_sinks: bool = True

    # MoE support (top-1 routing keeps per-token FLOPs close to dense)
    use_moe: bool = True
    num_experts: int = 8
    num_experts_per_tok: int = 1
    moe_load_balancing_coef: float | None = 0.01
    moe_router_z_loss_coef: float | None = 0.001

    # ---- LmConfig API ----
    @property
    def model_type(self) -> type[GrugWrapper]:
        return GrugWrapper

    @property
    def Embed(self) -> Axis:
        # Not used by GrugWrapper (it returns logits directly), but LmConfig requires it.
        return Axis("embed", self.hidden_dim)

    def build(self, Vocab: Axis, *, key: PRNGKeyArray) -> GrugWrapper:
        grug_cfg = GrugModelConfig(
            vocab_size=Vocab.size,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            use_clipped_gated_activation=self.use_clipped_gated_activation,
            clipped_gated_alpha=self.clipped_gated_alpha,
            use_attention_sinks=self.use_attention_sinks,
            use_moe=self.use_moe,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            moe_load_balancing_coef=self.moe_load_balancing_coef,
            moe_router_z_loss_coef=self.moe_router_z_loss_coef,
        )
        return GrugWrapper.init(Vocab, grug_cfg, key=key)

    def flops_per_token(self, vocab_size: int, context_length: int) -> float | None:
        num_experts = self.num_experts if self.use_moe else 1
        num_experts_per_tok = self.num_experts_per_tok if self.use_moe else 1
        return lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            seq_len=context_length,
            vocab_size=vocab_size,
            glu=True,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
        )

    def total_trainable_params(self, vocab_size: int) -> int:
        head_dim = self.head_dim or (self.hidden_dim // self.num_heads)
        token_embedding = vocab_size * self.hidden_dim
        attn = (
            self.hidden_dim * head_dim * self.num_heads
            + 2 * self.hidden_dim * head_dim * self.num_kv_heads
            + head_dim * self.num_heads * self.hidden_dim
        )
        if self.use_moe:
            router = self.hidden_dim * self.num_experts
            expert = 3 * self.hidden_dim * self.intermediate_dim
            mlp = self.num_experts * expert + router
        else:
            mlp = 3 * self.hidden_dim * self.intermediate_dim
        transformer = self.num_layers * (attn + mlp + 2 * self.hidden_dim) + self.hidden_dim
        attention_sinks = self.num_layers * (self.num_heads if self.use_attention_sinks else 0)
        return int(transformer + 2 * token_embedding + attention_sinks)

    def total_active_params(self, vocab_size: int) -> int:
        """Estimate trainable parameters used per token (MoE counts only active experts)."""
        head_dim = self.head_dim or (self.hidden_dim // self.num_heads)
        token_embedding = vocab_size * self.hidden_dim
        attn = (
            self.hidden_dim * head_dim * self.num_heads
            + 2 * self.hidden_dim * head_dim * self.num_kv_heads
            + head_dim * self.num_heads * self.hidden_dim
        )
        if self.use_moe:
            router = self.hidden_dim * self.num_experts
            expert = 3 * self.hidden_dim * self.intermediate_dim
            mlp = self.num_experts_per_tok * expert + router
        else:
            mlp = 3 * self.hidden_dim * self.intermediate_dim
        transformer = self.num_layers * (attn + mlp + 2 * self.hidden_dim) + self.hidden_dim
        attention_sinks = self.num_layers * (self.num_heads if self.use_attention_sinks else 0)
        return int(transformer + 2 * token_embedding + attention_sinks)


def build_run(size: str, *, use_tpu: bool = False) -> tuple[str, SpeedrunConfig]:
    sizes = _size_presets()
    if size not in sizes:
        raise ValueError(f"Unknown size: {size}")
    model_cfg = sizes[size]

    batch = _batch_sizes()[size]
    max_seq_len = model_cfg.max_seq_len
    base_params = int(model_cfg.total_active_params(llama3_tokenizer_vocab_size))
    steps = _get_num_train_steps(base_params, batch, max_seq_len, tpp=20)
    resources = _resource_presets(use_tpu=use_tpu)[size]

    train = SimpleTrainConfig(
        resources,
        train_seq_len=max_seq_len,
        train_batch_size=batch,
        num_train_steps=steps,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=500,
        steps_per_hf_export=-1,
        explicit_mesh_axes=True,
    )

    run_name = f"gpt_oss_grug_{size}"
    desc = f"GPT-OSS-style Grug MoE (clipped gated activation + attention sinks) ({size})."
    cfg = SpeedrunConfig(
        author=Author(
            name="__YOUR_NAME__",
            affiliation="__YOUR_AFFILIATION__",
            url="__YOUR_URL__",
        ),
        description=desc,
        model_config=model_cfg,
        train_config=train,
    )
    return run_name, cfg


def main() -> None:
    sizes = ["130m", "300m", "520m", "1_2b"]
    use_tpu = bool(int(os.environ.get("SR_USE_TPU", "0")))

    steps = []
    for s in sizes:
        name, cfg = build_run(s, use_tpu=use_tpu)
        if cfg.vocab_size != llama3_tokenizer_vocab_size:
            raise AssertionError("Speedrun vocab_size mismatch; expected llama3_tokenizer_vocab_size")
        cfg.print_run_info()
        steps.extend(_speedrun_with_expert_mesh(name, cfg, experts_axis_size=4))

    executor_main(steps=steps, description="GPT-OSS-style Grug (clipped gated activation + attention sinks).")


if __name__ == "__main__":
    main()
