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
Head-to-head speedrun: Hackable Transformer vs Grugformer (no sinks), ~125M params.

How to run:
  python marin/run/ray_run.py -- \
    python -m experiments.speedrun.grugformer_vs_hackable_125m.grugformer_vs_hackable_125m

By default this uses GPU resource presets. Set SR_USE_TPU=1 for TPU.
"""

# nodryrun

import os
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from haliax import Axis
from jaxtyping import PRNGKeyArray

from levanter.grug.model import GrugModelConfig
from levanter.models.grug_wrapper import GrugWrapper
from levanter.models.lm_model import LmConfig
from levanter.utils.flop_utils import lm_flops_per_token
from marin.execution.executor import executor_main
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

from experiments.llama import llama3_tokenizer_vocab_size
from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.hackable_transformer_starter.hackable_transformer_attn_sink import HackableTransformerConfig

AUTHOR = Author(
    name="David Hall",
    affiliation="Stanford University",
    url="https://github.com/dlwh",
)


def _resource_preset(*, use_tpu: bool) -> ResourceConfig:
    if use_tpu:
        return ResourceConfig.with_tpu("v5p-8")
    return ResourceConfig.with_gpu("A100-80G", count=1)


def _num_train_steps(*, param_count: int, batch_size: int, max_seq_len: int, tpp: int = 20) -> int:
    total_tokens = param_count * tpp
    return max(1, total_tokens // (batch_size * max_seq_len))


@LmConfig.register_subclass("grugformer_h2h_125m")
@dataclass(frozen=True)
class GrugformerH2HConfig(LmConfig[GrugWrapper]):
    """LmConfig wrapper around grug core hyperparameters (for head-to-head comparisons)."""

    max_seq_len: int = 2048

    hidden_dim: int = 512
    intermediate_dim: int = 1792
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: int = 8
    head_dim: int | None = None

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
        )
        return GrugWrapper.init(Vocab, grug_cfg, key=key)

    def flops_per_token(self, vocab_size: int, context_length: int) -> float | None:
        return lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            seq_len=context_length,
            vocab_size=vocab_size,
            glu=True,
        )

    def total_trainable_params(self, vocab_size: int) -> int:
        head_dim = self.head_dim or (self.hidden_dim // self.num_heads)
        token_embedding = vocab_size * self.hidden_dim
        attn = (
            self.hidden_dim * head_dim * self.num_heads
            + 2 * self.hidden_dim * head_dim * self.num_kv_heads
            + head_dim * self.num_heads * self.hidden_dim
        )
        mlp = 3 * self.hidden_dim * self.intermediate_dim
        transformer = self.num_layers * (attn + mlp + 2 * self.hidden_dim) + self.hidden_dim
        return int(transformer + token_embedding + token_embedding)


def _hackable_125m_config() -> HackableTransformerConfig:
    # Match the 130m preset dims from hackable transformer starter, but use 2048 context for parity with grug defaults.
    return HackableTransformerConfig(
        max_seq_len=2048,
        hidden_dim=512,
        intermediate_dim=1792,
        num_layers=6,
        num_heads=8,
        num_kv_heads=8,
        head_dim=None,
        use_attention_sink=False,
    )


def _grug_125m_config() -> GrugformerH2HConfig:
    return GrugformerH2HConfig(
        max_seq_len=2048,
        hidden_dim=512,
        intermediate_dim=1792,
        num_layers=6,
        num_heads=8,
        num_kv_heads=8,
        head_dim=None,
    )


def _train_config(
    *,
    use_tpu: bool,
    batch_size: int,
    max_seq_len: int,
    num_train_steps: int,
    explicit_mesh_axes: bool,
) -> SimpleTrainConfig:
    return SimpleTrainConfig(
        _resource_preset(use_tpu=use_tpu),
        train_seq_len=max_seq_len,
        train_batch_size=batch_size,
        num_train_steps=num_train_steps,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_hf_export=-1,
        explicit_mesh_axes=explicit_mesh_axes,
        profiler=True,
    )


def main() -> None:
    use_tpu = bool(int(os.environ.get("SR_USE_TPU", "0")))

    batch_size = 128
    max_seq_len = 2048

    hack_cfg = _hackable_125m_config()
    grug_cfg = _grug_125m_config()

    hack_params = int(hack_cfg.total_trainable_params(llama3_tokenizer_vocab_size))
    grug_params = int(grug_cfg.total_trainable_params(llama3_tokenizer_vocab_size))

    hack_steps = _num_train_steps(param_count=hack_params, batch_size=batch_size, max_seq_len=max_seq_len)
    grug_steps = _num_train_steps(param_count=grug_params, batch_size=batch_size, max_seq_len=max_seq_len)

    hack_train = _train_config(
        use_tpu=use_tpu,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_train_steps=hack_steps,
        explicit_mesh_axes=False,
    )
    grug_train = _train_config(
        use_tpu=use_tpu,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_train_steps=grug_steps,
        explicit_mesh_axes=use_tpu,
    )

    hack_speedrun = SpeedrunConfig(
        author=AUTHOR,
        description="Hackable Transformer (~125M) - standard attention (no sinks).",
        model_config=hack_cfg,
        train_config=hack_train,
    )
    grug_speedrun = SpeedrunConfig(
        author=AUTHOR,
        description="Grugformer (~125M) - TPU Splash Attention / reference fallback (no sinks).",
        model_config=grug_cfg,
        train_config=grug_train,
    )

    if hack_speedrun.vocab_size != llama3_tokenizer_vocab_size:
        raise AssertionError("Hackable speedrun vocab_size mismatch; expected llama3_tokenizer_vocab_size")
    if grug_speedrun.vocab_size != llama3_tokenizer_vocab_size:
        raise AssertionError("Grug speedrun vocab_size mismatch; expected llama3_tokenizer_vocab_size")

    steps = []
    steps.extend(default_speedrun(f"hackable_compare_125m_{max_seq_len}-profile4", hack_speedrun))
    steps.extend(default_speedrun(f"grug_compare_125m_{max_seq_len}-profile4", grug_speedrun))
    executor_main(steps=steps, description="Head-to-head: hackable transformer vs grugformer (~125M, no sinks)")


if __name__ == "__main__":
    main()
