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
Grugformer starter speedrun (ejkernel blocksparse attention).

This uses the grug core implementation (`levanter.grug`) and wires it into the existing
Marin speedrun harness via the `levanter.models.grug_wrapper.GrugWrapper` adapter.

How to run:
  python marin/run/ray_run.py -- \
    python -m experiments.speedrun.grugformer_starter.grugformer_speedrun
"""

# nodryrun

import logging
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from haliax import Axis
from jaxtyping import PRNGKeyArray

from levanter.grug.config import GrugModelConfig
from levanter.models.grug_wrapper import GrugWrapper
from levanter.models.lm_model import LmConfig
from levanter.utils.flop_utils import lm_flops_per_token
from marin.execution.executor import executor_main
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

from experiments.llama import llama3_tokenizer_vocab_size
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger("ray")


@LmConfig.register_subclass("grugformer")
@dataclass(frozen=True)
class GrugformerConfig(LmConfig[GrugWrapper]):
    """LmConfig wrapper around grug core hyperparameters."""

    # LmConfig field
    max_seq_len: int = 2048

    # Grug core hyperparams
    hidden_dim: int = 1024
    intermediate_dim: int = 2752
    num_layers: int = 12
    num_heads: int = 16
    num_kv_heads: int = 16
    head_dim: int | None = None

    tie_embeddings: bool = False

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
            tie_embeddings=self.tie_embeddings,
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
        head = 0 if self.tie_embeddings else token_embedding
        return int(transformer + token_embedding + head)


speedrun_config = SpeedrunConfig(
    author=Author(
        name="__YOUR_NAME__",
        affiliation="__YOUR_AFFILIATION__",
        url="__YOUR_URL__",
    ),
    description="Grugformer starter (ejkernel blocksparse attention).",
    model_config=GrugformerConfig(
        max_seq_len=2048,
        hidden_dim=1024,
        intermediate_dim=2752,
        num_layers=12,
        num_heads=16,
        num_kv_heads=16,
        head_dim=None,
        tie_embeddings=False,
    ),
    train_config=SimpleTrainConfig(
        ResourceConfig.with_gpu("A100", count=1),
        train_batch_size=32,
        num_train_steps=100,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=500,
    ),
)

speedrun_config.print_run_info()


def main() -> None:
    # Ensure model vocab matches the default Llama-3 tokenizer used by speedrun harness.
    if speedrun_config.vocab_size != llama3_tokenizer_vocab_size:
        raise AssertionError("Speedrun vocab_size mismatch; expected llama3_tokenizer_vocab_size")

    executor_main(steps=default_speedrun("grugformer_starter", speedrun_config))


if __name__ == "__main__":
    main()
