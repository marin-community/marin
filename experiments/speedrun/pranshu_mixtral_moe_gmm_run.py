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

# nodryrun

import dataclasses
import logging

import haliax.nn as hnn
import jax.random as jrandom
from haliax.jax_utils import maybe_rng_split, shaped_rng_split
from haliax.nn.scan import BlockSeq, Stacked

from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from levanter.layers.attention import Attention
from levanter.models.llama import LlamaEmbedding, LlamaMlp
from levanter.models.mixtral import (
    MixtralConfig,
    MixtralDecoderLayer,
    MixtralLMHeadModel,
    MixtralMoEMlp,
    MixtralSparseMoeBlock,
    MixtralTransformer,
)
from levanter.utils.activation import ActivationFunctionEnum
from marin.execution.executor import executor_main
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")


# Custom MoE classes that use GMM
class MixtralMoEMlpGMM(MixtralMoEMlp):
    """MoE MLP that uses GMM instead of ragged dot"""

    @staticmethod
    def init(
        Experts,
        Embed,
        Mlp,
        activation_fn,
        *,
        key,
        use_bias=False,
        use_gmm=True,  # Force GMM
    ):
        """Initialize MoE MLP with GMM enabled"""
        k1, k2, k3 = jrandom.split(key, 3)
        # Initialize MoELinear layers with use_gmm=True
        w1 = hnn.MoELinear.init(Experts=Experts, Out=Mlp, In=Embed, key=k1, use_bias=use_bias, use_gmm=True)
        w2 = hnn.MoELinear.init(Experts=Experts, Out=Embed, In=Mlp, key=k2, use_bias=use_bias, use_gmm=True)
        w3 = hnn.MoELinear.init(Experts=Experts, Out=Mlp, In=Embed, key=k3, use_bias=use_bias, use_gmm=True)

        if isinstance(activation_fn, ActivationFunctionEnum):
            activation_fn = activation_fn.to_fn()

        return MixtralMoEMlpGMM(w1, w2, w3, Embed, Mlp, activation_fn)


class MixtralSparseMoeBlockGMM(MixtralSparseMoeBlock):
    """Sparse MoE block that uses GMM"""

    @staticmethod
    def init(config, *, key):
        """Initialize MoE block with GMM-enabled MLP"""
        k_gate, k_experts = maybe_rng_split(key, 2)

        gate = hnn.Linear.init(config.Embed, config.Experts, key=k_gate, use_bias=config.use_bias)
        experts = MixtralMoEMlpGMM.init(
            Experts=config.Experts,
            Embed=config.Embed,
            Mlp=config.Mlp,
            activation_fn=config.activation_function,
            key=k_experts,
            use_bias=config.use_bias,
            use_gmm=True,  # Force GMM
        )

        return MixtralSparseMoeBlockGMM(config, gate, experts)


# Custom decoder layer that uses GMM-enabled MoE
class MixtralDecoderLayerGMM(MixtralDecoderLayer):
    """Decoder layer that uses GMM-enabled MoE blocks"""

    @staticmethod
    def init(config, *, key):
        """Initialize decoder layer with GMM-enabled MoE"""
        k_attn, k_moe, k_mlp = jrandom.split(key, 3)

        # Use the standard attention implementation
        attn_config = config.attention_config()
        attn = Attention.init(attn_config, key=k_attn)

        # Use our GMM-enabled MoE block
        block_sparse_moe = MixtralSparseMoeBlockGMM.init(config, key=k_moe)

        # Standard layer norms
        ln_1 = hnn.RmsNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)
        ln_2 = hnn.RmsNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)

        # Shared MLP if configured
        shared_mlp = None
        if config.n_shared_experts > 0:
            shared_mlp = LlamaMlp.init(
                config.Embed,
                config.Mlp,
                config.activation_function,
                key=k_mlp,
                use_bias=config.use_bias,
            )

        return MixtralDecoderLayerGMM(config, attn, block_sparse_moe, ln_1, ln_2, shared_mlp)


# Custom transformer that uses GMM-enabled decoder layers
class MixtralTransformerGMM(MixtralTransformer):
    """Transformer that uses GMM-enabled decoder layers"""

    @staticmethod
    def init(config, *, key):
        """Initialize transformer with GMM-enabled layers"""
        S = Stacked
        if not config.scan_layers:
            S = BlockSeq

        # Use our GMM-enabled decoder layers
        layers = S.init(config.Layers, MixtralDecoderLayerGMM, gradient_checkpointing=config.gradient_checkpointing)(
            config,
            key=shaped_rng_split(key, config.num_layers),
        )
        ln_f = config.mk_LayerNorm(config.Embed)

        return MixtralTransformerGMM(config, layers, ln_f)


# Custom LM head model that uses GMM-enabled transformer
class MixtralLMHeadModelGMM(MixtralLMHeadModel):
    """LM head model that uses GMM-enabled transformer"""

    @classmethod
    def init(cls, Vocab, config, *, key):
        """Initialize model with GMM-enabled transformer"""
        k_t, k_emb = jrandom.split(key, 2)

        # Use our GMM-enabled transformer
        transformer = MixtralTransformerGMM.init(config, key=k_t)
        embeddings = LlamaEmbedding.init(Vocab, config, key=k_emb)

        if config.tie_word_embeddings:
            lm_head = None
        else:
            lm_head = hnn.Linear.init(In=config.Embed, Out=Vocab, key=k_emb, use_bias=False, out_first=True)

        return MixtralLMHeadModelGMM(transformer, embeddings, lm_head)


# Custom config that uses GMM-enabled MoE blocks
@dataclasses.dataclass(frozen=True)
class MixtralConfigGMM(MixtralConfig):
    """MixtralConfig that uses GMM for MoE operations"""

    @property
    def model_type(self):
        """Return our custom GMM-enabled model type"""
        return MixtralLMHeadModelGMM


# This config uses MixtralConfigGMM for GMM-based MoE functionality
moe_300m_config = MixtralConfigGMM(
    seq_len=1024,
    hidden_dim=768,
    intermediate_dim=768,
    num_layers=12,
    num_heads=12,
    num_kv_heads=12,
    gradient_checkpointing=True,
    scan_layers=True,
    n_routed_experts=32,
    num_experts_per_tok=4,
    # NOTE: use_gmm parameter doesn't exist in standard MixtralConfig
    # This run now uses the same implementation as the non-GMM run
    # disable MoE auxiliary loss logging to prevent JAX tracer leaks
    lbl_coef=None,  # disables load balancing loss logging
    rzl_coef=None,  # disables router z-loss logging
)

speedrun_config = SpeedrunConfig(
    author=Author(
        name="Pranshu Chaturvedi",
        affiliation="Stanford University",
        url="https://stanford.edu/~pranshu",
    ),
    description="Training a 300M parameter Mixtral-style MoE model on a TPU with GMM (Grouped Matrix Multiply)",
    model_config=moe_300m_config,
    train_config=SimpleTrainConfig(
        ResourceConfig.with_tpu("v4-8", slice_count=1),
        train_batch_size=256,
        num_train_steps=4000,
        learning_rate=5e-4,
        weight_decay=0.1,
        steps_per_eval=1000,
    ),
)

if __name__ == "__main__":
    # Add logging to confirm GMM is being used
    logger.info("Running Mixtral with GMM-enabled MoE layers")
    executor_main(steps=default_speedrun("pranshu_mixtral_moe_gmm_v4_8", speedrun_config))

