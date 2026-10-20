# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the schema-v2 Hero inference snapshot."""

import dataclasses
import math
from dataclasses import dataclass

from haliax import Axis
from transformers import PretrainedConfig

from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig
from levanter.grug.attention import GrugAttentionImplementation, PagedAttentionImplementation, RotaryConfig
from levanter.grug.grug_moe import MoeImplementation
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.models.snowball import GrugMoeHfConfig


_HERO_SCHEMA_VERSION = 2


@LmConfig.register_subclass("hero")
@dataclass(frozen=True)
class HeroConfig(HFCompatConfig):
    """Hero's latent MoE, independent shared experts, and short-convolution recipe."""

    vocab_size: int = 128256
    hidden_dim: int = 6144
    intermediate_dim: int = 3072
    shared_expert_intermediate_dim: int = 3072
    num_shared_experts: int = 2
    num_experts: int = 384
    num_experts_per_token: int = 8
    latent_dim: int | None = 3072
    num_layers: int = 48
    num_heads: int = 48
    num_kv_heads: int = 12
    local_kv_heads: int | None = 12
    global_kv_heads: int | None = 6
    head_dim: int | None = 128
    max_seq_len: int = 4096
    sliding_window: int = 2048
    global_every: int = 4
    layer_norm_eps: float = 1e-5
    initializer_std: float = 0.5 / math.sqrt(6144)
    qk_mult: float = 1.3
    sconv: bool = True
    sconv_kernel: int = 4
    sconv_sites: tuple[str, ...] = ("k", "attn", "mlp")
    rope: RotaryConfig = dataclasses.field(default_factory=RotaryConfig)
    rope_fused: bool = True
    attention_implementation: GrugAttentionImplementation | None = None
    inference_attention_implementation: PagedAttentionImplementation | None = None
    moe_implementation: MoeImplementation | None = None
    capacity_factor: float = 1.0
    reference_checkpoint: str | None = None
    tokenizer: str | None = None

    def __post_init__(self):
        if (self.local_kv_heads is None) != (self.global_kv_heads is None):
            raise ValueError("local_kv_heads and global_kv_heads must be set together")
        heads = (self.num_kv_heads,) if self.local_kv_heads is None else (self.local_kv_heads, self.global_kv_heads)
        if any(h is None or h <= 0 or self.num_heads % h for h in heads):
            raise ValueError("num_heads must be divisible by every local/global KV-head count")
        if self.num_kv_heads != self.stored_kv_heads:
            raise ValueError("num_kv_heads must equal the stored maximum of local/global KV heads")
        if not 0 < self.num_experts_per_token < self.num_experts:
            raise ValueError("QB routing requires 0 < num_experts_per_token < num_experts")
        if self.latent_dim is not None and not 0 < self.latent_dim <= self.hidden_dim:
            raise ValueError("latent_dim must be positive and no larger than hidden_dim")
        if self.global_every <= 0 or self.num_layers <= 0:
            raise ValueError("global_every and num_layers must be positive")
        if self.sconv_kernel <= 0 or not set(self.sconv_sites) <= {"k", "attn", "mlp"}:
            raise ValueError("short convolution requires a positive kernel and known sites")
        if self.inferred_head_dim % 4:
            raise ValueError("half-RoPE requires head_dim divisible by four")
        if self.num_shared_experts <= 0 or self.shared_expert_intermediate_dim < 0:
            raise ValueError("shared experts require a positive count and non-negative intermediate width")

    @property
    def Embed(self) -> Axis:
        return Axis("embed", self.hidden_dim)

    @property
    def inferred_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads when head_dim is unspecified")
        return self.hidden_dim // self.num_heads

    @property
    def stored_kv_heads(self) -> int:
        if self.local_kv_heads is None or self.global_kv_heads is None:
            return self.num_kv_heads
        return max(self.local_kv_heads, self.global_kv_heads)

    @property
    def requires_explicit_mesh_axes(self) -> bool:
        return True

    @property
    def model_type(self) -> type[LmHeadModel]:  # pyrefly: ignore[bad-override]
        # The implementation imports HeroConfig, so resolve this boundary after configuration discovery.
        from levanter.models.hero_model import HeroLMHeadModel  # noqa: PLC0415

        return HeroLMHeadModel

    @classmethod
    def matches_hf_config(cls, hf_config: PretrainedConfig) -> bool:
        return getattr(hf_config, "grugmoe_artifact_schema_version", None) == _HERO_SCHEMA_VERSION

    @classmethod
    def from_hf_config(cls, hf_config: PretrainedConfig) -> "HeroConfig":
        if hf_config.model_type != "grug_moe" or not cls.matches_hf_config(hf_config):
            raise ValueError("Hero requires a grug_moe schema-v2 checkpoint")
        if getattr(hf_config, "grugmoe_attention_mode", None) != "production":
            raise ValueError("Hero requires production attention")
        if getattr(hf_config, "tie_word_embeddings", False):
            raise ValueError("Hero requires independent embeddings and output weights")
        # Schema-v2 exports write one canonical name per architecture field. Required fields
        # are read directly so incomplete artifacts cannot inherit a different model's defaults.
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_dim=hf_config.hidden_size,
            intermediate_dim=hf_config.moe_intermediate_size,
            shared_expert_intermediate_dim=hf_config.shared_expert_intermediate_size,
            num_shared_experts=hf_config.num_shared_experts,
            num_experts=hf_config.num_experts,
            num_experts_per_token=hf_config.num_experts_per_tok,
            latent_dim=hf_config.latent_dim,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            num_kv_heads=hf_config.num_key_value_heads,
            local_kv_heads=hf_config.local_kv_heads,
            global_kv_heads=hf_config.global_kv_heads,
            head_dim=hf_config.head_dim,
            max_seq_len=hf_config.max_position_embeddings,
            sliding_window=hf_config.sliding_window,
            global_every=hf_config.global_every,
            layer_norm_eps=hf_config.rms_norm_eps,
            initializer_std=hf_config.initializer_range,
            qk_mult=hf_config.qk_mult,
            rope=RotaryConfig(theta=hf_config.rope_theta),
            rope_fused=hf_config.rope_fused,
            sconv=hf_config.sconv,
            sconv_kernel=hf_config.sconv_kernel,
            sconv_sites=tuple(hf_config.sconv_sites),
        )

    def to_hf_config(self, vocab_size: int, config_overrides: dict | None = None) -> GrugMoeHfConfig:
        values = dict(
            architectures=["GrugMoeForCausalLM"],
            vocab_size=vocab_size,
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            head_dim=self.inferred_head_dim,
            max_position_embeddings=self.max_seq_len,
            sliding_window=self.sliding_window,
            rms_norm_eps=self.layer_norm_eps,
            initializer_range=self.initializer_std,
            rope_theta=self.rope.theta,
            tie_word_embeddings=False,
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_token,
            moe_intermediate_size=self.intermediate_dim,
            shared_expert_intermediate_size=self.shared_expert_intermediate_dim,
            num_shared_experts=self.num_shared_experts,
            latent_dim=self.latent_dim,
            qk_mult=self.qk_mult,
            local_kv_heads=self.local_kv_heads,
            global_kv_heads=self.global_kv_heads,
            global_every=self.global_every,
            rope_fused=self.rope_fused,
            sconv=self.sconv,
            sconv_kernel=self.sconv_kernel,
            sconv_sites=list(self.sconv_sites),
            grugmoe_attention_mode="production",
            grugmoe_artifact_schema_version=_HERO_SCHEMA_VERSION,
        )
        if config_overrides is not None:
            values.update(config_overrides)
        return GrugMoeHfConfig(**values)

    def hf_checkpoint_converter(self, ref_checkpoint: str | None = None) -> HFCheckpointConverter:
        ref = self.reference_checkpoint if ref_checkpoint is None else ref_checkpoint
        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=ref,
            HfConfigClass=GrugMoeHfConfig,
            tokenizer=self.tokenizer if self.tokenizer is not None else ref,
        )
