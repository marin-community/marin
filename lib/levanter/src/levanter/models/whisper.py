# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

import haliax as hax
import haliax.jax_utils
import haliax.nn as hnn
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call, shaped_rng_split
from haliax.nn.scan import Stacked
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig, ModelWithHfSerializationMixin
from levanter.layers.attention import AttentionBackend, AttentionMask, dot_product_attention
from levanter.models.asr_model import ASRConfig, ASRMixin
from levanter.models.lm_model import LmConfig
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.logging import silence_transformer_nag


silence_transformer_nag()
from transformers import PretrainedConfig as HfConfig  # noqa: E402
from transformers import WhisperConfig as HfWhisperConfig  # noqa: E402


@LmConfig.register_subclass("whisper")
@dataclass(frozen=True)
class WhisperConfig(HFCompatConfig, ASRConfig):
    vocab_size: int = 51865
    num_mel_bins: int = 80
    encoder_layers: int = 4
    encoder_attention_heads: int = 6
    encoder_ffn_dim: int = 1536
    decoder_layers: int = 4
    decoder_attention_heads: int = 6
    decoder_ffn_dim: int = 1536
    d_model: int = 384

    max_source_positions: int = 1500
    max_length: int = 448

    activation_function: ActivationFunctionEnum = ActivationFunctionEnum.gelu
    layer_norm_epsilon: float = 1e-5
    use_bias: bool = True

    initializer_range: float = 0.02
    gradient_checkpointing: bool = True

    # Attention-related config
    upcast_attn: bool = True
    use_flash_attention: bool = False
    attn_backend: Optional[AttentionBackend] = None
    flash_attention_block_size: Optional[int] = None

    @property
    def model_type(self) -> Type["WhisperModel"]:
        return WhisperModel

    @property
    def asr_model_type(self) -> Type["WhisperASRModel"]:
        return WhisperASRModel

    def hf_checkpoint_converter(
        self, ref_checkpoint: str = "openai/whisper-base"
    ) -> HFCheckpointConverter["WhisperModel"]:  # type: ignore
        return HFCheckpointConverter(self, reference_checkpoint=ref_checkpoint, ignore_prefix="model")

    # Axis
    MelPos = property(lambda self: Axis(name="position", size=self.max_source_positions * 2))
    Pos = property(lambda self: Axis(name="position", size=self.max_length))
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    SourcePos = property(lambda self: Axis(name="position", size=self.max_source_positions))
    Vocab = property(lambda self: Axis(name="vocab_size", size=self.vocab_size))
    Embed = property(lambda self: Axis(name="embed_dim", size=self.d_model))
    EncoderMlp = property(lambda self: Axis(name="mlp_dim", size=self.encoder_ffn_dim))
    EncoderHeads = property(lambda self: Axis(name="heads", size=self.encoder_attention_heads))
    EncoderHeadSize = property(lambda self: Axis(name="head_size", size=self.d_model // self.encoder_attention_heads))
    EncoderLayer = property(lambda self: Axis(name="encoder_layers", size=self.encoder_layers))
    DecoderMlp = property(lambda self: Axis(name="mlp_dim", size=self.decoder_ffn_dim))
    DecoderHeads = property(lambda self: Axis(name="heads", size=self.decoder_attention_heads))
    DecoderHeadSize = property(lambda self: Axis(name="head_size", size=self.d_model // self.decoder_attention_heads))
    DecoderLayer = property(lambda self: Axis(name="decoder_layers", size=self.decoder_layers))
    Mels = property(lambda self: Axis(name="n_mels", size=self.num_mel_bins))

    def to_hf_config(self, vocab_size, config_overrides=None):
        if config_overrides is None:
            config_overrides = {}

        return HfWhisperConfig(
            vocab_size=self.vocab_size,
            num_mel_bins=self.num_mel_bins,
            encoder_layers=self.encoder_layers,
            encoder_attention_heads=self.encoder_attention_heads,
            decoder_layers=self.decoder_layers,
            decoder_attention_heads=self.decoder_attention_heads,
            decoder_ffn_dim=self.decoder_ffn_dim,
            encoder_ffn_dim=self.encoder_ffn_dim,
            activation_function=self.activation_function.name,
            max_source_positions=self.max_source_positions,
            d_model=self.d_model,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig):
        return cls(
            vocab_size=hf_config.vocab_size,
            num_mel_bins=hf_config.num_mel_bins,
            encoder_layers=hf_config.encoder_layers,
            encoder_attention_heads=hf_config.encoder_attention_heads,
            decoder_layers=hf_config.decoder_layers,
            decoder_attention_heads=hf_config.decoder_attention_heads,
            decoder_ffn_dim=hf_config.decoder_ffn_dim,
            encoder_ffn_dim=hf_config.encoder_ffn_dim,
            activation_function=ActivationFunctionEnum(hf_config.hidden_act),
            max_source_positions=hf_config.max_source_positions,
            d_model=hf_config.d_model,
        )


class WhisperMlp(eqx.Module):
    fc1: hnn.Linear  # projection from Embed to Intermediate (typically 4x Embed)
    fc2: hnn.Linear  # projection from Intermediate to Embed
    act: Callable = eqx.field(static=True)

    @staticmethod
    def init(Embed: Axis, Mlp: Axis, activation_fn, *, key, use_bias: bool = True) -> "WhisperMlp":
        k_fc, k_proj = haliax.jax_utils.maybe_rng_split(key, 2)
        fc1 = hnn.Linear.init(Out=Mlp, In=Embed, key=k_fc, use_bias=use_bias, out_first=False)
        fc2 = hnn.Linear.init(Out=Embed, In=Mlp, key=k_proj, use_bias=use_bias, out_first=False)
        if isinstance(activation_fn, ActivationFunctionEnum):
            activation_fn = activation_fn.to_fn()

        return WhisperMlp(fc1, fc2, activation_fn)

    @named_call
    def __call__(self, x: NamedArray, *, key=None):
        k1, k2 = haliax.jax_utils.maybe_rng_split(key, 2)
        x = self.fc1(x, key=k1)
        x = self.act(x)
        x = self.fc2(x, key=k2)
        return x


class WhisperAttention(eqx.Module):
    config: WhisperConfig = eqx.field(static=True)

    q_proj: hnn.Linear  # input projection from [embed] -> [q, heads, head_dim]
    k_proj: hnn.Linear  # input projection from [embed] -> [k, heads, head_dim]
    v_proj: hnn.Linear  # input projection from [embed] -> [v, heads, head_dim]

    out_proj: hnn.Linear  # output projection from [heads, head_dim] -> [embed]
    inference: bool

    @staticmethod
    def init(Heads: Axis, HeadSize: Axis, config: WhisperConfig, *, key) -> "WhisperAttention":
        use_bias = config.use_bias
        Embed = config.Embed

        k_q, k_k, k_v, k_out = haliax.jax_utils.maybe_rng_split(key, 4)
        q_proj = hnn.Linear.init(In=Embed, Out=(Heads, HeadSize), key=k_q, use_bias=use_bias, out_first=False)
        k_proj = hnn.Linear.init(In=Embed, Out=(Heads, HeadSize), key=k_k, use_bias=False, out_first=False)
        v_proj = hnn.Linear.init(In=Embed, Out=(Heads, HeadSize), key=k_v, use_bias=use_bias, out_first=False)
        out_proj = hnn.Linear.init(In=(Heads, HeadSize), Out=Embed, key=k_out, use_bias=use_bias, out_first=False)

        return WhisperAttention(config, q_proj, k_proj, v_proj, out_proj, inference=False)

    @named_call
    def __call__(self, x: NamedArray, xa: Optional[NamedArray] = None, mask: Optional[AttentionMask] = None, *, key):
        k_k, k_v, k_q, k_out, k_drop = hax.jax_utils.maybe_rng_split(key, 5)
        q = self.q_proj(x, key=k_q).rearrange((..., "heads", "position", "head_size"))
        kv_in = x if xa is None else xa
        k = self.k_proj(kv_in, key=k_k).rearrange((..., "heads", "position", "head_size"))
        v = self.v_proj(kv_in, key=k_v).rearrange((..., "heads", "position", "head_size"))

        # Rename k and v's Pos as haliax doesn't support unnamed axes or duplicate axes
        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})

        attn_output = dot_product_attention(
            "position",
            "key_position",
            "head_size",
            q,
            k,
            v,
            mask=mask,
            inference=self.inference,
            use_flash=self.config.use_flash_attention,
            flash_block_size=self.config.flash_attention_block_size,
            prng=k_drop,
            attention_dtype=jnp.float32 if self.config.upcast_attn else None,
        )

        attn_output = attn_output.astype(x.dtype)
        attn_output = self.out_proj(attn_output, key=k_out)

        return attn_output


class WhisperLayer(ModuleWithStateDictSerialization, eqx.Module):
    self_attn: WhisperAttention
    attn_ln: hnn.LayerNorm

    encoder_attn: Optional[WhisperAttention]
    encoder_attn_ln: Optional[hnn.LayerNorm]

    mlp: WhisperMlp
    mlp_ln: hnn.LayerNorm

    @staticmethod
    def init(
        Heads: Axis, HeadSize: Axis, Mlp: Axis, config: WhisperConfig, has_cross: bool = True, *, key
    ) -> "WhisperLayer":
        k_attn, k_cross, k_mlp = haliax.jax_utils.maybe_rng_split(key, 3)

        attn_ln = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)
        self_attn = WhisperAttention.init(Heads, HeadSize, config, key=k_attn)

        if has_cross:
            encoder_attn_ln = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)
            encoder_attn = WhisperAttention.init(Heads, HeadSize, config, key=k_attn)
        else:
            encoder_attn_ln = None
            encoder_attn = None

        mlp = WhisperMlp.init(config.Embed, Mlp, config.activation_function, key=k_mlp, use_bias=config.use_bias)
        mlp_ln = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)

        return WhisperLayer(self_attn, attn_ln, encoder_attn, encoder_attn_ln, mlp, mlp_ln)

    @named_call
    def __call__(self, x: NamedArray, xa: Optional[NamedArray] = None, mask: Optional[AttentionMask] = None, *, key):
        k1, k2, k3 = haliax.jax_utils.maybe_rng_split(key, 3)

        attn_output = self.self_attn(self.attn_ln(x), mask=mask, key=k1)
        x = x + attn_output

        if self.encoder_attn:
            xs = x
            if self.encoder_attn_ln:
                xs = self.encoder_attn_ln(xs)
            x = x + self.encoder_attn(xs, xa, key=k2)

        ff_output = self.mlp(self.mlp_ln(x), key=k3)
        x = x + ff_output

        return x

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {
            "mlp": None,
            "mlp_ln": "final_layer_norm",
            "attn_ln": "self_attn_layer_norm",
            "encoder_attn_ln": "encoder_attn_layer_norm",
        }


class WhisperTransformer(ModuleWithStateDictSerialization):
    layers: Stacked[WhisperLayer]
    Layer: Axis
    layer_norm: hnn.LayerNorm

    @staticmethod
    def init(Layer: Axis, Heads: Axis, HeadSize: Axis, Mlp: Axis, config: WhisperConfig, has_cross: bool, *, key):
        # vectorize the blocks
        layers = Stacked.init(Layer, WhisperLayer, gradient_checkpointing=config.gradient_checkpointing)(
            Heads,
            HeadSize,
            Mlp,
            config,
            has_cross=has_cross,
            key=shaped_rng_split(key, Layer.size),
        )
        layer_norm = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_epsilon, use_bias=config.use_bias)

        return WhisperTransformer(layers, Layer, layer_norm)

    @named_call
    def __call__(
        self,
        x: NamedArray,
        xa: Optional[NamedArray] = None,
        attn_mask: Optional[AttentionMask] = None,
        *,
        key=None,
    ) -> NamedArray:
        keys = hax.jax_utils.maybe_rng_split(key, self.Layer.size) if key is not None else None
        x = self.layers.fold(x, xa, attn_mask, key=keys)
        x = self.layer_norm(x)

        return x


class WhisperEncoder(ModuleWithStateDictSerialization):
    config: WhisperConfig = eqx.field(static=True)
    conv1: hnn.Conv
    conv2: hnn.Conv
    act: Callable = eqx.field(static=True)

    transformer: WhisperTransformer

    @classmethod
    def init(cls, config: WhisperConfig, *, key) -> "WhisperEncoder":
        k_conv1, k_conv2, k_t = haliax.jax_utils.maybe_rng_split(key, 3)

        Len = hax.Axis("position", size=config.SourcePos.size * 2)
        Mid = hax.Axis("mid", config.Embed.size)
        conv1 = hnn.Conv.init(Len, config.Mels, Mid, kernel_size=3, padding=1, key=k_conv1)
        conv2 = hnn.Conv.init(Len, Mid, config.Embed, kernel_size=3, stride=2, padding=1, key=k_conv2)

        transformer = WhisperTransformer.init(
            config.EncoderLayer,
            config.EncoderHeads,
            config.EncoderHeadSize,
            config.EncoderMlp,
            config,
            has_cross=False,
            key=k_t,
        )

        return WhisperEncoder(config, conv1, conv2, config.activation_function.to_fn(), transformer)

    def __call__(self, spec: NamedArray, *, key=None) -> NamedArray:
        k_conv1, k_conv2, k_transformer = haliax.jax_utils.maybe_rng_split(key, 3)
        spec = spec.astype(self.conv1.weight.dtype)
        x = self.act(self.conv1(spec, key=k_conv1))
        x = self.act(self.conv2(x, key=k_conv2))

        seq_len = x.axis_size("position")
        pos_emb = whisper_sinusoids(self.config.Embed, self.config.SourcePos)[self.config.SourcePos, :seq_len]
        x = x + pos_emb

        x = self.transformer(x, key=k_transformer)
        return x

    def resize_vocab(self, new_size: int, key: Optional[PRNGKeyArray] = None) -> "WhisperDecoder":
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, embeddings=new_embeddings)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"transformer": None}


class WhisperDecoderEmbeddings(eqx.Module):
    Vocab: Axis = eqx.field(static=True)
    config: WhisperConfig = eqx.field(static=True)

    token_embeddings: hnn.Embedding
    position_embeddings: hnn.Embedding

    @staticmethod
    def init(Vocab: Axis, config: WhisperConfig, *, key) -> "WhisperDecoderEmbeddings":
        k_wte, k_wpe, k_out = haliax.jax_utils.maybe_rng_split(key, 3)

        token_embeddings = hnn.Embedding.init(
            Vocab, config.Embed, key=k_wte, initializer_range=config.initializer_range
        )

        # Whisper Initializes the Positional Embeddings as Empty
        position_embeddings = hnn.Embedding.init(config.Pos, config.Embed, key=k_wpe, initializer_range=0)

        return WhisperDecoderEmbeddings(Vocab, config, token_embeddings, position_embeddings)

    @named_call
    def embed(self, input_ids, *, key):
        input_embeds = self.token_embeddings(input_ids)
        input_Pos = input_ids.resolve_axis("position")
        position_embeds = self.position_embeddings.embed(hax.arange(input_Pos))
        x = input_embeds + position_embeds

        return x

    def unembed(self, x: NamedArray):
        return hax.dot(x, self.token_embeddings.weight, axis="embed_dim")

    def resize_embeddings(self, new_size: int, key: Optional[PRNGKeyArray] = None):
        new_token_embeddings = self.token_embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, Vocab=self.Vocab.resize(new_size), token_embeddings=new_token_embeddings)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"token_embeddings": "embed_tokens", "position_embeddings": "embed_positions"}


class WhisperDecoder(ModuleWithStateDictSerialization):
    transformer: WhisperTransformer
    embeddings: WhisperDecoderEmbeddings

    @property
    def config(self):
        return self.embeddings.config

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @property
    def Pos(self) -> Axis:
        return self.config.Pos

    @classmethod
    def init(cls, config: WhisperConfig, *, key) -> "WhisperDecoder":
        k_t, k_embeddings = haliax.jax_utils.maybe_rng_split(key, 2)
        transformer = WhisperTransformer.init(
            config.DecoderLayer,
            config.DecoderHeads,
            config.DecoderHeadSize,
            config.DecoderMlp,
            config,
            has_cross=True,
            key=k_t,
        )
        embeddings = WhisperDecoderEmbeddings.init(config.Vocab, config, key=k_embeddings)

        return WhisperDecoder(transformer, embeddings)

    def __call__(
        self,
        input_ids: NamedArray,
        audio_embeds: NamedArray,
        attn_mask: Optional[AttentionMask] = None,
        *,
        key=None,
    ) -> NamedArray:
        causal_mask = AttentionMask.causal()
        if attn_mask is not None:
            causal_mask = causal_mask & attn_mask
        k_embed, k_transformer = haliax.jax_utils.maybe_rng_split(key, 2)
        x = self.embeddings.embed(input_ids, key=k_embed)
        x = self.transformer(x, audio_embeds, causal_mask, key=k_transformer)
        lm_logits = self.embeddings.unembed(x)

        return lm_logits

    def resize_vocab(self, new_size: int, key: Optional[PRNGKeyArray] = None) -> "WhisperDecoder":
        new_embeddings = self.embeddings.resize_embeddings(new_size, key=key)
        return dataclasses.replace(self, embeddings=new_embeddings)

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        return {"transformer": None, "embeddings": None}


class WhisperModel(eqx.Module, ModelWithHfSerializationMixin[WhisperConfig]):
    encoder: WhisperEncoder
    decoder: WhisperDecoder

    @property
    def config(self):
        return self.encoder.config

    @property
    def Pos(self) -> Axis:
        return self.config.Pos

    @property
    def Vocab(self) -> Axis:
        return self.decoder.embeddings.Vocab

    def resize_vocab(self, new_size: int, key: Optional[PRNGKeyArray] = None) -> "WhisperModel":
        new_decoder = self.decoder.resize_vocab(new_size, key)
        return dataclasses.replace(self, decoder=new_decoder)

    @classmethod
    def init(cls, Vocab: Axis, config: WhisperConfig, *, key) -> "WhisperModel":
        k_t, k_embeddings = haliax.jax_utils.maybe_rng_split(key, 2)
        encoder = WhisperEncoder.init(config, key=k_embeddings)
        decoder = WhisperDecoder.init(config, key=k_t)

        return cls(encoder, decoder)

    def __call__(
        self,
        mel: NamedArray,
        input_ids: NamedArray,
        attn_mask: Optional[AttentionMask | NamedArray] = None,
        *,
        key=None,
    ) -> NamedArray:
        if attn_mask is not None and not isinstance(attn_mask, AttentionMask):
            attn_mask = AttentionMask.explicit(attn_mask)
        k_encoder, k_decoder = haliax.jax_utils.maybe_rng_split(key, 2)
        audio_features = self.encoder(mel, key=k_encoder)
        lm_logits = self.decoder(input_ids, audio_features, attn_mask=attn_mask, key=k_decoder)

        return lm_logits


class WhisperASRModel(WhisperModel, ASRMixin):
    pass


def whisper_sinusoids(Channels: Axis, SourcePos: Axis, base: int = 10000) -> NamedArray:
    with jax.ensure_compile_time_eval():
        log_timescale_increment = (hax.log(base) / (Channels.size // 2 - 1)).item()
        ChannelHalfSize = Channels.resize(Channels.size // 2)
        inv_timescales = hax.exp(-log_timescale_increment * hax.arange(ChannelHalfSize))

        position_ids: NamedArray = hax.arange(SourcePos)

        freqs = position_ids * inv_timescales.broadcast_axis(SourcePos)
        emb = hax.concatenate(Channels, (hax.sin(freqs), hax.cos(freqs)))
        return emb
