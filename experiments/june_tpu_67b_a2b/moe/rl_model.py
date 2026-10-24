# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Levanter learner boundary over the June Snowball training transformer."""

import dataclasses
from dataclasses import dataclass
from typing import cast

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from haliax.state_dict import ModuleWithStateDictSerialization, StateDict
from jax.sharding import PartitionSpec as P
from jax.sharding import get_abstract_mesh
from levanter.grug.attention import AttentionMask as GrugAttentionMask
from levanter.layers.attention import AttentionMask
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from levanter.models.loss import next_token_loss_weight
from levanter.models.snowball import (
    SnowballBlock,
    SnowballConfig,
    SnowballRouteCaptureBatch,
    SnowballTransformer,
    snowball_block_from_state_dict,
    snowball_block_to_state_dict,
    snowball_embeddings_from_state_dict,
    snowball_embeddings_to_state_dict,
    snowball_final_from_state_dict,
    snowball_final_to_state_dict,
    snowball_from_state_dict,
    snowball_to_state_dict,
)
from levanter.pipeline import evenly_partition_layers

from experiments.june_tpu_67b_a2b.moe.model import (
    _DEFAULT_EP_CAPACITY_FACTOR,
    LONG_ATTENTION_INTERVAL,
    Block,
    GatedNorm,
    GrugModelConfig,
    RMSNorm,
    Transformer,
)


@LmConfig.register_subclass("june_snowball")
@dataclass(frozen=True)
class JuneSnowballConfig(SnowballConfig):
    """Published Snowball architecture using June's training kernels."""

    capacity_factor: float = _DEFAULT_EP_CAPACITY_FACTOR

    @property
    def model_type(self):  # pyrefly: ignore[bad-override]  # Same HF recipe, different transformer implementation.
        return JuneSnowballLMHeadModel

    def training_config(self) -> GrugModelConfig:
        if self.projection_implementation != "native":
            raise ValueError("June Snowball training does not support fixed-schedule projection")
        if self.rms_norm_implementation != "native":
            raise ValueError("June Snowball training does not support non-native RMS normalization")
        return GrugModelConfig(
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            shared_expert_intermediate_dim=self.shared_expert_intermediate_dim,
            num_experts=self.num_experts,
            num_experts_per_token=self.num_experts_per_token,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            sliding_window=self.sliding_window,
            layer_norm_eps=self.layer_norm_eps,
            initializer_std=self.initializer_std,
            qk_mult=self.qk_mult,
            rope=self.rope,
            attention_implementation=self.attention_implementation,
            moe_implementation=self.moe_implementation,
            capacity_factor=self.capacity_factor,
            disable_pko=True,
            disable_long_rope=True,
        )


class JuneSnowballLMHeadModel(ModuleWithStateDictSerialization, LmHeadModel[JuneSnowballConfig]):
    transformer: Transformer
    _config: JuneSnowballConfig = eqx.field(static=True)

    @property
    def config(self) -> JuneSnowballConfig:
        return self._config

    @property
    def Vocab(self) -> hax.Axis:
        return hax.Axis("vocab", self.config.vocab_size)

    @classmethod
    def init(cls, Vocab: hax.Axis, config: JuneSnowballConfig, *, key):
        config = dataclasses.replace(config, vocab_size=Vocab.size)
        return cls(Transformer.init(config.training_config(), key=key), config)

    def capture_route_captures(self, input_ids, attn_mask=None) -> SnowballRouteCaptureBatch:
        """Capture June routes bound to the public token and segment layout."""
        Position = input_ids.resolve_axis(self.Pos.name)
        leading = tuple(axis for axis in input_ids.axes if axis != Position)
        tokens = input_ids.rearrange((*leading, Position)).array.reshape(-1, Position.size)
        if attn_mask is None:
            attn_mask = AttentionMask.causal()
        if not isinstance(attn_mask, AttentionMask):
            raise ValueError("June Snowball requires a structured causal attention mask")
        if (
            not attn_mask.is_causal
            or attn_mask.explicit_mask is not None
            or attn_mask.causal_offset is not None
            or attn_mask.sliding_window is not None
            or attn_mask.bidirectional_window is not None
        ):
            raise ValueError("June Snowball supports causal segment masks with its fixed layer window schedule")
        segments = None
        if attn_mask.segment_ids is not None:
            segments = tuple(
                segment.broadcast_axis(leading).rearrange((*leading, ...)).array.reshape(tokens.shape)
                for segment in attn_mask.segment_ids
            )
        grug_mask = GrugAttentionMask(is_causal=True, segment_ids=segments)
        return SnowballRouteCaptureBatch(self.transformer.capture_route_captures(tokens, grug_mask), tokens, segments)

    def activations(
        self,
        input_ids,
        attn_mask=None,
        *,
        key=None,
        pos_ids=None,
        route_captures: SnowballRouteCaptureBatch | None = None,
    ):
        tokens, segments, captures, positions = self._prepare_route_replay_inputs(
            input_ids, attn_mask, pos_ids=pos_ids, route_captures=route_captures
        )
        Position = input_ids.resolve_axis(self.Pos.name)
        leading = tuple(axis for axis in input_ids.axes if axis != Position)
        hidden, _router_metrics = self.transformer(
            tokens,
            mask=GrugAttentionMask(is_causal=True, segment_ids=segments),
            position_ids=positions,
            route_captures=captures,
        )
        return hax.named(
            hidden.reshape(*(axis.size for axis in leading), Position.size, self.Embed.size),
            (*leading, Position, self.Embed),
        )

    def _prepare_route_replay_inputs(self, input_ids, attn_mask, *, pos_ids, route_captures):
        if pos_ids is not None and route_captures is not None:
            raise ValueError("June Snowball route replay requires the default contiguous position layout")
        if route_captures is not None and not isinstance(route_captures, SnowballRouteCaptureBatch):
            raise TypeError("June Snowball replay requires a bound SnowballRouteCaptureBatch")
        Position = input_ids.resolve_axis(self.Pos.name)
        leading = tuple(axis for axis in input_ids.axes if axis != Position)
        tokens = input_ids.rearrange((*leading, Position)).array.reshape(-1, Position.size)
        if attn_mask is None:
            attn_mask = AttentionMask.causal()
        if not isinstance(attn_mask, AttentionMask):
            raise ValueError("June Snowball requires a structured causal attention mask")
        if (
            not attn_mask.is_causal
            or attn_mask.explicit_mask is not None
            or attn_mask.causal_offset is not None
            or attn_mask.sliding_window is not None
            or attn_mask.bidirectional_window is not None
        ):
            raise ValueError("June Snowball supports causal segment masks with its fixed layer window schedule")
        segments = None
        if attn_mask.segment_ids is not None:
            segments = tuple(
                segment.broadcast_axis(leading).rearrange((*leading, ...)).array.reshape(tokens.shape)
                for segment in attn_mask.segment_ids
            )
        captures = route_captures
        if captures is not None:
            binding_is_traced = any(
                isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves((tokens, captures, segments))
            )
            if binding_is_traced:
                tokens = eqx.error_if(
                    tokens,
                    jnp.any(captures.token_ids != tokens),
                    "route captures are bound to different token IDs or packed order",
                )
            elif not np.array_equal(np.asarray(captures.token_ids), np.asarray(tokens)):
                raise ValueError("route captures are bound to different token IDs or packed order")
            if (captures.segment_ids is None) != (segments is None):
                raise ValueError("route captures are bound to different packed segment IDs")
            if captures.segment_ids is not None and segments is not None:
                if binding_is_traced:
                    segments = tuple(
                        eqx.error_if(
                            expected,
                            jnp.any(captured != expected),
                            "route captures are bound to different packed segment IDs",
                        )
                        for captured, expected in zip(captures.segment_ids, segments, strict=True)
                    )
                elif any(
                    not np.array_equal(captured, expected)
                    for captured, expected in zip(captures.segment_ids, segments, strict=True)
                ):
                    raise ValueError("route captures are bound to different packed segment IDs")
            captures = captures.route_captures
        positions = None if pos_ids is None else pos_ids.rearrange((*leading, Position)).array.reshape(tokens.shape)
        return tokens, segments, captures, positions

    def __call__(self, input_ids, attn_mask=None, *, key=None, pos_ids=None, route_captures=None):
        hidden = self.activations(input_ids, attn_mask, key=key, pos_ids=pos_ids, route_captures=route_captures)
        return hax.dot(hidden, self.get_lm_head(), axis=self.Embed)

    def compute_next_token_loss(
        self,
        example: LmExample,
        *,
        key=None,
        reduction: hax.ReductionFunction | None = cast(hax.ReductionFunction | None, hax.mean),
        reduction_axis: hax.AxisSelection | None = None,
        logsumexp_weight: float | None = None,
        loss_dtype: jnp.dtype | None = jnp.float32,
        logit_soft_cap: float | None = None,
        route_captures: SnowballRouteCaptureBatch | None = None,
    ):
        """Compute June learner loss with optional bound route replay."""
        if route_captures is not None:
            if reduction_axis is not None:
                raise ValueError("June Snowball route replay does not support reduction_axis")
            if logit_soft_cap is not None:
                raise ValueError("June Snowball route replay does not support logit_soft_cap")
            if reduction is None:
                june_reduction = "none"
            elif reduction is hax.mean:
                june_reduction = "mean"
            elif reduction is hax.sum:
                june_reduction = "sum"
            else:
                raise ValueError("June Snowball route replay supports hax.mean, hax.sum, or no reduction")
            tokens, segments, captures, _ = self._prepare_route_replay_inputs(
                example.tokens, example.attn_mask, pos_ids=None, route_captures=route_captures
            )
            Position = example.tokens.resolve_axis(self.Pos.name)
            leading = tuple(axis for axis in example.tokens.axes if axis != Position)
            loss_weight = next_token_loss_weight(Position, example.loss_weight)
            loss_weight = loss_weight.rearrange((*leading, Position)).array.reshape(tokens.shape)
            return self.transformer.next_token_loss(
                tokens,
                loss_weight,
                mask=GrugAttentionMask(is_causal=True, segment_ids=segments),
                reduction=june_reduction,
                logsumexp_weight=logsumexp_weight,
                loss_dtype=loss_dtype or jnp.float32,
                route_captures=captures,
            )
        return self._compute_next_token_loss_from_activations(
            example,
            self.activations(example.tokens, example.attn_mask, key=key, route_captures=route_captures),
            reduction=reduction,
            reduction_axis=reduction_axis,
            logsumexp_weight=logsumexp_weight,
            loss_dtype=loss_dtype,
            logit_soft_cap=logit_soft_cap,
        )

    def get_lm_head(self):
        return hax.auto_sharded(hax.named(self.transformer.output_proj, (self.Embed, self.Vocab)))

    def resize_vocab(self, new_size, key=None):
        if new_size != self.Vocab.size:
            raise ValueError("June Snowball preserves the published checkpoint vocabulary")
        return self

    def to_state_dict(self, prefix=None) -> StateDict:
        # Tuple-block June and Snowball have the same canonical HF tensor layout.
        return snowball_to_state_dict(cast(SnowballTransformer, self.transformer), prefix=prefix)

    def from_state_dict(self, state_dict: StateDict, prefix=None):
        transformer = snowball_from_state_dict(cast(SnowballTransformer, self.transformer), state_dict, prefix=prefix)
        return eqx.tree_at(lambda model: model.transformer, self, cast(Transformer, transformer))


def june_trainable_filter(model: JuneSnowballLMHeadModel):
    """Exclude fixed QB routing biases from gradients and optimizer weight decay."""
    mask = jax.tree.map(lambda _: True, model)
    assert model.transformer.blocks is not None
    return eqx.tree_at(
        lambda tree: tuple(block.mlp.router_bias for block in tree.transformer.blocks),
        mask,
        tuple(False for _ in model.transformer.blocks),
    )


class JunePipelineStage(eqx.Module):
    """One contiguous set of June blocks with only its owned boundary weights."""

    blocks: tuple[Block, ...]
    token_embed: jax.Array | None
    embed_norm: RMSNorm | None
    embed_gated_norm: GatedNorm | None
    final_norm: RMSNorm | None
    final_gated_norm: GatedNorm | None
    output_proj: jax.Array | None
    config: GrugModelConfig = eqx.field(static=True)
    layer_offset: int = eqx.field(static=True)

    def embed(self, token_ids):
        assert self.token_embed is not None and self.embed_norm is not None and self.embed_gated_norm is not None
        hidden = self.token_embed.at[token_ids].get(out_sharding=P(("replica_dcn", "data", "expert")))
        # Keep gather fusion from changing normalization rounding between scoring and AD.
        hidden = jax.lax.optimization_barrier(hidden)
        return self.embed_gated_norm(self.embed_norm(hidden))

    def run_blocks_with_stats(self, hidden, segment_ids, position_ids):
        """Return hidden states and total routing counts, including padding assignments."""
        # Pipeline AD transports these counters through floating auxiliary values.
        # Bound the complete microbatch so every count remains exactly representable.
        assignment_bound = segment_ids.size * self.config.num_experts_per_token * self.config.num_layers
        if get_abstract_mesh().shape["expert"] > 1 and assignment_bound > 2**24:
            raise ValueError(
                "EP routing telemetry requires at most 2**24 assignments per microbatch; use more microbatches"
            )
        stats = {
            name: jnp.asarray(0, dtype=jnp.int32)
            for name in ("routing_assignments", "routing_sender_drops", "routing_receiver_drops")
        }
        max_layer_drops = jnp.asarray(0, dtype=jnp.int32)
        drop_layer = jnp.asarray(-1, dtype=jnp.int32)
        segments = (segment_ids, segment_ids)
        short_mask = GrugAttentionMask(is_causal=True, sliding_window=self.config.sliding_window, segment_ids=segments)
        long_mask = GrugAttentionMask(is_causal=True, segment_ids=segments)
        for index, block in enumerate(self.blocks, start=self.layer_offset):
            is_long = (
                index % LONG_ATTENTION_INTERVAL == LONG_ATTENTION_INTERVAL - 1 or index == self.config.num_layers - 1
            )
            hidden, layer_stats = eqx.filter_checkpoint(block)(
                hidden,
                short_mask,
                long_mask,
                is_long,
                False,
                True,
                position_ids,
            )
            stats = {name: count + layer_stats[name] for name, count in stats.items()}
            layer_drops = layer_stats["routing_sender_drops"] + layer_stats["routing_receiver_drops"]
            max_layer_drops = jnp.maximum(max_layer_drops, layer_drops)
            drop_layer = jnp.where(layer_drops > 0, index, drop_layer)
        return hidden, {**stats, "routing_max_layer_drops": max_layer_drops, "routing_drop_layer": drop_layer}

    def finish(self, hidden):
        assert self.final_norm is not None and self.final_gated_norm is not None
        return self.final_gated_norm(self.final_norm(hidden))

    def get_lm_head(self):
        assert self.output_proj is not None
        return self.output_proj

    def to_state_dict(self, prefix=None) -> StateDict:
        """Export only this stage's tensors using global HF layer indices."""
        tensors = {}
        if self.token_embed is not None:
            tensors.update(snowball_embeddings_to_state_dict(cast(SnowballTransformer, self), prefix))
        if self.output_proj is not None:
            tensors.update(snowball_final_to_state_dict(cast(SnowballTransformer, self), prefix))
        for index, block in enumerate(self.blocks, start=self.layer_offset):
            tensors.update(snowball_block_to_state_dict(cast(SnowballBlock, block), index, prefix))
        return tensors

    def from_state_dict(self, state_dict: StateDict, prefix=None):
        """Load an owned-stage tensor subset without constructing the full model."""
        stage = self
        if self.token_embed is not None:
            stage = cast(
                JunePipelineStage,
                snowball_embeddings_from_state_dict(
                    cast(SnowballTransformer, stage),
                    state_dict,
                    prefix,
                ),
            )
        if self.output_proj is not None:
            stage = cast(
                JunePipelineStage,
                snowball_final_from_state_dict(
                    cast(SnowballTransformer, stage),
                    state_dict,
                    prefix,
                ),
            )
        blocks = tuple(
            cast(Block, snowball_block_from_state_dict(cast(SnowballBlock, block), state_dict, index, prefix))
            for index, block in enumerate(stage.blocks, start=stage.layer_offset)
        )
        return eqx.tree_at(lambda stage: stage.blocks, stage, blocks)

    def trainable_filter(self):
        mask = jax.tree.map(lambda _: True, self)
        return eqx.tree_at(
            lambda stage: tuple(block.mlp.router_bias for block in stage.blocks), mask, tuple(False for _ in self.blocks)
        )


def split_june_pipeline_model(model: JuneSnowballLMHeadModel, num_stages: int) -> tuple[JunePipelineStage, ...]:
    """Partition tuple blocks and boundary weights without copying model arrays."""
    transformer = model.transformer
    assert transformer.blocks is not None
    stages = []
    for stage_index, (start, end) in enumerate(evenly_partition_layers(len(transformer.blocks), num_stages)):
        first, last = stage_index == 0, stage_index == num_stages - 1
        stages.append(
            JunePipelineStage(
                blocks=transformer.blocks[start:end],
                token_embed=transformer.token_embed if first else None,
                embed_norm=transformer.embed_norm if first else None,
                embed_gated_norm=transformer.embed_gated_norm if first else None,
                final_norm=transformer.final_norm if last else None,
                final_gated_norm=transformer.final_gated_norm if last else None,
                output_proj=transformer.output_proj if last else None,
                config=transformer.config,
                layer_offset=start,
            )
        )
    return tuple(stages)
