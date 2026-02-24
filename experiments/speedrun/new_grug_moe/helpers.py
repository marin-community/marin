# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import dataclasses
from typing import Any, Protocol, cast
import logging
import os

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
from haliax import Axis, NamedArray
from haliax.state_dict import ModuleWithStateDictSerialization
from jaxtyping import Array, Float, Int, PRNGKeyArray

from collections.abc import Sequence
from experiments.defaults import default_train
from experiments.llama import llama3_tokenizer_vocab_size, llama3_tokenizer
from levanter.grug.attention import AttentionMask
from levanter.layers.attention import AttentionMask as LevanterAttentionMask
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from marin.speedrun.speedrun import SpeedrunConfig, SpeedrunResultsConfig, speedrun_results
from marin.processing.tokenize import lm_data_config, lm_mixture_data_config
from marin.execution.executor import ExecutorStep, InputName, output_path_of
from experiments.pretraining_datasets import NEMOTRON_WEIGHTS, tokenize_nemotron
from marin.utilities.wandb_utils import WANDB_ENTITY, WANDB_PROJECT

OLMOE_1B7B_REFERENCE_CHECKPOINT = "allenai/OLMoE-1B-7B-0125"
logger = logging.getLogger("ray")

nemotron_cc_steps = tokenize_nemotron(tokenizer=llama3_tokenizer)
nemotron_cc_mixture = lm_mixture_data_config(
    components=nemotron_cc_steps,
    weights=NEMOTRON_WEIGHTS,
)


def nemotron_only_speedrun(
    name: str,
    config: SpeedrunConfig,
    tags: list[str] | None = None,
    override_output_path: str | None = None,
    *,
    append_ici_ag_pipelining_flags: bool = False,
    append_async_collective_permute_flag: bool = False,
    wandb_name: str | None = None,
    wandb_group: str | None = None,
    single_checkpoint: bool = False,
    checkpoint_save_minutes: int = 60,
    use_default_validation: bool = False,
    eval_suite: str = "none",
    eval_suite_mode: str = "post_train",
    eval_tpu_type: str | None = None,
    max_eval_instances: int | None = None,
) -> Sequence[ExecutorStep]:
    """Clone of default_speedrun that skips Paloma validation datasets."""

    logger.info(f"Running nemotron-only speedrun {name}")
    config.print_run_info()

    if eval_suite_mode not in ("post_train", "during_train", "both"):
        raise ValueError("eval_suite_mode must be one of: post_train, during_train, both")

    run_tags = ["speedrun"] + (tags or [])
    train_config = dataclasses.replace(config.train_config, data_seed=42)

    if isinstance(config.tokenized_dataset, (InputName, ExecutorStep)):
        pretraining_data = lm_data_config(
            training_set=config.tokenized_dataset,
            validation_sets=None,
            permutation_type="linear",
        )
    else:
        pretraining_data = config.tokenized_dataset

    suite_evals = None

    eval_harness_tasks = ()
    if eval_suite_mode in ("during_train", "both") and suite_evals:
        eval_harness_tasks = suite_evals

    train_step = default_train(
        name=f"speedrun/{name}",
        tokenized=pretraining_data,
        model_config=config.model_config,
        train_config=train_config,
        tags=run_tags,
        eval_harness_tasks=eval_harness_tasks,
        wandb_name=wandb_name or name,
        wandb_group=wandb_group,
        override_output_path=override_output_path,
        use_default_validation=use_default_validation,
    )

    wandb_entity = WANDB_ENTITY
    wandb_project = WANDB_PROJECT

    trainer_cfg = getattr(train_step.config, "train_config", None)
    trainer = getattr(trainer_cfg, "trainer", None) if trainer_cfg else None
    tracker = getattr(trainer, "tracker", None) if trainer else None
    if tracker:
        wandb_entity = tracker.entity or WANDB_ENTITY
        wandb_project = tracker.project or WANDB_PROJECT

    if os.getenv("WANDB_MODE", "").lower() in {"disabled", "offline", "dryrun"}:
        logger.info("Skipping speedrun_results step because WANDB_MODE=%s", os.getenv("WANDB_MODE"))
        return [train_step]

    results_step = ExecutorStep(
        name=f"speedrun/{name}-speedrun_results",
        description=f"compute and store metrics and stats for the speedrun {name}.",
        fn=speedrun_results,
        config=SpeedrunResultsConfig(
            wandb_run_id=train_step,  # resolved to the actual output path by the executor
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            speedrun_config=config,
            output_path=output_path_of(train_step, "speedrun_results.json"),
        ),
    )

    steps: list[ExecutorStep] = [train_step, results_step]
    return steps


class GrugConfigLike(Protocol):
    vocab_size: int
    max_seq_len: int
    hidden_dim: int


class GrugLossFn(Protocol):
    def __call__(
        self,
        transformer: eqx.Module,
        token_ids: Int[Array, "B S"],
        loss_weight: Float[Array, "B S"],
        cfg: GrugConfigLike,
        *,
        mask: AttentionMask | jax.Array | None = None,
        reduction: str = "mean",
        logsumexp_weight: float | None = None,
        loss_dtype: jnp.dtype = jnp.float32,
    ) -> jax.Array: ...


@LmConfig.register_subclass("grug_transformer")
@dataclass(frozen=True)
class WrapperConfig(LmConfig["WrapperLMHeadModel"]):
    # Core dims
    max_seq_len: int = 2048
    hidden_dim: int = 4096
    intermediate_dim: int = 11008
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32
    grug_config: GrugConfigLike = None
    initializer_std: float = 0.01
    vocab_size: int = llama3_tokenizer_vocab_size
    tokenizer: str | None = None
    layer_norm_eps: float = 0.01
    model_cls_fn: Any = None
    loss_fn: Any = None
    _total_trainable_params: int = None
    _flops_per_token: float = None

    def __post_init__(self):
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        assert self.hidden_dim % self.num_heads == 0, "hidden_dim % num_heads must be 0"

    # ---- LmConfig API ----
    @property
    def model_type(self) -> type["WrapperLMHeadModel"]:
        return WrapperLMHeadModel

    def build(self, Vocab: Axis, *, key: PRNGKeyArray) -> "WrapperLMHeadModel":
        return WrapperLMHeadModel.init(Vocab, self.grug_config, self.model_cls_fn(), self.loss_fn, key=key)

    Embed = property(lambda self: Axis("embed", self.hidden_dim))

    @property
    def actual_head_size(self) -> int:
        return self.hidden_dim // self.num_heads

    def flops_per_token(self, vocab_size: int, context_length: int) -> float | None:
        return self._flops_per_token

    def total_trainable_params(self, vocab_size: int) -> int:
        return self._total_trainable_params


def _mask_from_levanter(attn_mask: LevanterAttentionMask | NamedArray | None) -> AttentionMask | jax.Array | None:
    mask: AttentionMask | jax.Array | None = None
    if isinstance(attn_mask, LevanterAttentionMask):
        if attn_mask.explicit_mask is not None:
            raise NotImplementedError("Grug does not support explicit masks yet.")
        if attn_mask.causal_offset is not None:
            raise NotImplementedError("Grug does not support causal offsets yet.")
        segment_ids = None
        if attn_mask.segment_ids is not None:
            q_seg, kv_seg = attn_mask.segment_ids
            segment_ids = (q_seg.array, kv_seg.array)
        mask = AttentionMask(
            is_causal=attn_mask.is_causal,
            segment_ids=segment_ids,
            sliding_window=attn_mask.sliding_window,
        )
    elif isinstance(attn_mask, NamedArray):
        raise NotImplementedError(
            "NamedArray attention masks are not supported by Grug (pass a Levanter AttentionMask)."
        )
    return mask


class WrapperLMHeadModel(
    ModuleWithStateDictSerialization,
    LmHeadModel[GrugConfigLike],
):
    """Minimal Llama-like implementation of LmHeadModel"""

    transformer: eqx.Module
    loss_fn: GrugLossFn

    @property
    def config(self) -> GrugConfigLike:
        return self.transformer.config

    @property
    def Vocab(self) -> Axis:
        return Axis("vocab", self.transformer.config.vocab_size)

    @classmethod
    def init(
        cls, Vocab: Axis, config: GrugConfigLike, model_cls: eqx.Module, loss_fn: GrugLossFn, *, key
    ) -> "WrapperLMHeadModel":
        transformer = model_cls.init(config, key=key)
        return WrapperLMHeadModel(transformer, loss_fn)

    def activations(
        self,
        input_ids: NamedArray,
        attn_mask: AttentionMask | NamedArray | None = None,
        *,
        key=None,
        pos_ids: NamedArray | None = None,
    ) -> NamedArray:
        del key, pos_ids  # unused in this lightweight wrapper
        mask = _mask_from_levanter(attn_mask)
        hidden = self.transformer(input_ids.array, mask)
        axes = (*input_ids.axes, Axis("embed", self.transformer.config.hidden_dim))
        return hax.named(hidden, axes)

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
    ) -> jnp.ndarray | NamedArray:
        """Override to use grug's blockwise loss (avoids materializing full logits)."""
        # NOTE: this wrapper is intentionally minimal; grug core currently doesn't use PRNGs.
        assert logit_soft_cap is None, "logit_soft_cap is not supported by GrugWrapper.compute_next_token_loss"
        del key

        # LmExample-ish protocol: expects `.tokens`, `.loss_weight`, `.attn_mask`.
        tokens = example.tokens
        loss_weight = example.loss_weight
        attn_mask = example.attn_mask

        mask = _mask_from_levanter(attn_mask)
        dtype = jnp.float32 if loss_dtype is None else loss_dtype

        if reduction is None:
            per_pos = self.loss_fn(
                self.transformer,
                tokens.array,
                loss_weight.array,
                self.config,
                mask=mask,
                reduction="none",
                logsumexp_weight=logsumexp_weight,
                loss_dtype=dtype,
            )
            return hax.named(per_pos, tokens.axes)

        # Fast path: scalar mean/sum reduction over all axes.
        if reduction_axis is None and reduction is hax.mean:
            return self.loss_fn(
                self.transformer,
                tokens.array,
                loss_weight.array,
                self.config,
                mask=mask,
                reduction="mean",
                logsumexp_weight=logsumexp_weight,
                loss_dtype=dtype,
            )
        if reduction_axis is None and reduction is hax.sum:
            return self.loss_fn(
                self.transformer,
                tokens.array,
                loss_weight.array,
                self.config,
                mask=mask,
                reduction="sum",
                logsumexp_weight=logsumexp_weight,
                loss_dtype=dtype,
            )

        per_pos = self.loss_fn(
            self.transformer,
            tokens.array,
            loss_weight.array,
            self.grug_config,
            mask=mask,
            reduction="none",
            logsumexp_weight=logsumexp_weight,
            loss_dtype=dtype,
        )
        loss = hax.named(per_pos, tokens.axes)

        return reduction(loss, axis=reduction_axis)

    def get_lm_head(self) -> hax.NamedArray:
        return hax.named(self.transformer.output_proj, (Axis("embed", self.transformer.config.hidden_dim), self.Vocab))

    def resize_vocab(self, new_size: int, key: PRNGKeyArray | None = None) -> "WrapperLMHeadModel":
        pass


def build_speedrun(model_cls, model_cfg, train_cfg, loss_fn, speedrun_name, speedrun_desc, author):
    model_cfg = WrapperConfig(
        model_cls_fn=lambda: model_cls,
        _total_trainable_params=model_cfg.total_trainable_params,
        _flops_per_token=model_cfg.flops_per_token,
        grug_config=model_cfg,
        loss_fn=loss_fn,
        max_seq_len=model_cfg.max_seq_len,
        hidden_dim=model_cfg.hidden_dim,
        num_heads=model_cfg.num_heads,
        num_kv_heads=model_cfg.num_kv_heads,
        intermediate_dim=model_cfg.intermediate_dim,
        num_layers=model_cfg.num_layers,
    )
    speedrun = SpeedrunConfig(
        author=author,
        description=speedrun_desc,
        model_config=model_cfg,
        train_config=train_cfg,
        tokenized_dataset=nemotron_cc_mixture,
    )
    return nemotron_only_speedrun(speedrun_name, speedrun)
