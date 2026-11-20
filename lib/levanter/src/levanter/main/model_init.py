# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import cast

import equinox as eqx
import jmp
from haliax.partitioning import named_jit

import haliax as hax
from levanter.checkpoint import latest_checkpoint_path, load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig
from levanter.field_dtype_overrides import hf_load_dtype
from levanter.models.lm_model import LmConfig, LmHeadModel, ModelPlacement
from levanter.model_loading import load_hf_checkpoint, load_levanter_checkpoint
from levanter.utils.jax_utils import use_cpu_device


@dataclass(frozen=True)
class ModelInitContext:
    model: LmConfig
    converter: HFCheckpointConverter | None
    use_hf_model_config: bool


def prepare_model_init_context(
    model: LmConfig,
    *,
    tokenizer,
    initialize_from_hf: bool | str,
    use_hf_model_config: bool,
) -> ModelInitContext:
    if not isinstance(model, HFCompatConfig):
        if initialize_from_hf:
            raise ValueError("initialize_from_hf requires a HF-compatible model configuration.")
        return ModelInitContext(model=model, converter=None, use_hf_model_config=use_hf_model_config)

    ref = initialize_from_hf if isinstance(initialize_from_hf, str) else None
    converter = model.hf_checkpoint_converter_for_source(ref, tokenizer, check_tokenizer=bool(initialize_from_hf))
    if initialize_from_hf and use_hf_model_config:
        model = model.with_hf_architecture(converter)

    return ModelInitContext(model=model, converter=converter, use_hf_model_config=use_hf_model_config)


def load_model_from_source(
    *,
    context: ModelInitContext,
    Vocab,
    model_key,
    parameter_axis_mapping,
    mp: jmp.Policy,
    placement: ModelPlacement | None = None,
    hf_ref: bool | str = False,
    checkpoint_path: str | None = None,
) -> LmHeadModel:
    if hf_ref and checkpoint_path is not None:
        raise ValueError("Specify only one of hf_ref or checkpoint_path.")

    if hf_ref:
        if context.converter is None:
            raise ValueError("HF model loading requires a HF-compatible model configuration.")

        if context.model.owns_parameter_placement:
            source = hf_ref if isinstance(hf_ref, str) else str(context.converter.reference_checkpoint)
            return mp.cast_to_param(
                load_hf_checkpoint(
                    context.model,
                    source,
                    axis_mapping=parameter_axis_mapping,
                    tokenizer=context.converter.tokenizer,
                    compute_dtype=hf_load_dtype(mp),
                    placement=placement,
                )
            )

        converter_config = None
        if not context.use_hf_model_config:
            converter_config = cast(HFCompatConfig, context.model)

        model = context.converter.load_pretrained(
            context.model.model_type,
            ref=hf_ref if isinstance(hf_ref, str) else None,
            config=converter_config,
            axis_mapping=parameter_axis_mapping,
            dtype=hf_load_dtype(mp),
        )
        return named_jit(mp.cast_to_param, parameter_axis_mapping)(model)

    if checkpoint_path is None:
        raise ValueError("Either hf_ref or checkpoint_path must be provided.")

    if context.model.owns_parameter_placement:
        return mp.cast_to_param(
            load_levanter_checkpoint(
                context.model,
                checkpoint_path,
                Vocab=Vocab,
                axis_mapping=parameter_axis_mapping,
                key=model_key,
                placement=placement,
            )
        )

    resolved_checkpoint_path = latest_checkpoint_path(checkpoint_path)
    with use_cpu_device():
        model = eqx.filter_eval_shape(context.model.build, Vocab, key=model_key)
        model = load_checkpoint(model, resolved_checkpoint_path, subpath="model")

    model = hax.shard(model, parameter_axis_mapping)
    return named_jit(mp.cast_to_param, parameter_axis_mapping)(model)
