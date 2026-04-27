# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import cast

import equinox as eqx
from haliax.partitioning import named_jit

import haliax as hax
from levanter.checkpoint import latest_checkpoint_path, load_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig
from levanter.models.lm_model import LmConfig, LmHeadModel
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
    if initialize_from_hf:
        if not isinstance(model, HFCompatConfig):
            raise ValueError("initialize_from_hf requires a HF-compatible model configuration.")

        converter = model.hf_checkpoint_converter()
        if hasattr(tokenizer, "vocab") and tokenizer.vocab != converter.tokenizer.vocab:
            converter = converter.replaced(tokenizer=tokenizer)
        else:
            converter = converter.replaced(tokenizer=tokenizer)

        if isinstance(initialize_from_hf, str):
            converter = converter.replaced(reference_checkpoint=initialize_from_hf)

        if use_hf_model_config:
            model = converter.config_from_hf_config(converter.default_hf_config)
    elif isinstance(model, HFCompatConfig):
        converter = model.hf_checkpoint_converter()
        converter = converter.replaced(tokenizer=tokenizer)
    else:
        converter = None

    return ModelInitContext(model=model, converter=converter, use_hf_model_config=use_hf_model_config)


def load_model_from_source(
    *,
    context: ModelInitContext,
    Vocab,
    model_key,
    parameter_axis_mapping,
    compute_dtype,
    cast_to_param,
    hf_ref: bool | str = False,
    checkpoint_path: str | None = None,
) -> LmHeadModel:
    if hf_ref and checkpoint_path is not None:
        raise ValueError("Specify only one of hf_ref or checkpoint_path.")

    if hf_ref:
        if context.converter is None:
            raise ValueError("HF model loading requires a HF-compatible model configuration.")

        converter_config = None
        if not context.use_hf_model_config:
            converter_config = cast(HFCompatConfig, context.model)

        model = context.converter.load_pretrained(
            context.model.model_type,
            ref=hf_ref if isinstance(hf_ref, str) else None,
            config=converter_config,
            axis_mapping=parameter_axis_mapping,
            dtype=compute_dtype,
        )
        return named_jit(cast_to_param, parameter_axis_mapping)(model)

    if checkpoint_path is None:
        raise ValueError("Either hf_ref or checkpoint_path must be provided.")

    resolved_checkpoint_path = latest_checkpoint_path(checkpoint_path)
    with use_cpu_device():
        model = eqx.filter_eval_shape(context.model.build, Vocab, key=model_key)
        model = load_checkpoint(model, resolved_checkpoint_path, subpath="model")

    model = hax.shard(model, parameter_axis_mapping)
    return named_jit(cast_to_param, parameter_axis_mapping)(model)
