# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import logging
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Optional

import equinox as eqx
import haliax
import jax

from haliax import Axis

import levanter
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import RepoRef, load_tokenizer, HFCompatConfig
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig, LmHeadModel
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import is_inexact_arrayish, local_cpu_mesh

logger = logging.getLogger(__name__)


@dataclass
class ConvertLmConfig:
    trainer: TrainerConfig
    checkpoint_path: str
    output_dir: str
    upload_to_hf: Optional[RepoRef] = None  # if specified, attempt to upload this checkpoint to the hf hub

    model: LmConfig = LlamaConfig()
    save_tokenizer: bool = True  # if True, save the tokenizer to the output directory
    tokenizer: str | None = None  # if specified, use this tokenizer instead of the one from the model config
    override_vocab_size: Optional[int] = None  # if specified, override the vocab size in the config
    config_overrides: Optional[dict] = None  # if specified, override the config with these values

    use_cpu: bool = False


def main(config: ConvertLmConfig):
    import levanter.utils.logging as logging_utils

    logging_utils.init_logging("logs", "export")
    tokenizer_spec = config.tokenizer
    if tokenizer_spec is None:
        assert isinstance(config.model, HFCompatConfig)
        tokenizer = config.model.hf_checkpoint_converter().tokenizer
    else:
        tokenizer = load_tokenizer(tokenizer_spec)

    vocab_size = config.override_vocab_size or len(tokenizer)
    Vocab = Axis("vocab", vocab_size)

    key = jax.random.PRNGKey(0)

    exit_stack = ExitStack()
    if config.use_cpu:
        exit_stack.enter_context(local_cpu_mesh())
    else:
        # exit_stack.enter_context(Mesh(jax.local_devices(), "dev"))
        exit_stack.enter_context(config.trainer.device_mesh)
        exit_stack.enter_context(haliax.axis_mapping(config.trainer.parameter_axis_mapping))

    with exit_stack:
        model: LmHeadModel = eqx.filter_eval_shape(config.model.build, Vocab, key=key)
        trainable, non_trainable = eqx.partition(model, is_inexact_arrayish)
        # TODO: don't load the entire checkpoint into CPU memory when we only need our share of the model
        logger.info(f"Loading checkpoint from {config.checkpoint_path}...")
        trainable = load_checkpoint(trainable, config.checkpoint_path, subpath="model")

        assert trainable is not None
        model = eqx.combine(trainable, non_trainable)

        if config.override_vocab_size:
            model = model.resize_vocab(config.override_vocab_size)

        converter = model.config.hf_checkpoint_converter()
        if config.tokenizer:
            converter = converter.replaced(tokenizer=tokenizer)

        logger.info(f"Converting {config.checkpoint_path}...")

        converter.save_pretrained(
            model,
            config.output_dir,
            upload_to_hf=config.upload_to_hf or False,
            save_tokenizer=config.save_tokenizer,
        )


if __name__ == "__main__":
    levanter.config.main(main)()
