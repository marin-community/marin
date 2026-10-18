"""Convert a retained sweep checkpoint without resuming training."""

import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from levanter.compat.hf_checkpoints import HFCompatConfig, build_generation_config
from levanter.model_loading import load_levanter_checkpoint
from levanter.trainer import TrainerConfig, initialize

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    trainer: TrainerConfig
    model: HFCompatConfig
    reference_checkpoint: str
    tokenizer: str
    checkpoint_path: str
    output_path: str
    eos_token_ids: list[int] | None


def export_checkpoint(config: ExportConfig) -> None:
    """Restore native model weights and write a BF16 HF checkpoint."""
    initialize(config)
    converter = config.model.hf_checkpoint_converter().replaced(
        reference_checkpoint=config.reference_checkpoint, tokenizer=config.tokenizer
    )
    model_config = converter.config_from_hf_config(converter.default_hf_config)
    with config.trainer.use_device_mesh():
        vocab = round_axis_for_partitioning(Axis("vocab", len(converter.tokenizer)), config.trainer.compute_axis_mapping)
        model = load_levanter_checkpoint(
            model_config,
            config.checkpoint_path,
            Vocab=vocab,
            axis_mapping=config.trainer.parameter_axis_mapping,
            key=jax.random.PRNGKey(0),
        )
        converter.save_pretrained(
            model,
            config.output_path,
            dtype=jnp.bfloat16,
            generation_config=build_generation_config(converter.tokenizer, config.eos_token_ids),
            save_reference_code=False,
        )
    logger.info("Native-to-HF export completed: %s -> %s", config.checkpoint_path, config.output_path)
