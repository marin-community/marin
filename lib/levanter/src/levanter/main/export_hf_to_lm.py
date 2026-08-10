# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Script to import HuggingFace models and save them as Levanter Tensorstore checkpoints.
"""

import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional, Union

import jax.numpy as jnp
from jax.experimental.array_serialization.serialization import GlobalAsyncCheckpointManager

from rigging.filesystem import StoragePath, prefix_join

import levanter
import levanter.config
from levanter.checkpoint import save_checkpoint
from levanter.compat.hf_checkpoints import RepoRef, load_tokenizer
from levanter.models.llama import LlamaConfig
from levanter.models.lm_model import LmConfig
from levanter.utils.jax_utils import use_cpu_device

logger = logging.getLogger(__name__)


@dataclass
class ImportHfConfig:
    """Configuration for importing HuggingFace models to Levanter checkpoints."""

    hf_checkpoint: Union[str, RepoRef]
    """HF repository reference or path (e.g., 'meta-llama/Llama-2-7b-hf' or 'gs://bucket/path')"""

    output_path: str
    """Output path for Levanter checkpoint (local or cloud, e.g., 'gs://bucket/checkpoints/model')"""

    model: LmConfig = field(default_factory=LlamaConfig)
    """Levanter model configuration"""

    use_hf_model_config: bool = True
    """If True, use the model configuration from the HF checkpoint instead of the provided model config"""

    tokenizer: Optional[str] = None
    """Override tokenizer path/name. If None, will use tokenizer from HF checkpoint"""

    tokenizer_revision: Optional[str] = None
    """Commit pin for the tokenizer, when it is a hub id (for reproducibility)."""

    dtype: Optional[str] = "bfloat16"
    """Target dtype for the saved checkpoint (e.g., 'float32', 'bfloat16', 'float16')"""

    resize_vocab_to_match_tokenizer: bool = False
    """If True, resize model vocab to match tokenizer vocab size. Defaults to False because many models
    (e.g., Qwen) intentionally pad their embedding matrices beyond the tokenizer vocab size for hardware efficiency."""

    subpath: Optional[str] = None
    """Subtree to save the model under (e.g. 'model', matching a TrainerState's model field, so it loads
    with load_checkpoint(..., subpath='model')). None saves the bare model at the checkpoint root."""

    emit_padded_tokenizer: bool = False
    """If True, also write a tokenizer padded to the model's vocab size to output_path, so a downstream
    run that builds its Vocab axis from len(tokenizer) matches the checkpoint's embedding axis. Padding
    appends never-emitted dummy tokens, so text tokenizes unchanged."""


def _coerce_to_repo_ref(checkpoint: Union[str, RepoRef]) -> RepoRef:
    """Convert string or RepoRef to RepoRef."""
    if isinstance(checkpoint, str):
        return RepoRef.from_string(checkpoint)
    return checkpoint


def main(config: ImportHfConfig):
    """Main function to import HF model and save as Levanter checkpoint."""
    logger.setLevel(logging.INFO)

    start_time = time.time()

    logger.info("Starting HF to Levanter conversion")
    logger.info(f"Source: {config.hf_checkpoint}")
    logger.info(f"Target: {config.output_path}")
    logger.info(f"Model type: {config.model.__class__.__name__}")
    logger.info(f"Use HF config: {config.use_hf_model_config}")
    logger.info(f"Target dtype: {config.dtype}")

    hf_checkpoint = _coerce_to_repo_ref(config.hf_checkpoint)

    tokenizer_path = config.tokenizer
    if tokenizer_path is None:
        tokenizer_path = hf_checkpoint.model_name_or_path

    logger.info(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = load_tokenizer(tokenizer_path, revision=config.tokenizer_revision)

    logger.info("Setting up HF checkpoint converter")
    converter = config.model.hf_checkpoint_converter()
    converter = converter.replaced(reference_checkpoint=hf_checkpoint, tokenizer=tokenizer)

    logger.info("Loading HF model...")
    with use_cpu_device():
        model = converter.load_pretrained(
            config.model.model_type,
            config=config.model if not config.use_hf_model_config else None,
            dtype=getattr(jnp, config.dtype) if config.dtype else None,
            resize_vocab_to_match_tokenizer=config.resize_vocab_to_match_tokenizer,
        )

    StoragePath(config.output_path).mkdirs()
    logger.info(f"Output directory ready: {config.output_path}")

    logger.info(f"Saving checkpoint to Tensorstore format: {config.output_path}")

    manager = GlobalAsyncCheckpointManager()

    def commit_callback():
        elapsed = time.time() - start_time
        logger.info(f"Checkpoint committed to Tensorstore successfully! Total time: {elapsed:.2f}s")

    # A subpath saves the model as a subtree (e.g. `model/`), matching a TrainerState's model field so
    # it loads with load_checkpoint(..., subpath=...); None saves the bare model at the root.
    tree = {config.subpath: model} if config.subpath is not None else model
    save_checkpoint(
        tree=tree,
        checkpoint_path=config.output_path,
        manager=manager,
        commit_callback=commit_callback,
        step=0,
        is_temporary=False,
    )
    # Block until the async commit lands, so a caller (or an artifact record) never observes an
    # incomplete checkpoint after main() returns.
    manager.wait_until_finished()

    if config.emit_padded_tokenizer:
        padded_tokenizer = converter.with_tokenizer_padded_to_match_model().tokenizer
        with tempfile.TemporaryDirectory() as tokenizer_dir:
            padded_tokenizer.save_pretrained(tokenizer_dir)
            for name in os.listdir(tokenizer_dir):
                if name.startswith("."):
                    continue
                StoragePath(prefix_join(config.output_path, name)).upload_from(os.path.join(tokenizer_dir, name))
        logger.info(f"Emitted padded tokenizer (vocab {len(padded_tokenizer)}) to {config.output_path}")

    logger.info("Conversion completed successfully!")


if __name__ == "__main__":
    levanter.config.main(main)()
