# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export a Dr Doom science SFT checkpoint in the Snowball HF serving format."""

import argparse
import hashlib
import json
import logging
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from haliax import Axis
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import RepoRef
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from marin.datakit.sft import SftTokenStore
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.grug.science_sft.dr_doom import MODEL_PATH, MODEL_REPO, MODEL_REVISION, RECIPE
from experiments.grug.science_sft.hf_initialization import _copy_matching_parameters
from experiments.grug.science_sft.launch import build_run_config
from experiments.grug.science_sft.train import _apply_qb_betas
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer

logger = logging.getLogger(__name__)
EXPORT_SHARD_BYTES = 5_000_000_000


def _snowball_config(training: GrugModelConfig) -> SnowballConfig:
    return SnowballConfig(
        vocab_size=training.vocab_size,
        hidden_dim=training.hidden_dim,
        intermediate_dim=training.intermediate_dim,
        shared_expert_intermediate_dim=training.shared_expert_intermediate_dim,
        num_experts=training.num_experts,
        num_experts_per_token=training.num_experts_per_token,
        num_layers=training.num_layers,
        num_heads=training.num_heads,
        num_kv_heads=training.num_kv_heads,
        head_dim=training.head_dim,
        max_seq_len=training.max_seq_len,
        sliding_window=training.sliding_window,
        layer_norm_eps=training.layer_norm_eps,
        initializer_std=training.initializer_std,
        qk_mult=training.qk_mult,
        rope=training.rope,
    )


def _serving_model(params: Transformer, pending_qb_betas: jax.Array, config: SnowballConfig) -> SnowballLMHeadModel:
    effective = _apply_qb_betas(params, pending_qb_betas)
    if effective.stacked_blocks is None:
        raise ValueError("Expected stacked transformer blocks in the Grug checkpoint")
    blocks = tuple(effective.stacked_blocks.unstacked())
    unstacked = eqx.tree_at(lambda model: (model.blocks, model.stacked_blocks), effective, (blocks, None))
    template = eqx.filter_eval_shape(
        SnowballLMHeadModel.init,
        Axis("vocab", config.vocab_size),
        config,
        key=jax.random.key(0),
    )
    transformer = _copy_matching_parameters(unstacked, template.transformer)
    return SnowballLMHeadModel(transformer, config)


def export(checkpoint: str, store_path: str, version: str, output: str, expected_step: int, source_commit: str) -> None:
    """Load model weights and QB state, then write a regional HF checkpoint."""
    configure_coreweave_s3()
    metadata = json.loads(StoragePath(prefix_join(checkpoint, "metadata.json")).read_text())
    if metadata["step"] != expected_step:
        raise ValueError(f"Checkpoint step differs from expected step {expected_step}: {metadata['step']}")
    if StoragePath(prefix_join(output, "model.safetensors.index.json")).exists():
        raise FileExistsError(f"HF export already exists: {output}")
    store = SftTokenStore.raw_load(store_path)
    training = build_run_config(store, version, RECIPE).model
    config = _snowball_config(training)
    mesh = compact_grug_mesh(expert_axis_size=8)
    with jax.set_mesh(mesh):
        template = eqx.filter_eval_shape(Transformer.init, training, key=jax.random.key(0))
        pending = jax.ShapeDtypeStruct((training.num_layers, training.num_experts), jnp.float32)
        state = load_checkpoint(
            {"params": template, "pending_qb_betas": pending},
            checkpoint,
            mesh=mesh,
            allow_partial=True,
        )
        model = _serving_model(state["params"], state["pending_qb_betas"], config)
        converter = config.hf_checkpoint_converter(ref_checkpoint=RepoRef(MODEL_REPO, MODEL_REVISION))
        converter.save_pretrained(
            model,
            output,
            dtype=jnp.bfloat16,
            max_shard_size=EXPORT_SHARD_BYTES,
            save_reference_code=True,
            save_tokenizer=True,
        )

    # Preserve the source's full serving context and model-specific flags.
    base_config = StoragePath(prefix_join(MODEL_PATH, "config.json")).read_text()
    StoragePath(prefix_join(output, "config.json")).write_text(base_config)
    provenance = {
        "source_checkpoint": checkpoint,
        "source_step": expected_step,
        "source_checkout_commit": source_commit,
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_model": MODEL_REPO,
        "reference_revision": MODEL_REVISION,
        "serving_config_sha256": hashlib.sha256(base_config.encode()).hexdigest(),
        "router_bias": "effective centered negative pending_qb_betas",
        "weight_dtype": "bfloat16",
    }
    StoragePath(prefix_join(output, "export-provenance.json")).write_text(json.dumps(provenance, indent=2) + "\n")
    logger.info("HF export complete: %s", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--store-path", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    export(args.checkpoint, args.store_path, args.version, args.output, args.expected_step, args.source_commit)


if __name__ == "__main__":
    main()
