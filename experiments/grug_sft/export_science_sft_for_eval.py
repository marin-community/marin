# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export a frozen June Grug SFT checkpoint to the Snowball HF serving format."""

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
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.grug_sft.head_only_train import _add_qb_betas_to_residual, _apply_qb_betas
from experiments.grug_sft.hf_initialization import _copy_matching_parameters
from experiments.grug_sft.science_step38_model import MODEL_PATH, MODEL_REVISION, science_model_config
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig, Transformer

logger = logging.getLogger(__name__)

BASE_REPOSITORY = "open-athena/Snowball-67B-A2B-5.7T-Mixed-RLVR-Step38"
EXPORT_SHARD_BYTES = 5_000_000_000


def snowball_config(training: GrugModelConfig) -> SnowballConfig:
    """Project the trained architecture into the serving model's HF config."""
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


def serving_model(
    params: Transformer,
    pending_qb_betas: jax.Array,
    config: SnowballConfig,
    *,
    train_router_bias_residual: bool = False,
) -> SnowballLMHeadModel:
    """Convert the stacked training tree and materialize its effective router bias."""
    effective = (
        _add_qb_betas_to_residual(params, pending_qb_betas)
        if train_router_bias_residual
        else _apply_qb_betas(params, pending_qb_betas)
    )
    if effective.stacked_blocks is None:
        raise ValueError("The science SFT checkpoint must use stacked transformer blocks")
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


def export(
    checkpoint: str,
    output: str,
    expected_step: int,
    source_commit: str,
    *,
    train_router_bias_residual: bool = False,
) -> None:
    metadata = json.loads(StoragePath(prefix_join(checkpoint, "metadata.json")).read_text())
    if metadata["step"] != expected_step or metadata["is_temporary"] is not True:
        raise ValueError(f"Unexpected checkpoint metadata: {metadata}")
    if StoragePath(prefix_join(output, "model.safetensors.index.json")).exists():
        raise FileExistsError(f"HF export already exists at {output}")

    training_config = science_model_config(trainable_router_bias=train_router_bias_residual)
    config = snowball_config(training_config)
    mesh = compact_grug_mesh(expert_axis_size=8)
    with jax.set_mesh(mesh):
        template = eqx.filter_eval_shape(Transformer.init, training_config, key=jax.random.key(0))
        pending_template = jax.ShapeDtypeStruct((training_config.num_layers, training_config.num_experts), jnp.float32)
        loaded = load_checkpoint(
            {"params": template, "pending_qb_betas": pending_template},
            checkpoint,
            mesh=mesh,
            allow_partial=True,
        )
        model = serving_model(
            loaded["params"],
            loaded["pending_qb_betas"],
            config,
            train_router_bias_residual=train_router_bias_residual,
        )
        reference = RepoRef(BASE_REPOSITORY, MODEL_REVISION)
        converter = config.hf_checkpoint_converter(ref_checkpoint=reference)
        converter.save_pretrained(
            model,
            output,
            dtype=jnp.bfloat16,
            max_shard_size=EXPORT_SHARD_BYTES,
            save_reference_code=True,
            save_tokenizer=True,
        )

    # The generic converter omits Snowball-specific aliases and serving flags in
    # config.json. The SFT architecture is unchanged, so preserve the pinned base
    # artifact's exact serving config.
    base_config = StoragePath(prefix_join(MODEL_PATH, "config.json")).read_text()
    StoragePath(prefix_join(output, "config.json")).write_text(base_config)

    provenance = {
        "source_checkpoint": checkpoint,
        "source_step": expected_step,
        "source_timestamp": metadata["timestamp"],
        "source_checkout_commit": source_commit,
        "trainer_recipe_sha256": (
            hashlib.sha256(Path(__file__).with_name("science_rlvr1_step38.py").read_bytes()).hexdigest()
        ),
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_model": BASE_REPOSITORY,
        "reference_revision": MODEL_REVISION,
        "serving_config_sha256": hashlib.sha256(base_config.encode()).hexdigest(),
        "router_bias": (
            "learned residual plus centered negative pending_qb_betas"
            if train_router_bias_residual
            else "effective centered negative pending_qb_betas"
        ),
        "weight_dtype": "bfloat16",
    }
    StoragePath(prefix_join(output, "export-provenance.json")).write_text(json.dumps(provenance, indent=2) + "\n")
    logger.info("HF export complete: %s", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--train-router-bias-residual", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    export(
        args.checkpoint,
        args.output,
        args.expected_step,
        args.source_commit,
        train_router_bias_residual=args.train_router_bias_residual,
    )


if __name__ == "__main__":
    main()
