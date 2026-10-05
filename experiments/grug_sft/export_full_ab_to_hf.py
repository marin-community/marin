# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export the finished full A/B Grug SFT checkpoint to Hugging Face format."""

import argparse
import json
import logging
from pathlib import Path

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
from huggingface_hub import hf_hub_download
from iris.jax.init import initialize_jax
from levanter.checkpoint import load_checkpoint
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.model import GrugModelConfig as InferenceConfig
from experiments.grug.moe.model import GrugMoeHfConfig
from experiments.grug.moe.model import Transformer as InferenceTransformer
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig as TrainingConfig
from experiments.june_tpu_67b_a2b.moe.model import Transformer as TrainingTransformer

logger = logging.getLogger(__name__)
REFERENCE = "open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21"
MODEL_CONFIG = Path(__file__).with_name("full_ab_export_model_config.json")
CONTEXT_AXIS_SIZE = 4


def inference_model(training_model: TrainingTransformer, config: InferenceConfig) -> InferenceTransformer:
    """Expose the stacked training layers to the established Grug HF serializer."""
    assert training_model.stacked_blocks is not None
    return InferenceTransformer(
        token_embed=training_model.token_embed,
        embed_norm=training_model.embed_norm,
        embed_gated_norm=training_model.embed_gated_norm,
        output_proj=training_model.output_proj,
        blocks=tuple(training_model.stacked_blocks.unstacked()),
        final_norm=training_model.final_norm,
        final_gated_norm=training_model.final_gated_norm,
        config=config,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    initialize_jax()
    training_config = draccus.decode(TrainingConfig, json.loads(MODEL_CONFIG.read_text()))
    inference_config = InferenceConfig.from_hf_config(GrugMoeHfConfig.from_pretrained(REFERENCE))
    mesh = compact_grug_mesh(replica_axis_size=1, context_axis_size=CONTEXT_AXIS_SIZE)
    with jax.set_mesh(mesh):
        template = eqx.filter_eval_shape(TrainingTransformer.init, training_config, key=jax.random.PRNGKey(0))
        checkpoint = load_checkpoint(
            {
                "params": template,
                "pending_qb_betas": jax.ShapeDtypeStruct(
                    (training_config.num_layers, training_config.num_experts), jnp.float32
                ),
            },
            args.checkpoint,
            mesh=mesh,
        )
        jax.block_until_ready(checkpoint)
        router_bias = -checkpoint["pending_qb_betas"]
        router_bias -= jnp.mean(router_bias, axis=-1, keepdims=True)
        model = eqx.tree_at(
            lambda tree: tree.stacked_blocks.stacked.mlp.router_bias,
            checkpoint["params"],
            router_bias,
        )
        exported_model = inference_model(model, inference_config)
        logger.info("Loaded checkpoint and applied deferred QB router biases")
        chat_template = Path(hf_hub_download(REFERENCE, "chat_template.jinja")).read_text()
        converter = inference_config.hf_checkpoint_converter(REFERENCE)
        converter.save_pretrained(
            exported_model,
            args.output,
            save_reference_code=False,
            dtype=jnp.bfloat16,
            max_concurrent_shards=1,
            chat_template=chat_template,
            generation_config={"eos_token_id": [128001, 128009], "bos_token_id": 128000},
        )
        logger.info("Export complete: %s", args.output)


if __name__ == "__main__":
    main()
