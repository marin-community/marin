import dataclasses
import math
from typing import Any, cast

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
from haliax.partitioning import set_mesh
from levanter.checkpoint import load_checkpoint
from levanter.grug.sharding import compact_grug_mesh
from levanter.tokenizers import load_tokenizer

from experiments.grug.moe.model import GrugModelConfig, Transformer
from experiments.june_tpu_67b_a2b.moe.heuristic_muonh import MoeMuonHHeuristic
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig as VendoredGrugModelConfig
from experiments.june_tpu_67b_a2b.moe.model import Transformer as VendoredTransformer
from experiments.marin_tokenizer import MARIN_CHAT_TEMPLATE

CHECKPOINT = (
    "s3://marin-us-east-02a/marin/grug/"
    "snowball_step105149_sft_s2_thinking/"
    "2026.08.13.1/checkpoints/step-630/"
)
OUTPUT = (
    "s3://marin-us-east-02a/marin/exports/grug/"
    "snowball_step105149_sft_s2_thinking/2026.08.13.1/step-630/hf-bf16-vllm/"
)

qk_mult = 1.3 * (0.1 * math.log(65_536 / 8_192) + 1.0)
model_config = dataclasses.replace(
    MoeMuonHHeuristic(min_lr_ratio=0.05).build_model_config(2560, seq_len=65_536),
    disable_pko=True,
    disable_long_rope=True,
    sliding_window=2048,
    use_array_stacked_blocks=True,
    qk_mult=qk_mult,
    max_seq_len=32_768,
    attention_implementation="gpu_fa4_cute",
    ce_implementation="batched_xla",
)
model_dict = dataclasses.asdict(model_config)
vendored_config = draccus.decode(VendoredGrugModelConfig, model_dict)
main_fields = {field.name for field in dataclasses.fields(GrugModelConfig)}
main_config = draccus.decode(GrugModelConfig, {key: value for key, value in model_dict.items() if key in main_fields})

mesh = compact_grug_mesh()
with set_mesh(mesh):
    template = eqx.filter_eval_shape(VendoredTransformer.init, vendored_config, key=jax.random.PRNGKey(0))
    state = load_checkpoint(
        {
            "params": template,
            "pending_qb_betas": jax.ShapeDtypeStruct(
                (vendored_config.num_layers, vendored_config.num_experts), jnp.float32
            ),
        },
        CHECKPOINT,
        mesh=mesh,
    )
    params = state["params"]
    pending_qb_betas = state["pending_qb_betas"]
    del state
    assert params.stacked_blocks is not None
    router_bias = -pending_qb_betas
    router_bias -= jnp.mean(router_bias, axis=-1, keepdims=True)
    params = eqx.tree_at(lambda tree: tree.stacked_blocks.stacked.mlp.router_bias, params, router_bias)
    del pending_qb_betas
    params = jax.tree.map(
        lambda value: value.astype(jnp.bfloat16) if eqx.is_inexact_array(value) else value,
        params,
    )
    jax.block_until_ready(params)

    source = cast(Any, params)
    export_model = Transformer(
        token_embed=source.token_embed,
        embed_norm=source.embed_norm,
        embed_gated_norm=source.embed_gated_norm,
        output_proj=source.output_proj,
        blocks=tuple(source.stacked_blocks.unstacked()),
        final_norm=source.final_norm,
        final_gated_norm=source.final_gated_norm,
        config=main_config,
    )
    tokenizer = load_tokenizer("marin-community/marin-tokenizer")
    converter = main_config.hf_checkpoint_converter().replaced(tokenizer=tokenizer).with_config_overrides({"dtype": "bfloat16"})
    converter.save_pretrained(
        export_model,
        OUTPUT,
        dtype=jnp.bfloat16,
        generation_config={
            "bos_token_id": 128000,
            "eos_token_id": [128001, 128009],
            "pad_token_id": 128001,
        },
        chat_template=MARIN_CHAT_TEMPLATE,
    )
