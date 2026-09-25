# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The streamed export must match the Grug model's canonical HF state dict."""

import dataclasses
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh
from levanter.checkpoint import save_checkpoint
from levanter.grug.sharding import compact_grug_mesh
from safetensors import safe_open
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from experiments.grug.moe.model import GrugModelConfig as ExportConfig
from experiments.grug.moe.model import Transformer as ExportTransformer
from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig as TrainerConfig
from experiments.june_tpu_67b_a2b.moe.model import Transformer as TrainerTransformer
from experiments.post_training.curriculum_sft.hf_export_streaming import export_checkpoint_streaming


def test_streamed_export_matches_grug_state_dict(tmp_path: Path):
    trainer_config = TrainerConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=24,
        shared_expert_intermediate_dim=16,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=3,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=64,
        use_array_stacked_blocks=True,
        disable_pko=True,
        moe_implementation="ring",
    )
    export_fields = {field.name for field in dataclasses.fields(ExportConfig)}
    export_config = ExportConfig(
        **{
            field.name: getattr(trainer_config, field.name)
            for field in dataclasses.fields(TrainerConfig)
            if field.name in export_fields
        }
    )
    mesh = compact_grug_mesh(expert_axis_size=1)
    with set_mesh(mesh):
        params = TrainerTransformer.init(trainer_config, key=jax.random.PRNGKey(7))
        pending = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 10
        checkpoint = tmp_path / "checkpoint"
        save_checkpoint({"params": params, "pending_qb_betas": pending}, 0, checkpoint, is_temporary=False)

        bias = -pending
        bias -= bias.mean(axis=-1, keepdims=True)
        params = eqx.tree_at(lambda tree: tree.stacked_blocks.stacked.mlp.router_bias, params, bias)
        assert params.stacked_blocks is not None
        reference = ExportTransformer(
            token_embed=params.token_embed,
            embed_norm=params.embed_norm,
            embed_gated_norm=params.embed_gated_norm,
            output_proj=params.output_proj,
            blocks=tuple(params.stacked_blocks.unstacked()),
            final_norm=params.final_norm,
            final_gated_norm=params.final_gated_norm,
            config=export_config,
        ).to_state_dict()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2}, unk_token="[UNK]")),
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    output = tmp_path / "output"
    local = tmp_path / "local"
    output.mkdir()
    local.mkdir()
    export_checkpoint_streaming(str(checkpoint), str(output), export_config, tokenizer, local, max_shard_size=2048)

    actual = {}
    for path in output.glob("*.safetensors"):
        with safe_open(path, framework="numpy") as shard:
            actual.update({name: shard.get_tensor(name) for name in shard.keys()})
    assert actual.keys() == reference.keys()
    for name, value in reference.items():
        expected = np.asarray(value.astype(jnp.bfloat16))
        np.testing.assert_array_equal(actual[name].view(np.uint16), expected.view(np.uint16), err_msg=name)

    index = json.loads((output / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(reference)
    assert set(index["weight_map"].values()) == {path.name for path in output.glob("*.safetensors")}

    config = json.loads((output / "config.json").read_text())
    assert config["architectures"] == ["GrugMoeForCausalLM"]
    assert config["dtype"] == "bfloat16"
    assert config["bos_token_id"] == 1
    assert config["eos_token_id"] == 2
