# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax import Axis
from safetensors.numpy import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from levanter.compat.hf_checkpoints import HFCheckpointConverter

from levanter.grug.sharding import compact_grug_mesh
from levanter.models.hero import HeroConfig
from levanter.models.hero_model import HeroLMHeadModel
from levanter.models.snowball import SnowballConfig


@pytest.mark.parametrize("rope_fused", [False, True])
def test_hero_hf_config_preserves_architecture(rope_fused):
    cfg = dataclasses.replace(HeroConfig(), rope_fused=rope_fused)
    assert HeroConfig.from_hf_config(cfg.to_hf_config(cfg.vocab_size)) == cfg


def test_hero_and_snowball_reject_each_others_checkpoint_recipe():
    hero = HeroConfig().to_hf_config(128)
    snowball = SnowballConfig().to_hf_config(128)
    assert HeroConfig.matches_hf_config(hero) and not HeroConfig.matches_hf_config(snowball)
    assert SnowballConfig.matches_hf_config(snowball) and not SnowballConfig.matches_hf_config(hero)
    with pytest.raises(ValueError, match="schema-v2"):
        HeroConfig.from_hf_config(snowball)
    with pytest.raises(ValueError):
        SnowballConfig.from_hf_config(hero)


@pytest.mark.parametrize("config", [HeroConfig(), SnowballConfig()])
def test_hf_checkpoint_discovery_distinguishes_grug_recipes(tmp_path, config):
    config.to_hf_config(4).save_pretrained(tmp_path)
    _save_tiny_tokenizer(tmp_path)
    converter = HFCheckpointConverter.from_hf(str(tmp_path))
    assert converter.LevConfigClass is type(config)
    restored = converter.LevConfigClass.from_hf_config(converter.default_hf_config)
    assert restored == dataclasses.replace(config, vocab_size=4)


def _save_tiny_tokenizer(tmp_path):
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1, "a": 2, "b": 3}, unk_token="<unk>")),
        unk_token="<unk>",
        eos_token="<eos>",
    )
    tokenizer.save_pretrained(tmp_path)


def test_hero_checkpoint_load_preserves_logits(tmp_path):
    cfg = HeroConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=12,
        shared_expert_intermediate_dim=12,
        num_shared_experts=2,
        num_experts=4,
        num_experts_per_token=2,
        latent_dim=8,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        local_kv_heads=2,
        global_kv_heads=1,
        head_dim=8,
        max_seq_len=16,
        sliding_window=3,
        global_every=2,
        initializer_std=0.15,
        sconv_kernel=3,
        attention_implementation="reference",
        moe_implementation="ring",
    )
    cfg.to_hf_config(cfg.vocab_size).save_pretrained(tmp_path)
    _save_tiny_tokenizer(tmp_path)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = HeroLMHeadModel.init(Axis("vocab", cfg.vocab_size), cfg, key=jax.random.key(21))
        save_file(
            {name: np.asarray(value) for name, value in model.to_state_dict().items()}, tmp_path / "model.safetensors"
        )
        converter = HFCheckpointConverter.from_hf(str(tmp_path))
        loaded = converter.load_pretrained(HeroLMHeadModel, config=cfg, dtype=jnp.float32)
        tokens = hax.named(
            jnp.broadcast_to(jnp.arange(5, dtype=jnp.int32), (jax.device_count(), 5)),
            (Axis("batch", jax.device_count()), Axis("position", 5)),
        )
        logits = hax.named_jit(lambda m, t: m(t))
        np.testing.assert_array_equal(
            np.asarray(logits(model, tokens).array), np.asarray(logits(loaded, tokens).array)
        )
