import json

import pytest
from transformers import AutoConfig

from skyrl_train.models.grug_moe import GrugMoeConfig


def tiny_config(**overrides) -> GrugMoeConfig:
    values = {
        "vocab_size": 48,
        "hidden_size": 32,
        "intermediate_size": 64,
        "shared_expert_intermediate_size": 48,
        "num_local_experts": 8,
        "num_experts_per_tok": 2,
        "num_hidden_layers": 5,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 12,
        "max_position_embeddings": 32,
        "sliding_window": 4,
        "qk_mult": 1.37,
        "initializer_range": 0.02,
    }
    values.update(overrides)
    return GrugMoeConfig(**values)


@pytest.fixture
def saved_grug_config(tmp_path):
    path = tmp_path / "config.json"
    config = tiny_config().to_dict()
    path.write_text(json.dumps(config))
    return path, config


@pytest.mark.parametrize(
    ("omitted", "named"),
    [
        (("qk_mult",), "qk_mult"),
        (("sliding_window",), "sliding_window"),
        (("max_position_embeddings", "max_seq_len"), "max_position_embeddings"),
        (("num_experts_per_tok", "num_experts_per_token"), "num_experts_per_tok"),
    ],
)
def test_grug_checkpoint_without_a_sampler_defaulted_key_fails_to_load(saved_grug_config, omitted, named):
    path, config = saved_grug_config
    for key in omitted:
        config.pop(key, None)
    path.write_text(json.dumps(config))

    with pytest.raises((TypeError, ValueError), match=named):
        AutoConfig.from_pretrained(path.parent)


@pytest.mark.parametrize(
    "theta",
    [
        {"rope_parameters": {"rope_type": "default", "rope_theta": 500000.0}},
        {"rope": {"theta": 500000.0}},
    ],
    ids=["rope_parameters", "rope"],
)
def test_grug_checkpoint_rope_theta_aliases_resolve_to_the_stored_theta(saved_grug_config, theta):
    path, config = saved_grug_config
    config.pop("rope_theta")
    config.update(theta)
    path.write_text(json.dumps(config))

    loaded = AutoConfig.from_pretrained(path.parent)
    assert loaded.rope_theta == 500000.0
    exported = path.parent / "exported"
    loaded.save_pretrained(exported)
    assert AutoConfig.from_pretrained(exported).rope_theta == 500000.0
