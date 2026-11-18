import pytest

from skyrl_train.models.grug_moe import (
    GRUG_SPLIT_EXPERT_SCHEMA_VERSION,
    GRUG_STACKED_EXPERT_SCHEMA_VERSION,
    GrugMoeConfig,
    select_routed_expert_schema,
)


@pytest.mark.parametrize(
    ("routed_key", "expected_schema"),
    [
        ("model.layers.0.mlp.experts.gate_proj.weight", GRUG_STACKED_EXPERT_SCHEMA_VERSION),
        ("model.layers.0.mlp.experts.0.gate_proj.weight", GRUG_SPLIT_EXPERT_SCHEMA_VERSION),
    ],
)
def test_schema_two_shared_experts_select_the_observed_routed_mapping(routed_key, expected_schema):
    config = GrugMoeConfig(
        grugmoe_artifact_schema_version=GRUG_SPLIT_EXPERT_SCHEMA_VERSION,
        num_shared_experts=2,
        qk_mult=1.5703274004183786,
        sliding_window=2048,
        max_position_embeddings=65536,
        num_experts_per_tok=4,
    )
    weight_keys = {
        "model.layers.0.shared_experts.0.gate_proj.weight",
        "model.layers.0.shared_experts.1.gate_proj.weight",
        routed_key,
    }

    assert config.uses_hero_architecture
    assert select_routed_expert_schema(config.grugmoe_artifact_schema_version, weight_keys) == expected_schema


@pytest.mark.parametrize(
    "weight_keys",
    [set(), {"model.layers.0.mlp.experts.gate_proj.weight", "model.layers.0.mlp.experts.0.gate_proj.weight"}],
)
def test_missing_or_mixed_routed_tensors_are_rejected(weight_keys):
    with pytest.raises(ValueError):
        select_routed_expert_schema(GRUG_SPLIT_EXPERT_SCHEMA_VERSION, weight_keys)
