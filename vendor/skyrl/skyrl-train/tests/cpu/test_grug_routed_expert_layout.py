from skyrl_train.models.grug_moe import (
    GRUG_SPLIT_EXPERT_SCHEMA_VERSION,
    GRUG_STACKED_EXPERT_SCHEMA_VERSION,
    GrugMoeConfig,
    select_routed_expert_schema,
)


def test_schema_two_shared_experts_with_stacked_routed_weights_selects_stacked_mapping():
    config = GrugMoeConfig(
        grugmoe_artifact_schema_version=GRUG_SPLIT_EXPERT_SCHEMA_VERSION,
        num_shared_experts=2,
    )
    weight_keys = {
        "model.layers.0.shared_experts.0.gate_proj.weight",
        "model.layers.0.shared_experts.1.gate_proj.weight",
        "model.layers.0.mlp.experts.gate_proj.weight",
    }

    assert config.uses_hero_architecture
    assert select_routed_expert_schema(config.grugmoe_artifact_schema_version, weight_keys) == (
        GRUG_STACKED_EXPERT_SCHEMA_VERSION
    )
