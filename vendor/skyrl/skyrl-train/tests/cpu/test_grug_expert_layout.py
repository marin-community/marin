import pytest

from skyrl_train.grug_expert_layout import split_routed_experts


def test_layout_uses_routed_tensor_names_even_when_shared_experts_use_split_schema():
    keys = {
        "model.layers.0.mlp.experts.gate_proj.weight",
        "model.layers.0.shared_experts.0.gate_proj.weight",
    }

    assert not split_routed_experts(keys)
    assert split_routed_experts({"model.layers.0.mlp.experts.0.gate_proj.weight"})


@pytest.mark.parametrize(
    "keys",
    [set(), {"model.layers.0.mlp.experts.gate_proj.weight", "model.layers.0.mlp.experts.0.gate_proj.weight"}],
)
def test_layout_rejects_missing_or_mixed_routed_tensors(keys):
    with pytest.raises(ValueError, match="exactly one routed-expert weight layout"):
        split_routed_experts(keys)
