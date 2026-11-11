"""Detect the routed-expert tensor layout in a Grug HF checkpoint."""


def split_routed_experts(hf_keys: set[str]) -> bool:
    """Return whether routed experts are individually saved; reject ambiguous layouts."""
    stacked = "model.layers.0.mlp.experts.gate_proj.weight" in hf_keys
    split = "model.layers.0.mlp.experts.0.gate_proj.weight" in hf_keys
    if stacked == split:
        raise ValueError("Grug checkpoint must have exactly one routed-expert weight layout")
    return split
