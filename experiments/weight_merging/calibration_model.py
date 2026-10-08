# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug forward passes over frozen checkpoint banks and learned coefficients."""

import re
from dataclasses import dataclass
from importlib import import_module

import torch
from marin.merging.checkpoint import CheckpointReader, CheckpointSource
from marin.merging.learned import differentiable_weight_blend
from torch.func import functional_call


@dataclass(frozen=True)
class CalibrationOutput:
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...]


def coefficient_group(name: str) -> str | None:
    match = re.match(r"model.layers.(\d+).(.*)", name)
    if match is None:
        return None
    layer, suffix = match.groups()
    for prefix, component in (
        ("self_attn.", "attention"),
        ("shared_expert.", "shared"),
        ("mlp.experts.", "routed"),
        ("mlp.router.", "router"),
    ):
        if suffix.startswith(prefix):
            return f"layer_{int(layer):02d}_{component}"
    return None


class FrozenGrugBank:
    """Keep source checkpoints on CPU and execute their existing Grug modules.

    Sources are ordered Step92, Step12, Step20. Calibration uses the same token
    IDs for all teachers. Non-component weights use the Step92 anchor.
    """

    def __init__(self, sources: list[CheckpointSource], config: dict, devices: tuple[torch.device, ...]):
        if len(sources) != 3 or not devices:
            raise ValueError("Provide Step92, Step12, Step20 and at least one device")
        grug = import_module("skyrl_train.models.grug_moe")
        model_config = grug.GrugMoeConfig(**config)
        model_config._attn_implementation = "eager"
        with torch.device("meta"):
            self.model = grug.GrugMoeForCausalLM(model_config).eval().requires_grad_(False)
        self.layer_count = config["num_hidden_layers"]
        self.long_layers = grug.grug_long_layer_flags(self.layer_count)
        self.devices = devices
        self.states: list[dict[str, torch.Tensor]] = []
        expected = set(self.model.state_dict())
        for source in sources:
            reader = CheckpointReader(source)
            if set(reader.weight_map) != expected:
                raise ValueError(f"Grug implementation and checkpoint keys differ: {source.path}")
            state = {}
            for name in sorted(expected, key=lambda key: (reader.weight_map[key], key)):
                tensor = reader.tensor(name)
                if not torch.isfinite(tensor).all():
                    raise ValueError(f"Nonfinite checkpoint weight: {name}")
                state[name] = tensor.detach().contiguous()
            self.states.append(state)
        self.module_keys = {}
        prefixes = [
            "model.embed_tokens",
            "model.embed_norm",
            "model.embed_gated_norm",
            "model.norm",
            "model.final_gated_norm",
            "lm_head",
        ]
        prefixes.extend(f"model.layers.{layer}" for layer in range(self.layer_count))
        for prefix in prefixes:
            self.module_keys[prefix] = tuple(name for name in expected if name.startswith(prefix + "."))

    def device_for(self, name: str) -> torch.device:
        match = re.match(r"model.layers.(\d+).", name + ".")
        if match is not None:
            index = int(match[1]) * len(self.devices) // self.layer_count
            return self.devices[index]
        if name.startswith(("model.embed_tokens", "model.embed_norm", "model.embed_gated_norm")):
            return self.devices[0]
        return self.devices[-1]

    def frozen_weights(self, source: int) -> dict[str, torch.Tensor]:
        return {name: tensor.to(self.device_for(name)) for name, tensor in self.states[source].items()}

    def learned_weights(self, coefficients: torch.nn.ParameterDict, block_elements: int) -> dict[str, torch.Tensor]:
        result = {}
        for name, anchor in self.states[0].items():
            group = coefficient_group(name)
            if group is None:
                result[name] = anchor.to(self.device_for(name))
            else:
                result[name] = differentiable_weight_blend(
                    anchor,
                    (self.states[1][name], self.states[2][name]),
                    coefficients[group],
                    block_elements=block_elements,
                )
        return result

    def forward(
        self, weights: dict[str, torch.Tensor], tokens: torch.Tensor, logit_positions: torch.Tensor
    ) -> CalibrationOutput:
        def call(prefix: str, *args):
            device = self.device_for(prefix)
            moved = tuple(arg.to(device) if isinstance(arg, torch.Tensor) else arg for arg in args)
            parameters = {name[len(prefix) + 1 :]: weights[name] for name in self.module_keys[prefix]}
            return functional_call(self.model.get_submodule(prefix), parameters, moved, strict=True)

        hidden = call("model.embed_tokens", tokens)
        hidden = call("model.embed_norm", hidden)
        hidden = call("model.embed_gated_norm", hidden)
        positions = torch.arange(tokens.shape[1]).unsqueeze(0)
        states = []
        for layer in range(self.layer_count):
            hidden = call(f"model.layers.{layer}", hidden, None, positions, self.long_layers[layer])
            states.append(hidden)
        hidden = call("model.norm", hidden)
        hidden = call("model.final_gated_norm", hidden)
        selected = hidden.index_select(1, logit_positions.to(hidden.device))
        logits = call("lm_head", selected)
        return CalibrationOutput(logits=logits, hidden_states=tuple(states))

    def initial_coefficients(
        self, chunks: dict[str, int], initial: dict[str, tuple[float, float]]
    ) -> torch.nn.ParameterDict:
        coefficients = torch.nn.ParameterDict()
        for name in self.states[0]:
            group = coefficient_group(name)
            if group is not None and group not in coefficients:
                values = torch.tensor(initial[group.rsplit("_", 1)[1]], device=self.device_for(name))
                coefficients[group] = torch.nn.Parameter(values.repeat(chunks.get(group, 1), 1))
        return coefficients
