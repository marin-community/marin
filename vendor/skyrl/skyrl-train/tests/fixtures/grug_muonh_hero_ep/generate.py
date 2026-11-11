"""Freeze three updates from Hero EP at Marin 4af966070e5d0252e87d404196400c35bd04883b.

Run from the Marin root with its locked CPU environment. No Torch optimizer is
used to produce this fixture. Like the existing Grug golden generator, the outer
optimizer update is not jitted; JAX executes its individual compiled primitives. The final two matrix axes are transposed when
mapping Levanter arrays into Megatron's fused parameter layout.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec

REFERENCE = "4af966070e5d0252e87d404196400c35bd04883b"
OUTPUT = Path(__file__).with_name("golden.npz")
STEPS = 3
rng = np.random.default_rng(20261111)


def values(shape):
    return jnp.asarray(rng.standard_normal(shape, dtype=np.float32) * np.float32(0.3))


mesh = Mesh(np.asarray(jax.devices()[:1]), ("expert",), axis_types=(AxisType.Explicit,))
params = {
    "token_embed": values((11, 8)),
    "output_proj": values((8, 11)),
    "stacked_blocks": {
        "stacked": {
            "attn": {
                "q_proj": values((2, 8, 8)),
                "k_proj": values((2, 8, 4)),
                "v_proj": values((2, 8, 4)),
                "attn_gate": values((2, 8, 2)),
                "sconv": {"weight": values((2, 3, 8))},
            },
            "attn_gated_norm": {"down_proj": values((2, 8, 4))},
            "input_layernorm": {"weight": values((2, 8))},
            "mlp": {
                "router": values((2, 8, 3)),
                "router_bias": values((2, 3)),
                "expert_mlp": {
                    "w_gate": values((2, 3, 8, 4)),
                    "w_up": values((2, 3, 8, 4)),
                    "w_down": values((2, 3, 4, 8)),
                },
                "shared_mlp": {"w_gate": values((2, 8, 4)), "w_up": values((2, 8, 4)), "w_down": values((2, 4, 8))},
            },
        }
    },
}
# Unequal expert norms distinguish a whole-bank sphere from per-expert spheres.
for key in ("w_gate", "w_up", "w_down"):
    params["stacked_blocks"]["stacked"]["mlp"]["expert_mlp"][key] *= jnp.asarray([0.2, 2.0, 5.0])[None, :, None, None]


def megatron_layout(tree, selected_route=None):
    result = {}

    def add(name, value, route):
        if selected_route is None or selected_route == route:
            result[name] = np.asarray(value).copy()

    add("embedding.word_embeddings.weight", tree["token_embed"], "adam")
    if selected_route in (None, "adamh"):
        add("output_layer.weight", np.asarray(tree["output_proj"]).T, "adamh")
    block = tree["stacked_blocks"]["stacked"]
    for layer in range(2):
        prefix = f"decoder.layers.{layer}"
        if selected_route in (None, "muonh"):
            q, k, v = (np.asarray(block["attn"][key][layer]).T for key in ("q_proj", "k_proj", "v_proj"))
            add(
                f"{prefix}.self_attention.linear_qkv.weight",
                np.concatenate([q.reshape(2, 4, 8), k.reshape(2, 2, 8), v.reshape(2, 2, 8)], axis=1).reshape(16, 8),
                "muonh",
            )
            add(
                f"{prefix}.input_layernorm.down_proj.weight",
                np.asarray(block["attn_gated_norm"]["down_proj"][layer]).T,
                "muonh",
            )
            experts = block["mlp"]["expert_mlp"]
            for expert in range(3):
                gate, up, down = (np.asarray(experts[key][layer, expert]).T for key in ("w_gate", "w_up", "w_down"))
                add(f"{prefix}.mlp.experts.linear_fc1.weight{expert}", np.concatenate([gate, up], axis=0), "muonh")
                add(f"{prefix}.mlp.experts.linear_fc2.weight{expert}", down, "muonh")
            shared = block["mlp"]["shared_mlp"]
            gate, up, down = (np.asarray(shared[key][layer]).T for key in ("w_gate", "w_up", "w_down"))
            add(f"{prefix}.mlp.shared_experts.experts.0.linear_fc1.weight", np.concatenate([gate, up], axis=0), "muonh")
            add(f"{prefix}.mlp.shared_experts.experts.0.linear_fc2.weight", down, "muonh")
        if selected_route in (None, "adam"):
            add(f"{prefix}.self_attention.attn_gate.weight", np.asarray(block["attn"]["attn_gate"][layer]).T, "adam")
            add(f"{prefix}.self_attention.sconv_k.weight", block["attn"]["sconv"]["weight"][layer], "adam")
            add(f"{prefix}.input_layernorm.norm.weight", block["input_layernorm"]["weight"][layer], "adam")
            add(f"{prefix}.mlp.router.weight", np.asarray(block["mlp"]["router"][layer]).T, "adam")
            add(f"{prefix}.mlp.router.bias", block["mlp"]["router_bias"][layer], "adam")
    return result


with jax.set_mesh(mesh):
    params = jax.tree.map(lambda value: jax.device_put(value, NamedSharding(mesh, PartitionSpec())), params)
    config = GrugMoeMuonHConfig(
        learning_rate=0.03,
        adam_lr=0.004,
        momentum=0.95,
        nesterov=True,
        backend_steps=5,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        muon_epsilon=1e-8,
        max_grad_norm=None,
        coefficient_type="quintic",
        use_syrk=False,
        lr_schedule="constant",
        warmup=0,
    )
    optimizer = config.build(STEPS)
    state = optimizer.init(params)

    def step(parameters, optimizer_state, gradients):
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, parameters)
        return optax.apply_updates(parameters, updates), optimizer_state

    fixture = {
        "metadata_reference": np.asarray(REFERENCE),
        "metadata_steps": np.asarray(STEPS),
        "metadata_execution": np.asarray("cpu primitives; no outer optimizer jit"),
    }
    names = list(megatron_layout(params))
    routes = {name: route for route in ("muonh", "adamh", "adam") for name in megatron_layout(params, route)}
    fixture["metadata_names"] = np.asarray(names)
    fixture["metadata_routes"] = np.asarray([routes[name] for name in names])

    def record(prefix, tree, selected_route=None):
        for name, value in megatron_layout(tree, selected_route).items():
            fixture[f"{prefix}__{name}"] = value

    record("initial", params)
    for number in range(1, STEPS + 1):
        gradients = jax.tree.map(lambda value: jnp.asarray(rng.standard_normal(value.shape, dtype=np.float32)), params)
        record(f"gradient_{number}", gradients)
        params, state = step(params, state, gradients)
        record(f"parameter_{number}", params)
        inner = state.inner_state.inner_states
        record(f"momentum_{number}", inner["muonh"].inner_state[0].momentum_buffer, "muonh")
        for route in ("adamh", "adam"):
            adam = inner[route].inner_state[0]
            record(f"{route}_mu_{number}", adam.mu, route)
            record(f"{route}_nu_{number}", adam.nu, route)
            fixture[f"{route}_step_{number}"] = np.asarray(adam.count)
    np.savez_compressed(OUTPUT, **fixture)
    print(OUTPUT)
