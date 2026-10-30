# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import pinned Snowball HF weights and verify the persisted native payload."""

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.experimental.array_serialization.serialization import GlobalAsyncCheckpointManager
from levanter.checkpoint import save_checkpoint
from levanter.compat.hf_checkpoints import RepoRef
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballConfig
from levanter.utils.jax_utils import leaf_key_paths
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.checkpointing import load_grug_checkpoint
from experiments.grug.snowball_sft.model import GrugModelConfig, Transformer, grugmoe_inference_state_dict

logger = logging.getLogger(__name__)

ROOT_WEIGHTS = {
    "token_embed": ("model.embed_tokens.weight", False),
    "embed_norm.weight": ("model.embed_norm.weight", False),
    "embed_gated_norm.w_down": ("model.embed_gated_norm.down_proj.weight", True),
    "embed_gated_norm.w_up": ("model.embed_gated_norm.up_proj.weight", True),
    "output_proj": ("lm_head.weight", True),
    "final_norm.weight": ("model.norm.weight", False),
    "final_gated_norm.w_down": ("model.final_gated_norm.down_proj.weight", True),
    "final_gated_norm.w_up": ("model.final_gated_norm.up_proj.weight", True),
}
BLOCK_WEIGHTS = {
    "rms_attn.weight": ("input_layernorm.weight", False),
    "attn_gated_norm.w_down": ("attn_gated_norm.down_proj.weight", True),
    "attn_gated_norm.w_up": ("attn_gated_norm.up_proj.weight", True),
    "attn.w_q": ("self_attn.q_proj.weight", True),
    "attn.w_k": ("self_attn.k_proj.weight", True),
    "attn.w_v": ("self_attn.v_proj.weight", True),
    "attn.w_o": ("self_attn.o_proj.weight", True),
    "attn.attn_gate": ("self_attn.attn_gate.weight", True),
    "rms_mlp.weight": ("post_attention_layernorm.weight", False),
    "mlp_gated_norm.w_down": ("mlp_gated_norm.down_proj.weight", True),
    "mlp_gated_norm.w_up": ("mlp_gated_norm.up_proj.weight", True),
    "mlp.router": ("mlp.router.weight", True),
    "mlp.router_bias": ("mlp.router.bias", False),
    "mlp.expert_mlp.w_gate": ("mlp.experts.gate_proj.weight", True),
    "mlp.expert_mlp.w_up": ("mlp.experts.up_proj.weight", True),
    "mlp.expert_mlp.w_down": ("mlp.experts.down_proj.weight", True),
    "shared.0.w_gate": ("shared_expert.gate_proj.weight", True),
    "shared.0.w_up": ("shared_expert.up_proj.weight", True),
    "shared.0.w_down": ("shared_expert.down_proj.weight", True),
}


@dataclass(frozen=True)
class ImportConfig:
    hf_id: str
    hf_revision: str | None
    model: GrugModelConfig
    destination: str


def import_weights(config: GrugModelConfig, source: Mapping[str, jax.Array]) -> Transformer:
    """Map every canonical HF tensor into the stacked trainer without changing biases."""
    if config.num_shared_experts != 1 or config.latent_dim is not None or config.sconv:
        raise ValueError("This importer requires the original Snowball architecture")
    template = eqx.filter_eval_shape(Transformer.init, config, key=jax.random.key(0))
    consumed = set()

    def load(path, target):
        if path.startswith("stacked_blocks.stacked."):
            suffix, transpose = BLOCK_WEIGHTS[path.removeprefix("stacked_blocks.stacked.")]
            names = [f"model.layers.{i}.{suffix}" for i in range(config.num_layers)]
        else:
            name, transpose = ROOT_WEIGHTS[path]
            names = [name]
        values = [jnp.swapaxes(source[name], -1, -2) if transpose else source[name] for name in names]
        value = jnp.stack(values) if path.startswith("stacked_blocks.stacked.") else values[0]
        if value.shape != target.shape:
            raise ValueError(f"Shape mismatch for {path}: {value.shape} != {target.shape}")
        consumed.update(names)
        return jax.sharding.reshard(value, target.sharding)

    model = jax.tree.map(load, leaf_key_paths(template), template)
    if consumed != set(source):
        raise ValueError(f"Unmapped HF weights: {sorted(set(source) - consumed)}")
    return model


def verify_weights(model: Transformer, source: Mapping[str, jax.Array]) -> list[dict]:
    """Compare all values through the independent HF export mapping; report every mismatch."""
    errors = []
    records = []
    checked = set()
    # Exporting the whole model eagerly retains unstacked and transposed copies of
    # all expert banks. Bound those temporary allocations to one layer.
    for layer_index in range(model.config.num_layers):
        stacked = jax.tree.map(lambda a, i=layer_index: a[i : i + 1], model.stacked_blocks.stacked)
        layer_model = replace(model, stacked_blocks=replace(model.stacked_blocks, stacked=stacked, num_layers=1))
        exported = grugmoe_inference_state_dict(layer_model)
        for export_name, actual in exported.items():
            name = export_name.replace("model.layers.0.", f"model.layers.{layer_index}.")
            name = name.replace(".shared_experts.0.", ".shared_expert.")
            if name in checked:
                continue
            checked.add(name)
            if name not in source:
                errors.append(f"Missing source tensor: {name}")
                continue
            expected = source[name]
            if ".mlp.experts." in name and not bool(jnp.any(expected != 0)):
                errors.append(f"Empty source expert bank: {name}")
            if expected.shape != actual.shape:
                errors.append(f"Tensor shape mismatch: {name}")
                continue
            expected = jax.sharding.reshard(expected, actual.sharding)
            if not bool(jnp.array_equal(expected, actual)):
                errors.append(f"Tensor mismatch: {name}")
            records.append(dict(name=name, shape=list(actual.shape), dtype=str(actual.dtype)))
    errors.extend(f"Missing exported tensor: {name}" for name in sorted(set(source) - checked))
    if errors:
        raise ValueError("HF/native parity failed: " + "; ".join(errors))
    return records


def convert(config: ImportConfig) -> None:
    """Write a fresh native checkpoint, reload it, and require full HF tensor parity."""
    root = StoragePath(config.destination)
    if root.exists() and root.ls():
        raise FileExistsError(f"Import requires a fresh destination: {root}")
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
    converter = SnowballConfig().hf_checkpoint_converter()
    with jax.set_mesh(mesh):
        source = converter.load_state_dict(RepoRef(config.hf_id, config.hf_revision), dtype=jnp.bfloat16)
        model = import_weights(config.model, source)
        verify_weights(model, source)
        shared = model.stacked_blocks.stacked.shared
        assert shared is not None and len(shared) == 1
        # The SFT initializer reads the historical singleton shared-expert schema.
        legacy = eqx.tree_at(lambda m: m.stacked_blocks.stacked.shared, model, shared[0])
        state = {"params": legacy, "pending_qb_betas": jnp.zeros_like(model.stacked_blocks.stacked.mlp.router_bias)}
        manager = GlobalAsyncCheckpointManager(timeout_secs=1800)
        save_checkpoint(state, 0, str(root / "step-0"), manager=manager, is_temporary=False)
        manager.wait_until_finished()
        template = jax.tree.map(lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype, sharding=a.sharding), state)
        restored = load_grug_checkpoint(state=template, candidate=str(root / "step-0"), mesh=mesh, allow_partial=False)
        restored_model = eqx.tree_at(
            lambda m: m.stacked_blocks.stacked.shared,
            restored["params"],
            (restored["params"].stacked_blocks.stacked.shared,),
        )
        records = verify_weights(restored_model, source)
        if not bool(jnp.all(restored["pending_qb_betas"] == 0)):
            raise ValueError("Imported pending QB update must be zero")
    (root / "verification.json").write_text(
        json.dumps(dict(config=draccus.encode(config), tensors=records, exact_parity=True), indent=2) + "\n"
    )
    logger.info("Verified all %d HF tensors after native checkpoint restore", len(records))


@draccus.wrap()
def main(config: ImportConfig) -> None:
    configure_coreweave_s3()
    logging.basicConfig(level=logging.INFO)
    convert(config)


if __name__ == "__main__":
    main()
