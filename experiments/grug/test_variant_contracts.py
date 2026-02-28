# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Contract tests for grug variants under experiments/grug/*.

These checks are intentionally variant-discovered: if a subdirectory contains a
`model.py`, it is expected to satisfy this lowering contract.
"""

import importlib

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from jax._src import config as jax_config
from jax.sharding import NamedSharding, PartitionSpec as P, use_abstract_mesh

from levanter.grug.attention import AttentionMask as GrugAttentionMask


def _discover_grug_variant_modules_with_model() -> list[str]:
    grug_dir = Path(__file__).resolve().parent
    modules: list[str] = []
    for child in sorted(grug_dir.iterdir()):
        if not child.is_dir() or child.name.startswith("__"):
            continue
        if (child / "model.py").is_file():
            modules.append(f"experiments.grug.{child.name}.model")
    if not modules:
        raise AssertionError(f"No grug variants with model.py found under {grug_dir}")
    return modules


class _reset_abstract_mesh:
    def __enter__(self):
        self._prev = jax_config.abstract_mesh_context_manager.swap_local(jax_config.config_ext.unset)
        return self

    def __exit__(self, exc_type, exc, tb):
        jax_config.abstract_mesh_context_manager.set_local(self._prev)
        return False


def _infer_loss_fn_name(params) -> str:
    if hasattr(params, "compute_next_token_loss"):
        return "compute_next_token_loss"
    if hasattr(params, "next_token_loss"):
        return "next_token_loss"
    raise AssertionError("Transformer variant must define either compute_next_token_loss or next_token_loss")


@pytest.mark.parametrize(
    "module_name",
    _discover_grug_variant_modules_with_model(),
    ids=lambda name: name.split(".")[-2],
)
def test_grug_variant_loss_lowers_on_abstract_mesh(module_name: str):
    module = importlib.import_module(module_name)
    config_cls = module.GrugModelConfig
    transformer_cls = module.Transformer

    seq = 256 if jax.default_backend() == "tpu" else 16
    cfg = config_cls(vocab_size=256, max_seq_len=seq)
    mesh_fn = getattr(module, "debug_mesh_and_token_pspec", None)
    if mesh_fn is None:
        raise AssertionError(f"{module_name} must define debug_mesh_and_token_pspec(num_devices)")
    mesh, token_pspec = mesh_fn(num_devices=4)

    with _reset_abstract_mesh(), use_abstract_mesh(mesh):
        key = jax.ShapeDtypeStruct(shape=(2,), dtype=jnp.uint32, sharding=NamedSharding(mesh, P()))

        def init_model(k):
            return transformer_cls.init(cfg, key=k)

        params = jax.eval_shape(init_model, key)
        loss_fn_name = _infer_loss_fn_name(params)

        def loss_fn(p):
            token_ids = jnp.zeros((8, seq), dtype=jnp.int32)
            token_ids = jax.sharding.reshard(token_ids, token_pspec)
            loss_weight = jnp.ones((8, seq), dtype=jnp.float32)
            loss_weight = jax.sharding.reshard(loss_weight, token_pspec)
            if loss_fn_name == "compute_next_token_loss":
                return p.compute_next_token_loss(
                    token_ids,
                    loss_weight,
                    mask=GrugAttentionMask.causal(),
                    reduction="mean",
                )
            return p.next_token_loss(
                token_ids,
                loss_weight,
                mask=GrugAttentionMask.causal(),
                reduction="mean",
            )

        platform = jax.devices()[0].platform if jax.devices() else jax.default_backend()
        lowered = jax.jit(loss_fn).trace(params).lower(lowering_platforms=(platform,))

    assert lowered is not None
