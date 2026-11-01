# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""What the fast_track model code in the current directory does, as JSON: config defaults, a digest of a tiny CPU
model's loss and gradients, and how far past hidden states move when only future tokens change.

``code_change.probe`` runs this file's source with ``python -c`` in a checkout (so ``experiments`` imports from
that checkout); it imports nothing from ``experiments.belay``, so it works on any branch.

    python -c "$(cat experiments/belay/model_probe.py)" [name=value ...]   # model settings to apply
"""

import dataclasses
import hashlib
import json
import math
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh

from experiments.grug.fast_track.launch import _apply_settings, _parse_settings
from experiments.grug.fast_track.model import GrugModelConfig, LocalMixer, Transformer
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig

SEQ = 128  # the banded Inkling bias needs a multiple of 128


def _defaults() -> dict[str, dict[str, str]]:
    out = {}
    for cls in (GrugModelConfig, GrugMoeMuonHConfig):
        fields = {}
        for f in dataclasses.fields(cls):
            if f.default is not dataclasses.MISSING:
                fields[f.name] = repr(f.default)
            elif f.default_factory is not dataclasses.MISSING:
                fields[f.name] = repr(f.default_factory())
        out[cls.__name__] = fields
    return out


HIDDEN = 64


def tiny_config() -> GrugModelConfig:
    """A tiny fast_track model that exercises every mixer and routing path on CPU."""
    hidden = HIDDEN
    return GrugModelConfig(
        vocab_size=128,
        hidden_dim=hidden,
        intermediate_dim=hidden // 2,
        shared_expert_intermediate_dim=hidden // 2,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=2,
        latent_dim=hidden // 2,
        num_layers=4,
        num_heads=2,
        num_kv_heads=2,
        local_kv_heads=None,
        global_kv_heads=None,
        head_dim=16,
        max_seq_len=SEQ,
        sliding_window=SEQ,
        global_every=2,
        capacity_factor=8.0,
        pooled_transport_capacity_factor=8.0,
        initializer_std=0.5 / math.sqrt(hidden),
        qk_norm=False,
        sconv_sites=("attn", "mlp"),
        mla=True,
        mla_kv_latent_dim=32,
        inkling_relpos=True,
        rel_dim=4,
        rel_extent=8,
        attn_res=True,
        attn_res_num_blocks=2,
        local_mixer=LocalMixer.KDA,
    )


def tiny_mesh() -> Mesh:
    return Mesh(
        np.array(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )


def probe_here(model_settings: list[str]) -> dict:
    """Probe the checkout this process imported ``experiments`` from, with ``model_settings`` applied."""
    hidden = HIDDEN
    config = _apply_settings(tiny_config(), _parse_settings(tuple(model_settings)))
    with jax.set_mesh(tiny_mesh()):
        model = Transformer.init(config, key=jax.random.key(0))
        tokens = jax.random.randint(jax.random.key(1), (2, SEQ), 0, config.vocab_size)
        cotangent = jax.random.normal(jax.random.key(2), (2, SEQ, hidden), jnp.float32)

        def loss_fn(m):
            return jnp.sum(m(tokens)[0].astype(jnp.float32) * cotangent)

        loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(loss_fn))(model)
        future = tokens.at[:, SEQ // 2 :].set(jax.random.randint(jax.random.key(3), (2, SEQ // 2), 0, 128))
        forward = eqx.filter_jit(lambda m, t: m(t)[0].astype(jnp.float32))
        leak = float(jnp.max(jnp.abs(forward(model, tokens)[:, : SEQ // 2] - forward(model, future)[:, : SEQ // 2])))
    digest = hashlib.sha256(np.asarray(loss).tobytes())
    for path, leaf in jax.tree_util.tree_flatten_with_path(eqx.filter(grads, eqx.is_array))[0]:
        digest.update(jax.tree_util.keystr(path).encode())
        digest.update(np.asarray(leaf).tobytes())
    return {"defaults": _defaults(), "digest": digest.hexdigest(), "leak": leak}


if __name__ == "__main__":
    print(json.dumps(probe_here(sys.argv[1:])))
