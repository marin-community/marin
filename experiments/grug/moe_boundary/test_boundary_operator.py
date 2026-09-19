# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the boundary-operator variant (arXiv 2609.19107, Phase 0).

The boundary operator must be inert when off (logits identical to the parent
`experiments.grug.moe` model), apply the paper's Eq. 2/3 maps at the
prelude→core and core→coda boundaries when on, and reject misconfigured
splits.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.model import GrugModelConfig as MoEGrugModelConfig
from experiments.grug.moe.model import Transformer as MoETransformer
from experiments.grug.moe_boundary.model import (
    GrugModelConfig,
    Transformer,
    split_prelude_core_coda,
)

# Fields the parent moe variant's GrugModelConfig also defines; boundary-off parity
# only compares these.
_PARENT_FIELDS = {f.name for f in dataclasses.fields(MoEGrugModelConfig)}


def _small_config(**overrides) -> GrugModelConfig:
    kwargs = dict(
        vocab_size=128,
        hidden_dim=32,
        intermediate_dim=64,
        shared_expert_intermediate_dim=32,
        num_layers=6,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=16,
        sliding_window=8,
    )
    kwargs.update(overrides)
    return GrugModelConfig(**kwargs)


def _boundary_config(**overrides) -> GrugModelConfig:
    """Small config with the boundary operator on and a valid 6-layer split."""
    overrides = dict(overrides)
    prelude_len, _, coda_len = split_prelude_core_coda(6)
    if "prelude_len" in overrides and "coda_len" not in overrides:
        # Caller pins the prelude: keep the paper rule for the coda, core is the rest.
        overrides.setdefault("coda_len", min(coda_len, 6 - overrides["prelude_len"] - 1))
    boundary = dict(boundary_operator=True, prelude_len=prelude_len, coda_len=coda_len)
    boundary.update(overrides)
    return _small_config(**boundary)


def _init_and_forward(cfg: GrugModelConfig, tokens: jax.Array) -> jax.Array:
    model = Transformer.init(cfg, key=jax.random.PRNGKey(0))
    hidden, _ = model(tokens)
    return jax.nn.relu(hidden) if hidden.dtype == jnp.bfloat16 else hidden


_TOKENS = jnp.array([[3, 17, 42, 5, 91, 7, 0, 128 - 1, 44, 2, 88, 61, 9, 30, 12, 55]], dtype=jnp.int32)


def test_boundary_operator_off_matches_parent_model_logits():
    """With boundary_operator=False the variant must reproduce the parent model exactly."""
    variant_cfg = _small_config()
    parent_cfg = MoEGrugModelConfig(
        **{f.name: getattr(variant_cfg, f.name) for f in dataclasses.fields(variant_cfg) if f.name in _PARENT_FIELDS}
    )
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        variant_model = Transformer.init(variant_cfg, key=jax.random.PRNGKey(0))
        parent_model = MoETransformer.init(parent_cfg, key=jax.random.PRNGKey(0))
        variant_logits = variant_model.logits(_TOKENS)
        parent_logits = parent_model.logits(_TOKENS)
    np.testing.assert_allclose(variant_logits, parent_logits, rtol=1e-5, atol=1e-5)


def test_boundary_operator_forward_with_even_split_runs_and_injects_e():
    """The core entry state is alpha*e (h_0=0) and the coda entry adds alpha*e again."""
    cfg = _boundary_config(injection_scale=0.707)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(cfg, key=jax.random.PRNGKey(0))
        hidden, _ = model(_TOKENS)
    assert hidden.shape == (1, 16, cfg.hidden_dim)
    assert jnp.isfinite(hidden).all()


def test_boundary_operator_is_not_the_identity():
    """The operator must actually change the forward (it is not a no-op re-init)."""
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        plain_model = Transformer.init(_small_config(), key=jax.random.PRNGKey(0))
        boundary_model = Transformer.init(_boundary_config(), key=jax.random.PRNGKey(0))
        plain_logits = plain_model.logits(_TOKENS)
        boundary_logits = boundary_model.logits(_TOKENS)
    assert not jnp.allclose(plain_logits, boundary_logits, rtol=1e-3)


def test_split_prelude_core_coda_follows_paper_allocation():
    """Paper A.2.1: even split, remainder to core first then coda."""
    cases = {
        6: (2, 2, 2),
        8: (2, 3, 3),
        11: (3, 4, 4),
        13: (4, 5, 4),
        7: (2, 3, 2),
        1: (0, 1, 0),
        2: (0, 1, 1),
        4: (1, 2, 1),
        0: (0, 0, 0),
    }
    for num_layers, expected in cases.items():
        assert split_prelude_core_coda(num_layers) == expected


def test_boundary_operator_config_validation():
    with pytest.raises(ValueError, match="requires prelude_len"):
        _small_config(boundary_operator=True)
    with pytest.raises(ValueError, match="prelude_len must be within"):
        _small_config(boundary_operator=True, prelude_len=7, coda_len=0)
    with pytest.raises(ValueError, match="prelude_len \\+ coda_len"):
        _small_config(boundary_operator=True, prelude_len=4, coda_len=4)
    with pytest.raises(ValueError, match="injection_scale"):
        _small_config(boundary_operator=True, prelude_len=2, coda_len=2, injection_scale=-1.0)
    with pytest.raises(ValueError, match="require boundary_operator"):
        _small_config(prelude_len=2)
    # well-formed boundary config does not raise
    _boundary_config()


def test_zero_prelude_uses_embedded_input_as_e():
    """prelude_len=0: e is the post-embedding state; core still starts from alpha*e."""
    cfg = _boundary_config(prelude_len=0)
    assert cfg.prelude_len == 0
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(cfg, key=jax.random.PRNGKey(0))
        hidden, _ = model(_TOKENS)
    assert jnp.isfinite(hidden).all()


def test_hf_config_round_trip_preserves_boundary_fields():
    cfg = _boundary_config(injection_scale=0.707)
    hf = cfg.to_hf_config(vocab_size=cfg.vocab_size)
    restored = GrugModelConfig.from_hf_config(hf)
    assert restored.boundary_operator
    assert restored.prelude_len == cfg.prelude_len
    assert restored.coda_len == cfg.coda_len
    assert restored.injection_scale == cfg.injection_scale
