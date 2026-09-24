# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the boundary-operator variant (arXiv 2609.19107, Phase 0).

The boundary operator must be inert when off (logits identical to the parent
`experiments.grug.moe` model), apply the paper's Eq. 2/3 maps at the
prelude→core and core→coda boundaries when on, and reject misconfigured
splits.
"""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh, unshard

from experiments.grug.moe.model import GrugModelConfig as MoEGrugModelConfig
from experiments.grug.moe.model import Transformer as MoETransformer
from experiments.grug.moe_boundary.launch_compute_opt import _BASELINE_OPTIMIZER_VALUES, baseline_recipe
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
    split = split_prelude_core_coda(6)
    prelude_len, coda_len = split.prelude, split.coda
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

_ALPHA = 0.707


def _paper_rms_norm(x: jax.Array) -> jax.Array:
    """Paper Eq. 3 boundary norm: non-parametric RMSNorm, eps 1e-6."""
    variance = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
    return (x * jax.lax.rsqrt(variance + 1e-6)).astype(x.dtype)


def _run_block(model: Transformer, i: int, x: jax.Array) -> jax.Array:
    """Run block i exactly as the model's forward does (mask / PKO / RoPE wiring)."""
    cfg = model.config
    is_last = i == len(model.blocks) - 1
    is_long = i % 4 == 3 or is_last
    layer_mask = AttentionMask(is_causal=True, sliding_window=None if is_long else cfg.sliding_window)
    use_pko = is_long and not cfg.disable_pko
    disable_rope = is_long and cfg.disable_long_rope
    block = eqx.filter_checkpoint(model.blocks[i], policy=None)
    out, _ = block(x, layer_mask, use_pko, disable_rope)
    return out


def _manual_boundary_forward(
    model: Transformer,
    tokens: jax.Array,
    *,
    e_source,
    core_entry,
    coda_entry,
) -> jax.Array:
    """Execute the paper's prelude / core / coda pipeline by hand over the model's own submodules.

    Weights are shared with the model under test, so what gets checked is the wiring of
    arXiv 2609.19107 Eq. 2/3 (K = 1): ``e_source(embed, prelude_out)`` picks what ``e``
    is, ``core_entry(e, prelude_out)`` computes the state entering the core, and
    ``coda_entry(e, core_out)`` the state entering the coda.
    """
    cfg = model.config
    embed = model.embed_gated_norm(model.embed_norm(unshard(model.token_embed)[tokens]))
    num_blocks = len(model.blocks)
    prelude_len, coda_len = cfg.prelude_len, cfg.coda_len
    core_len = num_blocks - prelude_len - coda_len

    hidden = embed
    for i in range(prelude_len):
        hidden = _run_block(model, i, hidden)
    prelude_out = hidden

    e = e_source(embed, prelude_out)
    hidden = core_entry(e, prelude_out)
    for i in range(prelude_len, prelude_len + core_len):
        hidden = _run_block(model, i, hidden)
    core_out = hidden

    hidden = coda_entry(e, core_out)
    for i in range(prelude_len + core_len, num_blocks):
        hidden = _run_block(model, i, hidden)
    return model.final_gated_norm(model.final_norm(hidden))


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


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    "boundary_kwargs",
    [dict(injection_scale=_ALPHA), dict(prelude_len=0, injection_scale=_ALPHA)],
    ids=["even-split-2-2-2", "empty-prelude"],
)
def test_boundary_operator_forward_matches_paper_equations(boundary_kwargs):
    """The forward must implement the paper's Eq. 2/3 boundary maps exactly.

    Re-executes the prelude / core / coda pipeline by hand over the model's own
    submodules with the paper's equations — core entry BO(0, e) = alpha * e
    (h_0 = 0), coda entry BO(h, e) = rms_norm(h) + alpha * e, e = prelude output
    (the embedded input when the prelude is empty) — and asserts the model's
    hidden state matches. Each plausible mis-wiring must produce a *different*
    result, so shape-only assertions cannot pass by accident.
    """
    cfg = _boundary_config(**boundary_kwargs)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(cfg, key=jax.random.PRNGKey(0))
        actual, _ = model(_TOKENS)

        def paper_e(embed, prelude_out):
            return prelude_out

        def paper_core(e, prelude_out):
            return cfg.injection_scale * e

        def paper_coda(e, core_out):
            return _paper_rms_norm(core_out) + cfg.injection_scale * e

        expected = _manual_boundary_forward(
            model, _TOKENS, e_source=paper_e, core_entry=paper_core, coda_entry=paper_coda
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

        wrong_variants = {
            "core entry retains the prelude state (no h_0 = 0 reset)": (
                paper_e,
                lambda e, prelude_out: prelude_out,
                paper_coda,
            ),
            "coda entry omits rms_norm(h)": (
                paper_e,
                paper_core,
                lambda e, core_out: core_out + cfg.injection_scale * e,
            ),
            "coda entry omits the alpha * e injection": (
                paper_e,
                paper_core,
                lambda e, core_out: _paper_rms_norm(core_out),
            ),
        }
        if cfg.prelude_len > 0:
            wrong_variants["e is the embedding, not the prelude output"] = (
                lambda embed, prelude_out: embed,
                paper_core,
                paper_coda,
            )
        for label, (e_source, core_entry, coda_entry) in wrong_variants.items():
            wrong = _manual_boundary_forward(
                model, _TOKENS, e_source=e_source, core_entry=core_entry, coda_entry=coda_entry
            )
            assert not jnp.allclose(
                actual, wrong, rtol=1e-4, atol=1e-5
            ), f"model output matches the mis-wired variant: {label}"


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


def test_hf_config_round_trip_preserves_boundary_fields():
    cfg = _boundary_config(injection_scale=0.707)
    hf = cfg.to_hf_config(vocab_size=cfg.vocab_size)
    restored = GrugModelConfig.from_hf_config(hf)
    assert restored.boundary_operator
    assert restored.prelude_len == cfg.prelude_len
    assert restored.coda_len == cfg.coda_len
    assert restored.injection_scale == cfg.injection_scale


def test_baseline_recipe_matches_recorded_baseline_optimizers():
    """The compute-optimal arms' optimizer must match the recorded baseline recipes.

    Pins the launcher to the May Recipe baseline convention: optimizer built from
    the PINNED README cell batch (not the heuristic-derived batch - at d768 that
    would be 128 vs the cell's 64, detuning LR by sqrt(2)) with the cell's actual
    trained tokens, and a schedule decaying to zero (``min_lr_ratio = 0``) like
    the documented baseline. Expected values are the recorded reference recipes
    from issue #6822 (``larry_reference_d512/d768.json``).
    """
    for hidden_dim, (adam_lr, muonh_lr, epsilon) in _BASELINE_OPTIMIZER_VALUES.items():
        _, optimizer, (_, batch_size, _) = baseline_recipe(hidden_dim)
        np.testing.assert_allclose(optimizer.adam_lr, adam_lr, rtol=1e-9)
        np.testing.assert_allclose(optimizer.learning_rate, muonh_lr, rtol=1e-9)
        np.testing.assert_allclose(optimizer.epsilon, epsilon, rtol=1e-9)
        assert optimizer.min_lr_ratio == 0.0
        # beta2 is batch-derived: 0.999^(tpb/131072) at the pinned cell batch.
        expected_beta2 = 0.999 ** (batch_size * 4096 / 131_072)
        np.testing.assert_allclose(optimizer.beta2, expected_beta2, rtol=1e-9)
