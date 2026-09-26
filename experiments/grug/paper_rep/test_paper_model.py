# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the paper-replication model (arXiv 2609.19107, base size).

The boundary operator must be inert when off (the vanilla forward), apply the
paper's Eq. 2/3 maps at the prelude→core and core→coda boundaries when on, and
the paper's init scheme (WTE/UIS/zero) and RM/OM multipliers must hold.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import Pbatch, compact_grug_mesh

from experiments.grug.paper_rep.model import (
    GrugModelConfig,
    Transformer,
    rms_norm,
    split_prelude_core_coda,
)


def _small_config(**overrides) -> GrugModelConfig:
    kwargs = dict(
        vocab_size=128,
        hidden_dim=64,
        intermediate_dim=192,
        num_layers=6,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=16,
    )
    kwargs.update(overrides)
    return GrugModelConfig(**kwargs)


def _boundary_config(**overrides) -> GrugModelConfig:
    """Small config with the boundary operator on and the paper split for 6 layers."""
    split = split_prelude_core_coda(6)
    boundary = dict(boundary_operator=True, prelude_len=split.prelude, coda_len=split.coda)
    boundary.update(overrides)
    return _small_config(**boundary)


def _model_with_nonzero_sublayers(cfg: GrugModelConfig, seed: int = 0) -> Transformer:
    """Init with nonzero w_o / w_down / head weights.

    At init every sublayer output is zero (zero-init output projections), so
    the residual stream is just the normalized embedding and the final
    RMSNorm cancels any scale on it — the boundary operator and RM are
    invisible. Overwrite the zero-init weights to exercise them.
    """
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(cfg, key=jax.random.PRNGKey(seed))
        rng = jax.random.PRNGKey(1)
        blocks = []
        for i, block in enumerate(model.blocks):
            w_o = 0.05 * jax.random.normal(jax.random.fold_in(rng, i), block.attn.w_o.shape)
            w_down = 0.05 * jax.random.normal(jax.random.fold_in(rng, 100 + i), block.mlp.w_down.shape)
            attn = dataclasses.replace(block.attn, w_o=w_o)
            mlp = dataclasses.replace(block.mlp, w_down=w_down)
            blocks.append(dataclasses.replace(block, attn=attn, mlp=mlp))
        head = 0.05 * jax.random.normal(jax.random.fold_in(rng, 200), model.output_proj.shape)
        return dataclasses.replace(model, blocks=tuple(blocks), output_proj=head)


_TOKENS = jnp.array([[3, 17, 42, 5, 91, 7, 0, 127, 44, 2, 88, 61, 9, 30, 12, 55]], dtype=jnp.int32)


def test_split_prelude_core_coda_follows_paper_table_2():
    """Paper Table 2: even split, remainder to the core first then the coda."""
    cases = {
        6: (2, 2, 2),
        8: (2, 3, 3),
        11: (3, 4, 4),
        13: (4, 5, 4),
        26: (8, 9, 9),
        1: (0, 1, 0),
        2: (0, 1, 1),
        4: (1, 2, 1),
        0: (0, 0, 0),
    }
    for num_layers, (prelude, core, coda) in cases.items():
        split = split_prelude_core_coda(num_layers)
        assert (split.prelude, split.core, split.coda) == (prelude, core, coda)


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
    # Well-formed boundary config does not raise.
    _boundary_config()


def test_init_scheme_matches_paper():
    """WTE: normal(std) embeddings; UIS: uniform inputs; outputs zero-init."""
    cfg = _small_config(embed_init_std=0.05, input_init_scale=0.3)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = Transformer.init(cfg, key=jax.random.PRNGKey(0))

    np.testing.assert_allclose(np.asarray(model.token_embed.std()), 0.05, rtol=0.15)
    assert float(jnp.max(jnp.abs(model.token_embed))) < 0.05 * 6
    for block in model.blocks:
        for w in (block.attn.w_q, block.attn.w_k, block.attn.w_v, block.mlp.w_gate, block.mlp.w_up):
            assert float(jnp.max(jnp.abs(w))) <= 0.3 + 1e-6
            assert float(jnp.min(w)) >= -0.3 - 1e-6
        for w in (block.attn.w_o, block.mlp.w_down):
            assert float(jnp.max(jnp.abs(w))) == 0.0
    assert float(jnp.max(jnp.abs(model.output_proj))) == 0.0


def test_residual_and_output_multipliers_scale_forward():
    """RM/OM follow the paper's F(alpha*Theta): linear in the multipliers."""
    cfg = _small_config(residual_multiplier=1.0, output_multiplier=1.0)
    model_a = _model_with_nonzero_sublayers(cfg)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model_om = _model_with_nonzero_sublayers(dataclasses.replace(cfg, output_multiplier=3.0))
        model_rm = _model_with_nonzero_sublayers(dataclasses.replace(cfg, residual_multiplier=2.0))
        logits_a = model_a.logits(_TOKENS)
        logits_om = model_om.logits(_TOKENS)
        logits_rm = model_rm.logits(_TOKENS)

    np.testing.assert_allclose(np.asarray(logits_om), 3.0 * np.asarray(logits_a), rtol=1e-4, atol=1e-4)
    assert not jnp.allclose(logits_rm, logits_a, rtol=1e-3)
    assert jnp.isfinite(logits_rm).all()


def test_boundary_operator_changes_the_forward():
    """Hidden states (pre-head) differ between vanilla and boundary models.

    At init every sublayer output is zero, so the residual stream is scale-
    invariant through the final RMSNorm and the injection is invisible;
    nonzero sublayer weights expose the boundary operator.
    """
    plain = _model_with_nonzero_sublayers(_small_config())
    boundary = _model_with_nonzero_sublayers(_boundary_config())
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        hidden_plain = plain(_TOKENS)
        hidden_boundary = boundary(_TOKENS)
    assert not jnp.allclose(hidden_plain, hidden_boundary, rtol=1e-3)
    assert jnp.isfinite(hidden_boundary).all()


@pytest.mark.parametrize("prelude_len", [0, 2])
def test_boundary_operator_core_entry_matches_alpha_times_prelude(prelude_len: int):
    """Compare the model with Eq. 2's core entry, including an empty prelude."""
    cfg = _boundary_config(prelude_len=prelude_len, coda_len=0, injection_scale=0.707)
    model = _model_with_nonzero_sublayers(cfg)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        mask = AttentionMask.causal()
        prelude_output = rms_norm(model.token_embed.at[_TOKENS].get(out_sharding=Pbatch), cfg.layer_norm_eps)
        for block in model.blocks[:prelude_len]:
            prelude_output = block(prelude_output, mask)
        core_output = cfg.injection_scale * prelude_output
        skipped_injection = prelude_output
        for block in model.blocks[prelude_len:]:
            core_output = block(core_output, mask)
            skipped_injection = block(skipped_injection, mask)
        expected = rms_norm(core_output, cfg.layer_norm_eps)
        actual = model(_TOKENS)
        skipped_injection = rms_norm(skipped_injection, cfg.layer_norm_eps)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-4, atol=1e-4)
    assert not jnp.allclose(actual, skipped_injection, rtol=1e-3, atol=1e-3)


def test_attention_and_mlp_affect_loss():
    """The nonzero sublayers affect loss through both attention and SwiGLU."""
    cfg = _small_config()
    model = _model_with_nonzero_sublayers(cfg)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        loss, grads = jax.value_and_grad(
            lambda m: m.next_token_loss(_TOKENS, jnp.ones_like(_TOKENS, dtype=jnp.float32), reduction="mean")
        )(model)
    assert jnp.isfinite(loss).all()
    assert loss.shape == ()
    assert jnp.linalg.norm(grads.blocks[0].attn.w_q) > 0
    assert jnp.linalg.norm(grads.blocks[0].mlp.w_gate) > 0


def test_param_count_matches_paper_at_d8_and_d6():
    """Model array shapes match the paper's d8/d6 parameter counts."""
    for num_layers, expected in ((8, 210e6), (6, 123e6)):
        cfg = GrugModelConfig(
            vocab_size=50_304,
            hidden_dim=128 * num_layers,
            intermediate_dim=3 * 128 * num_layers,
            num_layers=num_layers,
            num_heads=128 * num_layers // 64,
            num_kv_heads=128 * num_layers // 64,
        )
        with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
            model_shape = jax.eval_shape(lambda cfg=cfg: Transformer.init(cfg, key=jax.random.PRNGKey(0)))
        total = sum(x.size for x in jax.tree.leaves(model_shape))
        assert abs(total - expected) / expected < 0.05, (num_layers, total)
