# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the Deletion Tuning masking core, run on a tiny MoE.

The tiny network mirrors the production scope (routed experts with a gate
projection) at toy scale so the three paper invariants are checked end to end:
the hard mask equals ``W * 1[G > 0]``, training only the gate logits leaves the
frozen weights untouched, and a mask-artifact export round-trip reloads a model
that produces the same outputs as the hard-masked original.
"""

import math

import numpy as np
import pytest
from marin.rl.deletion_tuning import (
    DeletionTuningDefaults,
    alpha_ramp,
    apply_keep_mask,
    hard_keep_mask,
    init_gate_logits,
    mask_scope_stats,
    read_mask_artifact,
    regularizer_loss,
    soft_mask,
    wanda_scores,
    write_mask_artifact,
)
from scipy.stats import norm


class TinyMoe:
    """A two-layer routed MoE with gate/up/down experts, forward in numpy.

    Weights follow the stacked expert layout the Snowball models use:
    ``w_gate``/``w_up`` are ``[E, D, I]`` and ``w_down`` is ``[E, I, D]``.
    """

    def __init__(self, num_experts: int = 4, hidden_dim: int = 8, intermediate_dim: int = 12, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.weights = {
            "router": rng.normal(0, 0.3, size=(num_experts, hidden_dim)),
            "w_gate": rng.normal(0, 0.3, size=(num_experts, hidden_dim, intermediate_dim)),
            "w_up": rng.normal(0, 0.3, size=(num_experts, hidden_dim, intermediate_dim)),
            "w_down": rng.normal(0, 0.3, size=(num_experts, intermediate_dim, hidden_dim)),
        }

    def copy(self) -> "TinyMoe":
        clone = TinyMoe.__new__(TinyMoe)
        clone.num_experts = self.num_experts
        clone.hidden_dim = self.hidden_dim
        clone.weights = {name: value.copy() for name, value in self.weights.items()}
        return clone

    def masked_gate_weights(self, keep_mask: np.ndarray) -> np.ndarray:
        return apply_keep_mask(self.weights["w_gate"], keep_mask)

    def forward(self, tokens: np.ndarray, masked_gate: np.ndarray | None = None) -> np.ndarray:
        """Per-token outputs [T, D]; each token averages its top-2 experts."""
        gate = self.weights["w_gate"] if masked_gate is None else masked_gate
        router_logits = np.einsum("td,ed->te", tokens, self.weights["router"])
        top_experts = np.argsort(-router_logits, axis=-1)[:, :2]
        outputs = np.zeros((tokens.shape[0], self.hidden_dim))
        for expert in range(self.num_experts):
            rows, _ = np.where(top_experts == expert)
            if rows.size == 0:
                continue
            x = tokens[rows]
            gate_out = np.einsum("td,di->ti", x, gate[expert])
            up_out = np.einsum("td,di->ti", x, self.weights["w_up"][expert])
            hidden = gate_out * up_out
            outputs[rows] = np.einsum("ti,id->td", hidden, self.weights["w_down"][expert])
        return outputs


def _wanda_input_norms(tokens: np.ndarray) -> np.ndarray:
    """Per-input-channel activation norms from one forward pass (paper Eq. 8)."""
    return np.linalg.norm(tokens, axis=0)


def test_alpha_ramp_is_linear_and_clamped():
    assert alpha_ramp(0, 400, 1e2, 1e3) == 1e2
    assert alpha_ramp(400, 400, 1e2, 1e3) == 1e3
    assert alpha_ramp(200, 400, 1e2, 1e3) == pytest.approx(5.5e2)
    assert alpha_ramp(-5, 400, 1e2, 1e3) == 1e2
    assert alpha_ramp(10**9, 400, 1e2, 1e3) == 1e3
    with pytest.raises(ValueError, match="total_steps"):
        alpha_ramp(1, 0, 1e2, 1e3)


def test_wanda_scores_match_definition():
    weight = np.array([[1.0, -2.0], [0.5, 4.0]])
    input_norms = np.array([3.0, 0.5])
    expected = np.array([[3.0, 1.0], [1.5, 2.0]])
    np.testing.assert_allclose(wanda_scores(weight, input_norms), expected)
    with pytest.raises(ValueError, match="input_norms"):
        wanda_scores(weight, np.array([1.0, 2.0, 3.0]))


def test_gate_logit_groups_follow_the_wanda_prior():
    scores = np.linspace(0.0, 1.0, 20_000).reshape(100, 200)
    eta = 1 / 2048
    logits = init_gate_logits(scores, eta, seed=7)
    assert logits.shape == scores.shape

    delete_count = math.floor(eta * scores.size)
    flat_scores = scores.reshape(-1)
    order = np.argsort(flat_scores, kind="stable")
    expected_delete = order[:delete_count]
    flat_logits = logits.reshape(-1)
    # The delete group's logits center below zero and the keep group's above
    # it; individual keep-group logits can still be negative because the init
    # is noisy by design (about 0.135% of the keep group starts deleted).
    assert flat_logits[expected_delete].mean() < 0
    kept = np.ones(scores.size, dtype=bool)
    kept[expected_delete] = False
    assert flat_logits[kept].mean() > 0
    negative_keep_fraction = float(np.mean(flat_logits[kept] < 0))
    assert negative_keep_fraction == pytest.approx(norm.cdf(-3), abs=5e-4)

    # Same seed reproduces the same logits; the group split is deterministic anyway.
    np.testing.assert_array_equal(init_gate_logits(scores, eta, seed=7), logits)


def test_gate_logit_init_deletion_rate_matches_the_paper():
    rng = np.random.default_rng(3)
    scores = rng.random((256, 4096))
    eta = DeletionTuningDefaults.eta
    logits = init_gate_logits(scores, eta, seed=11)
    deleted = float(np.mean(logits < 0))
    expected = eta * norm.cdf(3) + (1 - eta) * norm.cdf(-3)
    assert deleted == pytest.approx(expected, abs=2e-4)


def test_regularizer_is_negative_and_rewards_retained_norm():
    weights = {"a": np.array([[1.0, 2.0]]), "b": np.array([[0.5]])}
    lam = 1e-5
    expected = -lam * (1.0 + 4.0 + 0.25)
    assert regularizer_loss(weights, lam) == pytest.approx(expected)
    with pytest.raises(ValueError, match="lam"):
        regularizer_loss(weights, -1.0)


def test_mask_artifact_round_trip_and_digest_check(tmp_path):
    keep = np.zeros((6, 4), dtype=bool)
    keep[:2, :] = True
    keep[4:, 2:] = True
    write_mask_artifact(tmp_path, {"layer.gate": keep})
    loaded = read_mask_artifact(tmp_path)
    np.testing.assert_array_equal(loaded["layer.gate"], keep)

    # A corrupted artifact fails its digest check instead of silently loading.
    artifact_file = next(path for path in tmp_path.iterdir() if path.name.endswith(".mask.bin"))
    artifact_file.write_bytes(artifact_file.read_bytes()[:-1])
    with pytest.raises(ValueError, match="SHA-256"):
        read_mask_artifact(tmp_path)


def test_mask_artifact_rejects_rewrite_and_bad_dtype(tmp_path):
    keep = np.ones((4, 4), dtype=bool)
    write_mask_artifact(tmp_path, {"gate": keep})
    with pytest.raises(FileExistsError, match="already exists"):
        write_mask_artifact(tmp_path, {"gate": keep})
    with pytest.raises(ValueError, match="boolean"):
        write_mask_artifact(tmp_path / "second", {"gate": keep.astype(np.uint8)})


def test_mask_scope_stats_report_sparsity_and_size(tmp_path):
    keep = np.ones((64, 64), dtype=bool)
    keep[:5, :] = False
    stats = mask_scope_stats({"gate": keep}, total_model_weights=64 * 64 * 4)
    assert stats.masked_weights == 64 * 64
    assert stats.deleted_weights == 5 * 64
    assert stats.masked_scope_sparsity == pytest.approx(5 / 64)
    assert stats.all_weights_sparsity == pytest.approx(5 / (64 * 4))
    assert stats.compressed_bytes > 0
    write_mask_artifact(tmp_path, {"gate": keep})
    on_disk = sum(path.stat().st_size for path in tmp_path.iterdir())
    assert stats.compressed_bytes < on_disk  # manifest bytes are not mask bytes


def test_tiny_moe_hard_mask_equals_zeroed_weights():
    """Hard mask = W * 1[G > 0]: masking and zeroing agree in the forward."""
    model = TinyMoe(seed=1)
    rng = np.random.default_rng(2)
    tokens = rng.normal(0, 1, size=(16, model.hidden_dim))

    gate_logits = init_gate_logits(
        wanda_scores(model.weights["w_gate"], _wanda_input_norms(tokens), input_axis=-2), eta=0.25, seed=5
    )
    keep = hard_keep_mask(gate_logits)
    masked = model.masked_gate_weights(keep)

    manual = np.where(keep, model.weights["w_gate"], 0.0)
    np.testing.assert_array_equal(masked, manual)

    through_forward = model.forward(tokens, masked_gate=masked)
    through_zeroed = model.forward(tokens, masked_gate=manual)
    np.testing.assert_allclose(through_forward, through_zeroed, rtol=0, atol=0)


def test_tiny_moe_training_only_moves_gate_logits():
    """Simulated training perturbs G and never the frozen weights."""
    model = TinyMoe(seed=3)
    initial_weights = {name: value.copy() for name, value in model.weights.items()}
    rng = np.random.default_rng(4)
    tokens = rng.normal(0, 1, size=(16, model.hidden_dim))

    gate_logits = init_gate_logits(
        wanda_scores(model.weights["w_gate"], _wanda_input_norms(tokens), input_axis=-2), eta=0.25, seed=6
    )
    for step in range(5):
        # A training step: the soft mask sharpens and the logits move; the
        # weights stay frozen because the forward only reads W * sigmoid(alpha*G).
        alpha = alpha_ramp(step, 10, 1e2, 1e3)
        soft = soft_mask(gate_logits, alpha)
        assert (soft >= 0).all() and (soft <= 1).all()
        gate_logits = gate_logits - 0.01 * rng.normal(size=gate_logits.shape)  # pretend gradient
        masked_soft = model.weights["w_gate"] * soft
        model.forward(tokens, masked_gate=masked_soft)

    for name, value in model.weights.items():
        np.testing.assert_array_equal(value, initial_weights[name], err_msg=f"weight {name} changed")


def test_tiny_moe_export_round_trip_reloads_the_same_model(tmp_path):
    """The exported artifact rebuilds a model whose outputs match the original."""
    model = TinyMoe(seed=8)
    rng = np.random.default_rng(9)
    tokens = rng.normal(0, 1, size=(16, model.hidden_dim))

    gate_logits = init_gate_logits(
        wanda_scores(model.weights["w_gate"], _wanda_input_norms(tokens), input_axis=-2), eta=0.25, seed=10
    )
    keep = hard_keep_mask(gate_logits)
    original = model.forward(tokens, masked_gate=model.masked_gate_weights(keep))

    write_mask_artifact(tmp_path, {"model.layers.0.mlp.experts.gate_proj.weight": keep})
    reloaded_keep = read_mask_artifact(tmp_path)["model.layers.0.mlp.experts.gate_proj.weight"]
    np.testing.assert_array_equal(reloaded_keep, keep)

    # A freshly loaded base plus the reloaded mask reproduces the hard-masked
    # model exactly.
    reloaded_model = TinyMoe(seed=8)
    reloaded_outputs = reloaded_model.forward(tokens, masked_gate=reloaded_model.masked_gate_weights(reloaded_keep))
    np.testing.assert_allclose(reloaded_outputs, original, rtol=0, atol=0)


def test_soft_mask_converges_to_hard_mask_at_the_end_of_the_ramp():
    """At alpha_end the soft mask is at the hard mask for the paper's logits.

    With the noisy init the agreement is statistical: a weight whose logit
    happens to sit within ~4.6/alpha of zero keeps a soft value strictly
    between 0 and 1, so the test bounds the ambiguous fraction rather than a
    max over a million samples (where a logit near zero is expected).
    """
    deterministic = np.concatenate([np.full(512, 0.03), np.full(512, -0.03)])
    for alpha_end in (DeletionTuningDefaults.alpha_end_sft, DeletionTuningDefaults.alpha_end_rl):
        soft = soft_mask(deterministic, alpha_end)
        hard = hard_keep_mask(deterministic).astype(np.float64)
        assert np.max(np.abs(soft - hard)) < 1e-12

    rng = np.random.default_rng(12)
    logits = rng.normal(0.03, 0.01, size=(1024, 1024))
    flat = logits.reshape(-1)
    flat[: 64 * 64] = rng.normal(-0.03, 0.01, size=64 * 64)  # delete group
    logits = flat.reshape(1024, 1024)
    hard = hard_keep_mask(logits).astype(np.float64)
    for alpha_end in (DeletionTuningDefaults.alpha_end_sft, DeletionTuningDefaults.alpha_end_rl):
        soft = soft_mask(logits, alpha_end)
        ambiguous = float(np.mean(np.abs(soft - hard) > 0.01))
        assert ambiguous < 0.01, f"alpha_end={alpha_end}: {ambiguous:.4f} of soft masks sit off the hard mask"
