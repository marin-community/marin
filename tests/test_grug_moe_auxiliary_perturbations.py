# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections import Counter

import numpy as np

from experiments.grug.moe import swarm_auxiliary_perturbations as aux


def _weights_vector(weights: dict[str, float]) -> np.ndarray:
    return np.asarray([weights[bucket] for bucket in aux.COMPONENTS], dtype=float)


def _counts_vector(weights: dict[str, float]) -> np.ndarray:
    counts = np.rint(_weights_vector(weights) * aux.MIXTURE_QUANTUM_DENOMINATOR).astype(int)
    assert np.allclose(counts / aux.MIXTURE_QUANTUM_DENOMINATOR, _weights_vector(weights))
    return counts


def test_auxiliary_candidate_inventory_is_stable() -> None:
    candidates = aux.AUXILIARY_CANDIDATES
    assert len(aux.COMPONENTS) == 168
    assert aux.AUXILIARY_INDEX_START == len(aux._PRODUCTION_SWARM_CANDIDATES)
    assert aux.AUXILIARY_INDEX_START == 840
    assert len(candidates) == 302
    assert candidates[0].index == 840
    assert candidates[-1].index == 1141
    assert [c.index for c in candidates] == list(range(840, 1142))
    assert len({c.candidate_name for c in candidates}) == len(candidates)

    assert Counter(c.candidate_type for c in candidates) == {
        "baseline_proportional": 1,
        "baseline_uniform": 1,
        "baseline_unimax": 4,
        "partition_ablation": 168,
        "projected_controllability_plus": 64,
        "projected_controllability_minus": 64,
    }
    assert aux.MAX_CONCURRENT_AUXILIARY_STEPS == 240
    assert aux.MAX_CONCURRENT_AUXILIARY_STEPS <= aux.AUXILIARY_INDEX_START


def test_auxiliary_weights_are_lattice_simplex_vectors() -> None:
    for candidate in aux.AUXILIARY_CANDIDATES:
        assert tuple(candidate.phase_0) == tuple(aux.COMPONENTS)
        assert tuple(candidate.phase_1) == tuple(aux.COMPONENTS)
        for phase_weights in (candidate.phase_0, candidate.phase_1):
            counts = _counts_vector(phase_weights)
            assert int(counts.sum()) == aux.MIXTURE_QUANTUM_DENOMINATOR
            assert np.all(counts >= 0)
            assert np.isclose(_weights_vector(phase_weights).sum(), 1.0)


def test_delete_and_renormalize_ablation_formula() -> None:
    proportional = np.asarray([aux.PROPORTIONAL_WEIGHTS[bucket] for bucket in aux.COMPONENTS], dtype=float)
    proportional = proportional / proportional.sum()
    by_name = {candidate.candidate_name: candidate for candidate in aux.AUXILIARY_CANDIDATES}

    for bucket_idx, bucket in enumerate(aux.COMPONENTS):
        candidate = by_name[f"abl_del_{bucket}"]
        assert candidate.phase_0 == candidate.phase_1
        weights = _weights_vector(candidate.phase_0)
        assert weights[bucket_idx] == 0.0

        # The exact continuous deletion intervention has TV distance p_j from
        # proportional. Lattice rounding moves this by at most a few quanta.
        tv_distance = 0.5 * float(np.abs(weights - proportional).sum())
        assert abs(tv_distance - proportional[bucket_idx]) < len(aux.COMPONENTS) / aux.MIXTURE_QUANTUM_DENOMINATOR

        donor_mask = np.ones(len(aux.COMPONENTS), dtype=bool)
        donor_mask[bucket_idx] = False
        donor_total = weights[donor_mask].sum()
        assert np.isclose(donor_total, 1.0)
        expected_donor = proportional[donor_mask] / (1.0 - proportional[bucket_idx])
        donor_error = np.abs(weights[donor_mask] - expected_donor)
        assert donor_error.max() < 1 / aux.MIXTURE_QUANTUM_DENOMINATOR


def test_random_logit_tilts_are_centered_unit_fisher_pairs() -> None:
    proportional = np.asarray([aux.PROPORTIONAL_WEIGHTS[bucket] for bucket in aux.COMPONENTS], dtype=float)
    proportional = proportional / proportional.sum()
    directions = aux._sample_centered_fisher_directions(proportional)

    assert directions.shape == (64, len(aux.COMPONENTS))
    assert np.allclose((directions * proportional[None, :]).sum(axis=1), 0.0, atol=1e-12)
    assert np.allclose(np.sqrt((directions * directions * proportional[None, :]).sum(axis=1)), 1.0)

    by_name = {candidate.candidate_name: candidate for candidate in aux.AUXILIARY_CANDIDATES}
    for direction_idx in range(64):
        plus = by_name[f"pcdir_{direction_idx:03d}_plus"]
        minus = by_name[f"pcdir_{direction_idx:03d}_minus"]
        assert plus.phase_0 == plus.phase_1
        assert minus.phase_0 == minus.phase_1
        assert np.all(_counts_vector(plus.phase_0) >= 1)
        assert np.all(_counts_vector(minus.phase_0) >= 1)
        assert plus.candidate_type == "projected_controllability_plus"
        assert minus.candidate_type == "projected_controllability_minus"


def test_unimax_epoch_caps_bind_before_lattice_quantization() -> None:
    tokens = np.asarray([aux._TOKEN_COUNTS[bucket] for bucket in aux.COMPONENTS], dtype=float)

    previous_phase0: np.ndarray | None = None
    previous_phase1: np.ndarray | None = None
    for epoch_cap in aux.UNIMAX_EPOCH_CAPS:
        phase0 = aux._unimax_weights(tokens, phase_budget=aux.TARGET_BUDGET * aux.PHASE0_FRACTION, epoch_cap=epoch_cap)
        phase1 = aux._unimax_weights(tokens, phase_budget=aux.TARGET_BUDGET * aux.PHASE1_FRACTION, epoch_cap=epoch_cap)
        phase0_epochs = phase0 * aux.TARGET_BUDGET * aux.PHASE0_FRACTION / tokens
        phase1_epochs = phase1 * aux.TARGET_BUDGET * aux.PHASE1_FRACTION / tokens

        assert float(phase0_epochs.max()) <= epoch_cap + 1e-9
        assert float(phase1_epochs.max()) <= epoch_cap + 1e-9
        assert np.isclose(phase0.sum(), 1.0)
        assert np.isclose(phase1.sum(), 1.0)
        if previous_phase0 is not None and previous_phase1 is not None:
            assert not np.allclose(phase0, previous_phase0)
            assert not np.allclose(phase1, previous_phase1)
        previous_phase0 = phase0
        previous_phase1 = phase1


def test_executor_steps_preserve_live_swarm_naming_and_region() -> None:
    steps = aux.swarm_auxiliary_perturbation_steps
    assert len(steps) == len(aux.AUXILIARY_CANDIDATES)
    assert len({step.name for step in steps}) == len(steps)

    for candidate, step in zip(aux.AUXILIARY_CANDIDATES, steps, strict=True):
        slug = f"d512_{candidate.index:06d}"
        assert step.name == f"grug/swarm_fisher_dsp_{slug}"
        assert step.config.run_id == f"swarm_fisher_dsp_{slug}"
        assert step.config.resources.value.device.variant == "v4-8"
        assert step.config.resources.value.zone == "us-central2-b"
        assert step.config.tracker.project == "marin_moe"
        assert step.config.tracker.group == "swarm_fisher_dsp_tau20_lam0p25_uscentral2"
        assert "auxiliary_perturbation" in step.config.tracker.tags
        assert candidate.candidate_type in step.config.tracker.tags
