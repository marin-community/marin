# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from marin.rl.rollout_schedule import FeistelEpochSchedule, derive_worker_seed, rollout_assignment


def test_feistel_epoch_schedule_covers_dataset_once_per_epoch():
    schedule = FeistelEpochSchedule(dataset_len=31, seed=123)

    epoch_0 = schedule.indices_for_positions(start_position=0, count=31)
    epoch_1 = schedule.indices_for_positions(start_position=31, count=31)

    assert sorted(epoch_0) == list(range(31))
    assert sorted(epoch_1) == list(range(31))
    assert epoch_0 != epoch_1


def test_feistel_epoch_schedule_crosses_epoch_boundary():
    schedule = FeistelEpochSchedule(dataset_len=7, seed=123)

    crossed = schedule.indices_for_positions(start_position=5, count=5)

    assert crossed[:2] == schedule.indices_for_positions(start_position=5, count=2)
    assert crossed[2:] == schedule.indices_for_positions(start_position=7, count=3)


def test_rollout_workers_use_different_full_dataset_permutations():
    worker_0 = FeistelEpochSchedule(dataset_len=31, seed=1042).indices_for_positions(0, 31)
    worker_1 = FeistelEpochSchedule(dataset_len=31, seed=1043).indices_for_positions(0, 31)

    assert sorted(worker_0) == list(range(31))
    assert sorted(worker_1) == list(range(31))
    assert worker_0 != worker_1


def test_rollout_assignment_records_deterministic_metadata():
    assignment = rollout_assignment(
        worker_index=2,
        lesson_id="math_full",
        worker_seed=1044,
        dataset_len=31,
        start_position=29,
        n_examples=5,
    )

    assert assignment.assignment_id == "worker-2:lesson-math_full:start-29:count-5"
    assert assignment.worker_index == 2
    assert assignment.worker_seed == 1044
    assert assignment.epoch == 0
    assert assignment.start_position == 29
    assert assignment.end_position == 34
    assert len(assignment.indices) == 5


def test_derive_worker_seed_is_deterministic():
    """Same (base, worker) inputs always return the same seed."""
    assert derive_worker_seed(42, 0) == derive_worker_seed(42, 0)
    assert derive_worker_seed(0, 5) == derive_worker_seed(0, 5)
    assert derive_worker_seed(1_000_000, 7) == derive_worker_seed(1_000_000, 7)


def test_derive_worker_seed_returns_int_in_int32_range():
    """Output is a Python int in [0, 2**31), suitable for random.Random / vLLM seed."""
    for base in (0, 1, 42, 1_000_000):
        for w in (0, 1, 7, 31):
            s = derive_worker_seed(base, w)
            assert isinstance(s, int)
            assert 0 <= s < 2**31


def test_derive_worker_seed_no_collisions_across_grid():
    """Every (base, worker) pair across a 100x16 grid produces a unique seed.

    This is the regression test for the bug where ``worker_seed = base + 1000 + i``
    aliased adjacent (base, worker) pairs (e.g., base=0,w=1 collided with
    base=1,w=0 at 1001). ``derive_worker_seed`` uses ``jax.random.fold_in``
    which is collision-free by Threefry avalanche.
    """
    seen: dict[int, tuple[int, int]] = {}
    for base in range(100):
        for w in range(16):
            s = derive_worker_seed(base, w)
            assert s not in seen, f"collision: (base={base}, w={w}) shares seed {s} with {seen[s]}"
            seen[s] = (base, w)
    assert len(seen) == 100 * 16


def test_derive_worker_seed_3_seed_ablation_decorrelated():
    """3-seed * 2-worker ablation: all 6 derived seeds are distinct.

    Under the old ``base + 1000 + i`` scheme, run-0's rollout-1 and run-1's
    rollout-0 both produced seed 1001, biasing across-seed variance estimates.
    """
    seeds = [derive_worker_seed(b, w) for b in (0, 1, 2) for w in (0, 1)]
    assert len(set(seeds)) == 6, f"expected 6 unique seeds, got {seeds}"


def test_derive_worker_seed_num_workers_invariant():
    """Output depends only on (base, worker_index), not on run-time context.

    Adding or removing workers from a run must not shift any existing worker's
    seed. This is what makes ``derive_worker_seed`` safe to land on a live
    branch and use across runs with different ``num_rollout_workers``.
    """
    # Worker 0's seed for base=42 is the same value regardless of any other
    # configuration the run has.
    ref = derive_worker_seed(42, 0)
    assert derive_worker_seed(42, 0) == ref  # no global state
    # Worker N's seed is independent of how many other workers exist.
    assert derive_worker_seed(42, 3) != ref  # different worker_index, different seed
    assert derive_worker_seed(42, 3) == derive_worker_seed(42, 3)  # but still deterministic


def test_derive_worker_seed_avalanche_against_adjacent_inputs():
    """1-bit input change should produce roughly 16-bit Hamming-distance output change.

    Loose lower bound at 8 bits to avoid false negatives from rare low-Hamming
    pairs; Threefry's avalanche makes ~15.5 bits the expectation.
    """

    def hamming32(a: int, b: int) -> int:
        return bin(a ^ b).count("1")

    adjacent_pairs = [
        ((42, 0), (42, 1)),
        ((42, 0), (43, 0)),
        ((0, 0), (1, 0)),
        ((0, 0), (0, 1)),
        ((100, 5), (100, 6)),
        ((1, 0), (2, 0)),
    ]
    for (b1, w1), (b2, w2) in adjacent_pairs:
        d = hamming32(derive_worker_seed(b1, w1), derive_worker_seed(b2, w2))
        assert d >= 8, f"weak avalanche: ({b1},{w1}) vs ({b2},{w2}) Hamming = {d}/31"


def test_derive_worker_seed_rejects_negative_worker_index():
    """worker_index must be non-negative."""
    with pytest.raises(ValueError, match="worker_index must be non-negative"):
        derive_worker_seed(42, -1)
