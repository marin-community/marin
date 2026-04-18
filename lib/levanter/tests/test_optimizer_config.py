# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from levanter.optim.config import AdamConfig


def test_no_stable_weirdness():
    optimizer = AdamConfig(
        learning_rate=2e-6,  # 2x10^-6
        weight_decay=0.0,
        warmup=0.03,
        min_lr_ratio=0.0,
        lr_schedule="linear",
        max_grad_norm=None,
        cycles=None,
        weight_decay_modules=None,
        default_weight_decay_mask=None,
    )

    sched_fn = optimizer.lr_scheduler(861)

    assert sched_fn(0) == 0.0
    assert np.isclose(sched_fn(int(861 * 0.03)), 2e-6)
    assert np.isclose(sched_fn(int(860)), 0.0)

    # get a middle value
    mid_cooldown = 0.03 + 0.97 / 2
    assert np.isclose(sched_fn(int(861 * mid_cooldown)), 2e-6 / 2)


def test_constant_schedule():
    optimizer = AdamConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup=0.0,
        min_lr_ratio=1.0,  # No decay
        lr_schedule="constant",
        cycles=None,
    )

    sched_fn = optimizer.lr_scheduler(1000)

    assert sched_fn(0) == 1e-3
    assert sched_fn(500) == 1e-3
    assert sched_fn(999) == 1e-3


def test_warmup_and_cosine_decay():
    optimizer = AdamConfig(
        learning_rate=1e-2,
        weight_decay=0.0,
        warmup=0.1,  # 10% of steps
        min_lr_ratio=0.1,
        lr_schedule="cosine",
        cycles=None,
    )

    sched_fn = optimizer.lr_scheduler(1000)

    # Warmup phase
    assert np.isclose(sched_fn(0), 0.0)
    assert np.isclose(sched_fn(50), 0.5e-2)
    assert np.isclose(sched_fn(100), 1e-2)

    # Decay phase
    assert np.isclose(sched_fn(999), 1e-3, atol=1e-5)


def test_linear_schedule_with_cycles():
    optimizer = AdamConfig(
        learning_rate=5e-4,
        weight_decay=0.0,
        warmup=50,
        min_lr_ratio=0.2,
        lr_schedule="linear",
        cycles=2,
    )

    sched_fn = optimizer.lr_scheduler(1000)

    # Warmup phase
    assert np.isclose(sched_fn(0), 0.0)
    assert np.isclose(sched_fn(50), 5e-4)

    num_main_steps = 1000

    first_nadir = num_main_steps // 2 - 1

    # First cycle decay
    assert np.isclose(sched_fn(first_nadir), 0.2 * 5e-4, atol=1e-5)

    # Second cycle starts
    assert np.isclose(sched_fn(first_nadir + 1), 5e-4)

    # midway through second cycle
    midpoint = first_nadir + num_main_steps // 4
    assert np.isclose(sched_fn(midpoint), (5e-4 + 0.2 * 5e-4) / 2, atol=1e-5)

    # Final value
    assert np.isclose(sched_fn(999), 0.2 * 5e-4, atol=1e-5)


def test_wsds_schedule():
    optimizer = AdamConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup=0.0,
        decay=0.1,
        min_lr_ratio=0.1,
        lr_schedule="cosine",
        cycles=[300, 700],
    )

    sched_fn = optimizer.lr_scheduler(1000)

    # First cycle
    assert np.isclose(sched_fn(0), 1e-3)
    assert np.isclose(sched_fn(269), 1e-3)
    assert sched_fn(271) < 1e-3

    # Second cycle
    assert np.isclose(sched_fn(300), 1e-3)
    assert np.isclose(sched_fn(659), 1e-3)
    assert sched_fn(661) < 1e-3

    # Third cycle
    assert np.isclose(sched_fn(701), 1e-3)
    assert np.isclose(sched_fn(969), 1e-3)
    assert sched_fn(971) < 1e-3


def test_inv_sqrt_decay_schedule():
    optimizer = AdamConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup=0.1,
        min_lr_ratio=0.1,
        lr_schedule="inv_sqrt",
        cycles=None,
    )

    sched_fn = optimizer.lr_scheduler(100_000)

    # Warmup phase
    assert np.isclose(sched_fn(0), 0.0)
    assert np.isclose(sched_fn(5000), 0.5e-3)

    # Decay phase: our invsqrt has a non configurable, very long period
    assert sched_fn(50000) < sched_fn(30000)  # Decreasing after warmup


def test_rewarmup_schedule():
    optimizer = AdamConfig(
        learning_rate=1e-2,
        weight_decay=0.0,
        warmup=0.2,  # 20% of cycle
        min_lr_ratio=0.2,
        lr_schedule="linear",
        cycles=2,
        rewarmup=0.05,  # 5% of steps in each cycle
    )

    # cycle length is 500 steps
    sched_fn = optimizer.lr_scheduler(1000)

    # Warmup phase
    assert np.isclose(sched_fn(0), 0.0)
    assert np.isclose(sched_fn(100), 1e-2)  # Warmup reaches max LR

    # First decay phase
    assert np.isclose(sched_fn(300), 0.6e-2)  # Mid of first decay
    assert np.isclose(sched_fn(500), 0.2e-2)  # End of first decay

    # Rewarmup at start of second cycle
    rewarmup_start = 500
    rewarmup_end = rewarmup_start + int(0.05 * 500)
    assert np.isclose(sched_fn(rewarmup_start), 0.2e-2)  # End of previous decay
    assert np.isclose(sched_fn(rewarmup_end), 1e-2)  # Back to max LR after rewarmup
    # make sure this is the high point
    assert sched_fn(rewarmup_end - 1) < sched_fn(rewarmup_end)
    assert sched_fn(rewarmup_end + 1) < sched_fn(rewarmup_end)

    # Final decay phase
    assert sched_fn(999 - 1) > sched_fn(999)
    assert np.isclose(sched_fn(999), 0.2e-2, atol=1e-4)  # End of second decay


def test_linear_schedule_with_cycle_length():
    optimizer = AdamConfig(
        learning_rate=5e-4,
        weight_decay=0.0,
        warmup=50,
        min_lr_ratio=0.2,
        lr_schedule="linear",
        cycle_length=500,
    )

    sched_fn = optimizer.lr_scheduler(1000)

    # Warmup phase
    assert np.isclose(sched_fn(0), 0.0)
    assert np.isclose(sched_fn(50), 5e-4)

    num_main_steps = 1000

    # First cycle decay
    assert np.isclose(sched_fn(499), 0.2 * 5e-4, atol=1e-5)

    # Second cycle starts
    assert np.isclose(sched_fn(500), 5e-4)

    # midway through second cycle
    midpoint = 500 - 1 + num_main_steps // 4
    assert np.isclose(sched_fn(midpoint), (5e-4 + 0.2 * 5e-4) / 2, atol=1e-5)

    # Final value
    assert np.isclose(sched_fn(999), 0.2 * 5e-4, atol=1e-5)


def test_wsds_schedule_with_cycle_points():
    optimizer = AdamConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup=0.0,
        decay=0.1,
        min_lr_ratio=0.1,
        lr_schedule="cosine",
        cycle_length=[300, 400],
    )

    sched_fn = optimizer.lr_scheduler(1000)

    # First cycle
    assert np.isclose(sched_fn(0), 1e-3)
    assert np.isclose(sched_fn(269), 1e-3)
    assert sched_fn(271) < 1e-3

    # Second cycle
    assert np.isclose(sched_fn(300), 1e-3)
    assert np.isclose(sched_fn(659), 1e-3)
    assert sched_fn(661) < 1e-3

    # Third cycle
    assert np.isclose(sched_fn(701), 1e-3)
    assert np.isclose(sched_fn(969), 1e-3)
    assert sched_fn(971) < 1e-3


def test_polynomial_schedule_quadratic():
    """Quadratic decay: (1-t)^2 shape via PolynomialLrSchedule."""
    from levanter.optim.config import PolynomialLrSchedule

    optimizer = AdamConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup=0.1,
        min_lr_ratio=0.0,
        lr_schedule=PolynomialLrSchedule(power=2.0),
    )

    sched_fn = optimizer.lr_scheduler(1000)

    # Warmup phase
    assert np.isclose(sched_fn(0), 0.0)
    assert np.isclose(sched_fn(100), 1e-3)

    # Decay phase: at midpoint (t=450 into 900-step decay), LR = 1e-3 * (1 - 450/900)^2 = 0.25e-3
    assert np.isclose(sched_fn(550), 0.25e-3, atol=1e-6)

    # End of decay
    assert np.isclose(sched_fn(999), 0.0, atol=1e-5)


def test_polynomial_schedule_linear():
    """Power=1 should match linear decay."""
    from levanter.optim.config import PolynomialLrSchedule

    optimizer = AdamConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup=0.0,
        min_lr_ratio=0.0,
        lr_schedule=PolynomialLrSchedule(power=1.0),
    )

    sched_fn = optimizer.lr_scheduler(100)

    assert np.isclose(sched_fn(0), 1e-3)
    assert np.isclose(sched_fn(50), 0.5e-3, atol=1e-6)
    assert np.isclose(sched_fn(100), 0.0, atol=1e-6)


def test_polynomial_schedule_sqrt():
    """Power=0.5 (sqrt decay) holds LR higher early, drops faster at end."""
    from levanter.optim.config import PolynomialLrSchedule

    optimizer = AdamConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup=0.0,
        min_lr_ratio=0.0,
        lr_schedule=PolynomialLrSchedule(power=0.5),
    )

    sched_fn = optimizer.lr_scheduler(100)

    # At midpoint: (1-0.5)^0.5 ≈ 0.707
    assert np.isclose(sched_fn(50), 1e-3 * 0.5**0.5, atol=1e-5)
    assert np.isclose(sched_fn(100), 0.0, atol=1e-6)


def test_polynomial_schedule_with_min_lr():
    """Polynomial decay with a floor (min_lr_ratio > 0)."""
    from levanter.optim.config import PolynomialLrSchedule

    optimizer = AdamConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup=0.0,
        min_lr_ratio=0.05,
        lr_schedule=PolynomialLrSchedule(power=2.0),
    )

    sched_fn = optimizer.lr_scheduler(100)

    # End of decay should reach min_lr = 0.05 * 1e-3
    assert np.isclose(sched_fn(100), 0.05e-3, atol=1e-6)


def test_inv_sqrt_decay_lr_schedule():
    """InvSqrtDecayLrSchedule: lr / sqrt(1 + c * t / T)."""
    from levanter.optim.config import InvSqrtDecayLrSchedule

    optimizer = AdamConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup=0.0,
        min_lr_ratio=0.0,
        lr_schedule=InvSqrtDecayLrSchedule(decay_constant=28.6),
    )

    sched_fn = optimizer.lr_scheduler(1000)

    # At t=0, lr = 1e-3 / sqrt(1) = 1e-3
    assert np.isclose(sched_fn(0), 1e-3)

    # Monotonically decreasing
    assert sched_fn(100) < sched_fn(0)
    assert sched_fn(500) < sched_fn(100)
    assert sched_fn(999) < sched_fn(500)

    # At t=T, lr = 1e-3 / sqrt(1 + 28.6) ≈ 1e-3 / 5.44 ≈ 0.000184
    expected_end = 1e-3 / np.sqrt(1 + 28.6)
    assert np.isclose(sched_fn(1000), expected_end, atol=1e-6)

    # Never reaches zero
    assert sched_fn(1000) > 0


def test_inv_sqrt_decay_lr_schedule_honors_min_lr():
    """InvSqrtDecayLrSchedule should clamp to min_lr when decay_constant is large."""
    from levanter.optim.config import InvSqrtDecayLrSchedule

    optimizer = AdamConfig(
        learning_rate=1e-3,
        weight_decay=0.0,
        warmup=0.0,
        min_lr_ratio=0.1,
        lr_schedule=InvSqrtDecayLrSchedule(decay_constant=200.0),
    )

    sched_fn = optimizer.lr_scheduler(1000)
    min_lr = 1e-3 * 0.1  # 1e-4

    # Without clamping, lr at t=T would be 1e-3 / sqrt(1+200) ≈ 7.06e-5 < min_lr
    unclamped = 1e-3 / np.sqrt(1 + 200.0)
    assert unclamped < min_lr, "test precondition: unclamped value should be below min_lr"

    # With clamping, schedule should never go below min_lr
    assert np.isclose(sched_fn(1000), min_lr, atol=1e-7)
    assert sched_fn(500) >= min_lr - 1e-10


def test_warmup_longer_than_run_does_not_jump():
    optimizer = AdamConfig(
        learning_rate=3e-3,
        weight_decay=0.0,
        warmup=1000,
        decay=0.2,
        min_lr_ratio=0.1,
        lr_schedule="cosine",
    )

    sched_fn = optimizer.lr_scheduler(200)

    assert np.isclose(sched_fn(160), 0.0024, atol=1e-6)
    assert sched_fn(161) > sched_fn(160)
    assert np.isclose(sched_fn(200), 3e-3, atol=1e-6)
