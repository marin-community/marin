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
