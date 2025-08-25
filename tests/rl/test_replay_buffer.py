import time

import jax

from marin.rl.datatypes import RolloutRecord, Turn
from marin.rl.replay_buffer import ReplayBuffer


def _make_rollout(uid: str, *, reward: float) -> RolloutRecord:
    return RolloutRecord(
        environment="env",
        example_id="ex",
        policy_version="v1",
        rollout_uid=uid,
        turns=[
            Turn.from_prompt("m", input_seed=None),
            Turn(message="reply", role="assistant", reward=reward),
        ],
        created_ts=time.time(),
        metadata={},
    )


def test_extend_groups_rollouts():
    rb = ReplayBuffer(prng_key=jax.random.PRNGKey(0), min_group_size=2)
    r1 = _make_rollout("u1", reward=1.0)
    r2 = _make_rollout("u2", reward=0.0)

    rb.extend([r1])
    rb.extend([r2])

    # One group keyed by (env, example_id) with two rollouts
    assert len(rb.rollout_groups) == 1
    sole_group = next(iter(rb.rollout_groups.values()))
    assert len(sole_group) == 2


def test_sample_returns_nonzero_advantage_rollouts():
    rb = ReplayBuffer(prng_key=jax.random.PRNGKey(42), min_group_size=2)
    # Two rollouts with different rewards -> non-zero advantages
    rb.extend([_make_rollout("u1", reward=1.0), _make_rollout("u2", reward=0.0)])

    sampled = rb.sample(bsize=1, step=0)
    # Should return at most 1 rollout
    assert len(sampled) == 1
