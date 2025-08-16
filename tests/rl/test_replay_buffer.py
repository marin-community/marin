import ray
import pytest

from marin.rl.datatypes import RolloutRecord
from marin.rl.replay_buffer import ReplayBuffer


@pytest.fixture(scope="module", autouse=True)
def _ray():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


def test_idempotent_rollout(tmp_path):
    rb = ReplayBuffer.remote(str(tmp_path), target_group_size=1)
    r = RolloutRecord(environment="env", example_id="ex", policy_version="v1", rollout_uid="u1")
    ray.get(rb.add_rollout.remote(r))
    ray.get(rb.add_rollout.remote(r))
    ray.get(rb.flush.remote())
    groups = ray.get(rb.list_groups.remote())
    assert len(groups) == 1
    assert len(groups[0].rollouts) == 1


def test_timeout_sealing(tmp_path):
    rb = ReplayBuffer.remote(
        str(tmp_path), target_group_size=10, min_group_size=2, seal_timeout_s=0
    )
    r1 = RolloutRecord(environment="env", example_id="ex", policy_version="v1", rollout_uid="u1")
    r2 = RolloutRecord(environment="env", example_id="ex", policy_version="v1", rollout_uid="u2")
    ray.get(rb.add_rollout.remote(r1))
    ray.get(rb.add_rollout.remote(r2))
    groups = ray.get(rb.list_groups.remote())
    assert len(groups) == 1
    assert len(groups[0].rollouts) == 2
