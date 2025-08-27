
import jax
import numpy as np

from .datatypes import GroupKey, RolloutRecord

class ReplayBuffer:
    """

    Basic replay buffer that stores rollouts in memory. This class is deliberately not a Ray actor
    (though it can be decorated with @ray.remote just-in-time or wrapped)

    We'll add support for writing to disk later.
    """

    def __init__(
        self,
        *,
        prng_key: PRNGKeyArray,
        min_group_size: int = 4,
    ) -> None:

        self.prng_key = prng_key
        self.min_group_size = min_group_size
        if min_group_size < 2:
            raise ValueError("min_group_size must be at least 2")

        self.rollout_groups: dict[GroupKey, list[RolloutRecord]] = {}

    def extend(self, rollouts: list[RolloutRecord]):
        for rollout in rollouts:
            key = GroupKey(rollout.environment, rollout.example_id)
            group = self.rollout_groups.get(key)
            if group is None:
                self.rollout_groups[key] = [rollout]
            else:
                group.append(rollout)

    def purge(self):
        self.rollout_groups = {}

    def sample(self, *, bsize: int, step: int):
        # TODO: packing?
        # TODO: should we track advantage stats online
        # TODO: log reward and advantage stats
        # find useful rollout groups (those with some non-zero advantage entry)
        useful_rollouts: list[RolloutRecord] = []
        for _key, group in self.rollout_groups.items():
            if len(group) <= 1:
                continue
            advantages = _compute_advantages(group)

            for rollout, advantage in zip(group, advantages, strict=False):
                if not np.allclose(advantage, 0.0):
                    useful_rollouts.append(rollout)

        this_prng = jax.random.fold_in(self.prng_key, step)
        n = len(useful_rollouts)
        if n == 0:
            return []
        # Permute indices to avoid object array issues
        perm = jax.random.permutation(this_prng, n)
        # Convert to Python ints
        import numpy as _np

        idx = list(map(int, _np.array(perm[:bsize])))
        # TODO: remove selected from pool
        return [useful_rollouts[i] for i in idx]


def _compute_advantages(rollouts: list[RolloutRecord]) -> list[np.ndarray]:
    """Compute advantages for a group of rollouts.

    Args:
        rollouts: List of rollouts in the group

    Returns:
        List of advantage arrays for each rollout
    """
    # Extract rewards from turns
    rewards = []
    for rollout in rollouts:
        # Sum rewards across all turns in the rollout
        rollout_reward = sum(turn.reward or 0.0 for turn in rollout.turns)
        rewards.append(rollout_reward)

    # Compute advantages using RLOO method
    advantages = []
    for i in range(len(rollouts)):
        # RLOO: compute advantage relative to other rollouts in the group
        other_rewards = np.concatenate([rewards[:i], rewards[i + 1 :]])
        if len(other_rewards) > 0:
            advantage = rewards[i] - np.mean(other_rewards)
            # Normalize
            if np.std(other_rewards) > 1e-8:
                advantage = advantage / np.std(other_rewards)
        else:
            advantage = 0.0

        # Create advantage array for each token position
        # For now, we'll use the same advantage for all positions in a rollout
        # In practice, you might want more sophisticated token-level advantage computation
        total_tokens = sum(len(turn.message.split()) for turn in rollouts[i].turns)
        advantages.append(np.full(total_tokens, advantage, dtype=np.float32))

    return advantages