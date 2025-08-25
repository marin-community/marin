"""Batch maker implementations for creating RL training batches from rollouts."""

from abc import ABC, abstractmethod
from typing import Any
import time

import numpy as np

from .datatypes import RLExample, RolloutRecord


class BatchMaker(ABC):
    """Abstract base class for creating batches of RL examples.

    Different implementations can support different batching strategies.
    """

    @abstractmethod
    def create_batch(self, batch_size: int) -> list[RLExample] | None:
        """Create a batch of RL examples.

        Args:
            batch_size: Target size of the batch

        Returns:
            List of RL examples, or None if insufficient data
        """
        pass

    @abstractmethod
    def get_batch_metadata(self, batch: list[RLExample]) -> dict[str, Any]:
        """Get metadata about a batch for storage purposes.

        Args:
            batch: List of RL examples

        Returns:
            Dictionary containing batch metadata
        """
        pass


class RlooBatchMaker(BatchMaker):
    """RLOO-style batch maker that groups rollouts and computes advantages.

    This implementation:
    - Groups rollouts by (env, problem_id), keeping older rollouts according to policy
    - Computes advantages in the normal way
    - Maintains a reservoir of rollouts with non-zero advantage from valid groups
    """

    def __init__(self, rng_seed: int | None = None):
        # Reservoir of rollouts with non-zero advantage
        self.advantage_reservoir: list[dict[str, Any]] = []

        # Group rollouts by (env, problem_id, policy_version)
        self.rollout_groups: dict[tuple[str, str, str], list[RolloutRecord]] = {}

        if rng_seed is not None:
            self.np_rng = np.random.Generator(np.random.PCG64(rng_seed))
        else:
            self.np_rng = np.random.Generator(np.random.PCG64())

    def add_rollout(self, rollout: RolloutRecord) -> None:
        """Add a rollout to the batch maker.

        Args:
            rollout: RolloutRecord to add
        """
        key = (rollout.environment, rollout.example_id, rollout.policy_version)

        if key not in self.rollout_groups:
            self.rollout_groups[key] = []

        self.rollout_groups[key].append(rollout)

        # Apply policy for keeping older rollouts (e.g., only latest policy version)
        self._apply_retention_policy(key)

        # Compute advantages and add to reservoir if non-zero
        self._process_rollout_group(key)

    def _apply_retention_policy(self, key: tuple[str, str, str]) -> None:
        """Apply retention policy to keep only relevant rollouts.

        Current policy: keep only rollouts from the latest policy version for each problem.
        """
        env, problem_id, _ = key

        # Find the latest policy version for this problem
        latest_policy = None
        for e, p, pv in self.rollout_groups.keys():
            if e == env and p == problem_id:
                if latest_policy is None or pv > latest_policy:
                    latest_policy = pv

        # Remove rollouts from older policy versions
        keys_to_remove = []
        for e, p, pv in self.rollout_groups.keys():
            if e == env and p == problem_id and pv != latest_policy:
                keys_to_remove.append((e, p, pv))

        for old_key in keys_to_remove:
            del self.rollout_groups[old_key]

    def _process_rollout_group(self, key: tuple[str, str, str]) -> None:
        """Process a rollout group to compute advantages and add to reservoir."""
        rollouts = self.rollout_groups[key]

        # Compute advantages for the group
        advantages = _compute_advantages(rollouts)

        # Add rollouts with non-zero advantage to reservoir
        for rollout, advantage in zip(rollouts, advantages, strict=False):
            # Include both positive and negative advantages, just exclude literal 0
            if not np.allclose(advantage, 0.0):
                # Create RLExample and add to reservoir
                rl_example = self._create_rl_example(rollout, advantage)
                self.advantage_reservoir.append(
                    {
                        "example": rl_example,
                        "advantage_sum": np.sum(advantage),
                        "timestamp": time.time(),
                        "rollout": rollout,  # Keep reference to original rollout
                    }
                )

    def _create_rl_example(self, rollout: RolloutRecord, advantage: np.ndarray) -> RLExample:
        """Create an RLExample from rollout data and advantage.

        Args:
            rollout: RolloutRecord data
            advantage: Computed advantage values

        Returns:
            RLExample instance
        """
        # Extract tokens and log probs from turns
        tokens = []
        log_probs = []

        for turn in rollout.turns:
            # For now, we'll use simple tokenization (split by whitespace)
            # In practice, you'd want proper tokenization
            turn_tokens = turn.message.split()
            tokens.extend([hash(token) % 10000 for token in turn_tokens])  # Simple hash-based tokenization

            if turn.logprobs is not None:
                log_probs.extend(turn.logprobs)
            else:
                # If no logprobs, use zeros
                log_probs.extend([0.0] * len(turn_tokens))

        tokens = np.array(tokens, dtype=np.int32)
        log_probs = np.array(log_probs, dtype=np.float32)

        # Create loss mask (all positions except padding)
        loss_mask = np.ones_like(tokens, dtype=bool)

        # Ensure advantage has the same length as tokens
        if len(advantage) != len(tokens):
            # Pad or truncate advantage to match token length
            if len(advantage) < len(tokens):
                advantage = np.pad(advantage, (0, len(tokens) - len(advantage)), mode="constant", constant_values=0.0)
            else:
                advantage = advantage[: len(tokens)]

        return RLExample(
            tokens=tokens, loss_mask=loss_mask, advantage=advantage.astype(np.float32), generator_log_probs=log_probs
        )

    def create_batch(self, batch_size: int) -> list[RLExample] | None:
        """Create a batch of RL examples from the reservoir.

        Args:
            batch_size: Target size of the batch

        Returns:
            List of RL examples, or None if insufficient data
        """
        if len(self.advantage_reservoir) < batch_size:
            return None

        self.np_rng.shuffle(self.advantage_reservoir)

        batch_examples = [item["example"] for item in self.advantage_reservoir[:batch_size]]

        self.advantage_reservoir = self.advantage_reservoir[batch_size:]

        return batch_examples

    def get_batch_metadata(self, batch: list[RLExample]) -> dict[str, Any]:
        """Get metadata about a batch for storage purposes.

        Args:
            batch: List of RL examples

        Returns:
            Dictionary containing batch metadata
        """
        return {
            "batch_size": len(batch),
            "timestamp": time.time(),
            "advantage_stats": {
                "mean": float(np.mean([np.mean(ex.advantage) for ex in batch])),
                "std": float(np.std([np.mean(ex.advantage) for ex in batch])),
                "max": float(np.max([np.max(ex.advantage) for ex in batch])),
                "min": float(np.min([np.min(ex.advantage) for ex in batch])),
            },
        }


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
