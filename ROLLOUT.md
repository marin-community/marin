# Planning

We want to clean up environment rollouts.
Environments should focus on the minimal surface area for their work, computing:

* Per token rewards
* Prompt & output tokens

The output of an environment should now be:

class Rollout:
  prompt_tokens: list[int]
  response_tokens: list[int]
  rewards: list[int] # indexed against response tokens

Rollouts are grouped for GRPO/RLOO:

class RolloutGroup:
  key: str # example used for this group
  rollouts: list[Rollout]

Reference logprobs are now computed on the training worker
on demand. We do not optimize the reference logprobs so
we recalc for each group member. We could cache these later

# New Format

// rollout_types.py

class NewRollout(eqx.Module):
    env_name: str
    env_example_id: str
    prompt_tokens: jax.Array
    response_tokens: jax.Array
    response_logprobs: jax.Array
    token_rewards: jax.Array
    episode_reward: float


class NewRolloutGroup(eqx.Module):
    key: str
    rollouts: list[NewRollout]

    def compute_rloo_advantages(self) -> np.ndarray:
        """Compute RLOO advantages for this group."""
        rewards = np.array([r.episode_reward for r in self.rollouts])
        n = len(rewards)
        if n <= 1:
            return np.zeros_like(rewards)

        total = rewards.sum()
        leave_one_out_baselines = (total - rewards) / (n - 1)
        advantages = rewards - leave_one_out_baselines

        # Add random noise to avoid failure cases when all rewards are identical
        generator = np.random.default_rng()
        advantages += generator.normal(loc=0.0, scale=1e-6, size=advantages.shape)
        return advantages


@dataclass
class NewRolloutMetadata:
    worker_id: str
    timestamp: float


class NewRolloutBatch(eqx.Module):
    groups: list[NewRolloutGroup]
    metadata: NewRolloutMetadata


# New system
So now:

Rollout worker:

no longer calls into the dataset
calls into the environment
makes a rollout group
hands off to replay buffer

# TESTING
You always test with:

```
uv run pytest tests/post_training/test_async_train.py::test_full_integration_moar_cats -o log_cli_level=WARNING
```

# Testable steps

"big move"

step 1. DONE

move reference logprob cacluation into training worker.
ignore logprobs from the rollout
validate this still works as expected

step 2. DONE

delete reference logprobs from rollout, ignore them from the rl_dataset

step 3. DONE

compute rollout groups manually instead of via rl_env
still compute in current format, just happens in rollout worker now instead of rl_dataset
we will eventually move these into a library for use with the environments
we want to be able to test enviroments more independently as well

step 4.

define new format, serialize in old format from rollout worker. DONE
compute rloo award on rollout worker still. DONE

step 5

serialize in NEW format: update rollout storage
move new -> old format conversion into replay_buffer.py
restore into old format before storing in the replay buffer

VALIDATE with test_async_traing

step 6

replay_buffer stores NEW FORMAT
rloo award assignment occurs on the replay buffer
replay_buffer computes a "training_batch" dictionary at sample generation time
reuse logic from NEW -> OLD conversion

VALIDATE with test_async_traing

step 7

REMOVE OLD ROULLOUT BATCH CODE.
