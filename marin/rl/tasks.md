## Meta

- [ ] understand charlie's code
- [ ] do we want to do disaggegrated inference

## Transfer
- [x] Implement weight transfer coordinator
- [x] make benchmark_weight_transfer to use new coordinator
- [ ] Multiple servers (one per training node)

## Coordination
- [ ] parameterize and initialize environments
- [ ] gather and log rollouts
- [ ] central coordinator that holds weight server address, does logging, and spins up environments



## ReplayBuffer

- [ ] Store rollouts in a replay buffer



## Training
- [ ] skeleton loop
- [ ] initialize model
- [ ] objective function
- [ ] optimizer
- [ ] aux losses (kl) and logging


## Metrics
- cf https://huggingface.co/blog/putting_rl_back_in_rlhf_with_rloo
- [ ] **eps**: Tracks the number of episodes per second.
- [ ] **objective/kl**: The mean Kullback-Leibler (KL) divergence between the current policy and reference policy.
- [ ] **objective/entropy**: The mean entropy of the policy, indicating the randomness of the actions chosen by the policy.
- [ ] **objective/non_score_reward**: The mean reward from non-score-related sources, basically `beta * kl.sum(1)`, where beta is the KL penalty coefficient and kl is the per-token KL divergence.
- [ ] **objective/rlhf_reward**: The mean RLHF reward, which is `score - non_score_reward`.
- [ ] **objective/scores**: The mean scores returned by the reward model / environment.
- [ ] **policy/approxkl_avg**: The average approximate KL divergence between consecutive PPO policies. Note that this is not the same as `objective/kl`.
- [ ] **policy/clipfrac_avg**: The average fraction of policy updates that are clipped, indicating how often the policy updates are constrained to prevent large changes.
- [ ] **loss/policy_avg**: The average policy loss, indicating how well the policy is performing.
- [ ] **val/clipfrac_avg**: The average fraction of value function updates that are clipped, similar to `policy/clipfrac_avg` but for the value function.
- [ ] **policy/entropy_avg**: The average entropy of the policy during training, indicating how diverse the policy’s actions are.
- [ ] **val/ratio**: The mean ratio of the current policy probability to the old policy probability, providing a measure of how much the policy has changed.
- [ ] **val/ratio_var**: The variance of the `val/ratio`, indicating the variability in policy changes.
- [ ] **val/num_eos_tokens**: The number of end-of-sequence (EOS) tokens generated, which can indicate the number of complete responses.
- [ ] **lr**: The current learning rate used by the optimizer.
- [ ] **episode**: The current global step or episode count in the training process.
- [ ] **train/rewards**
- [ ] **train/format_rewards**
- [ ] **train/correct_rewards**
- [ ] **train/output_length**





## Sketches
- RLExample:

```python
import haliax.haxtyping as ht
import jaxtyping as jt
from marin.rl.types import Rollout

class RlExample(eqx.Module):
    input_ids: ht.i32[NamedArray, "batch position"]
    loss_mask: ht.bool_[NamedArray, "batch position"]  # indicates prompt vs not prompt
    segment_ids: ht.i32[NamedArray, "batch position"]  # mostly 1/0 for padding
    loss_weights: ht.f32[NamedArray, "batch position"]  # RLOO advantages or similar
    policy_logprobs: ht.Float[NamedArray, "batch position"]
    reference_logprobs: ht.Float[NamedArray, "batch position"]

    def to_lm_example() -> LmExample:
        return LmExample(
            tokens=self.input_ids,
            loss_mask=self.loss_mask,
            attn_mask=AttentionMask.causal().with_segment_ids(self.segment_ids),
        )



@dataclass(frozen=True)
class ProcessedRollout:
    source: str
    id: str
    input_ids: jt.i32[np.ndarray, "batch position"]
    loss_mask: jt.bool_[np.ndarray, "batch position"]
    segment_ids: jt.i32[np.ndarray, "batch position"]
    returns: jt.f32[np.ndarray, "batch"]
    reference_logprobs: jt.f32[np.ndarray, "batch position"]
    policy_logprobs: jt.f32[np.ndarray, "batch position"]
    

def process_rollout_group(
    tokenizer: "HfTokenizer",  # type: ignore
    rollout: RolloutGroup,  # type: ignore
    max_length: int,
    apply_chat_template: bool = True,
) -> "ProcessedRollout":  # type: ignore
    """Process a rollout into a format suitable for training."""
    rollout_items = []
    for rollout in rollout.groups:
        if apply_chat_template:
            messages = [{"role": turn.role, "content": turn.message} for turn in rollout.turns]
            tokenized = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_attenteon_mask=True,
                return_assistant_tokens_mask=True,
                max_length=max_length,
                truncation=True,
            )
            input_ids = np.array(tokenized["input_ids"], dtype=np.int32)
            loss_mask = np.array(tokenized["assistant_masks"], dtype=np.bool_)
            segment_ids = np.ones_like(input_ids, dtype=np.int32)

        else:
            # Simple concatenation of turns without a chat template.
            # This is a simplified approach. A more robust implementation would handle special tokens and roles more carefully.
            all_input_ids = []
            all_loss_mask = []
            all_segment_ids = []

            for turn in rollout.turns:
                turn_ids = tokenizer(turn.message, add_special_tokens=False)["input_ids"]
                all_input_ids.extend(turn_ids)
                if turn.role == "assistant":
                    all_loss_mask.extend([True] * len(turn_ids))
                else:
                    all_loss_mask.extend([False] * len(turn_ids))
                all_segment_ids.extend([1] * len(turn_ids))

            # Pad or truncate
            input_ids = np.array(all_input_ids[:max_length], dtype=np.int32)
            loss_mask = np.array(all_loss_mask[:max_length], dtype=np.bool_)
            segment_ids = np.array(all_segment_ids[:max_length], dtype=np.int32)

        if len(input_ids) < max_length:
            padding_len = max_length - len(input_ids)
            input_ids = np.pad(input_ids, (0, padding_len), constant_values=tokenizer.pad_token_id)
            loss_mask = np.pad(loss_mask, (0, padding_len), constant_values=False)
            segment_ids = np.pad(segment_ids, (0, padding_len), constant_values=0)

    total_reward = sum(turn.reward for turn in rollout.turns if turn.reward is not None)
    returns = np.array([total_reward], dtype=np.float32)

    policy_logprobs = np.zeros_like(input_ids, dtype=np.float32)
    reference_logprobs = np.zeros_like(input_ids, dtype=np.float32)

    # The ProcessedRollout is defined to not be batched, so we add a batch dimension.
    return ProcessedRollout(
        input_ids=input_ids[None, :],
        loss_mask=loss_mask[None, :],
        segment_ids=segment_ids[None, :],
        returns=returns,
        reference_logprobs=reference_logprobs[None, :],
        policy_logprobs=policy_logprobs[None, :],
    )

def compute_rloo_advantages_for_group(rewards: np.ndarray) -> np.ndarray:
    """Compute RLOO advantages for a group of rewards.

    Args:
        rewards: Array of rewards for a group

    Returns:
        Normalized advantages
    """
    advantages = (rewards - rewards.mean()) / np.clip(rewards.std(), 1e-8, None)
    return advantages



```


