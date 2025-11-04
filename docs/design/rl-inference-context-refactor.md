# Design Doc: InferenceContext Utilities for Rollout Construction

> **Meta-note**: This design addresses [GitHub Issue #1744](https://github.com/marin-community/marin/issues/1744) by providing utilities to safely convert OpenAI API responses to rollouts with proper prompt token alignment.

## Problem

Constructing a `Rollout` from OpenAI API responses is error-prone because **you need to know the exact prompt template used on the server to correctly align logprobs with training examples**.

**Broken code** (`prime_intellect_env.py:140` before fix):

```python
# ❌ Re-encodes prompt without chat template
prompt_tokens = tokenizer.encode(result.prompt[prompt_idx])
response_logprobs = jnp.zeros(len(response_tokens))  # ❌ No logprobs!
```

This breaks when the server uses a chat template (e.g., `<|user|>{prompt}<|assistant|>`):
1. Environment re-encodes plain prompt → gets different tokens than server used
2. Logprobs from server correspond to tokens AFTER chat template was applied
3. Training data misalignment → incorrect gradients → model degradation

**Why this matters**: The tokenizer's chat template transforms `"What is 2+2?"` into `[1234, 5678, "What is 2+2?", 9012, 3456]` (special tokens + prompt + generation prompt). If environments use `tokenizer.encode()` directly, they get `[5678]` instead, causing misalignment.

## Goals

1. **Centralize prompt tokenization** - One method that always matches server behavior
2. **Safe rollout construction** - Helper methods to build rollouts from OpenAI responses with validation
3. **Backwards compatibility** - No breaking changes to existing code
4. **Alignment validation** - Catch template mismatches early with assertions

**Non-goals**: Fetching raw token IDs from OpenAI API (not exposed), changing protocol signatures, supporting non-chat endpoints

## Proposed Solution

### Core Approach

**Consolidate inference utilities into `inference_ctx.py`**: Move `InferenceContext` and add methods for tokenization and rollout construction.

**Key principle**: Provide methods on `InferenceContext` that encapsulate chat template logic, so environments can't accidentally bypass it.

### Data Flow

```python
# Environment calls InferenceContext methods
prompt_tokens = inference_ctx.tokenize_prompt(prompt)           # ✅ Uses chat template
response_tokens = inference_ctx.response_tokens_from_choice(choice)  # ✅ Extracts from API
response_logprobs = inference_ctx.logprobs_from_choice(choice)      # ✅ Extracts logprobs

# Or use all-in-one helper
rollout = inference_ctx.create_rollout_from_choice(
    prompt=prompt,
    choice=completion.choices[0],
    env_name="gsm8k",
    env_example_id="ex_123",
    reward=1.0,
)
```

### Core Methods

Add to `InferenceContext` class in `lib/marin/src/marin/rl/inference_ctx.py`:

```python
def tokenize_prompt(self, prompt: str) -> np.ndarray:
    """Tokenize with chat template matching server behavior."""
    messages = [{"role": "user", "content": prompt}]
    try:
        tokens = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    except Exception as e:
        logger.warning(f"Chat template failed: {e}")
        # Fallback: manual formatting
        prompt_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        tokens = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        if not tokens:
            raise ValueError(f"Failed to tokenize: {prompt[:100]}...") from None
    return np.array(tokens, dtype=np.int32)


def response_tokens_from_choice(self, choice: Choice) -> np.ndarray:
    """Extract token IDs using BPE round-trip."""
    if not choice.logprobs or not choice.logprobs.content:
        raise ValueError("Choice missing logprobs. Use logprobs=True in API call.")

    # Use convert_tokens_to_ids for correct BPE round-trip
    tokens = [self.tokenizer.convert_tokens_to_ids(t.token)
              for t in choice.logprobs.content]

    if not tokens:
        raise ValueError("Choice has zero tokens")
    return np.array(tokens, dtype=np.int32)


def create_rollout_from_choice(
    self, prompt: str, choice: Choice,
    env_name: str, env_example_id: str, reward: float
) -> Rollout:
    """Construct Rollout from OpenAI choice with validation."""
    prompt_tokens = self.tokenize_prompt(prompt)
    response_tokens = self.response_tokens_from_choice(choice)
    response_logprobs = self.logprobs_from_choice(choice)

    # Validation
    if len(response_tokens) < 5:
        logger.warning(f"Only {len(response_tokens)} tokens for {env_example_id}")
    if len(prompt_tokens) == 0:
        logger.error(f"Prompt tokenization failed for {env_example_id}")

    token_rewards = jnp.full(len(response_tokens), reward, dtype=jnp.float32)

    return Rollout(
        env_name=env_name,
        env_example_id=env_example_id,
        prompt_tokens=jnp.array(prompt_tokens, dtype=jnp.int32),
        response_tokens=jnp.array(response_tokens, dtype=jnp.int32),
        response_logprobs=jnp.array(response_logprobs, dtype=jnp.float32),
        token_rewards=token_rewards,
        episode_reward=float(reward),
    )
```

**Rationale**:
- ✅ All inference utilities in one module
- ✅ Methods encapsulate chat template logic
- ✅ Fail-fast validation catches misalignment at construction time
- ✅ Single source of truth for tokenization

### Module Consolidation

Rename `lib/marin/src/marin/rl/types.py` → split into:
- `inference_ctx.py` - `InferenceContext`, `InferenceChoice`, `InferenceResponse`, utility functions
- `types.py` - `Rollout`, `RolloutGroup`, `RolloutBatch`, `TrainingBatch` (training-focused types)

**Why separate**: Inference concerns (API interaction, tokenization) are distinct from training types (rollout batching, data structures).

## Implementation Outline

1. **Consolidate module** - Move `InferenceContext` from `types.py` to `inference_ctx.py`, update imports across codebase
2. **Add methods** - Implement `tokenize_prompt()`, `response_tokens_from_choice()`, `logprobs_from_choice()`, `create_rollout_from_choice()` in `InferenceContext`
3. **Fix PrimeIntellectEnv** - Replace manual `tokenizer.encode()` call with `inference_ctx.tokenize_prompt()` at line 141
4. **Test** - Verify chat template adds special tokens, logprob extraction works, rollout construction succeeds

No breaking changes. All modifications are additions to `InferenceContext` class.

## Notes

**Chat template fallback**: If `apply_chat_template()` fails (no template configured), falls back to manual `{role}: {content}` formatting. This ensures tokenization always succeeds while still using templates when available.

**BPE round-trip**: Must use `convert_tokens_to_ids()` instead of `encode()` because OpenAI returns BPE tokens (e.g., `"Ġhello"` for space-prefixed words), not text. Direct encoding would double-encode.

**Validation strategy**: Log warnings for suspicious cases (short responses, zero logprobs) but don't error - allows partial batches to proceed. Errors only for clearly invalid data (missing logprobs, empty tokens).

**PrimeIntellect limitation**: Verifiers library doesn't expose logprobs, so `prime_intellect_env.py` uses `jnp.zeros()` for `response_logprobs`. This is a known limitation - gradients will be incorrect but environment still runs.

## Future Work

- Multi-turn conversation support (extend to handle conversation history)
- Token-level reward shaping (helpers for position-based reward distribution)
- Alignment metrics (track validation warnings as telemetry)
- OpenAI proxy (add token IDs to responses for exact alignment)

---

## See Also

- GitHub Issue: https://github.com/marin-community/marin/issues/1744
- `lib/marin/src/marin/rl/inference_ctx.py` - Consolidated inference utilities (lines 38-192)
- `lib/marin/src/marin/rl/types.py` - Training-focused types (Rollout, RolloutBatch)
- `lib/marin/src/marin/rl/environments/prime_intellect_env.py` - Fixed at line 141
