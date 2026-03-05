# RL Rollout Metadata Migration

## Motivation

Currently, rollout metadata (worker_id, timestamp, weight_step) lives at the batch level in `RolloutMetadata`, attached to `RolloutBatch`. This creates several issues:

1. **Replay buffer filtering is imprecise**: When filtering stale rollouts by timestamp or weight_step, we can only filter entire batches, not individual rollouts
2. **Metadata is per-batch, not per-rollout**: All rollouts in a batch share the same metadata, but conceptually each rollout has its own creation context
3. **Incomplete timestamp filtering**: The `max_rollout_timestamp_delay` feature is partially implemented but not fully functional

This migration moves metadata from the batch level to the rollout level, enabling precise filtering and completing the timestamp-based staleness detection.

## Design

### Current Structure

```python
@dataclass
class RolloutMetadata:
    worker_id: str
    timestamp: float
    weight_step: int

class Rollout(eqx.Module):
    env_name: str
    env_example_id: str
    prompt_tokens: jax.Array
    response_tokens: jax.Array
    response_logprobs: jax.Array
    token_rewards: jax.Array
    episode_reward: float
    # No metadata

class RolloutBatch(eqx.Module):
    groups: list[RolloutGroup]
    metadata: RolloutMetadata  # Batch-level metadata
```

### Proposed Structure

```python
@dataclass
class RolloutMetadata:
    """Metadata about when/where a rollout was generated."""
    worker_id: str
    timestamp: float
    weight_step: int

class Rollout(eqx.Module):
    """A single rollout: one prompt + one generated response + rewards."""

    env_name: str
    env_example_id: str
    prompt_tokens: jax.Array
    response_tokens: jax.Array
    response_logprobs: jax.Array
    token_rewards: jax.Array
    episode_reward: float

    metadata: RolloutMetadata  # Now on each rollout

class RolloutBatch(eqx.Module):
    """A batch of rollout groups with metadata."""

    groups: list[RolloutGroup]
    metadata: RolloutMetadata  # Kept for backward compatibility (redundant)
```

### Key Principles

1. **Environments don't know about metadata**: The environment interface remains clean and focused on domain logic
2. **Worker attaches metadata**: After getting rollouts from environment, the rollout worker wraps each with metadata
3. **Replay buffer reads from rollout**: Filtering logic uses `rollout.metadata` instead of `batch.metadata`
4. **Backward compatible**: Keep `RolloutBatch.metadata` during transition to avoid breaking serialized data

## Implementation Plan

### Phase 1: Add metadata field to Rollout

**File:** `src/marin/rl/types.py`

```python
class Rollout(eqx.Module):
    """A single rollout: one prompt + one generated response + rewards."""

    env_name: str
    env_example_id: str
    prompt_tokens: jax.Array
    response_tokens: jax.Array
    response_logprobs: jax.Array
    token_rewards: jax.Array
    episode_reward: float

    # Metadata about when/where this rollout was generated
    metadata: RolloutMetadata
```

Keep `RolloutMetadata` and `RolloutBatch` unchanged.

### Phase 2: Update rollout worker to attach metadata

**File:** `src/marin/rl/rollout_worker.py`

The rollout worker should:
1. Generate rollouts from environment (environment stays unchanged)
2. Create metadata once per batch
3. Attach metadata to each rollout
4. Create RolloutBatch (with duplicate metadata for backward compat)

```python
def _sample_batch(self, lesson_id: str, mode: str, rng) -> tuple[RolloutBatch | None, dict | None]:
    """Sample a batch of rollouts from the environment for the given lesson ID."""

    # ... existing setup code ...

    # Generate rollouts from environment (no metadata needed here)
    with (
        self.config.trainer.device_mesh,
        hax.axis_mapping(self.config.trainer.compute_axis_mapping),
    ):
        rollout_groups, metrics = env.sample(
            inference_ctx=policy_ctx,
            n_examples=n_examples,
            n_generations=n_generations,
            temperature=temperature,
            prng_key=rng,
            mode=mode,
        )

    if len(rollout_groups) == 0:
        logger.warning("No valid rollouts generated in this batch...")
        return None, None

    # Create metadata once for this batch
    batch_metadata = RolloutMetadata(
        worker_id=f"{socket.gethostname()}_{os.getpid()}",
        timestamp=time.time(),
        weight_step=self._current_weight_step,
    )

    # Attach metadata to each rollout in each group
    rollout_groups_with_metadata = []
    for group in rollout_groups:
        rollouts_with_metadata = []
        for rollout in group.rollouts:
            # Create new rollout with metadata attached
            rollout_with_meta = eqx.tree_at(
                lambda r: r.metadata,
                rollout,
                batch_metadata
            )
            rollouts_with_metadata.append(rollout_with_meta)

        rollout_groups_with_metadata.append(
            RolloutGroup(rollouts=rollouts_with_metadata)
        )

    rollout_batch = RolloutBatch(
        groups=rollout_groups_with_metadata,
        metadata=batch_metadata,  # Keep for backward compatibility
    )

    return rollout_batch, metrics
```

### Phase 3: Update replay buffer to read from rollout metadata

**File:** `src/marin/rl/replay_buffer.py`

#### Update config

```python
@dataclass
class ReplayBufferConfig:
    """Configuration for the replay buffer."""

    capacity: int
    alpha: float
    max_samples: int

    max_rollout_step_delay: int
    """Maximum age of rollouts in training steps."""

    max_rollout_timestamp_delay: float = 3600.0
    """Maximum age of rollouts in seconds."""
```

#### Update ReplayBuffer fields

```python
@dataclass
class ReplayBuffer:
    capacity: int
    local_batch_size: int
    alpha: float
    total_processes: int
    process_id: int
    max_samples: int
    max_rollout_step_delay: int
    max_rollout_timestamp_delay: float  # Renamed for clarity
    loss_module: RLLossModule
    # ... rest unchanged ...
```

#### Update set_current_step to use rollout.metadata

```python
def set_current_step(self, step: int) -> None:
    """Set current training step and filter stale rollouts."""
    self._current_step = step
    min_time = time.time() - self.max_rollout_timestamp_delay
    min_step = step - self.max_rollout_step_delay

    logger.info(
        "Discarding rollouts older than step %d or timestamp %.0f (current step %d)",
        min_step, min_time, step
    )

    with self._lock:
        total_removed = 0
        for env_name in self.rollout_storage:
            rollouts = self.rollout_storage[env_name]
            before = len(rollouts)

            # Filter using rollout.metadata instead of batch metadata
            self.rollout_storage[env_name] = [
                r for r in rollouts
                if (r.rollout.metadata.weight_step >= min_step and
                    r.rollout.metadata.timestamp > min_time)
            ]

            total_removed += before - len(self.rollout_storage[env_name])

        total_remaining = sum(len(rollouts) for rollouts in self.rollout_storage.values())

        if total_removed > 0:
            logger.info(
                f"Filtered {total_removed} stale rollouts "
                f"(min_step={min_step}, min_time={min_time:.0f}), "
                f"{total_remaining} remaining"
            )
```

#### Update add_batches to read from rollout.metadata

```python
def add_batches(self, new_batches: list[RolloutBatch]) -> None:
    """Add new rollout batches into the replay buffer."""
    env_examples: dict[str, list[RolloutWithCount]] = defaultdict(list)

    for batch in new_batches:
        if not batch.groups or not batch.groups[0].rollouts:
            continue

        # Read weight_step from first rollout's metadata
        first_rollout = batch.groups[0].rollouts[0]
        weight_step = first_rollout.metadata.weight_step

        if weight_step < self._current_step - self.max_rollout_step_delay:
            logger.info(
                f"Skipping stale rollout batch "
                f"(weight_step={weight_step}, current_step={self._current_step})"
            )
            continue

        self._total_batches_added += 1

        for group in batch.groups:
            advantages = self.loss_module.compute_advantages(group.rollouts)
            for rollout, advantage in zip(group.rollouts, advantages, strict=True):
                individual = RolloutWithCount(
                    rollout=rollout,
                    advantage=advantage,
                    usage_count=0,
                    weight_step=weight_step,
                )
                env_examples[rollout.env_name].append(individual)

    with self._lock:
        for env_name, examples in env_examples.items():
            if env_name in self.rollout_storage:
                self.rollout_storage[env_name].extend(examples)
            else:
                self.rollout_storage[env_name] = examples

            if len(self.rollout_storage[env_name]) > self.capacity:
                self.rollout_storage[env_name] = (
                    self.rollout_storage[env_name][-self.capacity:]
                )
```

### Phase 4: Update train_worker.py

**File:** `src/marin/rl/train_worker.py`

Update ReplayBuffer instantiation to use renamed parameter:

```python
self.replay_buffer = ReplayBuffer(
    capacity=config.replay_buffer.capacity,
    local_batch_size=config.trainer.train_batch_size,
    alpha=config.replay_buffer.alpha,
    total_processes=jax.process_count(),
    process_id=jax.process_index(),
    max_samples=config.replay_buffer.max_samples,
    max_rollout_step_delay=config.replay_buffer.max_rollout_step_delay,
    max_rollout_timestamp_delay=config.replay_buffer.max_rollout_timestamp_delay,
    loss_module=self.loss_module,
)
```

### Phase 5: Update rl_job.py config

**File:** `src/marin/rl/rl_job.py`

Add timestamp delay parameter:

```python
@dataclass
class TrainParams:
    """RL-specific training configuration parameters."""

    optimizer: OptimizerConfig
    rl_loss: "RLLossModule"

    max_samples_per_rollout: int = 1
    max_rollout_delay: int = 1

    max_rollout_timestamp_delay: float = 3600.0
    """Maximum age of rollouts in seconds. Negative means no limit."""

    replay_buffer_capacity: int = 4096
    replay_buffer_alpha: float = 3.0
```

Wire it through to ReplayBufferConfig:

```python
replay_buffer = ReplayBufferConfig(
    capacity=self.config.train_params.replay_buffer_capacity,
    alpha=self.config.train_params.replay_buffer_alpha,
    max_samples=self.config.train_params.max_samples_per_rollout,
    max_rollout_step_delay=self.config.train_params.max_rollout_delay,
    max_rollout_timestamp_delay=self.config.train_params.max_rollout_timestamp_delay,
)
```

### Phase 6: Update all test files

Update test helpers to create rollouts with metadata:

**Pattern:**

```python
# Before: Rollout without metadata
rollout = Rollout(
    env_name="test_env",
    env_example_id="example_1",
    prompt_tokens=prompt_tokens,
    response_tokens=response_tokens,
    response_logprobs=response_logprobs,
    token_rewards=token_rewards,
    episode_reward=episode_reward,
)

# After: Rollout with metadata
rollout = Rollout(
    env_name="test_env",
    env_example_id="example_1",
    prompt_tokens=prompt_tokens,
    response_tokens=response_tokens,
    response_logprobs=response_logprobs,
    token_rewards=token_rewards,
    episode_reward=episode_reward,
    metadata=RolloutMetadata(
        worker_id="test_worker",
        timestamp=time.time(),
        weight_step=0,
    ),
)
```

**Files to update:**
- `tests/rl/test_replay_buffer.py` - Update `create_test_batch` helper
- `tests/rl/test_rollout_storage.py` - Update `create_test_rollout` helper
- `tests/rl/integration/tasks.py` - Update both task helpers

Also update all ReplayBuffer instantiations in tests to use the new parameter name:

```python
replay_buffer = ReplayBuffer(
    capacity=100,
    local_batch_size=4,
    alpha=3.0,
    total_processes=1,
    process_id=0,
    max_samples=-1,
    max_rollout_step_delay=1000,
    max_rollout_timestamp_delay=3600.0,  # New parameter name
    loss_module=RLOOLoss(),
)
```

### Phase 7: Update environment implementations

Since environments don't need to know about metadata in this design, no changes are required to:
- `src/marin/rl/environments/base.py`
- `src/marin/rl/environments/mock_env.py`
- `src/marin/rl/environments/prime_intellect_env.py`

The environment interface stays clean and focused on domain logic.

## Testing Strategy

Run tests in this order to validate the migration:

```bash
# 1. Test storage layer (simplest)
uv run pytest tests/rl/test_rollout_storage.py -v

# 2. Test replay buffer functionality
uv run pytest tests/rl/test_replay_buffer.py -v

# 3. Test integration
uv run pytest tests/rl/integration/ -v
```

Key test scenarios:
1. **Timestamp filtering**: Verify old rollouts are filtered by timestamp
2. **Step filtering**: Verify old rollouts are filtered by weight_step
3. **Both filters**: Verify both filters work together correctly
4. **Serialization**: Verify rollouts with metadata can be pickled/unpickled
5. **Backward compatibility**: Verify reading old RolloutBatch with batch-level metadata still works (if needed)

## Benefits

1. **Precise filtering**: Can filter individual rollouts by age, not just entire batches
2. **Complete timestamp feature**: Fully implements `max_rollout_timestamp_delay`
3. **Clean separation**: Environments don't need to know about worker metadata
4. **Backward compatible**: Keeps RolloutBatch.metadata for transition period
5. **Better data model**: Metadata naturally belongs to each rollout, not the batch

## Future Work

After this migration is complete and stable:

1. **Remove RolloutBatch.metadata**: Once we're confident in the migration, remove the redundant batch-level metadata field
2. **Per-rollout timestamps**: Consider making timestamp/worker_id unique per rollout rather than shared across a batch
3. **Metadata versioning**: Add version field to RolloutMetadata for future schema evolution

## Migration Checklist

- [ ] Update `Rollout` to include `metadata: RolloutMetadata` field
- [ ] Update rollout worker to attach metadata after env.sample
- [ ] Update replay buffer config: rename parameter to `max_rollout_timestamp_delay`
- [ ] Update replay buffer: read from `rollout.metadata` instead of `batch.metadata`
- [ ] Update train_worker.py: use new parameter name
- [ ] Update rl_job.py: add timestamp delay to TrainParams
- [ ] Update test helpers: create rollouts with metadata
- [ ] Run all tests and verify they pass
- [ ] Update integration tests if needed
- [ ] Document the change in release notes
