# Reference Logprobs Migration Plan

## Overview

Move reference logprobs calculation from rollout workers to train workers. This eliminates the need for rollout workers to load reference models, reducing memory usage and improving scalability.

## Current Architecture

1. **Rollout Worker**: Loads both policy and reference models, computes both policy and reference logprobs
2. **Train Worker**: Receives rollout batches with pre-computed reference logprobs
3. **RolloutBatch**: Contains both policy and reference logprobs

## Target Architecture

1. **Rollout Worker**: Loads only policy model, computes only policy logprobs
2. **Train Worker**: Loads both policy and reference models, computes reference logprobs just-in-time
3. **RolloutBatch**: Contains only policy logprobs and raw tokens for reference computation

## Detailed Changes

### 1. rollout_storage.py

#### Remove reference_logprobs from RolloutBatch
```python
@dataclass
class RolloutBatch:
    input_ids: np.ndarray
    attention_mask: np.ndarray
    position_ids: np.ndarray
    target_ids: np.ndarray
    loss_weights: np.ndarray
    loss_masks: np.ndarray
    # REMOVE: reference_logprobs: np.ndarray
    policy_logprobs: np.ndarray
```

#### Remove reference_logprobs from JaxRolloutBatch
```python
class JaxRolloutBatch(eqx.Module):
    input_ids: jax.Array
    attention_mask: jax.Array
    position_ids: jax.Array
    target_ids: jax.Array
    loss_weights: jax.Array
    loss_masks: jax.Array
    # REMOVE: reference_logprobs: jax.Array
    policy_logprobs: jax.Array
```

#### Update as_named() method
```python
def as_named(self) -> dict:
    return {
        "input_ids": hax.named(self.input_ids, ("batch", "position")),
        "attention_mask": hax.named(self.attention_mask, ("batch", "position")),
        "position_ids": hax.named(self.position_ids, ("batch", "position")),
        "target_ids": hax.named(self.target_ids, ("batch", "position")),
        "loss_weights": hax.named(self.loss_weights, ("batch", "position")),
        "loss_masks": hax.named(self.loss_masks, ("batch", "position")),
        # REMOVE: "reference_logprobs": hax.named(self.reference_logprobs, ("batch", "position")),
        "policy_logprobs": hax.named(self.policy_logprobs, ("batch", "position")),
    }
```

#### Update to_jax() method in RolloutBatch
```python
def to_jax(self) -> JaxRolloutBatch:
    return JaxRolloutBatch(
        input_ids=jnp.array(self.input_ids),
        attention_mask=jnp.array(self.attention_mask),
        position_ids=jnp.array(self.position_ids),
        target_ids=jnp.array(self.target_ids),
        loss_weights=jnp.array(self.loss_weights),
        loss_masks=jnp.array(self.loss_masks),
        # REMOVE: reference_logprobs=jnp.array(self.reference_logprobs),
        policy_logprobs=jnp.array(self.policy_logprobs),
    )
```

### 2. rollout_worker.py

#### Remove reference model from _build_models()
```python
def _build_models(self):
    # ... existing policy model code ...

    # REMOVE all reference model initialization:
    # self.reference_model = load_model_from_checkpoint(...)

    # Keep only policy model code
    shape_tree = hax.tree_util.tree_map(lambda x: x.shape, self.policy_model)
    self.policy_model = self.transfer_client.receive_weights(shape_tree)
    # ...
```

#### Remove reference_ctx from _generate_rollout_batch()
```python
def _generate_rollout_batch(self, rng) -> tuple[list[dict], dict]:
    barrier_sync()

    # Create only policy inference context
    policy_ctx = LevanterInferenceContext(
        self.policy_model,
        tokenizer=self._tokenizer,
        inference_server=self.inference_server,
        max_tokens=self.config.max_input_length + self.config.max_output_length,
        stop_tokens=self.config.stop_tokens,
    )
    # REMOVE: reference_ctx = LevanterInferenceContext(...)

    with (
        self.config.trainer.device_mesh,
        hax.axis_mapping(self.config.trainer.compute_axis_mapping),
    ):
        rl_dataset, dataset_metrics = create_dataset_from_environment(
            environment=self._environment,
            policy_ctx=policy_ctx,
            # REMOVE: reference_ctx=reference_ctx,
            n_examples=self.config.n_prompts_per_step,
            prng_key=rng,
            n_generations=self.config.n_generations,
            max_input_length=self.config.max_input_length,
            max_output_length=self.config.max_output_length,
            pad_token_id=self.config.pad_token_id,
            mode="train",
            temperature=self.config.temperature,
        )
    # ...
```

### 3. rl_dataset.py

#### Update create_dataset_from_environment() signature
```python
def create_dataset_from_environment(
    environment,
    policy_ctx,
    # REMOVE: reference_ctx,
    n_examples: int,
    prng_key,
    n_generations: int,
    max_input_length: int,
    max_output_length: int,
    pad_token_id: int,
    mode: str = "train",
    temperature: float = 1.0,
) -> tuple["RLDataset", dict[str, float]]:
```

#### Update RLDataset.from_env_step() to not compute reference logprobs
```python
@classmethod
def from_env_step(
    cls,
    env_step: EnvStep,
    # REMOVE: reference_ctx,
    max_input_length: int,
    max_output_length: int,
    pad_token_id: int,
    kl_coef: float = 0.0,
) -> "RLDataset":
    # ... existing tokenization code ...

    # REMOVE: reference logprob computation
    # reference_logprobs = reference_ctx.compute_logprobs(...)

    # Use zeros as placeholder (will be computed in train worker)
    all_reference_logprobs = np.zeros_like(all_policy_logprobs)

    # ... rest of the method unchanged ...
```

#### Update data validation in RLDataset
```python
def _validate_data(self) -> None:
    expected_keys = {
        "returns",
        "policy_logprobs",
        # REMOVE: "reference_logprobs",
        "prompt_tokens",
        "prompt_masks",
        "output_tokens",
        "output_masks",
    }
    # ... rest unchanged ...
```

#### Update prepare_training_batch() to not require reference_logprobs
```python
def prepare_training_batch(
    prompt_tokens: np.ndarray,
    prompt_masks: np.ndarray,
    output_tokens: np.ndarray,
    output_masks: np.ndarray,
    loss_weights: np.ndarray,
    # reference_logprobs: np.ndarray,  # Make optional with default None
    policy_logprobs: np.ndarray,
    reference_logprobs: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    # ... existing code ...

    # Handle missing reference logprobs
    if reference_logprobs is None:
        reference_logprobs = np.zeros_like(policy_logprobs)

    # ... rest unchanged ...
```

### 4. train_worker.py

#### Add reference logprob computation to StreamingRolloutLoader
```python
class StreamingRolloutLoader:
    def __init__(self, data_loader: ReplayDataLoader, config: TrainerConfig, reference_model):
        self.data_loader = data_loader
        self.config = config
        self.reference_model = reference_model
        self.timeout = 60.0

    def __iter__(self):
        while True:
            batch = self.data_loader.get_training_batch(timeout=self.timeout)
            if not batch:
                logger.warning("No batch received from data loader within timeout, retrying...")
                continue

            # Compute reference logprobs just-in-time
            batch = self._add_reference_logprobs(batch)

            # Convert to dict of `hax.NamedArray`s and shard
            batch = batch.as_named()
            with self.config.device_mesh:
                sharded_batch = hax.shard(batch, self.config.compute_axis_mapping)

            yield sharded_batch

    def _add_reference_logprobs(self, batch: JaxRolloutBatch) -> JaxRolloutBatch:
        """Compute reference logprobs for the batch."""
        # TODO: Add caching mechanism here to avoid recomputation
        # Could cache based on input_ids hash or add version numbers to batches

        # Extract input and target sequences for reference model
        # Need to convert from combined sequence back to input/target format
        seq_len = batch.input_ids.shape[1]

        # Find the boundary between input and target tokens using attention_mask and loss_masks
        # This is environment-specific logic that may need refinement

        # For now, use a simple approach based on loss_masks
        # loss_masks are 0 for prompt tokens, 1 for output tokens
        loss_masks_np = np.array(batch.loss_masks)

        # Find first position where loss_mask becomes 1 (start of output)
        output_start_positions = []
        for i in range(len(loss_masks_np)):
            output_start = np.argmax(loss_masks_np[i] > 0)
            output_start_positions.append(output_start)

        # Reconstruct input/target separation
        batch_size = len(batch.input_ids)
        input_tokens_list = []
        input_masks_list = []
        target_tokens_list = []
        target_masks_list = []

        for i in range(batch_size):
            output_start = output_start_positions[i]

            # Input tokens: everything before output_start
            input_tokens = batch.input_ids[i][:output_start+1]  # Include transition token
            input_mask = batch.attention_mask[i][:output_start+1]

            # Target tokens: everything from output_start onward (shifted)
            target_tokens = batch.target_ids[i][output_start:]
            target_mask = batch.loss_masks[i][output_start:]

            # Pad to consistent lengths
            max_input_len = max(len(input_tokens) for input_tokens in [input_tokens])
            max_target_len = max(len(target_tokens) for target_tokens in [target_tokens])

            input_tokens_list.append(np.pad(input_tokens, (0, max_input_len - len(input_tokens))))
            input_masks_list.append(np.pad(input_mask, (0, max_input_len - len(input_mask))))
            target_tokens_list.append(np.pad(target_tokens, (0, max_target_len - len(target_tokens))))
            target_masks_list.append(np.pad(target_mask, (0, max_target_len - len(target_mask))))

        input_tokens_array = np.stack(input_tokens_list)
        input_masks_array = np.stack(input_masks_list)
        target_tokens_array = np.stack(target_tokens_list)
        target_masks_array = np.stack(target_masks_list)

        # Compute reference logprobs using the helper function
        reference_logprobs = compute_model_logprobs(
            self.reference_model,
            input_tokens_array,
            input_masks_array,
            target_tokens_array,
            target_masks_array,
        )

        # Pad reference logprobs back to match the original sequence format
        formatted_reference_logprobs = np.zeros_like(batch.policy_logprobs)
        for i in range(batch_size):
            output_start = output_start_positions[i]
            ref_logprobs_len = len(reference_logprobs[i])
            formatted_reference_logprobs[i][output_start:output_start+ref_logprobs_len] = reference_logprobs[i]

        # Create new batch with reference logprobs
        return JaxRolloutBatch(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            position_ids=batch.position_ids,
            target_ids=batch.target_ids,
            loss_weights=batch.loss_weights,
            loss_masks=batch.loss_masks,
            reference_logprobs=jnp.array(formatted_reference_logprobs),
            policy_logprobs=batch.policy_logprobs,
        )
```

#### Update train() method to pass reference_model to StreamingRolloutLoader
```python
def train(self):
    # ... existing setup code ...

    train_loader = StreamingRolloutLoader(
        self.data_loader,
        config.trainer,
        self.reference_model  # Add reference model
    )

    # ... rest unchanged ...
```

### 5. Test Updates

#### Update config_helpers.py
```python
def create_rollout_batch(
    policy_model,
    reference_model,  # Keep for test compatibility but don't use for rollout creation
    batch_size: int,
    tokenizer=None,
    max_input_length: int = 16,
    max_output_length: int = 16,
    pad_token_id: int = 0,
    worker_id: str = "test_worker",
    include_reference_logprobs: bool = False,  # New flag for test compatibility
) -> TaggedRolloutBatch:
    # ... existing encoding code ...

    policy_logprobs = compute_model_logprobs(
        policy_model,
        prompt_tokens,
        prompt_masks,
        response_tokens,
        response_masks,
    )

    # Only compute reference logprobs if requested (for test compatibility)
    if include_reference_logprobs:
        reference_logprobs = compute_model_logprobs(
            reference_model,
            prompt_tokens,
            prompt_masks,
            response_tokens,
            response_masks,
        )
    else:
        reference_logprobs = np.zeros_like(policy_logprobs)

    # ... rest unchanged ...
```

#### Update test_async_train.py calls
```python
# Update calls to create_rollout_batch to not include reference logprobs
batch = create_rollout_batch(
    policy_model=runner.reference_model,
    reference_model=runner.reference_model,
    batch_size=batch_size,
    tokenizer=tokenizer,
    include_reference_logprobs=False,  # New flag
)
```

## Implementation Order

1. **rollout_storage.py**: Remove reference_logprobs fields
2. **rl_dataset.py**: Update to not compute reference logprobs
3. **rollout_worker.py**: Remove reference model
4. **train_worker.py**: Add reference logprob computation
5. **config_helpers.py**: Update test helpers
6. **test_async_train.py**: Verify all tests still pass

## Validation

1. Run `test_rollout_worker` to verify rollout generation works without reference model
2. Run `test_train_worker` to verify training works with just-in-time reference logprob computation
3. Run `test_inference_and_training_workers` for end-to-end validation
4. Monitor memory usage to confirm reduction in rollout worker memory footprint

## Performance Considerations

- Reference logprobs are computed fresh for each training batch
- No caching initially (add TODO comment for future optimization)
- Computation happens on training worker which already has reference model loaded
- Adds some latency to training but removes compute from rollout workers

## Future Optimizations

1. **Caching**: Add caching mechanism based on input token hashes
2. **Batch Computation**: Compute reference logprobs for multiple batches at once
3. **Async Computation**: Compute reference logprobs in background thread
4. **Model Sharding**: Share reference model computation across multiple training workers