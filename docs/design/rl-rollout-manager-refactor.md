# RL Rollout Manager Refactor

## Motivation

The `RolloutWorker` class currently handles too many responsibilities:
- Inference server management
- Model state
- Environment loading
- Rollout generation
- Curriculum evaluation
- Weight transfer
- Rollout storage
- Metrics logging
- Training loop coordination

This tight coupling makes it difficult to:
1. **Test rollout generation independently** without weight transfer infrastructure
2. **Reuse rollout logic** for offline evaluation on saved checkpoints
3. **Reason about responsibilities** - inference vs orchestration concerns are mixed

This refactor splits the class into two focused components with clear responsibilities.

## Design

### Current Structure

```python
class RolloutWorker:
    """Does everything: inference, evaluation, weight sync, storage, logging, loop."""

    _inference_server: InferenceServer
    _policy_model: Any
    _transfer_client: WeightTransferClient
    _rollout_writer: RolloutWriter
    _tokenizer: PreTrainedTokenizer
    _environments: dict[str, MarinEnv]
    _curriculum_actor: ray.ActorHandle

    def __init__(config): ...
    def _build_models(): ...
    def _load_environment(lesson_id): ...
    def _sample_batch(lesson_id, ...): ...
    def _evaluate_lesson(lesson_id, ...): ...
    def _evaluate_curriculum(rng, step): ...
    def _sync_weights(): ...
    def _log_prompt_example(...): ...
    def _build_eval_metrics(...): ...
    def run(): ...  # Main loop
    def stop(): ...
```

### Proposed Structure

#### RolloutManager (Inference & Curriculum)

```python
class RolloutManager:
    """Manages inference server, rollout generation, and curriculum.

    Handles all curriculum-related activities:
    - Inference server lifecycle
    - Model state management
    - Environment loading/caching
    - Rollout generation
    - Curriculum evaluation
    - Curriculum stats updates

    Returns data structures for the worker to log.
    Worker has NO curriculum responsibilities.
    """

    _inference_server: InferenceServer
    _policy_model: Any
    _tokenizer: PreTrainedTokenizer
    _environments: dict[str, MarinEnv]
    _curriculum_actor: ray.ActorHandle

    def __init__(
        inference_config: InferenceServerConfig,
        model_config: LmConfig,
        trainer_config: TrainerConfig,
        curriculum_config: CurriculumConfig,
        tokenizer: PreTrainedTokenizer,
        initial_model: Any,
    ): ...

    def sample_batch(
        lesson_id: str,
        n_examples: int,
        n_generations: int,
        mode: str,
        rng,
        weight_step: int,
        worker_id: str,
    ) -> tuple[RolloutBatch | None, dict | None]:
        """Generate rollouts with metadata attached."""

    def evaluate_lesson(
        lesson_id: str,
        n_examples: int,
        eval_type: str,
        rng,
        step: int,
        weight_step: int,
        worker_id: str,
    ) -> tuple[RolloutBatchStats, RolloutBatch]:
        """Evaluate single lesson, update curriculum, return data for logging."""

    def evaluate_curriculum(
        eval_n_examples: int,
        rng,
        step: int,
        weight_step: int,
        worker_id: str,
    ) -> dict[str, tuple[RolloutBatchStats, RolloutBatch]]:
        """Evaluate all lessons, update curriculum, return {lesson_id: (stats, batch)}."""

    def update_training_stats(
        lesson_id: str,
        batch: RolloutBatch,
        step: int,
    ) -> None:
        """Update curriculum with training rollout stats."""

    def update_model(model: Any) -> None:
        """Update policy model and reload inference server."""

    def shutdown() -> None:
        """Cleanup inference server."""
```

#### RolloutWorker (Orchestration & I/O)

```python
class RolloutWorker:
    """Orchestrates RL training loop.

    Handles I/O and coordination:
    - Weight transfer from training workers
    - Rollout storage (writing batches)
    - Metrics logging via tracker
    - Main training loop

    Delegates ALL inference and curriculum activities to RolloutManager.
    Worker only logs data returned by manager.
    """

    _manager: RolloutManager
    _transfer_client: WeightTransferClient
    _rollout_writer: RolloutWriter
    tracker: WandBTracker

    def __init__(config: RolloutWorkerConfig): ...
    def _build_initial_model() -> Any: ...
    def _sync_weights() -> None: ...
    # call out to the rollout manager to actually run the env, send the stats back to the tracker.
    def _run_environment(...) -> None: ...
    def run() -> None: ...
    def stop() -> None: ...
```

### Key Principles

1. **Manager computes, Worker coordinates**: Manager generates data, Worker logs/stores it
2. **Stateless manager methods**: Pass `weight_step` and `worker_id` as parameters instead of tracking mutable state
3. **Manager owns curriculum**: All curriculum activities (evaluation, stats updates) in manager
4. **Worker owns tracker**: All logging decisions in worker
5. **Worker owns storage**: Rollout persistence is worker responsibility
6. **Clean boundaries**: No circular dependencies, clear data flow

## Implementation Plan

### Phase 1: Create RolloutManager class skeleton

Add new class to `src/marin/rl/rollout_worker.py` above `RolloutWorker`:

```python
class RolloutManager:
    """Manages inference server and rollout generation."""

    def __init__(
        self,
        inference_config: InferenceServerConfig,
        model_config: LmConfig,
        trainer_config: TrainerConfig,
        curriculum_config: CurriculumConfig,
        tokenizer: PreTrainedTokenizer,
        initial_model: Any,
    ):
        self.inference_config = inference_config
        self.model_config = model_config
        self.trainer_config = trainer_config
        self.curriculum_config = curriculum_config
        self.tokenizer = tokenizer

        self._policy_model = initial_model
        self._environments: dict[str, MarinEnv] = {}

        # Start inference server
        self._inference_server = InferenceServer.create(
            inference_config,
            model=self._policy_model,
            tokenizer=self.tokenizer,
        )
        self._inference_thread = threading.Thread(
            target=lambda: self._inference_server.serve(),
            daemon=True,
        )
        self._inference_thread.start()
        time.sleep(1.0)  # TODO: replace with wait_until_ready()

        # Create curriculum actor
        self._curriculum_actor = get_or_create_curriculum_actor(curriculum_config)

    def update_model(self, model: Any) -> None:
        """Update the policy model and reload inference server."""
        self._policy_model = model
        self._inference_server.reload(lambda m: self._policy_model)
        logger.info("Model updated in inference server")

    def shutdown(self) -> None:
        """Shutdown inference server."""
        if self._inference_server:
            self._inference_server.shutdown()
```

### Phase 2: Move environment loading to manager

Extract `_load_environment` from `RolloutWorker` to `RolloutManager`:

```python
class RolloutManager:
    # ... __init__ and update_model ...

    def _load_environment(self, lesson_id: str) -> MarinEnv:
        """Load and cache environment for lesson."""
        if lesson_id in self._environments:
            return self._environments[lesson_id]

        lesson_config = self.curriculum_config.lessons[lesson_id]
        env = load_environment_from_spec(lesson_config.env_config)
        self._environments[lesson_id] = env
        return env
```

### Phase 3: Move sample_batch to manager

Extract and modify `_sample_batch` to take `weight_step` and `worker_id` as parameters:

```python
class RolloutManager:
    # ... previous methods ...

    def sample_batch(
        self,
        lesson_id: str,
        n_examples: int,
        n_generations: int,
        mode: str,
        rng,
        weight_step: int,
        worker_id: str,
    ) -> tuple[RolloutBatch | None, dict | None]:
        """Generate a batch of rollouts.

        Args:
            lesson_id: Lesson to sample from
            n_examples: Number of examples to generate
            n_generations: Generations per example
            mode: 'train' or 'eval'
            rng: JAX PRNG key
            weight_step: Current weight step for metadata
            worker_id: Worker identifier for metadata

        Returns:
            (RolloutBatch with metadata attached, env metrics dict)
        """
        env = self._load_environment(lesson_id)
        lesson_config = self.curriculum_config.lessons[lesson_id]

        # Get sampling params
        temperature = lesson_config.sampling_params.temperature
        stop_tokens = lesson_config.sampling_params.stop_tokens
        max_tokens = lesson_config.sampling_params.max_tokens

        policy_ctx = LevanterInferenceContext(
            tokenizer=self.tokenizer,
            inference_server=self._inference_server,
            max_tokens=max_tokens,
            stop_tokens=stop_tokens,
        )

        with (
            self.trainer_config.device_mesh,
            hax.axis_mapping(self.trainer_config.compute_axis_mapping),
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
            logger.warning("No valid rollouts generated")
            return None, None

        logger.info(
            "Generated %d rollout groups from lesson %s at step %d",
            len(rollout_groups),
            lesson_id,
            weight_step,
        )

        # Create metadata
        batch_metadata = RolloutMetadata(
            worker_id=worker_id,
            timestamp=time.time(),
            weight_step=weight_step,
        )

        # Attach metadata to rollouts
        rollout_groups_with_metadata = []
        for group in rollout_groups:
            rollouts_with_metadata = [
                eqx.tree_at(lambda r: r.metadata, rollout, batch_metadata)
                for rollout in group.rollouts
            ]
            rollout_groups_with_metadata.append(
                RolloutGroup(rollouts=rollouts_with_metadata)
            )

        rollout_batch = RolloutBatch(
            groups=rollout_groups_with_metadata,
            metadata=batch_metadata,
        )
        return rollout_batch, metrics
```

### Phase 4: Move evaluation methods to manager

Extract `evaluate_lesson` and `evaluate_curriculum`:

```python
class RolloutManager:
    # ... previous methods ...

    def evaluate_lesson(
        self,
        lesson_id: str,
        n_examples: int,
        eval_type: str,
        rng,
        step: int,
        weight_step: int,
        worker_id: str,
    ) -> tuple[RolloutBatchStats, RolloutBatch]:
        """Evaluate a single lesson and update curriculum.

        Args:
            eval_type: 'eval' or 'micro_eval' - only 'eval' updates curriculum

        Returns:
            (stats, batch) for worker to log
        """
        batch, _ = self.sample_batch(
            lesson_id=lesson_id,
            n_examples=n_examples,
            n_generations=1,
            mode="eval",
            rng=rng,
            weight_step=weight_step,
            worker_id=worker_id,
        )

        stats = _compute_batch_stats(batch, lesson_id)

        # Manager handles curriculum updates
        if eval_type == "eval":
            self._curriculum_actor.update_lesson_stats.options(
                enable_task_events=False
            ).remote(stats.rollout_stats, mode="eval", current_step=step)

        return stats, batch

    def evaluate_curriculum(
        self,
        eval_n_examples: int,
        rng,
        step: int,
        weight_step: int,
        worker_id: str,
    ) -> dict[str, tuple[RolloutBatchStats, RolloutBatch]]:
        """Evaluate all lessons and update curriculum.

        Returns:
            {lesson_id: (stats, batch)} for worker to log
        """
        lesson_names = list(self.curriculum_config.lessons.keys())
        if not lesson_names:
            logger.info("No lessons to evaluate")
            return {}

        logger.info(f"Evaluating {len(lesson_names)} lessons")

        results = {}
        for lesson_id in lesson_names:
            stats, batch = self.evaluate_lesson(
                lesson_id=lesson_id,
                n_examples=eval_n_examples,
                eval_type="eval",  # Full eval updates curriculum
                rng=rng,
                step=step,
                weight_step=weight_step,
                worker_id=worker_id,
            )
            results[lesson_id] = (stats, batch)

        return results

    def update_training_stats(
        self,
        lesson_id: str,
        batch: RolloutBatch,
        step: int,
    ) -> None:
        """Update curriculum with training rollout stats.

        Args:
            lesson_id: Lesson that generated the batch
            batch: Rollout batch from training
            step: Current training step
        """
        stats = _compute_batch_stats(batch, lesson_id)
        self._curriculum_actor.update_lesson_stats.options(
            enable_task_events=False
        ).remote(stats.rollout_stats, mode="training", current_step=step)
```

### Phase 5: Refactor RolloutWorker to use manager

#### Update __init__ to build model and create manager

```python
class RolloutWorker:
    def __init__(self, config: RolloutWorkerConfig):
        config.trainer.id = f"{config.run_id}-rollout"
        levanter.initialize(config.trainer)

        # Infer model_axis_size
        config.inference_server_config = dataclasses.replace(
            config.inference_server_config,
            trainer=dataclasses.replace(
                config.inference_server_config.trainer,
                model_axis_size=jax.local_device_count(),
            ),
        )

        self.tracker = levanter.current_tracker()
        self.config = config
        self._running = True
        self._shutdown_complete = threading.Event()
        self._shutdown_condition = threading.Condition()
        self._current_weight_step: int = 0
        self._worker_id = f"{socket.gethostname()}_{os.getpid()}"

        # Weight transfer
        logger.info("Starting weight transfer client")
        self._transfer_client = create_weight_transfer_client(
            config.weight_transfer,
            mesh=config.trainer.device_mesh,
            axis_mapping=config.trainer.compute_axis_mapping,
        )

        # Rollout storage
        self._rollout_writer = config.rollout_storage.create_writer()

        # Build initial model (worker responsibility)
        initial_model = self._build_initial_model()

        # Create rollout manager (handles inference & evaluation)
        self._manager = RolloutManager(
            inference_config=config.inference_server_config,
            model_config=config.model,
            trainer_config=config.trainer,
            curriculum_config=config.curriculum_config,
            tokenizer=config.tokenizer,
            initial_model=initial_model,
        )
```

#### Extract _build_initial_model

```python
class RolloutWorker:
    def _build_initial_model(self):
        """Build or load the initial policy model."""
        if self.config.initial_checkpoint is not None:
            logger.info(
                f"Loading initial policy model from: {self.config.initial_checkpoint}"
            )
        else:
            logger.info("Building new policy model from scratch")

        key = jrandom.PRNGKey(42)
        vocab_size = self.config.tokenizer.vocab_size
        Vocab = hax.Axis("vocab", vocab_size)

        return load_model_from_checkpoint(
            checkpoint=self.config.initial_checkpoint,
            model_config=self.config.model,
            trainer_config=self.config.trainer,
            mesh=self.config.trainer.device_mesh,
            axis_mapping=self.config.trainer.compute_axis_mapping,
            vocab_axis=Vocab,
            tokenizer=self.config.tokenizer,
            key=key,
        )
```

#### Update _sync_weights to use manager

```python
class RolloutWorker:
    def _sync_weights(self) -> None:
        """Poll for and apply new weights from training workers."""
        while True:
            logger.info("Checking for new weights...")
            update = self._transfer_client.receive_weights(self._manager._policy_model)
            if update:
                break
            time.sleep(1.0)

        if update:
            self._current_weight_step = update.weight_id
            logger.info(f"Received new weights from step {update.weight_id}")
            self._manager.update_model(update.model)
```

#### Update evaluation methods to use manager and log results

```python
class RolloutWorker:
    def _evaluate_lesson(
        self,
        lesson_id: str,
        n_examples: int,
        eval_type: str,
        rng,
        step: int,
    ) -> RolloutBatchStats:
        """Evaluate lesson and log to tracker.

        Manager handles curriculum updates.
        Worker only logs to tracker.
        """
        stats, batch = self._manager.evaluate_lesson(
            lesson_id=lesson_id,
            n_examples=n_examples,
            eval_type=eval_type,
            rng=rng,
            step=step,
            weight_step=self._current_weight_step,
            worker_id=self._worker_id,
        )

        # Worker only logs - manager already updated curriculum
        self._log_prompt_example(lesson_id, batch, step, eval_type=eval_type)
        metrics = self._build_eval_metrics(
            prefix=f"inference.{eval_type}",
            lesson_id=lesson_id,
            batch=batch,
        )
        self.tracker.log(metrics, step=step)
        logger.info("Eval metrics for %s at step %d: %s", lesson_id, step, metrics)

        return stats

    def _evaluate_curriculum(self, rng, step: int) -> None:
        """Evaluate all lessons and log results.

        Manager handles curriculum updates.
        Worker only logs to tracker.
        """
        results = self._manager.evaluate_curriculum(
            eval_n_examples=self.config.curriculum_config.eval_n_examples,
            rng=rng,
            step=step,
            weight_step=self._current_weight_step,
            worker_id=self._worker_id,
        )

        # Worker only logs - manager already updated curriculum
        for lesson_id, (stats, batch) in results.items():
            self._log_prompt_example(lesson_id, batch, step, eval_type="eval")
            metrics = self._build_eval_metrics(
                prefix="inference.eval",
                lesson_id=lesson_id,
                batch=batch,
            )
            self.tracker.log(metrics, step=step)
            logger.info("Eval metrics for %s: %s", lesson_id, metrics)

        barrier_sync()
```

#### Update run() to use manager for rollout generation

```python
class RolloutWorker:
    def run(self):
        """Main training loop."""
        logger.info("Starting rollout worker...")

        step = 0
        seed = 0
        rng = jax.random.PRNGKey(seed)
        rng = multihost_utils.broadcast_one_to_all(rng)
        logger.info(f"Starting with seed {seed}")

        while self._running:
            self._sync_weights()

            if self.config.max_rollouts and step >= self.config.max_rollouts:
                logger.info(f"Reached max rollouts ({self.config.max_rollouts})")
                break

            logger.info("Generating rollout batch...")
            rng, seed_key = jax.random.split(rng)
            seed = int(seed_key[0])

            try:
                lesson_id = ray.get(
                    self._manager._curriculum_actor.sample_lesson.remote(seed)
                )
            except Exception as e:
                logger.warning(f"Failed to sample lesson: {e}, retrying...")
                time.sleep(10.0)
                continue

            # Micro-eval
            if step > 0 and step % self.config.curriculum_config.micro_eval_frequency == 0:
                rng, micro_eval_rng = jrandom.split(rng)
                self._evaluate_lesson(
                    lesson_id,
                    self.config.curriculum_config.micro_eval_n_examples,
                    eval_type="micro_eval",
                    rng=micro_eval_rng,
                    step=step,
                )

            # Full eval
            if step > 0 and step % self.config.curriculum_config.eval_frequency == 0:
                rng, eval_rng = jrandom.split(rng)
                self._evaluate_curriculum(eval_rng, step)

            logger.info(f"Sampled lesson '{lesson_id}'")

            # Generate rollouts via manager
            rng, input_rng = jax.random.split(rng)
            lesson_config = self.config.curriculum_config.lessons[lesson_id]
            rollout_batch, env_metrics = self._manager.sample_batch(
                lesson_id=lesson_id,
                n_examples=lesson_config.sampling_params.n_prompts,
                n_generations=lesson_config.sampling_params.n_generations_per_prompt,
                mode="train",
                rng=input_rng,
                weight_step=self._current_weight_step,
                worker_id=self._worker_id,
            )

            if rollout_batch is None:
                continue

            # Manager updates curriculum with training stats
            self._manager.update_training_stats(lesson_id, rollout_batch, step)

            # Worker builds metrics for logging
            eval_metrics = self._build_eval_metrics(
                prefix="rollout",
                lesson_id=lesson_id,
                batch=rollout_batch,
            )

            step += 1
            self._rollout_writer.write_batch(rollout_batch)

            # Logging
            if self.config.log_freq > 0 and step % self.config.log_freq == 0:
                log_metrics = eval_metrics
                log_metrics.update(self._transfer_client.get_metrics())
                log_metrics.update({f"env.{k}": v for k, v in (env_metrics or {}).items()})
                log_metrics = {f"inference.{k}": v for k, v in log_metrics.items()}
                logger.info(f"Logging metrics at step {step}... {log_metrics}")
                self.tracker.log(log_metrics, step=step)

        logger.info(f"Completed after generating {step} rollouts")
        barrier_sync()
        self._shutdown_complete.set()
```

#### Update stop() to shutdown manager

```python
class RolloutWorker:
    def stop(self):
        """Stop the worker and cleanup."""
        with self._shutdown_condition:
            self._running = False
            self._transfer_client.cleanup()
            self._shutdown_condition.notify()

        self._shutdown_complete.wait()
        self._manager.shutdown()
```

#### Remove old methods from RolloutWorker

Delete the following methods (now in `RolloutManager`):
- `_build_models()` → replaced with `_build_initial_model()` + manager owns model
- `_load_environment()` → moved to manager
- `_sample_batch()` → moved to manager as `sample_batch()`

Keep in `RolloutWorker`:
- `_log_prompt_example()` → uses `self.tracker`
- `_build_eval_metrics()` → helper for logging

### Phase 6: Update tests

No test changes required - `RolloutWorker` API remains unchanged.

Future: Add unit tests for `RolloutManager`:

```python
def test_rollout_manager_standalone():
    """Test RolloutManager can be used without worker."""
    model = build_test_model()
    manager = RolloutManager(
        inference_config=test_inference_config,
        model_config=test_model_config,
        trainer_config=test_trainer_config,
        curriculum_config=test_curriculum_config,
        tokenizer=test_tokenizer,
        initial_model=model,
    )

    batch, metrics = manager.sample_batch(
        lesson_id="test_lesson",
        n_examples=4,
        n_generations=2,
        mode="eval",
        rng=jax.random.PRNGKey(0),
        weight_step=100,
        worker_id="test_worker",
    )

    assert batch is not None
    assert batch.metadata.weight_step == 100
    assert batch.metadata.worker_id == "test_worker"

    manager.shutdown()
```

## Benefits

1. **Separation of concerns**:
   - Manager = inference + curriculum (self-contained)
   - Worker = I/O coordination (weight transfer, logging, storage)
2. **Reusability**: Can use `RolloutManager` standalone for:
   - Offline evaluation on saved checkpoints
   - Testing rollout generation without weight transfer
   - Batch processing scenarios
3. **Testability**: Can unit test rollout generation and curriculum updates independently
4. **Cleaner interfaces**: Each class has focused responsibilities with clear boundaries
5. **Self-contained manager**: Manager handles ALL curriculum logic, no leaky abstractions
6. **No API changes**: `RolloutWorker` public interface unchanged

## Tradeoffs

1. **Indirection**: Need to pass `weight_step`/`worker_id` through to manager
   - Chosen: Stateless methods > mutable state coupling
2. **Access to internal state**: Worker needs `manager._policy_model` for weight sync
   - Acceptable: Same file, private attribute access is fine
3. **Tokenizer duplication**: Both hold references
   - Acceptable: Lightweight reference, no extra methods needed

## Testing Strategy

```bash
# Run existing tests to verify no regressions
uv run pytest tests/rl/test_rollout_worker.py -v
uv run pytest tests/rl/integration/ -v
```

All existing tests should pass without modification.

## Future Work

1. **Extract RolloutManager to separate file** if it grows significantly
2. **Add RolloutManager unit tests** for offline evaluation scenarios
3. **Create RolloutManager ABC** if we need multiple implementations
4. **Add `wait_until_ready()`** to inference server instead of `time.sleep(1.0)`

## Implementation Checklist

- [ ] Create `RolloutManager` class skeleton with `__init__`, `update_model`, `shutdown`
- [ ] Move `_load_environment` to manager
- [ ] Move `_sample_batch` to manager (rename to `sample_batch`, add params)
- [ ] Move evaluation methods to manager (`evaluate_lesson`, `evaluate_curriculum`)
- [ ] Add `update_training_stats` method to manager
- [ ] Manager methods should update curriculum_actor directly, not return stats for worker to update
- [ ] Update `RolloutWorker.__init__` to build model and create manager
- [ ] Add `_build_initial_model` method to worker
- [ ] Update `_sync_weights` to use `manager.update_model()`
- [ ] Update `_evaluate_lesson` and `_evaluate_curriculum` to use manager (worker only logs)
- [ ] Update `run()` to use `manager.sample_batch()` and `manager.update_training_stats()`
- [ ] Update `stop()` to call `manager.shutdown()`
- [ ] Remove old methods from worker (`_build_models`, `_load_environment`, `_sample_batch`)
- [ ] Run tests to verify no regressions
- [ ] Add docstrings to all new methods
