# Unified RL Interface

Our "programming model" for asynchronous RL currently imposes a number of
implementation details on the user, including manually determining the number of
rollout & train workers and launching them explicitly.

This is not great, as it also means we aren't properly encapsulating the RL "math"
from the implementation logistics. e.g. why does the training worker care about the
importance sampling clip epsilon?

```python
@jax.jit
def _loss_function(model, batch, key):
    return rloo_loss_with_importance_sampling(
        model, self.reference_model, batch, key=key, kl_coef=config.kl_coef, clip_epsilon=0.2
    )
```

Manually managing jobs also means that it's harder for us to implement things
like auto-scaling rollout or inference in response to resources or environment
speed.

A better interface would let users focus on the math and important
hyperparameters for the model, with maximum flexibility for how the actual
training happens, without having to pay excessive attention to the logistics of
_how_ the training happens.

# Unified Interface Design

## Core Abstractions

### SamplingParams
Configuration for how rollouts are generated from environments.

```python
@dataclass
class SamplingParams:
    """Parameters for sampling rollouts from an environment."""
    temperature: float = 1.0
    n_prompts: int = 8
    n_generations_per_prompt: int = 4
    stop_tokens: list[int] | None = None
```

### LessonConfig (Enhanced)
Existing `LessonConfig` enhanced with per-lesson sampling configuration and global defaults support.

```python
@dataclass
class LessonConfig:
    """Configuration for a single lesson in the curriculum."""
    lesson_id: str
    env_config: EnvConfig

    # Curriculum progression
    dependencies: list[LessonDependency] = field(default_factory=list)
    start_threshold: float = 0.0  # Minimum eval performance to activate
    stop_threshold: float = 1.0   # Performance threshold for graduation

    # Plateau detection
    plateau_window: int = 50
    plateau_threshold: float = 0.01

    # Per-lesson sampling (overrides global defaults)
    sampling_params: SamplingParams | None = None
```

### RLLossModule (Protocol)
Abstract interface for RL loss computation, allowing different algorithms (RLOO, PPO, etc.).

```python
class RLLossModule(Protocol):
    """Protocol for RL loss computation.

    Implementations define how to compute advantages and losses from rollouts.
    """

    def build(self, reference_model: eqx.Module) -> eqx.Module:
        """Initialize any learned components (e.g., value heads)."""
        ...

    def compute_advantages(self, rollout_group: list[Rollout]) -> list[float]:
        """Compute advantages for a group of rollouts. """
        ...

    def create_loss_fn(self, reference_model: eqx.Module, train_model: eqx.Module) -> Callable:
        """Create the loss function for training."""
        ...

# Default implementations
@dataclass
class RLOOLoss:
    """RLOO loss with importance sampling."""
    kl_coef: float = 0.1
    clip_epsilon: float = 0.2

    def build(self, reference_model: eqx.Module) -> eqx.Module:
        return self  # No learned parameters

    def compute_advantages(self, rollout_group: list[Rollout]) -> list[float]:
        return compute_rloo_advantages(rollout_group)

    def create_loss_fn(self, reference_model: eqx.Module) -> Callable:
        def loss_fn(model, batch, key):
            return rloo_loss_with_importance_sampling(
                model, reference_model, batch,
                key=key, kl_coef=self.kl_coef, clip_epsilon=self.clip_epsilon
            )
        return loss_fn
```

### TrainParams
Training-specific configuration.

```python
@dataclass
class TrainParams:
    """Training configuration parameters."""

    # Optimizer
    optimizer: OptimizerConfig
    num_train_steps: int

    # Batch sizing
    batch_size: int  # Global batch size (divided across processes)

    # Replay buffer
    replay_buffer_capacity: int = 10000
    replay_buffer_alpha: float = 3.0  # Recency bias
    max_samples_per_rollout: int = 4  # How many times to use each rollout
    max_batch_latency: int = 1000  # Max age of rollouts in steps
```

### RLJobConfig
Top-level configuration bringing everything together.

```python
@dataclass
class RLJobConfig:
    """Configuration for a complete RL training job."""

    # Model & initialization
    model: LmConfig
    initial_checkpoint: str | None = None

    # Training configuration
    trainer: TrainerConfig
    train_params: TrainParams

    # Curriculum & environments
    curriculum: CurriculumConfig

    # RL algorithm
    rl_loss: RLLossModule = field(default_factory=RLOOLoss)

    # Infrastructure
    num_rollout_workers: int = 1
    rollout_storage: RolloutStorageConfig = field(
        default_factory=lambda: RolloutStorageConfig(
            storage_type=StorageType.IN_MEMORY,
            queue_name="default"
        )
    )
    weight_transfer: WeightTransferConfig = field(default_factory=WeightTransferConfig)

    # Inference server (auto-configured by default)
    inference_server_config: InferenceServerConfig | None = None

    # Sequence lengths (global)
    max_input_length: int = 512
    max_output_length: int = 512

    # Logging
    run_id: str = field(default_factory=lambda: f"rl-{uuid.uuid4().hex[:8]}")
    log_freq: int = 10
```

### RLJob
Main entry point for running RL training.

```python
class RLJob:
    """High-level interface for RL training jobs.

    Handles worker creation, coordination, and lifecycle management.
    """

    def __init__(self, config: RLJobConfig):
        self.config = config
        self._validate_config()

    def run(self) -> LmHeadModel:
        """Run the RL training job to completion.

        Returns:
            The trained model
        """
        # Create shared infrastructure
        self._setup_curriculum_actor()
        self._setup_weight_transfer()

        # Launch workers as Ray actors
        train_worker = self._create_train_worker()
        rollout_workers = [
            self._create_rollout_worker(i)
            for i in range(self.config.num_rollout_workers)
        ]

        # Wait for completion (num_train_steps reached)
        trained_model = ray.get(train_worker.run.remote())

        # Cleanup
        for worker in rollout_workers:
            ray.get(worker.stop.remote())

        return trained_model

    def to_worker_configs(self) -> tuple[TrainWorkerConfig, RolloutWorkerConfig]:
        """Export worker configurations for inspection/testing."""
        ...

    def _validate_config(self):
        """Validate configuration consistency."""
        # Check curriculum dependencies form DAG
        # Validate batch_size >= num processes
        # etc.
        ...
```

## Usage Example

```python
from marin.rl import RLJob, RLJobConfig, LessonConfig, SamplingParams, RLOOLoss

# Define curriculum
curriculum = CurriculumConfig(
    lessons={
        "easy": LessonConfig(
            lesson_id="easy",
            env_config=EnvConfig(
                env_class="marin.rl.environments.MathEnv",
                env_args={"difficulty": "easy"}
            ),
            stop_threshold=0.8,
        ),
        "hard": LessonConfig(
            lesson_id="hard",
            env_config=EnvConfig(
                env_class="marin.rl.environments.MathEnv",
                env_args={"difficulty": "hard"}
            ),
            dependencies=[LessonDependency(dependency_id="easy", reward_threshold=0.8)],
            sampling_params=SamplingParams(n_generations_per_prompt=8),  # Override
        ),
    },
    eval_frequency=100,
)

# Create job
job = RLJob(RLJobConfig(
    model=LmConfig.init(...),
    initial_checkpoint="meta-llama/Llama-3.2-1B-Instruct",
    trainer=TrainerConfig(...),
    train_params=TrainParams(
        optimizer=OptimizerConfig(lr=1e-5),
        num_train_steps=10000,
        batch_size=256,
    ),
    curriculum=curriculum,
    default_sampling_params=SamplingParams(
        temperature=1.0,
        n_prompts=32,
        n_generations_per_prompt=4,
    ),
    rl_loss=RLOOLoss(kl_coef=0.1, clip_epsilon=0.2),
    num_rollout_workers=8,
))

# Run training
trained_model = job.run()
```

# Implementation Plan

## Phase 1: Core Interface (Current)

**Goal:** Define and validate the unified interface without breaking existing code.

### Step 1.1: Define Configuration Classes
- Create `src/marin/rl/rl_job.py` with:
  - `SamplingParams` dataclass
  - Enhanced `LessonConfig` (extend existing)
  - `RLLossModule` Protocol
  - `RLOOLoss` default implementation
  - `TrainParams` dataclass
  - `RLJobConfig` dataclass
  - `RLJob` class with `to_worker_configs()` stub

### Step 1.2: Write Validation Tests
- Create `tests/rl/test_rl_job_config.py`:
  - Test `RLJobConfig` construction with various configs
  - Test `to_worker_configs()` produces valid worker configs
  - Test config validation catches errors (circular deps, invalid batch sizes)
  - Test per-lesson sampling param overrides

### Step 1.3: Implement RLJob.to_worker_configs()
- Map `RLJobConfig` → `(TrainWorkerConfig, RolloutWorkerConfig)`
- Handle:
  - Merging default and per-lesson sampling params
  - Creating shared `rollout_storage` config
  - Splitting `weight_transfer` config appropriately
  - Computing `local_batch_size` from global batch size and num processes
  - Passing curriculum with enhanced lesson configs

**Exit Criteria:**
- All tests pass
- `to_worker_configs()` produces configs identical to current test helpers
- Config validation prevents common mistakes

## Phase 2: Job Execution

**Goal:** Implement `RLJob.run()` using existing workers.

### Step 2.1: Infrastructure Setup
- Implement `RLJob._setup_curriculum_actor()`
  - Create/get curriculum actor with enhanced configs
- Implement `RLJob._setup_weight_transfer()`
  - Create coordinator if needed

### Step 2.2: Worker Creation & Management
- Implement `RLJob._create_train_worker()` as Ray actor
- Implement `RLJob._create_rollout_worker(worker_id)` as Ray actor
- Implement wait logic for `num_train_steps` completion
- Implement cleanup on completion/failure

### Step 2.3: Integration Tests
- Port `test_rollout_and_train_workers` to use `RLJob`
- Port `test_train_worker_checkpoint_restart` to use `RLJob`
- Add test for job completion and model return

**Exit Criteria:**
- `RLJob.run()` successfully trains model end-to-end
- Checkpointing works through RLJob interface
- All existing integration tests pass with RLJob

## Phase 3: RLLossModule Integration

**Goal:** Refactor training to use `RLLossModule` abstraction.

### Step 3.1: Implement Default Losses as RLLossModule
- Create `RLOOLoss` implementing full protocol
- Create `PPOLoss` implementing full protocol
- Move advantage computation into loss modules

### Step 3.2: Refactor TrainWorker
- Accept `RLLossModule` in config
- Call `loss_module.compute_advantages()` in replay buffer
- Use `loss_module.create_loss_fn()` for training
- Update `create_training_batch_from_rollouts()` to use advantages from loss module

### Step 3.3: Update Tests
- Test custom `RLLossModule` implementations
- Validate PPO and RLOO produce equivalent results

**Exit Criteria:**
- Loss computation fully abstracted via `RLLossModule`
- Can swap RLOO/PPO by changing config
- Tests demonstrate custom loss implementation

## Phase 4: Sampling Parameter Refactor

**Goal:** Move sampling config to per-lesson with global defaults.

### Step 4.1: Update Worker Sampling Logic
- Modify `RolloutWorker._sample_batch()` to:
  - Get lesson's `sampling_params` or fall back to global
  - Use appropriate eval vs train sampling params

### Step 4.2: Update Tests
- Test per-lesson sampling overrides
- Test eval vs train sampling differences
- Migrate existing tests to new pattern

**Exit Criteria:**
- Each lesson can have custom sampling params
- Eval and train can use different params per lesson
- All tests use new sampling config pattern

## Phase 5: Migration & Cleanup

**Goal:** Migrate all existing code to use `RLJob`.

### Step 5.1: Update All Experiments
- Migrate experiment scripts to use `RLJob`
- Remove direct worker instantiation

### Step 5.2: Update Documentation
- Add `RLJob` usage guide
- Add migration guide from old API
- Add examples for common patterns

### Step 5.3: Remove Deprecated Patterns
Per CLAUDE.md: no deprecation paths, update all usages directly.
- Remove test helper functions (`create_nano_*_config`)
- Simplify worker configs (remove duplicated fields)

**Exit Criteria:**
- All code uses `RLJob`
- No direct worker instantiation remains
- Documentation complete

---

# Open Questions Requiring Decisions

## 1. Error Handling & Resilience

**Q:** What should happen when a worker fails during training?

Options:
2. **Retry workers:** Auto-restart failed workers with exponential backoff. This is builtin to ray, see experiments/exp1247_rl_async.py

---

## 2. Inference Server Port Management

inference server ports will be:
Auto-assigned (current behavior, but can cause issues with firewalls)

---

## 3. Multi-Host Ray Cluster Setup

**Q:** How does RLJob handle multi-host deployments?

Use the existing job launching pattern seen in exp1247_rl_async.py

---

## 4. Shared vs Split Tracker

**Q:** Should workers share a single tracker or have separate trackers?

Separate trackers with prefixes (`inference.`, `train.`)

2. **Separate trackers:** Current behavior, easier to debug per-worker

---

## 5. Evaluation Sampling Configuration

**Q:** Should evaluation use different sampling parameters than training?

Current: Global `eval_n_examples`, `eval_n_generations` in `CurriculumConfig`, same temperature/stop_tokens as training.

Options:
1. **Same params:** Eval uses same `SamplingParams` as training for each lesson
2. **Separate eval params:** Each lesson has `eval_sampling_params` field
3. **Global eval defaults:** Global eval params in `CurriculumConfig` with per-lesson override

**Recommendation:** Option 3 - global eval defaults (temperature=0 for deterministic eval, fewer generations) with per-lesson override via `eval_sampling_params` field.

---

## 6. Stopping Criteria

**Q:** When should `RLJob.run()` terminate?

when training & rollout jobs terminate, or never.

---

# Execution Checklist

- [ ] Phase 1.1: Define configuration classes in `rl_job.py`
- [ ] Phase 1.2: Write validation tests
- [ ] Phase 1.3: Implement `to_worker_configs()`
- [ ] Decide: Error handling strategy (#1)
- [ ] Decide: Inference port management (#2)
- [ ] Decide: Shared vs split tracker (#4)
- [ ] Decide: Eval sampling configuration (#5)
- [ ] Decide: Stopping criteria (#6)
- [ ] Phase 2.1: Infrastructure setup
- [ ] Phase 2.2: Worker creation & management
- [ ] Phase 2.3: Integration tests with RLJob
- [ ] Phase 3.1: Implement RLLossModule defaults
- [ ] Phase 3.2: Refactor TrainWorker for RLLossModule
- [ ] Phase 3.3: Update loss tests
- [ ] Phase 4.1: Update worker sampling logic
- [ ] Phase 4.2: Update sampling tests
- [ ] Phase 5.1: Migrate all experiments
- [ ] Phase 5.2: Update documentation
- [ ] Phase 5.3: Remove deprecated code
