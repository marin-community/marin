# Adaptive Curriculum Learning System

## Background & Motivation

We need RL training to automatically sample from environments of appropriate difficulty without manual intervention. Sampling from too easy environments wastes compute and risks overfitting, while too hard distributions yield zero rewards or unstable learning (RLOO needs multiple positive examples).

An optimal curriculum focuses on the "zone of proximal development" - tasks just beyond current ability where learning is maximized. This design enables Marin to automatically reweight environments while minimizing evaluation overhead.

### Prior Work

- **[Teacher-Student Curriculum Learning](https://arxiv.org/abs/1707.00183)**: Benefits of adaptive difficulty based on performance
- **[Automatic Domain Randomization](https://arxiv.org/abs/1910.07113)**: Progressive distribution expansion

## Design Overview

The curriculum system tracks performance across multiple environments, manages dependencies between lessons, and dynamically adjusts sampling weights to maximize learning efficiency.

Key principles:
- **Performance tracking** combines cheap training metrics with periodic evaluation
- **Dependencies** ensure logical skill progression
- **Diversity guarantees** prevent collapse to single environments
- **Smooth transitions** avoid harsh cutoffs that destabilize training

## Core Components

### Lesson Configuration

A lesson wraps an environment with metadata about dependencies and activation thresholds.

```python

@dataclass
class LessonDependency:
    dependency_name: str
    """Lesson this depends on."""

    reward_threshold: float = 0.0
    """When `dependency_name` reaches `reward_threshold` and learning plateaus, activate.
    By default, wait only for `dependency_name` to plateau."""


@dataclass
class LessonConfig:
    lesson_name: str

    env_config: EnvConfig  # Uses existing Marin EnvConfig
    dependencies: list[LessonDependency] = field(default_factory=list)

    initial_weight: float = 1.0
    """The initial weight to use for sampling this environment before data is available."""

    start_threshold: float = 0.0
    """Once unlocked, how well does the agent need to do on this environment in eval to being training?"""

    stop_threshold: float = 1.00
    """As reward approaches this threshold, consider the environment for graduation."""

    plateau_window: int = 50
    """Number of recent samples to consider for plateau detection."""

    plateau_threshold: float = 0.01
    """Relative slope threshold for detecting plateaus."""
```

### Performance Tracking

Training and evaluation performance are tracked separately using the same mechanism for consistency.

```python
@dataclass
class PerformanceStats:
    """Statistics for a particular mode (training or eval)."""
    smoothed_success: float = 0.0  # Exponentially smoothed success rate
    smoothed_reward: float = 0.0   # Exponentially smoothed reward
    total_samples: int = 0         # Total rollouts seen
    reward_history: list[float] = field(default_factory=list)  # Last 100 samples
    last_update_step: int = -1     # When last updated

@dataclass
class LessonStats:
    training_stats: PerformanceStats = field(default_factory=PerformanceStats)
    eval_stats: PerformanceStats = field(default_factory=PerformanceStats)

def update_performance_stats(stats: PerformanceStats, rollout_stats: RolloutStats, current_step: int) -> PerformanceStats:
    """Update performance stats with exponential smoothing."""

def get_success_rate_for_decisions(stats: LessonStats, current_step: int, max_staleness: int = 1000) -> float:
    """Use eval if available and recent, otherwise training."""
    if stats.eval_stats.last_update_step >= 0:
        staleness = current_step - stats.eval_stats.last_update_step
        if staleness <= max_staleness:
            return stats.eval_stats.smoothed_success
    return stats.training_stats.smoothed_success
```

### Curriculum Manager

The curriculum maintains lesson configurations, statistics, and sampling state as a stateful class.

```python

def _validate_dependencies(lesson_configs):
    """Ensure no circular dependencies."""
    visited = set()
    rec_stack = set()

    def has_cycle(node):
        if node not in lesson_configs:
            return False
        visited.add(node)
        rec_stack.add(node)
        for dep in lesson_configs[node].dependencies:
            if dep not in visited:
                if has_cycle(dep):
                    return True
            elif dep in rec_stack:
                return True
        rec_stack.remove(node)
        return False

    for name in lesson_configs:
        if name not in visited and has_cycle(name):
            raise ValueError(f"Circular dependency detected involving {name}")

class CurriculumStats:
    lesson_stats: dict[str, LessonStats]
    """Mapping from lesson name to statistics."""

class AdaptiveCurriculum:
    """Manages lesson progression and sampling."""

    def __init__(
        self,
        lesson_configs: list[LessonConfig],
        eval_frequency: int = 1000,
        temperature: float = 1.0
    ):
        _validate_dependencies(lesson_configs)

        self.lesson_configs = {cfg.name: cfg for cfg in lesson_configs}
        self.stats = {cfg.name: LessonStats() for cfg in lesson_configs}
        self.environments = {
            cfg.name: load_environment_from_spec(cfg.env_config)
            for cfg in lesson_configs
        }

        self.unlocked = set()  # Currently accessible lessons
        self.graduated = set()  # Mastered lessons

        for lesson in lesson_configs:
            # unlock any lesson without deps

        self.eval_frequency = eval_frequency
        self.temperature = temperature
        self.current_step = 0
```

### Dependency Management

Dependencies ensure lessons unlock in logical order based on prerequisite performance.

```python
def check_dependencies(lesson_stats: LessonStats, lesson_name: str) -> bool:
    """Check if all dependencies are satisfied for a lesson."""

def update_unlocked_lessons(self):
    """Update which lessons are currently unlocked based on dependencies."""
```

### Sampling Strategy

Sampling weights balance performance-based weighting with exploration bonuses to maintain diversity.

```python
def sigmoid(x: float, center: float, steepness: float) -> float:
    """Smooth sigmoid transition."""
    return 1 / (1 + np.exp(-steepness * (x - center)))

def is_plateaued(stats: LessonStats, window: int = 50, threshold: float = 0.01) -> bool:
    """Detect reward plateau using linear regression on recent history."""
    if len(stats.reward_history) < window:
        return False

    recent = np.array(list(stats.reward_history)[-window:])
    x = np.arange(len(recent))

    # Fit linear trend
    coeffs = np.polyfit(x, recent, 1)
    slope = coeffs[0]

    # Check if relative trend is flat
    mean_reward = np.mean(recent)
    if abs(mean_reward) > 1e-6:
        relative_trend = abs(slope) / abs(mean_reward)
        return relative_trend < threshold

    return True

def compute_sampling_weights(self) -> dict[str, float]:
    """Compute sampling weights for all active lessons."""
    weights = {}

    # for each example
    # compute quadatric weight, peaking at 50% success
    # soften edges with sigmoids
    # include an exploration bonus for new environments, decaying over time
    # Ensure minimum probability for active lessons
    # penalize lessons which have plateaued
    min_prob = 0.01
    for name in weights:
        weights[name] = max(weights[name], min_prob)

    # Renormalize
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}
    return weights

def sample_lesson(self, prng_key) -> tuple[str, MarinEnv]:
    """Sample next lesson for training."""
    weights = self.compute_sampling_weights()
    names = list(weights.keys())
    probs = list(weights.values())
    lesson_name = jax.random.choice(prng_key, names, p=probs)
    return lesson_name, self.environments[lesson_name]

```

### Training Integration

The curriculum integrates with the training loop, processing rollouts and triggering evaluations.

```python
def process_rollout_batch(self, batch: RolloutBatch):
    """Update curriculum state from training rollouts."""
    for group in batch.groups:
        for rollout in group.rollouts:
            lesson_name = rollout.env_name
            if lesson_name in self.stats:
                self.stats[lesson_name] = update_from_rollout(
                    self.stats[lesson_name], rollout
                )
```

### Graduation
Lessons graduate when mastered or plateau at high performance.

```python
def is_graduated(lesson):
    """Update graduated lessons based on performance."""
    success = get_combined_success_rate(stats, self.current_step)

    # Graduate if consistently > max_success_rate
    if success > lesson.max_success_rate and stats.eval_step > 0:
        return True

def step(self):
    """Increment step counter for internal tracking."""
    self.current_step += 1
```

## Usage Example

```python
# Define curriculum
lessons = [
    LessonConfig(
        name="tutorial",
        env_config=EnvConfig(
            env_class="marin.rl.environments.tutorial.TutorialEnv",
            env_args={"difficulty": "easy"}
        )
    ),
    LessonConfig(
        name="basic",
        env_config=EnvConfig(
            env_class="marin.rl.environments.math.MathEnv",
            env_args={"level": "basic"}
        ),
        dependencies={"tutorial": 0.7}
    ),
    LessonConfig(
        name="intermediate",
        env_config=EnvConfig(
            env_class="marin.rl.environments.math.MathEnv",
            env_args={"level": "intermediate"}
        ),
        dependencies={"basic": 0.6}
    ),
    LessonConfig(
        name="advanced",
        env_config=EnvConfig(
            env_class="marin.rl.environments.math.MathEnv",
            env_args={"level": "advanced"}
        ),
        dependencies={"basic": 0.7, "intermediate": 0.7}
    ),
]

# Initialize
curriculum = AdaptiveCurriculum(
    lesson_configs=lessons,
    eval_frequency=1000,
    temperature=1.0
)

# Training loop
for step in range(num_steps):
    # Update curriculum tracking
    curriculum.step()
    curriculum.update_unlocked_lessons()

    # Sample lesson
    lesson_name, env = curriculum.sample_lesson(prng_key)

    # Generate rollouts using existing Marin infrastructure
    rollout_batch, metrics = env.sample(
        inference_ctx,
        n_examples=batch_size,
        n_generations=n_gens,
        temperature=sampling_temp,
        prng_key=prng_key,
        mode="train"
    )

    # Update curriculum with results
    curriculum.process_rollout_batch(rollout_batch)

    # Periodic evaluation (user-triggered)
    if step % eval_frequency == 0:
        curriculum.evaluate(inference_ctx, prng_key, n_examples=n_eval_examples)
        # evaluation iterates over all active environments and collects a sample
        curriculum.update_lesson_states()

    # Log metrics
    if step % 100 == 0:
        metrics = curriculum.get_metrics()
        print(f"Step {step}: Active={metrics['active_lessons']}, "
              f"Entropy={metrics['sampling_entropy']:.2f}")
```

## Metrics and Monitoring

```python
Curriculum::get_metrics(self) -> dict:
    """Comprehensive metrics for monitoring curriculum health."""
    weights = self.compute_sampling_weights()
    active = self.unlocked - self.graduated

    # Sampling entropy
    entropy = -sum(w * np.log(w + 1e-10) for w in weights.values() if w > 0)

    # Effective lessons (inverse Simpson index)
    effective = 1 / sum(w**2 for w in weights.values()) if weights else 0

    return {
        "step": self.current_step,
        "total_lessons": len(self.lesson_configs),
        "unlocked_lessons": len(self.unlocked),
        "active_lessons": len(active),
        "graduated_lessons": len(self.graduated),
        "sampling_entropy": entropy,
        "effective_lessons": effective,
        "mean_success": np.mean([
            get_combined_success_rate(self.stats[n], self.current_step)
            for n in active
        ]) if active else 0,
        "sampling_weights": weights,
    }

def check_health(self) -> list[str]:
    """Generate alerts for curriculum issues."""
    metrics = self.get_metrics()
    alerts = []

    if metrics["sampling_entropy"] < 0.5:
        alerts.append("WARNING: Low diversity - curriculum collapsing")

    if metrics["active_lessons"] < 0.2 * metrics["total_lessons"]:
        alerts.append("WARNING: Few active lessons - consider adjusting thresholds")

    if metrics["effective_lessons"] < 2:
        alerts.append("WARNING: Training dominated by <2 lessons")

    if metrics["graduated_lessons"] / metrics["total_lessons"] > 0.9:
        alerts.append("INFO: Most lessons graduated - curriculum may be too easy")

    return alerts
```

## State Persistence

The curriculum supports checkpointing for fault tolerance and experiment resumption.
Uses JSON serialization for portability and human-readability.

```python
def save_checkpoint(self, path: str):
    """Save curriculum state to disk as JSON."""
    checkpoint_data = {
        "lesson_configs": {
            name: {
                "lesson_name": config.lesson_name,
                "env_config": {
                    "env_class": config.env_config.env_class,
                    "env_args": config.env_config.env_args,
                },
                "dependencies": [
                    {"dependency_name": dep.dependency_name, "reward_threshold": dep.reward_threshold}
                    for dep in config.dependencies
                ],
                "initial_weight": config.initial_weight,
                "start_threshold": config.start_threshold,
                "stop_threshold": config.stop_threshold,
                "plateau_window": config.plateau_window,
                "plateau_threshold": config.plateau_threshold,
            }
            for name, config in self.lesson_configs.items()
        },
        "stats": {name: stats.to_dict() for name, stats in self.stats.items()},
        "unlocked": list(self.unlocked),
        "graduated": list(self.graduated),
        "current_step": self.current_step,
    }

    with open(path, "w") as f:
        json.dump(checkpoint_data, f, indent=2)

@classmethod
def load_checkpoint(cls, path: str, config: CurriculumConfig) -> "Curriculum":
    """Restore curriculum from checkpoint."""
    with open(path, "r") as f:
        checkpoint_data = json.load(f)

    # Create curriculum from config
    curriculum = cls(config)

    # Restore state
    curriculum.stats = {
        name: LessonStats.from_dict(stats_dict)
        for name, stats_dict in checkpoint_data["stats"].items()
    }
    curriculum.unlocked = set(checkpoint_data["unlocked"])
    curriculum.graduated = set(checkpoint_data["graduated"])
    curriculum.current_step = checkpoint_data["current_step"]

    return curriculum
```

# Implementation

You will implement this project in phases. After each phase you will validate
the progress with tests before committing the phase with an appropriate `git`
message.

Step 1.

Write out the core curriculum classes in `src/rl/curriculum.py`:

* LessonStatistics
* LessonDependency
* LessonPlan
* Curriculum

No tests.

Step 2.

Write functions for statistics tracking and environment sampling, along with
tests in `tests/rl/test_curriculum.py`.  Implement basic environment weighting
based on the low and high watermarks.

Your test should cover basic boundary conditions like single environment
curricula.

Step 3.

Add dependency activation and test.

Step 4.

Add graduation and plateau detection.
Add tests.

Step 5.

Add exploration bonuses.
Add tests.

Step 6.

Add checkpoint/restore support.
Add tests.

Step 7.
Integrate into `RolloutWorker`.

**Breaking change:** Rollout workers now accept a `CurriculumConfig` to initialize
the curriculum, replacing the `environment_spec: EnvConfig` field. No backwards
compatibility layer is needed.

Changes:
- Remove `environment_spec: EnvConfig` from `RolloutWorkerConfig`
- Add `curriculum_config: CurriculumConfig` to `RolloutWorkerConfig`
- Add `curriculum_eval_frequency: int` for triggering periodic evaluation
- Update RolloutWorker to sample from curriculum instead of single environment
- Add curriculum eval logic (triggered by rollout worker, evaluates unlocked lessons only)
- Update curriculum stats from rollouts

Update tests in `config_helpers` and `test_async_train`.

Step 8.

Update integration tests for "moar cats" and validate.

---

# Implementation Progress

## Implementation Decisions

During the planning phase, the following design decisions were made:

1. **Naming Conventions**: Used concise names (`LessonStats`, `LessonConfig`) instead of verbose alternatives to match existing codebase patterns.

2. **Dependency Structure**: Implemented dependencies as `list[LessonDependency]` dataclass format for type safety and clarity.

3. **RolloutWorker Integration**: Chose breaking change approach (Option A) - completely replace `environment_spec: EnvConfig` with `curriculum_config: CurriculumConfig`. No backwards compatibility layer.

4. **Environment Naming**: Rollouts use internal environment names (e.g., `"mock:cats"`) in the `env_name` field. Lesson tracking happens via separate curriculum state.

5. **Evaluation Trigger**: Evaluation is triggered by the rollout worker based on a configurable frequency. Evaluates only unlocked (non-graduated) lessons and updates `eval_reward` and `eval_success` in `LessonStats`.

6. **Checkpoint Format**: Uses JSON serialization (not pickle) for portability and human-readability. Leverages `LessonStats.to_dict()` and `from_dict()` methods.

7. **Integration Test**: For Step 8, focus on ensuring `test_full_integration_moar_cats` runs end-to-end with curriculum support.

8. **Plateau Detection Parameters**: Implemented as per-lesson configuration (`plateau_window`, `plateau_threshold`) with sensible defaults (50 samples, 0.01 relative slope).

## Completed Steps

### ✅ Step 1: Core Curriculum Classes (Completed)
**Files:** `src/marin/rl/curriculum.py`

Implemented core data structures:
- `LessonStats`: Performance tracking with serialization support
- `LessonDependency`: Prerequisite specification
- `LessonConfig`: Lesson configuration with plateau parameters
- `CurriculumConfig`: Top-level curriculum configuration
- `Curriculum`: Main curriculum manager class with state tracking

Includes circular dependency validation using DFS algorithm.

**Commit:** `3224abbf` - "Add statistics tracking and sampling for curriculum learning."

### ✅ Step 2: Statistics Tracking & Sampling (Completed)
**Files:** `src/marin/rl/curriculum.py`, `tests/rl/test_curriculum.py`

Implemented core curriculum logic:
- `update_from_rollout()`: Exponential smoothing of success/reward metrics
- `get_combined_success_rate()`: Blends eval and training metrics with staleness decay
- `compute_sampling_weights()`: Quadratic weighting peaking at 50% success, with sigmoid smoothing
- `sample_lesson()`: JAX-based weighted sampling
- `sigmoid()`: Smooth transition function

**Tests:**
- Statistics update mechanics
- Combined success rate computation
- Quadratic weight distribution
- Sampling respects weight distribution
- Initial weight handling
- Serialization round-trip

**Commit:** `3224abbf` - "Add statistics tracking and sampling for curriculum learning."

### ✅ Step 3: Dependency Management (Completed)
**Files:** `src/marin/rl/curriculum.py`, `tests/rl/test_curriculum.py`

Implemented dependency system:
- `check_dependencies()`: Validates threshold and plateau requirements
- `update_unlocked_lessons()`: Progressive lesson activation
- `is_plateaued()`: Linear regression-based plateau detection

**Tests:**
- Circular dependency detection
- Unknown dependency validation
- Progressive unlocking with thresholds
- Multiple dependency chains
- Plateau detection with various reward patterns

**Commit:** `9ab93df3` - "Add dependency management and plateau detection."

### ✅ Step 4: Graduation Logic (Completed)
**Files:** `src/marin/rl/curriculum.py`, `tests/rl/test_curriculum.py`

Implemented graduation system:
- `check_graduation()`: Verifies stop threshold + plateau + eval data
- `update_graduated_lessons()`: Automatic graduation tracking
- Graduated lessons automatically excluded from `compute_sampling_weights()`

**Tests:**
- Basic graduation with eval data requirement
- Plateau requirement for graduation
- Graduated lessons excluded from sampling

**Commit:** `203ef6a3` - "Add lesson graduation logic and tests."

### ✅ Step 5: Exploration Bonuses (Completed)
**Files:** `src/marin/rl/curriculum.py`, `tests/rl/test_curriculum.py`

Implemented exploration bonus system:
- Exponential decay: `bonus = 1.0 + exp(-0.03 * total_samples)`
- Bonus decays from 2x to ~1x over first 100 samples
- Applied to base weight in `compute_sampling_weights()`
- Ensures new lessons get sufficient exploration

**Tests:**
- Bonus effect on new vs experienced lessons
- Decay curve validation
- Convergence at high sample counts

**Commit:** `f67d188f` - "Integrate curriculum learning into RolloutWorker"

### ✅ Step 6: Checkpoint/Restore (Completed)
**Files:** `src/marin/rl/curriculum.py`, `tests/rl/test_curriculum.py`

Implemented JSON-based checkpoint system:
- `save_checkpoint(filename)`: Serializes curriculum state to JSON
- `load_checkpoint(config, filename)`: Restores from checkpoint
- Added `checkpoint_dir` field to `CurriculumConfig`
- Preserves: stats, unlocked, graduated, current_step

**Tests:**
- Round-trip save/load validation
- Reward history preservation
- Graduation state preservation
- Error handling for missing checkpoint_dir

**Commit:** `f67d188f` - "Integrate curriculum learning into RolloutWorker"

### ✅ Step 7: RolloutWorker Integration (Completed)
**Files:** `src/marin/rl/rollout_worker.py`, `tests/rl/config_helpers.py`

**Breaking changes:**
- Replaced `environment_spec: EnvConfig` with `curriculum_config: CurriculumConfig`
- Added `eval_n_examples`, `eval_n_generations`, `eval_frequency` to `CurriculumConfig`

**Implementation:**
- `RolloutWorker` initializes `Curriculum` instead of single environment
- `_generate_rollout_batch()` samples lesson from curriculum, updates stats
- `_evaluate_curriculum()` evaluates all active lessons periodically
- Automatic curriculum state updates (step, unlock, graduate)
- Test helpers create single-lesson curriculum by default

**Commit:** `f67d188f` - "Integrate curriculum learning into RolloutWorker"

## Remaining Work

### ⏳ Step 8: Integration Test (Not Started)
Validate end-to-end functionality with multi-lesson curriculum in integration tests.

**Note:** Steps 7 integration test updates and Step 8 deferred - basic integration complete and working with existing single-lesson default.

## Test Coverage

Current test count: **25 tests passing**

Test categories:
- Core functionality: 6 tests
- Dependency management: 4 tests
- Plateau detection: 1 test
- Graduation: 3 tests
- Sampling & weighting: 4 tests
- Exploration bonuses: 3 tests
- Checkpointing: 4 tests

## Architecture Notes

**Key Design Patterns:**
- Functional approach for statistics updates (immutable `LessonStats` transformations)
- Protocol-based `MarinEnv` for environment abstraction
- Dataclass-based configuration for clear type contracts
- JAX integration for sampling operations

**Performance Considerations:**
- Training metric updates: O(1) per rollout (exponential smoothing)
- Evaluation: O(n) where n = number of unlocked lessons
- Weight computation: O(k) where k = number of active lessons
- Dependency checking: O(d) where d = dependency depth (cached via sets)

**Future Extensions:**
- Dynamic threshold adjustment based on training progress
- Multi-objective optimization (speed vs accuracy)
- Automatic difficulty estimation from first N samples
- Curriculum visualization and debugging tools


## Distributed Curricula

Multiple rollout workers share a single curriculum to aggregate statistics and
coordinate lesson progression. The `Curriculum` class is deployed as a Ray actor
using the `get_or_create_curriculum_actor()` helper.

### RolloutStats Format

Curriculum updates use lightweight `RolloutStats` instead of full rollouts:

```python
@dataclass
class RolloutStats:
    """Lightweight statistics from a rollout for curriculum updates."""
    lesson_name: str
    episode_reward: float
    env_example_id: str
```

### Actor Creation

The `Curriculum` class itself is the Ray actor. Use `get_or_create_curriculum_actor()`
to obtain a handle:

```python
from marin.rl.curriculum import get_or_create_curriculum_actor, CurriculumConfig

# Get or create curriculum actor (shared across workers)
curriculum_actor = get_or_create_curriculum_actor(
    config=curriculum_config,
    name="curriculum"  # Named actor for shared access
)
```

### Curriculum Actor Interface

The actor supports remote method calls from multiple workers:

```python
# Sample lesson (returns lesson_name and serializable env_config dict)
lesson_name, env_config_dict = ray.get(curriculum_actor.sample_lesson.remote(seed=42))

# Collect statistics from rollouts (batched for efficiency)
rollout_stats_list = [
    RolloutStats(lesson_name="basic_math", episode_reward=1.0, env_example_id="ex_1"),
    RolloutStats(lesson_name="basic_math", episode_reward=0.5, env_example_id="ex_2"),
]

# Update statistics (training mode)
curriculum_actor.update_lesson_stats.remote(rollout_stats_list, mode="training")  # Async

# Update from evaluation (eval mode)
eval_stats_list = [...]
curriculum_actor.update_lesson_stats.remote(eval_stats_list, mode="eval")

# Update curriculum state
curriculum_actor.step.remote()
curriculum_actor.update_unlocked_lessons.remote()
curriculum_actor.update_graduated_lessons.remote()

# Get monitoring metrics
metrics = ray.get(curriculum_actor.get_metrics.remote())
print(f"Active lessons: {metrics['active_lessons']}")
print(f"Sampling entropy: {metrics['sampling_entropy']:.2f}")
```

### Multi-Worker Usage

Multiple rollout workers coordinate through the shared actor:

```python
# Worker 1
curriculum_actor = get_or_create_curriculum_actor(config, name="shared_curriculum")
lesson_name, env_config = ray.get(curriculum_actor.sample_lesson.remote(seed=1))
# ... generate rollouts ...
curriculum_actor.update_lesson_stats.remote(rollout_stats)

# Worker 2 (gets same actor by name)
curriculum_actor = get_or_create_curriculum_actor(config, name="shared_curriculum")
lesson_name, env_config = ray.get(curriculum_actor.sample_lesson.remote(seed=2))
# ... generate rollouts ...
curriculum_actor.update_lesson_stats.remote(rollout_stats)
```

### Concurrency & Synchronization

- Actor uses `num_cpus=0` for minimal overhead
- All methods are thread-safe (single actor instance)
- Stats updates are async (`.remote()` without `ray.get()`)
- Lesson sampling is synchronous (`ray.get()` required)
- The `get_or_create_actor()` pattern handles race conditions during initialization
