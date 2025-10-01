# Adaptive Curriculum Learning System

## Background & Motivation

We need RL training to automatically sample from environments of appropriate difficulty without manual intervention. Sampling from too easy environments wastes compute and risks overfitting, while too hard distributions yield zero rewards or unstable learning (RLOO needs multiple positive examples).

An optimal curriculum focuses on the "zone of proximal development" - tasks just beyond current ability where learning is maximized. This design enables Marin to automatically reweight environments while minimizing evaluation overhead.

### Prior Work

- **[Automatic Curriculum Learning](https://arxiv.org/abs/2012.02035)**: Curricula emerge from competence-based progression
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
class LessonConfig:
    name: str
    env_config: EnvConfig  # Uses existing Marin EnvConfig
    dependencies: dict[str, float] = field(default_factory=dict)
    # e.g., {"basic_math": 0.7, "algebra": 0.6}
    initial_weight: float = 1.0
    start_threshold: float = 0.1  # Activate when unlocked
    stop_threshold: float = 0.95  # Deactivate when exceeded
```

### Performance Tracking

Statistics track recent training and evaluation performance for each lesson. Training updates are cheap (every rollout), evaluation updates are expensive (periodic).

```python
@dataclass
class LessonStats:
    smoothed_success: float = 0.0  # Exponentially smoothed success rate
    smoothed_reward: float = 0.0   # Exponentially smoothed reward
    reward_history: deque = field(default_factory=lambda: deque(maxlen=100))
    eval_success: float = 0.0      # Last evaluation success rate
    eval_step: int = -1            # When last evaluated
    total_samples: int = 0         # Total rollouts seen

def update_from_rollout(stats: LessonStats, rollout: Rollout) -> LessonStats:

def get_combined_success_rate(stats: LessonStats, current_step: int) -> float:
    """Blend training and eval metrics based on recency."""
    if stats.eval_step < 0:
        return stats.smoothed_success

    # Weight eval more if recent, training more if stale
    staleness = current_step - stats.eval_step
    eval_weight = 0.7 * np.exp(-0.001 * staleness)
    return eval_weight * stats.eval_success + (1 - eval_weight) * stats.smoothed_success
```

### Curriculum Manager

The curriculum maintains lesson configurations, statistics, and sampling state as a stateful class.

```python

def _validate_dependencies(lesson_config):
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


class AdaptiveCurriculum:
    """Manages lesson progression and sampling."""

    def __init__(
        self,
        lesson_configs: list[LessonConfig],
        eval_frequency: int = 1000,
        temperature: float = 1.0
    ):
        self.lesson_configs = {cfg.name: cfg for cfg in lesson_configs}
        self.stats = {cfg.name: LessonStats() for cfg in lesson_configs}
        self.environments = {
            cfg.name: load_environment_from_spec(cfg.env_config)
            for cfg in lesson_configs
        }

        self.unlocked = set()  # Currently accessible lessons
        self.graduated = set()  # Mastered lessons
        self.eval_frequency = eval_frequency
        self.temperature = temperature
        self.current_step = 0

        # Validate dependency DAG
        _validate_dependencies(self._lesson_configs)


```

### Dependency Management

Dependencies ensure lessons unlock in logical order based on prerequisite performance.

```python
def check_dependencies(self, lesson_name: str) -> bool:
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
        curriculum.trigger_evaluation(inference_ctx, prng_key)
        curriculum.update_lesson_states()

    # Log metrics
    if step % 100 == 0:
        metrics = curriculum.get_metrics()
        print(f"Step {step}: Active={metrics['active_lessons']}, "
              f"Entropy={metrics['sampling_entropy']:.2f}")
```

## Metrics and Monitoring

```python
def get_metrics(self) -> dict:
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
        "top_weights": sorted(
            [(n, w) for n, w in weights.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
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

```python
@dataclass
class CurriculumCheckpoint:
    """Serializable curriculum state."""
    lesson_configs: dict[str, LessonConfig]
    stats: dict[str, LessonStats]
    unlocked: set[str]
    graduated: set[str]
    current_step: int

def save_checkpoint(self, path: str):
    """Save curriculum state to disk."""
    checkpoint = CurriculumCheckpoint(
        lesson_configs=self.lesson_configs,
        stats=self.stats,
        unlocked=self.unlocked,
        graduated=self.graduated,
        current_step=self.current_step
    )
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)

def load_checkpoint(cls, path: str, eval_frequency: int = 1000) -> "AdaptiveCurriculum":
    """Restore curriculum from checkpoint."""
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)

    # Reconstruct curriculum
    curriculum = cls(
        lesson_configs=list(checkpoint.lesson_configs.values()),
        eval_frequency=eval_frequency
    )
    curriculum.stats = checkpoint.stats
    curriculum.unlocked = checkpoint.unlocked
    curriculum.graduated = checkpoint.graduated
    curriculum.current_step = checkpoint.current_step

    return curriculum
```
