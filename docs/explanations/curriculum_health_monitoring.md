# Curriculum Alert System

## Overview

The Curriculum Alert System provides comprehensive monitoring and alerting for RL curriculum learning. Alerts are configured per lesson and evaluated using existing curriculum statistics, providing a natural extension of environment testing without duplicating logic. The system continuously tracks lesson performance, detects regressions and stagnation, and sends automated alerts through Weights & Biases (W&B).

## Key Features

### 1. Per-Lesson Alert Configuration

Alerts are configured directly in `LessonConfig`, making them a natural part of lesson definition:

```python
from marin.rl.alerts import GraduationRegressionAlert, TrainingStalledAlert
from marin.rl.curriculum import LessonConfig

lesson = LessonConfig(
    lesson_id="math_task",
    env_config=env_config,
    alerts=[
        GraduationRegressionAlert(regression_threshold=0.85),
        TrainingStalledAlert(stagnation_window=100),
    ],
)
```

### 2. Alert Types

The system provides several built-in alert types:

#### `GraduationRegressionAlert`

Monitors graduated lessons to detect performance regression below graduation baseline.

- **`regression_threshold`**: Warn if performance drops below this fraction of graduation level (default: 0.85)
- **`critical_threshold`**: Critical warning if performance drops below this fraction (default: 0.70)
- **`cooldown_steps`**: Steps between repeated alerts (default: 200)

#### `TrainingStalledAlert`

Detects when training shows no progress for an extended period.

- **`stagnation_window`**: Number of steps without progress before alerting (default: 100)
- **`plateau_window`**: Window size for plateau detection. If None, uses lesson's `plateau_window` (default: None)
- **`plateau_threshold`**: Threshold for plateau detection. If None, uses lesson's `plateau_threshold` (default: None)
- **`cooldown_steps`**: Steps between repeated alerts (default: 200)

#### `PerformanceVolatilityAlert`

Detects high variance/volatility in performance metrics.

- **`volatility_threshold`**: Coefficient of variation threshold (default: 0.15)
- **`window`**: Number of recent samples to analyze (default: 30)
- **`cooldown_steps`**: Steps between repeated alerts (default: 200)

#### `ProlongedTrainingAlert`

Alerts when a lesson has been training for an extended period without graduating.

- **`max_training_steps`**: Maximum steps before alerting (default: 1000)
- **`cooldown_steps`**: Steps between repeated alerts (default: 500)

### 3. Health Status

All alerts use `HealthStatus` enum to indicate urgency:

- **`HEALTHY`**: No issues detected (typically not used in alerts, but available for consistency)
- **`WARNING`**: Issue requiring attention (e.g., regression, stagnation, volatility)
- **`CRITICAL`**: Immediate attention required (e.g., severe regression)

### 4. W&B Integration

When enabled, alerts are automatically sent to W&B:

```python
wandb.alert(
    title="Curriculum Alert: graduation_regression",
    text="[math_task] Graduated lesson 'math_task' performance dropped from 0.95 to 0.75",
    level="WARN",  # Maps to HealthStatus (WARNING -> WARN, CRITICAL -> ERROR)
    wait_duration=300  # Rate limiting
)
```

## Configuration

### Basic Configuration

```python
from marin.rl.alerts import GraduationRegressionAlert, TrainingStalledAlert
from marin.rl.curriculum import LessonConfig, CurriculumConfig

# Configure alerts per lesson
lesson = LessonConfig(
    lesson_id="math_task",
    env_config=env_config,
    alerts=[
        GraduationRegressionAlert(
            regression_threshold=0.85,  # Warn at 85% of graduation performance
            critical_threshold=0.70,    # Critical at 70%
        ),
        TrainingStalledAlert(
            stagnation_window=100,  # Steps without progress
        ),
    ],
)

# Create curriculum with wandb alerts enabled
curriculum_config = CurriculumConfig(
    lessons={"math_task": lesson},
    enable_wandb_alerts=True,
    wandb_alert_rate_limit=300,  # Seconds between W&B alerts
)
```

### Advanced Configuration

```python
from marin.rl.alerts import (
    GraduationRegressionAlert,
    TrainingStalledAlert,
    PerformanceVolatilityAlert,
    ProlongedTrainingAlert,
)

lesson = LessonConfig(
    lesson_id="complex_task",
    env_config=env_config,
    alerts=[
        # Monitor for regression after graduation
        GraduationRegressionAlert(
            regression_threshold=0.85,
            critical_threshold=0.70,
            cooldown_steps=200,
        ),
        # Detect training stagnation
        TrainingStalledAlert(
            stagnation_window=100,
            plateau_window=50,  # Custom plateau window
            plateau_threshold=0.01,
            cooldown_steps=200,
        ),
        # Monitor performance volatility
        PerformanceVolatilityAlert(
            volatility_threshold=0.15,
            window=30,
            cooldown_steps=200,
        ),
        # Alert if training too long
        ProlongedTrainingAlert(
            max_training_steps=1000,
            cooldown_steps=500,
        ),
    ],
)
```

## Usage

### Integration with Curriculum

Alerts are automatically evaluated when lesson statistics are updated:

```python
from marin.rl.curriculum import Curriculum, CurriculumConfig
from marin.rl.alerts import GraduationRegressionAlert

# Configure lesson with alerts
lesson = LessonConfig(
    lesson_id="math_task",
    env_config=env_config,
    alerts=[GraduationRegressionAlert(regression_threshold=0.85)],
)

# Create curriculum
curriculum_config = CurriculumConfig(
    lessons={"math_task": lesson},
    enable_wandb_alerts=True,
)

curriculum = Curriculum(curriculum_config)

# Alerts are automatically evaluated when stats are updated
curriculum.update_lesson_stats(rollout_stats, mode="training", current_step=100)
```

### Custom Alert Configuration

You can configure different alerts for different lessons:

```python
easy_lesson = LessonConfig(
    lesson_id="easy_task",
    env_config=easy_env_config,
    alerts=[
        TrainingStalledAlert(stagnation_window=50),  # Shorter window for easy tasks
    ],
)

hard_lesson = LessonConfig(
    lesson_id="hard_task",
    env_config=hard_env_config,
    alerts=[
        TrainingStalledAlert(stagnation_window=200),  # Longer window for hard tasks
        ProlongedTrainingAlert(max_training_steps=2000),  # More patience
    ],
)

curriculum_config = CurriculumConfig(
    lessons={
        "easy_task": easy_lesson,
        "hard_task": hard_lesson,
    },
    enable_wandb_alerts=True,
)
```

## Alert Evaluation Logic

### Graduation Regression Detection

Alerts trigger when a graduated lesson's performance drops below the graduation baseline:

```python
# Recorded at graduation
graduation_performance = 0.95

# Current performance drops to 0.75
current_performance = 0.75
ratio = current_performance / graduation_performance  # 0.79

# Triggers alert if ratio < regression_threshold (0.85)
if ratio < 0.85:
    # HealthStatus: WARNING (not CRITICAL since ratio > 0.70)
    alert = AlertResult(
        health_status=HealthStatus.WARNING,
        message="Graduated lesson 'math_task' performance dropped from 0.95 to 0.75"
    )
```

### Training Stagnation Detection

Alerts trigger when training has plateaued and shows no progress:

```python
# Check if performance has plateaued (using existing curriculum logic)
if is_plateaued(stats, window=50, threshold=0.01):
    # Check if stuck for stagnation_window steps
    recent_rewards = stats.training_stats.reward_history[-100:]
    if no_significant_change(recent_rewards):
        alert = AlertResult(
            health_status=HealthStatus.WARNING,
            message="Lesson 'math_task' showing no progress for 100 steps"
        )
```

### Performance Volatility Detection

Alerts trigger when performance shows high variance:

```python
recent_performance = stats.training_stats.reward_history[-30:]
mean_perf = np.mean(recent_performance)
std_perf = np.std(recent_performance)
cv = std_perf / abs(mean_perf)  # Coefficient of variation

if cv > volatility_threshold:  # 0.15
    alert = AlertResult(
        health_status=HealthStatus.WARNING,
        message="Lesson 'math_task' showing high performance volatility (CV=0.20)"
    )
```

## Best Practices

### 1. Threshold Tuning

- **`regression_threshold`**: Start with 0.85 (15% drop tolerance)
- **`critical_threshold`**: Set to 0.70 for severe issues
- **`stagnation_window`**: Adjust based on lesson complexity (50-200 steps)
- **`volatility_threshold`**: 0.15-0.20 for most lessons

### 2. Alert Selection

- **Graduated lessons**: Use `GraduationRegressionAlert` to monitor for catastrophic forgetting
- **Active lessons**: Use `TrainingStalledAlert` to detect when training isn't progressing
- **Unstable lessons**: Use `PerformanceVolatilityAlert` to identify instability
- **Long-running lessons**: Use `ProlongedTrainingAlert` to catch lessons that never graduate

### 3. Cooldown Management

- Set appropriate `cooldown_steps` to prevent alert spam
- Use `wandb_alert_rate_limit` in `CurriculumConfig` for W&B rate limiting
- Different alert types can have different cooldowns based on urgency

### 4. Debugging Performance Issues

When an alert is triggered:

1. Check the alert message and metrics in logs
2. Review recent training changes or hyperparameter adjustments
3. Examine lesson statistics using `curriculum.get_metrics()`
4. Look for correlated issues in other lessons

## Example Scenarios

### Scenario 1: Graduated Lesson Regression

```
1. Math lesson graduates at 95% success rate
2. After 1000 more training steps, performance drops to 75%
3. GraduationRegressionAlert triggers with WARNING health status (75% < 85% threshold, but > 70% critical threshold)
4. W&B alert sent: "[math_task] Graduated lesson 'math_task' performance dropped from 0.95 to 0.75"
5. Training team investigates potential catastrophic forgetting
```

### Scenario 2: Training Stagnation

```
1. Reasoning lesson stuck at 60% success for 150 steps
2. TrainingStalledAlert triggers with WARNING health status
3. Alert: "[reasoning] Lesson 'reasoning' showing no progress for 150 steps"
4. Team may need to adjust learning rate or curriculum dependencies
```

### Scenario 3: High Volatility

```
1. Language lesson showing 30% coefficient of variation
2. PerformanceVolatilityAlert triggers with WARNING health status
3. Alert: "[language] Lesson 'language' showing high performance volatility (CV=0.30)"
4. May indicate unstable training or environment issues
```

### Scenario 4: Prolonged Training

```
1. Complex lesson has been training for 1200 steps without graduating
2. ProlongedTrainingAlert triggers with WARNING health status
3. Alert: "[complex] Lesson 'complex' has been training for 1200 steps without graduating"
4. May indicate lesson is too difficult or needs curriculum adjustment
```

## Troubleshooting

### Common Issues

1. **Too Many Alerts**
   - Increase `cooldown_steps` on individual alerts
   - Increase `wandb_alert_rate_limit` in `CurriculumConfig`
   - Adjust thresholds to be less sensitive
   - Focus on CRITICAL health status alerts only

2. **Missing Regressions**
   - Decrease `regression_threshold` for earlier detection
   - Ensure lessons are being evaluated regularly
   - Check that graduated lessons have `GraduationRegressionAlert` configured

3. **False Positives**
   - Adjust `volatility_threshold` if lesson is naturally variable
   - Increase `stagnation_window` for lessons that naturally take longer
   - Consider lesson-specific alert configurations

4. **Alerts Not Triggering**
   - Verify alerts are configured in `LessonConfig.alerts`
   - Check that `enable_wandb_alerts=True` in `CurriculumConfig`
   - Ensure lesson statistics are being updated via `update_lesson_stats()`

## API Reference

### Alert Base Class

All alerts inherit from `Alert` and implement:

```python
class Alert(ABC):
    @abstractmethod
    def evaluate(
        self,
        lesson_id: str,
        stats: LessonStats,
        lesson_config: LessonConfig,
        current_step: int,
        lesson_state: str,  # "locked", "active", "graduated"
        graduation_performance: float | None = None,
    ) -> AlertResult | None:
        """Evaluate whether this alert condition is met."""
        pass
```

### AlertResult

Result returned when an alert condition is met:

```python
@dataclass
class AlertResult:
    triggered: bool
    message: str
    health_status: HealthStatus  # HEALTHY, WARNING, or CRITICAL
    metrics: dict[str, Any]
    timestamp: float
```

### CurriculumConfig

Configuration for alert system:

- **`enable_wandb_alerts`**: Whether to send alerts to W&B (default: True)
- **`wandb_alert_rate_limit`**: Minimum seconds between W&B alerts (default: 300)

### LessonConfig

Each lesson can have alerts configured:

- **`alerts`**: List of `Alert` instances to monitor for this lesson

## Migration from Health Monitor

If you were using the old `CurriculumHealthMonitor` system:

1. **Remove** `health_monitor_config` from `CurriculumConfig`
2. **Add** `alerts` to each `LessonConfig` with appropriate alert types
3. **Configure** `enable_wandb_alerts` and `wandb_alert_rate_limit` in `CurriculumConfig`
4. **Remove** calls to `curriculum.health_monitor.get_health_summary()` (use `curriculum.get_metrics()` instead)

The new system provides the same functionality but integrates naturally with lesson configuration and reuses existing curriculum statistics.
