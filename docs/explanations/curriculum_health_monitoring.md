# Curriculum Health Monitoring System

## Overview

The Curriculum Health Monitoring System provides comprehensive monitoring and alerting for RL curriculum learning. It continuously tracks the performance of all environments (locked, active, and graduated), detects performance regressions, and sends automated alerts through Weights & Biases (W&B).

## Key Features

### 1. Continuous Performance Tracking

The system tracks performance metrics for **all** environments, not just active ones:

- **Training Performance**: Success rates during training rollouts
- **Evaluation Performance**: Success rates during evaluation
- **Historical Performance**: Maintains a rolling window of recent performance data
- **Peak Performance**: Tracks the best performance achieved by each environment

### 2. Automated Warning System

The monitoring system generates warnings for various conditions:

#### Warning Types

- **`GRADUATED_REGRESSION`**: Graduated environment performance has dropped significantly
- **`TRAINING_STAGNATION`**: Active environment showing no progress
- **`PERFORMANCE_VOLATILITY`**: High variance in performance metrics
- **`UNEXPECTED_TRANSITION`**: Unusual state transition patterns
- **`PROLONGED_TRAINING`**: Environment training for excessive duration

#### Severity Levels

- **`critical`**: Immediate attention required (e.g., severe regression)
- **`high`**: Significant issue requiring prompt action
- **`medium`**: Notable issue that should be investigated
- **`low`**: Informational warning for awareness

### 3. Graduated Environment Monitoring

Once an environment graduates (reaches `stop_threshold` and plateaus), the system:

1. Records the graduation performance as a baseline
2. Continues monitoring performance in subsequent evaluations
3. Generates warnings if performance drops below configurable thresholds
4. Tracks time since graduation and performance trends

### 4. W&B Integration

When enabled, the system sends alerts to W&B:

```python
wandb.alert(
    title="Curriculum Health: graduated_regression",
    text="Environment 'math_task' performance dropped from 0.95 to 0.75",
    level="WARN",  # Maps to severity levels
    wait_duration=300  # Rate limiting
)
```

## Configuration

### Basic Configuration

```yaml
health_monitor_config:
  enabled: true  # Enable/disable monitoring
  regression_threshold: 0.85  # Warn at 85% of graduation performance
  critical_regression_threshold: 0.70  # Critical at 70%
  stagnation_window: 100  # Steps without progress
  evaluation_frequency: 50  # Eval all environments every N steps
```

### Advanced Configuration

```yaml
health_monitor_config:
  # Performance monitoring
  performance_history_size: 100  # Rolling window size
  volatility_threshold: 0.15  # Coefficient of variation threshold
  prolonged_training_threshold: 1000  # Max training steps

  # Alert management
  warning_cooldown: 200  # Steps between repeated warnings
  enable_wandb_alerts: true  # Send to W&B
  wandb_alert_rate_limit: 300  # Seconds between W&B alerts
```

## Usage

### Integration with Curriculum

The health monitor is automatically integrated when you create a curriculum:

```python
from marin.rl.curriculum import Curriculum, CurriculumConfig
from marin.rl.health_monitor import HealthMonitorConfig

# Configure health monitoring
health_config = HealthMonitorConfig(
    enabled=True,
    regression_threshold=0.85,
    enable_wandb_alerts=True
)

# Create curriculum with monitoring
curriculum_config = CurriculumConfig(
    lessons=lessons,
    health_monitor_config=health_config
)

curriculum = Curriculum(curriculum_config)
```

### Accessing Health Metrics

```python
# Get overall health summary
health_summary = curriculum.health_monitor.get_health_summary()
print(f"Graduated regressions: {health_summary['graduated_regressions']}")
print(f"Active warnings: {health_summary['active_warnings']}")

# Get detailed report for specific environment
report = curriculum.health_monitor.get_environment_report("math_task")
print(f"Health status: {report['health_status']}")
print(f"Performance trend: {report['performance']['trend']}")
```

### Metrics in Curriculum

Health metrics are included in the curriculum's metrics:

```python
metrics = curriculum.get_metrics()

# Overall health summary
health = metrics["health"]
print(f"Healthy environments: {health['health_distribution']['healthy']}")
print(f"Warning environments: {health['health_distribution']['warning']}")
print(f"Critical environments: {health['health_distribution']['critical']}")

# Per-environment health
env_health = metrics["environment_health"]
for env_id, status in env_health.items():
    print(f"{env_id}: {status['status']} (trend: {status['trend']})")
```

## Warning Detection Logic

### Graduated Regression Detection

```python
def check_graduated_regression(env_id, current_performance):
    baseline = graduation_baselines[env_id]
    ratio = current_performance / baseline

    if ratio < critical_regression_threshold:
        return Warning(severity="critical", ...)
    elif ratio < regression_threshold:
        return Warning(severity="high", ...)
    return None
```

### Training Stagnation Detection

```python
def check_training_stagnation(env_id, steps_without_progress):
    if steps_without_progress > stagnation_window:
        return Warning(
            type=WarningType.TRAINING_STAGNATION,
            severity="medium",
            message=f"No progress for {steps_without_progress} steps"
        )
    return None
```

### Performance Volatility Detection

```python
def check_volatility(env_id, performance_history):
    cv = coefficient_of_variation(performance_history)
    if cv > volatility_threshold:
        return Warning(
            type=WarningType.PERFORMANCE_VOLATILITY,
            severity="low",
            message=f"High volatility (CV={cv:.3f})"
        )
    return None
```

## Best Practices

### 1. Threshold Tuning

- **`regression_threshold`**: Start with 0.85 (15% drop tolerance)
- **`critical_regression_threshold`**: Set to 0.70 for severe issues
- **`stagnation_window`**: Adjust based on environment complexity (50-200 steps)
- **`volatility_threshold`**: 0.15-0.20 for most environments

### 2. Alert Management

- Enable W&B alerts for production training
- Set appropriate rate limits to prevent alert fatigue
- Use severity levels to prioritize responses
- Monitor the health summary regularly

### 3. Debugging Performance Issues

When a regression is detected:

1. Check the environment report for detailed metrics
2. Review recent training changes or hyperparameter adjustments
3. Examine the performance trend (improving/stable/regressing)
4. Look for correlated issues in other environments

### 4. Checkpoint Integration

Health monitor state is saved with curriculum checkpoints:

```python
# Save checkpoint (includes health monitor state)
curriculum.save_checkpoint("checkpoints/curriculum_state.json")

# Restore checkpoint (restores health monitor state)
curriculum.restore_checkpoint("checkpoints/curriculum_state.json")
```

## Example Scenarios

### Scenario 1: Graduated Environment Regression

```
1. Math environment graduates at 95% success rate
2. After 1000 more training steps, performance drops to 75%
3. System generates HIGH severity warning (75% < 85% threshold)
4. W&B alert sent: "Math environment regressed from 0.95 to 0.75"
5. Training team investigates potential catastrophic forgetting
```

### Scenario 2: Training Stagnation

```
1. Reasoning environment stuck at 60% success for 150 steps
2. System generates MEDIUM severity warning
3. Alert: "Reasoning showing no progress for 150 steps"
4. Team may need to adjust learning rate or curriculum dependencies
```

### Scenario 3: High Volatility

```
1. Language task showing 30% coefficient of variation
2. System generates LOW severity warning
3. Alert: "Language task showing high performance volatility"
4. May indicate unstable training or environment issues
```

## Troubleshooting

### Common Issues

1. **Too Many Alerts**
   - Increase `warning_cooldown` and `wandb_alert_rate_limit`
   - Adjust thresholds to be less sensitive
   - Focus on critical/high severity warnings only

2. **Missing Regressions**
   - Decrease `regression_threshold` for earlier detection
   - Ensure `evaluation_frequency` is appropriate
   - Check that graduated environments are being evaluated

3. **False Positives**
   - Increase `performance_history_size` for more stable metrics
   - Adjust `volatility_threshold` if environment is naturally variable
   - Consider environment-specific configurations

## API Reference

### HealthMonitorConfig

Configuration dataclass for the health monitoring system.

### CurriculumHealthMonitor

Main monitoring class that tracks performance and generates warnings.

Key methods:
- `update(env_id, performance, mode, state, step)`: Update metrics
- `check_warnings()`: Check for and return new warnings
- `get_health_summary()`: Get overall health statistics
- `get_environment_report(env_id)`: Get detailed report for an environment

### Warning

Dataclass representing a health warning with type, severity, and metrics.

### EnvironmentHealthMetrics

Comprehensive metrics for a single environment including performance, trends, and health status.
