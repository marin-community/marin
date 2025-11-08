# RolloutWorker Hook System

## Overview

The RolloutWorker hook system provides a flexible, extensible mechanism for customizing the behavior of rollout workers during RL training. This system replaces the previous hardcoded evaluation logic with a plugin-based architecture that allows arbitrary callbacks to be registered and executed at specific points during the rollout generation process.

## Motivation

Previously, the RolloutWorker had hardcoded evaluation logic in its main run loop:
- Micro-evaluations executed every 10 steps
- Full curriculum evaluations executed every 100 steps
- No way to customize this behavior without modifying the source code
- Tests required monkey-patching to change evaluation behavior

This tight coupling made it difficult to:
- Add custom monitoring or logging
- Implement different evaluation strategies
- Test the worker without complex mocking
- Add new functionality without modifying core code

## Architecture

### Core Components

#### Hook Base Class
The `Hook` abstract base class defines the interface for all hooks:

```python
class Hook(ABC):
    @abstractmethod
    def should_run(self, context: HookContext) -> bool:
        """Determine if this hook should run at the current step."""
        pass
    
    @abstractmethod
    def run(self, context: HookContext) -> Optional[dict[str, Any]]:
        """Execute the hook logic."""
        pass
```

#### HookContext
Provides all necessary information to hooks:

```python
@dataclass
class HookContext:
    worker: "RolloutWorker"  # Reference to the worker
    step: int                # Current step number
    rng: jnp.ndarray        # JAX random key
    lesson_id: Optional[str] # Current lesson being processed
    metadata: dict[str, Any] # Additional context data
```

#### HookManager
Manages hook registration and execution:

```python
class HookManager:
    def register_hook(self, hook: Hook) -> None
    def unregister_hook(self, hook: Hook) -> bool
    def clear_hooks(self) -> None
    def run_hooks(self, context: HookContext) -> dict[str, Any]
```

### Built-in Hook Types

#### PeriodicHook
Base class for hooks that run at regular intervals:

```python
class PeriodicHook(Hook):
    def __init__(self, frequency: int, start_step: int = 0):
        self.frequency = frequency
        self.start_step = start_step
```

#### EvaluationHook
Preserves the original evaluation behavior:

```python
EvaluationHook(
    frequency=100,           # How often to run
    n_examples=64,          # Number of evaluation examples
    eval_type="eval",       # Type of evaluation
    evaluate_all_lessons=True  # Evaluate all or just current lesson
)
```

#### Other Built-in Hooks
- `LoggingHook`: Periodic logging of worker state
- `MetricsHook`: Custom metrics collection
- `CheckpointHook`: Trigger checkpoints at intervals
- `CompositeHook`: Combine multiple hooks

## Usage Examples

### Basic Usage

```python
from marin.rl.rollout_worker import RolloutWorker
from marin.rl.hooks import EvaluationHook, LoggingHook

# Create worker with config
worker = RolloutWorker(config)

# Register a custom evaluation hook
custom_eval = EvaluationHook(
    frequency=50,
    n_examples=32,
    eval_type="custom_eval"
)
worker.register_hook(custom_eval)

# Add logging
logging_hook = LoggingHook(frequency=100)
worker.register_hook(logging_hook)
```

### Custom Hook Implementation

```python
class CustomMonitoringHook(PeriodicHook):
    """Monitor rollout generation health."""
    
    def __init__(self):
        super().__init__(frequency=25)
        self.metrics = []
    
    def run(self, context: HookContext) -> Optional[dict[str, Any]]:
        # Collect custom metrics
        metrics = {
            "monitoring.step": context.step,
            "monitoring.lesson": context.lesson_id,
            "monitoring.timestamp": time.time()
        }
        self.metrics.append(metrics)
        
        # Log to worker's tracker
        context.worker.tracker.log(metrics, step=context.step)
        return metrics

# Register the custom hook
worker.register_hook(CustomMonitoringHook())
```

### Testing with Hooks

Instead of monkey-patching:

```python
# OLD: Monkey-patching approach
def mock_evaluate_lesson(self, *args, **kwargs):
    return {"mocked": True}

worker._evaluate_lesson = mock_evaluate_lesson
```

Use custom test hooks:

```python
# NEW: Hook-based approach
class TestEvaluationHook(Hook):
    def __init__(self):
        self.evaluations = []
    
    def should_run(self, context):
        return context.step % 5 == 0
    
    def run(self, context):
        result = {"test_eval": context.step}
        self.evaluations.append(result)
        return result

test_hook = TestEvaluationHook()
worker.register_hook(test_hook)

# Run worker and check test_hook.evaluations
```

### Dynamic Hook Management

```python
# Add hooks during runtime
health_monitor = HealthMonitoringHook()
worker.register_hook(health_monitor)

# Remove hooks when no longer needed
worker.unregister_hook(health_monitor)

# Clear all hooks (useful for testing)
worker.clear_hooks()

# Get current hooks
current_hooks = worker.get_hooks()
```

## Migration Guide

### For Existing Code

The default behavior is preserved automatically. When a RolloutWorker is created, it automatically registers default evaluation hooks based on the curriculum config:

```python
# No changes needed - default hooks are registered automatically
worker = RolloutWorker(config)
# Micro-eval and full eval will run as before
```

### For Custom Evaluation

Replace direct modifications with hooks:

```python
# Before: Modifying evaluation frequencies required config changes
config.curriculum_config.micro_eval_frequency = 5
config.curriculum_config.eval_frequency = 50

# After: Add custom hooks without changing config
worker.register_hook(EvaluationHook(frequency=5, ...))
worker.register_hook(EvaluationHook(frequency=50, ...))
```

### For Tests

Replace monkey-patching with test hooks:

```python
# Before
original_method = worker._evaluate_lesson
worker._evaluate_lesson = mock_method
try:
    # run test
finally:
    worker._evaluate_lesson = original_method

# After
test_hook = CustomTestHook()
worker.register_hook(test_hook)
# run test
worker.unregister_hook(test_hook)
```

## Best Practices

### Hook Design

1. **Single Responsibility**: Each hook should have one clear purpose
2. **Idempotent**: Hooks should be safe to run multiple times
3. **Error Handling**: Hooks should handle errors gracefully
4. **Performance**: Avoid expensive operations in frequently-run hooks

### Hook Naming

Use descriptive names that indicate:
- What the hook does
- When it runs
- What it monitors/evaluates

Examples:
- `MicroEvaluationHook`
- `CurriculumHealthMonitor`
- `RolloutMetricsCollector`

### Hook Configuration

Make hooks configurable:

```python
class ConfigurableHook(PeriodicHook):
    def __init__(self, frequency=100, threshold=0.9, enabled=True):
        super().__init__(frequency)
        self.threshold = threshold
        self.enabled = enabled
    
    def should_run(self, context):
        return self.enabled and super().should_run(context)
```

### Testing Hooks

Write unit tests for custom hooks:

```python
def test_custom_hook():
    hook = CustomHook()
    context = HookContext(
        worker=mock_worker,
        step=100,
        rng=jrandom.PRNGKey(42)
    )
    
    # Test should_run logic
    assert hook.should_run(context)
    
    # Test run logic
    result = hook.run(context)
    assert "expected_key" in result
```

## Advanced Usage

### Conditional Hooks

```python
class ConditionalHook(Hook):
    def should_run(self, context):
        # Run only when specific conditions are met
        return (
            context.step > 1000 and
            context.lesson_id == "advanced_lesson" and
            context.metadata.get("performance") > 0.8
        )
```

### Stateful Hooks

```python
class StatefulHook(PeriodicHook):
    def __init__(self):
        super().__init__(frequency=50)
        self.state = {"runs": 0, "total_reward": 0}
    
    def run(self, context):
        self.state["runs"] += 1
        reward = context.metadata.get("reward", 0)
        self.state["total_reward"] += reward
        
        return {
            "avg_reward": self.state["total_reward"] / self.state["runs"]
        }
```

### Hook Composition

```python
# Combine multiple hooks into one
composite = CompositeHook([
    EvaluationHook(frequency=100, ...),
    LoggingHook(frequency=50),
    MetricsHook(frequency=25, ...)
])
worker.register_hook(composite)
```

## Troubleshooting

### Common Issues

1. **Hook not running**: Check `should_run()` logic and frequency settings
2. **Missing context data**: Ensure required data is in `HookContext`
3. **Performance impact**: Profile hooks and optimize expensive operations
4. **Hook conflicts**: Ensure hooks don't interfere with each other

### Debugging

Enable debug logging to see hook execution:

```python
import logging
logging.getLogger("marin.rl.hooks").setLevel(logging.DEBUG)
```

## Future Enhancements

Potential future improvements to the hook system:

1. **Priority-based execution**: Run hooks in priority order
2. **Async hooks**: Support for asynchronous hook execution
3. **Hook dependencies**: Define dependencies between hooks
4. **Hook marketplace**: Share and reuse hooks across projects
5. **Configuration files**: Load hooks from YAML/JSON configs
6. **Hook metrics**: Built-in performance monitoring for hooks

## API Reference

See the [hooks module documentation](../references/hooks.md) for complete API details.
