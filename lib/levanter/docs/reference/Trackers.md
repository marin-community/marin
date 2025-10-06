# Trackers and Metrics

Logging values and other metadata about a run is a core requirement for any ML framework.
Until recently, Levanter had a hard dependency on [W&B](https://wandb.ai/site) for tracking such values.
We now provide a pluggable tracker interface with built‑in support for W&B, Tensorboard, and [Trackio](https://github.com/gradio-app/trackio).

In the latest version, we introduce the [levanter.tracker.Tracker][] interface, which allows you to use any tracking backend you want.
The interface name is taken from the [HuggingFace Accelerate](https://github.com/huggingface/accelerate/blob/0f2686c8d3e6d949c4b7efa15d7f2dee44f7ce91/src/accelerate/tracking.py#L395)
framework.

Levanter ships with trackers for W&B, TensorBoard, and a lightweight JSON logger that emits structured log lines. The interface is designed to look similar to W&B's API.
The methods currently exposed are:

* [levanter.tracker.current_tracker][]: returns the current tracker instance or sets it.
* [levanter.tracker.log][]: logs a dictionary of metrics for a given step.
* [levanter.tracker.log_summary][]: logs a dictionary of "summary" information, analogous to W&B's version.
* [levanter.tracker.get_tracker][]: returns a tracker with the given name.
* [levanter.tracker.jit_log][]: a version of [levanter.tracker.log][] that accumulates metrics inside of a `jit`-ted function.

A basic example of using the tracker interface is shown below:

```python
import wandb
import levanter.tracker as tracker
from levanter.tracker.wandb import WandbTracker

with tracker.current_tracker(WandbTracker(wandb.init())):
    for step in range(100):
        tracker.log({"loss": 100 - 0.01 * step}, step=step)

    tracker.log_summary({"best_loss": 0.0})
```

A more typical example would be to use it in a config file, as we do with Trainer:

```yaml
trainer:
  tracker:
    type: wandb
    project: my-project
    entity: my-entity
```

### Multiple Trackers

In some cases, you may want to use multiple trackers at once.
For example, you may want to use both W&B and Tensorboard.

To do this, you can use the [levanter.tracker.tracker.CompositeTracker][] class, or, if using a config file, you
can specify multiple trackers:

```yaml
trainer:
  tracker:
    - type: wandb
      project: my-project
      entity: my-entity
    - type: tensorboard
      logdir: logs
```

Installation note: the TensorBoard tracker depends on `tensorboardX`. Install the profiling extra to get
both TensorBoard and TensorBoardX: `pip install "levanter[profiling]"` (or `uv sync --extra profiling`).

## Adding your own tracker

To add your own tracker, you need to implement the [levanter.tracker.Tracker][] interface.
You will also want to register your config with TrackerConfig as a "choice" in the choice type.
Follow the pattern for Tensorboard and W&B.

TODO: expand this section.


## API Reference

### Core Functions

::: levanter.tracker.current_tracker

::: levanter.tracker.log

::: levanter.tracker.log_summary

::: levanter.tracker.get_tracker

::: levanter.tracker.jit_log

### Trackers

::: levanter.tracker.Tracker

::: levanter.tracker.tracker.CompositeTracker

::: levanter.tracker.tracker.NoopTracker

::: levanter.tracker.tensorboard.TensorboardTracker

::: levanter.tracker.wandb.WandbTracker


::: levanter.tracker.trackio.TrackioTracker

::: levanter.tracker.json_logger.JsonLoggerTracker

### Tracker Config

::: levanter.tracker.TrackerConfig

::: levanter.tracker.tracker.NoopConfig

::: levanter.tracker.tensorboard.TensorboardConfig

::: levanter.tracker.wandb.WandbConfig


::: levanter.tracker.trackio.TrackioConfig

::: levanter.tracker.json_logger.JsonLoggerConfig
