# How to add a new optimizer in Marin

Marin builds on Levanter for training code; this means that training changes added to Levanter will be available in Marin. It is also possible to add new optimizers directly in Marin, since Levanter supports [Optax](https://github.com/deepmind/optax) for optimization.
This makes it easier to experiment with new optimizers (especially for speedruns) without having to merge a PR into Levanter.

In this guide, we will use a speedrun submission that uses [AdaMax](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adamax) as a working example.

### Steps to add an optimizer

1. Import Optax and OptimizerConfig:

```python
import optax
from levanter.optim import OptimizerConfig
```

2. Define a new optimizer by subclassing `OptimizerConfig` and add optimizer-specific parameters as class variables:

```python
@dataclass
class AdamaxConfig(OptimizerConfig):
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float | None = 1.0
```
`OptimizerConfig` has a number of fields that are common to all optimizer; these include `weight_decay`, `learning_rate`, `lr_schedule`, `min_lr_ratio`, `warmup`, `decay`, `rewarmup`, `cycles`, and `cycle_length`. You can find documentation for the OptimizerConfig class, along with further details about the fields [here](https://levanter.readthedocs.io/en/latest/reference/Configuration/#standard-options).

3. Implement the `build()` method to define the optimizer's update rule. This method should return an Optax optimizer. Optax allows you to define components that are gradient transformations, and then chain them together to obtain a final gradient update rule.

Additionally, you should register your optimizer class with an identifier, as shown below.

```python
def build(self, num_train_steps):
    print(f"Building optimizer: {self.__class__.__name__}")

    # Register the optimizer class if not already registered
    try:
        OptimizerConfig.register_subclass("adamax")(AdamaxConfig)
    except ValueError:
        pass

    def _optimizer(learning_rate):
        components = []

        # Add gradient clipping if specified
        if self.max_grad_norm is not None:
            components.append(optax.clip_by_global_norm(self.max_grad_norm))

        # Add the Adamax optimizer
        components.append(
            optax.adamax(
                b1=self.beta1,
                b2=self.beta2,
                eps=self.epsilon,
            )
        )

        # Add weight decay if specified
        if self.weight_decay > 0:
            components.append(
                optax.add_decayed_weights(
                    self.weight_decay,
                    self.build_weight_decay_mask()
                )
            )

        # Scale the learning rate
        components.append(optax.scale(-learning_rate))

        # Chain all components together
        optimizer = optax.chain(*components)
        return optimizer

    # Inject hyperparameters using the learning rate scheduler
    return optax.inject_hyperparams(_optimizer)(
        learning_rate=self.lr_scheduler(num_train_steps)
    )
```

Note that `optax.inject_hyperparams` is a wrapper in Optax that can be used to pass schedules (or stateful hyperparameters) into the optimizer. This also allows us to log the learning rate in the tracker.

5. You can now use the optimizer in your training script. To use the new optimizer in your training script, simply instantiate it with your desired parameters:

```python
optimizer = AdamaxConfig(
    beta1=0.9,
    beta2=0.95,
    epsilon=1e-8,
    max_grad_norm=1.0,
    weight_decay=0.1,
    lr=1e-4
)
```

and you can simply pass it into `TrainLmConfig`:

```python
from levanter.trainer import TrainLmConfig

trainer_config = TrainLmConfig(
    ...
    optimizer=AdamaxConfig(
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        max_grad_norm=1.0,
        weight_decay=0.1,
        lr=1e-4
    ),
    ...
)
```

Alternatively, you can set the optimizer config in `SimpleTrainConfig`:

```python
train_config = SimpleTrainConfig(
    ...
    optimizer_config=AdamaxConfig(
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        max_grad_norm=1.0,
        weight_decay=0.1,
        lr=1e-4
    ),
    ...
)
```

and pass it into `default_train`, which will set the optimizer config correctly in the training step.

That's it! You can use this approach to define new optimizers you are interested in experimenting with; for optimizers that work well and/or ones that are "standard", considering opening a pull request to Levanter with an implementation.

Here's an example of a speedrun configuration that leverages `AdamaxConfig`:

```
speedrun_config = SpeedrunConfig(
    model_config=llama_75m,
    train_config=SimpleTrainConfig(
        TpuPodConfig(tpu_type="v4-128"),
        train_batch_size=512,
        num_train_steps=6000,
        learning_rate=3e-3,
        weight_decay=0.0,
        steps_per_eval=2000,
        steps_per_task_eval=2000,
        optimizer_config=AdamaxConfig(),
    ),
    hardware_config=HardwareConfig(
        device_type="v4-128",
        num_devices=64,
        device_flops=275e12,  # from https://cloud.google.com/tpu/docs/v4
    ),
)

if __name__ == "__main__":
    executor_main(steps=default_speedrun("75M_llama_adamax", speedrun_config))
```

That's it! You can now define new optimizers in this manner and train models using them, all within Marin. For optimizers that are relatively standard, or that work really well, consider opening a pull request to add the optimizer to Levanter.

See below for further examples and relevant links:

[This link](https://github.com/marin-community/marin/blob/main/experiments/speedrun/llama_75m_fineweb_edu_adamax/llama_75m_fineweb_edu_adamax.py) contains the full code for defining a new optimizer and running a speedrun.

See [Levanter's documentation](https://levanter.readthedocs.io/en/latest/optimizers.html) for more information on how to use Optax optimizers. Also, [Optax's documentation](https://optax.readthedocs.io/en/latest/api/optimizers.html) has a list of all available optimizers, as well as further details on how to implement a new/custom optimizer.
