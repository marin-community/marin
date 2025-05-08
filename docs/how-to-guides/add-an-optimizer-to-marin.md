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

3. Implement the `build()` method to define the optimizer's update rule. This method should return an Optax optimizer:

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

5. That's it! You can now use the optimizer in your training script. To use the new optimizer in your training script, simply instantiate it with your desired parameters:

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
from experiments.simple_train_config import SimpleTrainConfig

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

default_train(
    name="",
    tokenized=,
    model_config=,
    train_config=train_config,
    tags=(),
    use_default_validation=True,
    eval_harness_tasks=(),
)
```

For an example of a full speedrun script using the optimizer, see [this speedrun](https://github.com/mar-in/marin/blob/main/experiments/speedrun/llama_75m_fineweb_edu_adamax.py).

See [Levanter's documentation](https://levanter.readthedocs.io/en/latest/optimizers.html) for more information on how to use Optax optimizers. Also, [Optax's documentation](https://optax.readthedocs.io/en/latest/api/optimizers.html) has a list of all available optimizers, as well as further details on how to implement a new/custom optimizer.
