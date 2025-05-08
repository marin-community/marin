# How to add a new optimizer in Marin

Marin builds on Levanter for training code; this means that training changes added to Levanter will be available in Marin. It is also possible to add new optimizers directly in Marin, since Levanter supports [Optax](https://github.com/deepmind/optax) for optimization.
This makes it easier to experiment with new optimizers (especially for speedruns) without having to merge a PR into Levanter.

In this guide, we will use a speedrun submission as a working example.

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

4. Optax allows us to define multiple operations and chain them together to obtain the final update rule. In the example above, we:
   - Add gradient clipping (if specified)
   - Create the Adamax optimizer with its hyperparameters
   - Add weight decay (if specified)
   - Scale the learning rate
   - Chain all components together

5. To use the new optimizer in your training script, simply instantiate it with your desired parameters:

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

6. Use the optimizer in training: (TODO (Nikil): replace with an actual run, but with comments)

```python
from levanter.trainer import TrainerConfig
from marin.config import ModelConfig

# For regular training
trainer_config = TrainerConfig(
    optimizer=AdamaxConfig(
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        max_grad_norm=1.0,
        weight_decay=0.1,
        lr=1e-4
    ),
    # other trainer config params...
)

model_config = ModelConfig(
    # model config params...
)

# Run training
from marin.execution.executor import executor_main
executor_main(model_config, trainer_config)
```

7. Using the optimizer in a speedrun:

```python
from dataclasses import dataclass
from marin.speedrun import SpeedrunConfig

@dataclass
class MySpeedrunConfig(SpeedrunConfig):
    def __post_init__(self):
        super().__post_init__()
        self.trainer_config.optimizer = AdamaxConfig(
            beta1=0.9,
            beta2=0.95,
            epsilon=1e-8,
            max_grad_norm=1.0,
            weight_decay=0.1,
            lr=1e-4
        )

# Run speedrun
from marin.speedrun import default_speedrun

config = MySpeedrunConfig(
    # speedrun config params...
)
default_speedrun("my_speedrun", config)
```