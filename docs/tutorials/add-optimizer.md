# How to Add a New Optimizer

Marin builds on [Levanter](https://levanter.readthedocs.io/) for training code, meaning any training changes made in Levanter are automatically available in Marin. However, you can also add new optimizers **directly in Marin**- thanks to Levanter’s support for [Optax](https://optax.readthedocs.io/)-- without needing to merge a pull request upstream.

In this guide, we’ll walk through adding an [AdaMax](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adamax) optimizer as an example.

---

## Steps to Add an Optimizer

1. Import Optax and OptimizerConfig:

    ```python
    from dataclasses import dataclass

    import optax
    from levanter.optim.config import OptimizerConfig
    ```

    Import `OptimizerConfig` from `levanter.optim.config`, not from the `levanter.optim`
    package: the package body is a docstring so that draccus discovers each optimizer
    submodule lazily, and it re-exports nothing.

2. Define a new optimizer by subclassing `OptimizerConfig`, add optimizer-specific parameters as
   fields, and register the class under an identifier. `OptimizerConfig` is a frozen dataclass, so
   the subclass must be frozen too:
    ```python
    @OptimizerConfig.register_subclass("adamax")
    @dataclass(frozen=True)
    class AdamaxConfig(OptimizerConfig):
        beta1: float = 0.9
        beta2: float = 0.95
        epsilon: float = 1e-8
        max_grad_norm: float | None = 1.0
    ```

    `OptimizerConfig` has a number of fields that are common to all optimizers; these include `weight_decay`, `learning_rate`, `lr_schedule`, `min_lr_ratio`, `warmup`, `decay`, `rewarmup`, `cycles`, and `cycle_length`. You can find documentation for the OptimizerConfig class, along with further details about the fields [here](https://levanter.readthedocs.io/en/latest/reference/Configuration/#standard-options).

3. Implement the `build()` method to define the optimizer's update rule. This method should return an Optax optimizer. Optax allows you to define components that are gradient transformations, and then chain them together to obtain a final gradient update rule.

    ```python
    def build(self, num_train_steps):
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

4. Use the optimizer in your training script. You can instantiate and pass it directly into your training config:

    ```python
    optimizer = AdamaxConfig(
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        max_grad_norm=1.0,
        weight_decay=0.1,
        learning_rate=1e-4,
    )
    ```

    In a Marin experiment, pass it as `train_lm`'s `optimizer` argument:

    ```python
    from marin.experiment.train import train_lm

    checkpoint = train_lm(
        ...
        optimizer=optimizer,
        ...
    )
    ```

    To drive Levanter directly instead, use `TrainLmConfig`:

    ```python
    from levanter.main.train_lm import TrainLmConfig

    trainer_config = TrainLmConfig(
        ...
        optimizer=optimizer,
        ...
    )
    ```

That’s it! You can now define new optimizers in this manner and train models using them, all within Marin. For optimizers that are widely useful or “standard,” consider submitting a pull request to Levanter.

Further reading:

- [Levanter Optimizer Documentation](https://levanter.readthedocs.io/en/latest/optimizers.html)
- [Optax Optimizer Reference](https://optax.readthedocs.io/en/latest/api/optimizers.html)
