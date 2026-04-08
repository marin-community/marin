# How to Add a New Optimizer

Marin builds on [Levanter](https://levanter.readthedocs.io/) for training code, meaning any training changes made in Levanter are automatically available in Marin. However, you can also add new optimizers **directly in Marin**- thanks to Levanter’s support for [Optax](https://optax.readthedocs.io/)-- without needing to merge a pull request upstream.

In this guide, we’ll walk through adding an [AdaMax](https://optax.readthedocs.io/en/latest/api/optimizers.html#optax.adamax) optimizer as an example.

---

## Steps to Add an Optimizer

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

    `OptimizerConfig` has a number of fields that are common to all optimizers; these include `weight_decay`, `learning_rate`, `lr_schedule`, `min_lr_ratio`, `warmup`, `decay`, `rewarmup`, `cycles`, and `cycle_length`. You can find documentation for the OptimizerConfig class, along with further details about the fields [here](https://levanter.readthedocs.io/en/latest/reference/Configuration/#standard-options).

3. Implement the `build()` method to define the optimizer's update rule. This method should return an Optax optimizer. Optax allows you to define components that are gradient transformations, and then chain them together to obtain a final gradient update rule.

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

    !!! note

        You should also register your optimizer class with an identifier, as shown above.

4. Use the optimizer in your training script. You can instantiate and pass it directly into your training config:

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

    and use it in `TrainLmConfig`:

    ```python
    from levanter.main.train_lm import TrainLmConfig

    trainer_config = TrainLmConfig(
        ...
        optimizer=optimizer,
        ...
    )
    ```

    Or inside a `SimpleTrainConfig`:

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

    Then pass it into `default_train`, which will set the optimizer config correctly in the training step.

That’s it! You can now define new optimizers in this manner and train models using them, all within Marin. For optimizers that are widely useful or “standard,” consider submitting a pull request to Levanter.

Further reading:

- [Levanter Optimizer Documentation](https://levanter.readthedocs.io/en/latest/optimizers.html)
- [Optax Optimizer Reference](https://optax.readthedocs.io/en/latest/api/optimizers.html)
