"""
Muon optimizer definition and example speedrun in a single file.
"""

import logging
from dataclasses import dataclass

import optax
from levanter.optim import OptimizerConfig

from experiments.llama import llama_75m
from experiments.simple_train_config import SimpleTrainConfig
from marin.speedrun.speedrun import ComputeBudget, HardwareConfig, SpeedrunConfig, default_speedrun
from marin.execution.executor import executor_main


@dataclass
class AdamaxConfig(OptimizerConfig):
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float | None = 1.0

    def build(self, num_train_steps):
        print(f"Building optimizer: {self.__class__.__name__}")

        # Try to register the class if it's not already registered
        try:
            OptimizerConfig.register_subclass("adamax")(AdamaxConfig)
        except ValueError:
            # ignore gracefully and use the already registered class
            pass

        def _optimizer(learning_rate):
            components = []

            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))

            components.append(
                optax.scale_by_adamax(
                    b1=self.beta1,
                    b2=self.beta2,
                    eps=self.epsilon,
                )
            )

            if self.weight_decay > 0:
                components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))

            components.append(optax.scale(-learning_rate))

            optimizer = optax.chain(*components)
            return optimizer

        return optax.inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler(num_train_steps))


# --------------------- Example Speedrun Using Adamax ------------------------

logger = logging.getLogger("ray")

speedrun_config = SpeedrunConfig(
    compute_budget=ComputeBudget.SMALL,
    model_config=llama_75m,
    train_config=SimpleTrainConfig(
        tpu_type="v4-128",
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
        device_flops=275e12,
    ),
)

is_valid, error = speedrun_config.validate()
logger.info(f"Speedrun validation: {is_valid}, {error}")

if __name__ == "__main__":
    executor_main(steps=default_speedrun("75M_llama_adamax", speedrun_config))
