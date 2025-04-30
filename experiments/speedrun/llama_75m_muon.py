"""
Muon optimizer definition and example speedrun in a single file.
"""

from dataclasses import dataclass
from levanter.optim import OptimizerConfig
import optax
import logging

from experiments.llama import llama_75m
from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.speedrun import ComputeBudget, HardwareConfig, SpeedrunConfig, default_speedrun
from marin.execution.executor import executor_main
from typing import Optional

@OptimizerConfig.register_subclass("muon")
@dataclass
class MuonConfig(OptimizerConfig):
    ns_coeffs: tuple[float, float, float] = (3.4445, -4.775, 2.0315)
    """Coefficients for the Newton-Schulz method in Muon."""
    ns_steps: int = 5
    """Number of Newton-Schulz iterations for orthogonalization."""
    beta: float = 0.95
    """Decay rate for the exponentially weighted average of gradients."""
    epsilon: float = 1e-8
    """Term added to denominators for numerical stability."""
    nesterov: bool = True
    """Whether to use Nesterov momentum."""
    adaptive: bool = False
    """Whether to scale updates by the dual norm of the original updates."""
    max_grad_norm: Optional[float] = 1.0
    """Optional gradient clipping by global norm."""

    def build(self, num_train_steps):
        """Creates the Muon optimizer with the specified learning rate schedule."""
        def _optimizer(learning_rate):
            components = []

            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))

            # Muon optimizer
            muon_transform = optax.contrib.scale_by_muon(
                ns_coeffs=self.ns_coeffs,
                ns_steps=self.ns_steps,
                beta=self.beta,
                eps=self.epsilon,
                nesterov=self.nesterov,
                adaptive=self.adaptive,
            )
            components.append(muon_transform)

            # Add weight decay if specified
            if self.weight_decay > 0:
                components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))

            # Scale by the negative learning rate for gradient descent
            components.append(optax.scale(-learning_rate))

            optimizer = optax.chain(*components)
            return optimizer

        return optax.inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler(num_train_steps))

# --------------------- Example Speedrun Using Muon ------------------------

logger = logging.getLogger("ray")

speedrun_config = SpeedrunConfig(
    compute_budget=ComputeBudget.SMALL,
    model_config=llama_75m,
    train_config=SimpleTrainConfig(
        tpu_type="v4-128",
        train_batch_size=512,
        num_train_steps=6000,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=2000,
        #steps_per_task_eval=2000,
        #optimizer_config=MuonConfig(learning_rate=3e-3),
        optimizer_config=AdamConfig(),
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
    executor_main(steps=default_speedrun("75M_llama_muon_adamsanity", speedrun_config))
