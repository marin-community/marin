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

@OptimizerConfig.register_subclass("muon")
@dataclass
class MuonConfig(OptimizerConfig):
    ns_coeffs: tuple[float, float, float] = (3.4445, -4.775, 2.0315)
    ns_steps: int = 5
    beta: float = 0.95
    eps: float = 1e-8
    nesterov: bool = True
    adaptive: bool = False
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    adam_eps_root: float = 0.0
    adam_weight_decay: float = 0.0
    mu_dtype: str | None = None
    max_grad_norm: float | None = 1.0

    def build(self, num_train_steps):
        """
        Creates the Muon optimizer using optax.contrib.muon.
        """
        def _optimizer(learning_rate):
            components = []
            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))
            components.append(optax.contrib.muon(
                learning_rate=learning_rate,
                ns_coeffs=self.ns_coeffs,
                ns_steps=self.ns_steps,
                beta=self.beta,
                eps=self.eps,
                weight_decay=self.weight_decay,
                weight_decay_mask=self.build_weight_decay_mask(),
                mu_dtype=self.mu_dtype,
                nesterov=self.nesterov,
                adaptive=self.adaptive,
                adam_b1=self.adam_b1,
                adam_b2=self.adam_b2,
                adam_eps_root=self.adam_eps_root,
                adam_weight_decay=self.adam_weight_decay,
            ))
            return optax.chain(*components)
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
        steps_per_task_eval=2000,
        optimizer_config=MuonConfig(learning_rate=3e-3, weight_decay=0.1),
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
    executor_main(steps=default_speedrun("75M_llama_muon", speedrun_config))
