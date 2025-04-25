from dataclasses import dataclass
from levanter.optim import OptimizerConfig
import optax
from optax.contrib import scale_by_muon

@OptimizerConfig.register_subclass("muon")
@dataclass
class MuonConfig(OptimizerConfig):
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float | None = 1.0

    def build(self, num_train_steps):
        """
        Creates the Muon optimizer using optax.contrib.scale_by_muon.
        
        """
        def _optimizer(learning_rate):
            components = []
            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))
            components.append(scale_by_muon(self.beta1, self.beta2, self.epsilon))
            if self.weight_decay > 0:
                components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
            components.append(optax.scale(-learning_rate))
            return optax.chain(*components)
        return optax.inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler(num_train_steps))
