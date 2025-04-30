from dataclasses import dataclass
from levanter.optim import OptimizerConfig
import optax

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
