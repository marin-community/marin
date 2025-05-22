from levanter.optim import MuonConfig
from typing import Optional
def muon_config(
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    beta1: Optional[float] = None,
    beta2: Optional[float] = None,
    momentum: Optional[float] = None,
    epsilon: Optional[float] = None,
    muon_epsilon: Optional[float] = None,
    muon_to_adam_lr: Optional[float] = None,
    nesterov: Optional[float] = None,
    max_grad_norm: Optional[float] = None,
    warmup: Optional[float] = None,
    decay: Optional[float] = None,
    lr_schedule: Optional[str] = None,
    stable_lr_schedule: Optional[str] = None,
    min_lr_ratio: Optional[float] = None,
    cycle_length: int | list[int] | None = None,
) -> MuonConfig:
    muon_config = MuonConfig(
            learning_rate=learning_rate,
            weight_decay=(
                weight_decay if weight_decay is not None else MuonConfig().weight_decay
            ),
            momentum=(momentum if momentum is not None else MuonConfig().momentum),
            beta1=(beta1 if beta1 is not None else MuonConfig().beta1),
            beta2=(beta2 if beta2 is not None else MuonConfig().beta2),
            epsilon=(epsilon if epsilon is not None else MuonConfig().epsilon),
            nesterov=(nesterov if nesterov is not None else MuonConfig().nesterov),
            max_grad_norm=(
                max_grad_norm if max_grad_norm is not None else MuonConfig().max_grad_norm
            ),
            muon_epsilon=(
                muon_epsilon if muon_epsilon is not None else MuonConfig().muon_epsilon
            ),
            muon_to_adam_lr=(
                muon_to_adam_lr
                if muon_to_adam_lr is not None
                else MuonConfig().muon_to_adam_lr
            ),
            warmup=(warmup if warmup is not None else MuonConfig().warmup),
            decay=(decay if decay is not None else MuonConfig().decay),
            lr_schedule=(lr_schedule if lr_schedule is not None else MuonConfig().lr_schedule),
            stable_lr_schedule=(
                stable_lr_schedule
                if stable_lr_schedule is not None
                else MuonConfig().stable_lr_schedule
            ),
            cycle_length=cycle_length,  # can be int, list[int], or None
            min_lr_ratio=(
                min_lr_ratio if min_lr_ratio is not None else MuonConfig().min_lr_ratio
            ),
        )
    return muon_config
