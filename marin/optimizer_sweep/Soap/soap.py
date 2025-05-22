from levanter.optim import SoapConfig
from typing import Optional


def soap_config(
    learning_rate: Optional[float] = None,
    weight_decay: Optional[float] = None,
    beta1: Optional[float] = None,
    beta2: Optional[float] = None,
    shampoo_beta: Optional[float] = None,
    precondition_frequency: Optional[int] = None,
    partition_grads_into_blocks: Optional[bool] = None,
    block_size: Optional[int] = None,
    epsilon: Optional[float] = None,
    max_grad_norm: Optional[float] = None,
    warmup: Optional[float] = None,
    decay: Optional[float] = None,
    lr_schedule: Optional[str] = None,
    stable_lr_schedule: Optional[str] = None,
    min_lr_ratio: Optional[float] = None,
    cycle_length: Optional[int] = None,
) -> SoapConfig:
    optimizer=SoapConfig(
            learning_rate=learning_rate,
            weight_decay=(
                weight_decay if weight_decay is not None else SoapConfig().weight_decay
            ),
            beta1=(beta1 if beta1 is not None else SoapConfig().beta1),
            beta2=(beta2 if beta2 is not None else SoapConfig().beta2),
            shampoo_beta=(
                shampoo_beta if shampoo_beta is not None else SoapConfig().shampoo_beta
            ),
            precondition_frequency=(
                precondition_frequency
                if precondition_frequency is not None
                else SoapConfig().precondition_frequency
            ),
            partition_grads_into_blocks=(
                partition_grads_into_blocks
                if partition_grads_into_blocks is not None
                else SoapConfig().partition_grads_into_blocks
            ),
            block_size=(block_size if block_size is not None else SoapConfig().block_size),
            epsilon=(epsilon if epsilon is not None else SoapConfig().epsilon),
            max_grad_norm=(
                max_grad_norm if max_grad_norm is not None else SoapConfig().max_grad_norm
            ),
            warmup=(warmup if warmup is not None else SoapConfig().warmup),
            decay=(decay if decay is not None else SoapConfig().decay),
            lr_schedule=(lr_schedule if lr_schedule is not None else SoapConfig().lr_schedule),
            stable_lr_schedule=(
                stable_lr_schedule
                if stable_lr_schedule is not None
                else SoapConfig().stable_lr_schedule
            ),
            cycle_length=cycle_length,  # can be int, list[int], or None
            min_lr_ratio=(
                min_lr_ratio if min_lr_ratio is not None else SoapConfig().min_lr_ratio
            ),
        )
    return optimizer