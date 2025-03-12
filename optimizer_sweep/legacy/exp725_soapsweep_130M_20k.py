# https://github.com/stanford-crfm/marin/issues/725
# Sweep to determine optimal hyperparameters for Adam on small scale
from optimizer_sweep.template import template
if __name__ == "__main__":
    # round 1
    sweep_grids = {
        'learning_rate': [4e-3, 8e-3, 1.6e-2],
        'weight_decay': [0, 0.1, 0.2, 0.4],
        'warmup': [500, 1000, 2000],
        'beta1': [0.9, 0.95, 0.98],
        'beta2': [0.9, 0.95, 0.98, 0.99],
        'shampoo_beta': [0.9, 0.95, 0.98, 0.99],
        'precondition_frequency': [1, 5, 10],
        'partition_grads_into_blocks': [False, True],
        'block_size': [128, 256, 512],
        'epsilon': [1e-20, 1e-15, 1e-10],
    }

    baseline_config = {
        'learning_rate': 8e-3, 
        'weight_decay': 0.2,
        'min_lr_ratio': 0,
        'warmup': 1000,
        'beta1': 0.95,
        'beta2': 0.95,
        'shampoo_beta': 0.95,
        'precondition_frequency': 10,
        'partition_grads_into_blocks': True,
        'block_size': 512,
        'epsilon': 1e-15,
        'max_grad_norm': 1.0
    }
    model_size = "130m"
    target_step = 20000
    template(model_size, target_step, 'soap', baseline_config, sweep_grids) 




