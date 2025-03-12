# https://github.com/stanford-crfm/marin/issues/725
# Sweep to determine optimal hyperparameters for Adam on small scale
from optimizer_sweep.template import template
if __name__ == "__main__":
    sweep_grids = {
        'learning_rate': [8e-3, 1.6e-2, 3.2e-2, 6.4e-2, 1.28e-1],
        'weight_decay': [0, 0.1, 0.2],
        'momentum': [0.8, 0.9, 0.95, 0.98],
        'beta1': [0.8, 0.9, 0.95, 0.98],
        'scion_epsilon': [1e-20, 1e-15, 1e-10, 1e-5],
        'max_grad_norm': [0, 1.0, 2.0],
        'decay': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'scion_to_signum_lr': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    }

    baseline_config = {
        'learning_rate': 1.6e-2, 
        'weight_decay': 0,
        'min_lr_ratio': 0,
        'warmup': 0,
        'momentum': 0.98,
        'beta1': 0.95,
        'scion_epsilon': 1e-15,
        'max_grad_norm': 1.0,
        'lr_schedule': 'linear',
        'scion_to_signum_lr': 0.2,
        'decay': 0.4
    }
    model_size = "130m"
    target_step = 5000
    template(model_size, target_step, 'scion', baseline_config, sweep_grids) 


