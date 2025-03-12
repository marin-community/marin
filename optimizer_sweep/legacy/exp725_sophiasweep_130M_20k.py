# https://github.com/stanford-crfm/marin/issues/725
# Sweep to determine optimal hyperparameters for Adam on small scale
from optimizer_sweep.template import template
if __name__ == "__main__":
    sweep_grids = {
        'learning_rate': [2e-3, 4e-3, 8e-3],
        'weight_decay': [0, 0.1, 0.2],
        'warmup': [2000, 4000, 8000],
        'beta1': [0.9, 0.95, 0.98],
        'beta2': [0.9, 0.95, 0.98],
        'gamma': [0.00625, 0.0125, 0.025, 0.05],
        'epsilon': [1e-17, 1e-12, 1e-7],
    }

    baseline_config = {
        'learning_rate': 4e-3, 
        'weight_decay': 0.1,
        'min_lr_ratio': 0,
        'warmup': 4000,
        'beta1': 0.95,
        'beta2': 0.95,
        'gamma': 0.0125,
        'epsilon': 1e-12,
        'max_grad_norm': 1.0
    }
    model_size = "130m"
    target_step = 20000
    template(model_size, target_step, 'sophia', baseline_config, sweep_grids, tpu_type = 'v4-256') 


