# https://github.com/stanford-crfm/marin/issues/725
# Sweep to determine optimal hyperparameters for Adam on small scale
from optimizer_sweep.template import template
if __name__ == "__main__":

    # round 1
    sweep_grids = {
        'learning_rate': [8e-3, 1.6e-2, 3.2e-2],
        'weight_decay': [0, 0.1, 0.2],
        'min_lr_ratio': [0, 0.05, 0.1],
        'warmup': [500, 1000, 2000, 4000],
        'beta1': [0.8, 0.9, 0.95, 0.98],
        'beta2': [0.9, 0.95, 0.98],
        'epsilon': [1e-25, 1e-20, 1e-15, 1e-10],
        'max_grad_norm': [0, 1.0, 2.0],
    }

    baseline_config = {
        'learning_rate': 1.6e-2, 
        'weight_decay': 0.1,
        'min_lr_ratio': 0,
        'warmup': 2000,
        'beta1': 0.95,
        'beta2': 0.95,
        'epsilon': 1e-25,
        'max_grad_norm': 1.0,
        'nesterov': True
    }
    model_size = "130m"
    target_step = 5000
    template(model_size, target_step, 'nadamw', baseline_config, sweep_grids) 




