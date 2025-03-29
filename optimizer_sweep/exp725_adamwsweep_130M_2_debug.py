# https://github.com/stanford-crfm/marin/issues/725
# Sweep to determine optimal hyperparameters for Adam on small scale
from optimizer_sweep.template import template
if __name__ == "__main__":
    sweep_grids = {
        'learning_rate': [4e-3, 8e-3, 1.6e-2, 3.2e-2],
        'weight_decay': [0, 0.1, 0.2],
        'warmup': [500, 1000, 2000, 4000],
        'beta1': [0.9, 0.95, 0.98],
        'beta2': [0.9, 0.95, 0.98],
        'epsilon': [1e-25, 1e-20, 1e-15, 1e-10],
        'max_grad_norm': [0, 1.0, 2.0],
        'train_batch_size': [128, 256, 512, 1024]
    }
    baseline_config = {
        'learning_rate': 1.6e-2, 
        'weight_decay': 0.1,
        'min_lr_ratio': 0,
        'warmup': 2000,
        'beta1': 0.9,
        'beta2': 0.95,
        'epsilon': 1e-20,
        'max_grad_norm': 1.0,
        'nesterov': False, 
        'train_batch_size': 128
    }
    model_size = "130m"
    target_chinchilla = 2 
    my_suffix = '_splash_v5' 
    template(model_size, target_chinchilla, 'adamw', baseline_config, sweep_grids,  random_suffix=my_suffix, force_run=True) 