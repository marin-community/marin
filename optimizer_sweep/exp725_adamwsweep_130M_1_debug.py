# https://github.com/stanford-crfm/marin/issues/725
# Sweep to determine optimal hyperparameters for Adam on small scale
from optimizer_sweep.template import template
if __name__ == "__main__":
    sweep_grids = {}
    baseline_config = {
        'learning_rate': 6e-4, 
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
    my_suffix = '_baseline' 
    template(model_size, target_chinchilla, 'adamw', baseline_config, sweep_grids,  random_suffix=my_suffix, force_run=True) 