# https://github.com/stanford-crfm/marin/issues/725
# Sweep to determine optimal hyperparameters for Adam on small scale
from optimizer_sweep.template import template
if __name__ == "__main__":
    sweep_grids = {
        'learning_rate': [2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2],
        'weight_decay': [0, 0.1, 0.2, 0.3],
        'warmup': [500, 1000, 2000, 4000],
        'beta1': [0.8, 0.9, 0.95, 0.98],
        'beta2': [0.9, 0.95, 0.98, 0.99, 0.995],
        'gamma': [0.00625, 0.0125, 0.025, 0.05],
        'epsilon': [1e-17, 1e-12, 1e-7],
        'train_batch_size': [128, 256, 512, 1024]
    }

    baseline_config = {
        'learning_rate': 4e-3, 
        'weight_decay': 0.1,
        'min_lr_ratio': 0,
        'warmup': 4000,
        'beta1': 0.95,
        'beta2': 0.99,
        'gamma': 0.0125,
        'epsilon': 1e-12,
        'max_grad_norm': 1.0,
        'train_batch_size': 128
    }
    model_size = "130m"
    target_chinchilla = 1 
    my_suffix = 'g'
    template(model_size, target_chinchilla, 'sophia', baseline_config, sweep_grids,  random_suffix=my_suffix) 


