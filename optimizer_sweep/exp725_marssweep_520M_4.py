from optimizer_sweep.template import template

if __name__ == '__main__':
    sweep_grids = {'warmup': [2000, 1000, 4000], 'beta1': [0.9, 0.95, 0.98], 'learning_rate': [0.004, 0.008], 'train_batch_size': [256, 128]}
    baseline_config = {'learning_rate': 0.008, 'weight_decay': 0.1, 'min_lr_ratio': 0, 'warmup': 2000, 'beta1': 0.95, 'beta2': 0.98, 'gamma': 0.025, 'epsilon': 9.999999999999999e-26, 'max_grad_norm': 1, 'train_batch_size': 256}
    model_size = '520m'
    target_chinchilla = 4
    my_suffix = None
    template(model_size, target_chinchilla, 'mars', baseline_config, sweep_grids, random_suffix=my_suffix)
