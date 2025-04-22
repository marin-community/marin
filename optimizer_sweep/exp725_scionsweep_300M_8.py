from optimizer_sweep.template import template

if __name__ == '__main__':
    sweep_grids = {'learning_rate': [0.004, 0.008], 'beta1': [0.95, 0.98], 'decay': [0.8, 1], 'train_batch_size': [256, 128]}
    baseline_config = {'learning_rate': 0.008, 'weight_decay': 0.1, 'min_lr_ratio': 0, 'warmup': 0, 'momentum': 0.95, 'beta1': 0.98, 'scion_epsilon': 1e-05, 'max_grad_norm': 2, 'lr_schedule': 'linear', 'scion_to_signum_lr': 0.1, 'decay': 0.8, 'train_batch_size': 128}
    model_size = '300m'
    target_chinchilla = 8
    my_suffix = None
    template(model_size, target_chinchilla, 'scion', baseline_config, sweep_grids, random_suffix=my_suffix)
