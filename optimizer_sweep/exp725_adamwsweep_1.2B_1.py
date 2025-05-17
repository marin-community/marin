from optimizer_sweep.template import template

if __name__ == '__main__':
    sweep_grids = {}
    baseline_config = {'learning_rate': 0.002, 'weight_decay': 0.2, 'min_lr_ratio': 0.0, 'warmup': 1000, 'beta1': 0.9, 'beta2': 0.98, 'epsilon': 1e-10, 'max_grad_norm': 1, 'train_batch_size': 256, 'nesterov': False}
    model_size = '1.2b'
    target_chinchilla = 1
    my_suffix = None
    template(model_size, target_chinchilla, 'adamw', baseline_config, sweep_grids, random_suffix=my_suffix)
