from optimizer_sweep.template import template

if __name__ == '__main__':
    sweep_grids = {}
    baseline_config = {'learning_rate': 0.004, 'weight_decay': 0.1, 'min_lr_ratio': 0.0, 'warmup': 0.0, 'momentum': 0.98, 'beta1': 0.8, 'beta2': 0.98, 'epsilon': 1e-15, 'muon_epsilon': 1e-05, 'max_grad_norm': 2, 'decay': 1.0, 'train_batch_size':256, 'lr_schedule': 'linear', 'muon_to_adam_lr': 0.3}
    model_size = '1.2b'
    target_chinchilla = 4
    my_suffix = None
    template(model_size, target_chinchilla, 'muon', baseline_config, sweep_grids, random_suffix=my_suffix, tpu_type = 'v5litepod-256')
