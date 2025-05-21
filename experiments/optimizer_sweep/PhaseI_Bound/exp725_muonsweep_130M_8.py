from marin.optimizer_sweep.template import template

if __name__ == '__main__':
    sweep_grids = {'learning_rate': [0.004, 0.008], 'muon_to_adam_lr': [0.3, 0.4]}
    baseline_config = {'learning_rate': 0.008, 'weight_decay': 0.1, 'min_lr_ratio': 0, 'warmup': 0, 'momentum': 0.98, 'beta1': 0.8, 'beta2': 0.98, 'epsilon': 1e-25, 'muon_epsilon': 1e-05, 'max_grad_norm': 1, 'lr_schedule': 'linear', 'muon_to_adam_lr': 0.3, 'decay': 1, 'train_batch_size': 128}
    model_size = '130m'
    target_chinchilla = 8
    my_suffix = None
    template(model_size, target_chinchilla, 'muon', baseline_config, sweep_grids, random_suffix=my_suffix)
