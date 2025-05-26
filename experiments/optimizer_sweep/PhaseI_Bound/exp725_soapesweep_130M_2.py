from marin.optimizer_sweep.template import template

if __name__ == '__main__':
    sweep_grids = {'learning_rate': [0.016, 0.032]}
    baseline_config = {'learning_rate': 0.008, 'weight_decay': 0.1, 'min_lr_ratio': 0, 'warmup': 500, 'beta1': 0.95, 'beta2': 0.99, 'shampoo_beta': 0.98, 'precondition_frequency': 1, 'partition_grads_into_blocks': False, 'block_size': 256, 'epsilon': 1e-15, 'max_grad_norm': 1, 'train_batch_size': 128}
    model_size = '130m'
    target_chinchilla = 2
    my_suffix = 'i'
    template(model_size, target_chinchilla, 'soape', baseline_config, sweep_grids, random_suffix=my_suffix)
