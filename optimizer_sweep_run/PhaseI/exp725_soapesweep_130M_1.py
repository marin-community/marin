from optimizer_sweep.template import template

if __name__ == '__main__':
    sweep_grids = {'learning_rate': [0.004, 0.008, 0.016], 'weight_decay': [0, 0.1, 0.2, 0.3], 'warmup': [500, 1000, 2000], 'beta1': [0.8, 0.9, 0.95], 'beta2': [0.9, 0.95, 0.98, 0.99], 'shampoo_beta': [0.9, 0.95, 0.98, 0.99], 'precondition_frequency': [10], 'block_size': [128, 256, 512], 'epsilon': [1e-20, 1e-15, 1e-10], 'train_batch_size': [128, 256, 512]}
    baseline_config = {'learning_rate': 0.016, 'weight_decay': 0.1, 'min_lr_ratio': 0, 'warmup': 1000, 'beta1': 0.95, 'beta2': 0.99, 'shampoo_beta': 0.95, 'precondition_frequency': 1, 'partition_grads_into_blocks': True, 'block_size': 256, 'epsilon': 1e-15, 'max_grad_norm': 1, 'train_batch_size': 128}
    model_size = '130m'
    target_chinchilla = 1
    my_suffix = None
    template(model_size, target_chinchilla, 'soape', baseline_config, sweep_grids, random_suffix=my_suffix)
