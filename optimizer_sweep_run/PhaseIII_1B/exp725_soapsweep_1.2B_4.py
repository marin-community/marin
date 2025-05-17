from optimizer_sweep.template import template

if __name__ == '__main__':
    sweep_grids = {}
    baseline_config = {'learning_rate': 0.004, 'weight_decay': 0.1, 'min_lr_ratio': 0.0, 'warmup': 1000, 'beta1': 0.95, 'beta2': 0.99, 'shampoo_beta': 0.95, 'precondition_frequency': 10, 'block_size': 256, 'epsilon': 1e-10, 'max_grad_norm': 1, 'train_batch_size': 512, 'partition_grads_into_blocks': True}
    model_size = '1.2b'
    target_chinchilla = 4
    my_suffix = None
    template(model_size, target_chinchilla, 'soape', baseline_config, sweep_grids, random_suffix=my_suffix, tpu_type = 'v5litepod-512')
