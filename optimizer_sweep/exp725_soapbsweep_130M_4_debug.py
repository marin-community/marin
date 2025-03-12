from optimizer_sweep.template import template

if __name__ == '__main__':
    sweep_grids = {}
    baseline_config = {'learning_rate': 0.016, 'weight_decay': 0.1, 'min_lr_ratio': 0, 'warmup': 1000, 'beta1': 0.95, 'beta2': 0.98, 'shampoo_beta': 0.98, 'precondition_frequency': 1, 'partition_grads_into_blocks': True, 'block_size': 256, 'epsilon': 1e-15, 'max_grad_norm': 1, 'train_batch_size': 256}
    model_size = '130m'
    target_chinchilla = 4
    my_suffix = 'a'
    template(model_size, target_chinchilla, 'soap', baseline_config, sweep_grids, random_suffix=my_suffix, force_run=True)
