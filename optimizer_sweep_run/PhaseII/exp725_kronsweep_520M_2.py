from optimizer_sweep.template import template

if __name__ == '__main__':
    sweep_grids = {'learning_rate': [0.0005, 0.001], 'train_batch_size': [256, 128]}
    baseline_config = {'learning_rate': 0.001, 'weight_decay': 0.5, 'beta1': 0.95, 'preconditioner_lr': 0.2, 'preconditioner_init_scale': 1, 'max_grad_norm': 1, 'normalize_grads': True, 'partition_grads_into_blocks': True, 'block_size': 256, 'preconditioner_update_probability': 0.1, 'update_prob_flat_start': 2000, 'warmup': 1000, 'min_lr_ratio': 0, 'train_batch_size': 128}
    model_size = '520m'
    target_chinchilla = 2
    my_suffix = None
    template(model_size, target_chinchilla, 'kron', baseline_config, sweep_grids, random_suffix=my_suffix)
