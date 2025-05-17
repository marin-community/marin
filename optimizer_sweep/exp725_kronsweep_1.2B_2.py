from optimizer_sweep.template import template

if __name__ == '__main__':
    sweep_grids = {}
    baseline_config = {'learning_rate': 0.0003, 'weight_decay': 0.6, 'beta1': 0.95, 'preconditioner_lr': 0.2, 'preconditioner_init_scale': 1.0, 'max_grad_norm': 1.0, 'block_size': 256, 'preconditioner_update_probability': 0.1, 'update_prob_flat_start': 2000, 'warmup': 1000, 'min_lr_ratio': 0.0, 'train_batch_size': 256, 'normalize_grads': True, 'partition_grads_into_blocks': True}
    model_size = '1.2b'
    target_chinchilla = 2
    my_suffix = None
    template(model_size, target_chinchilla, 'kron', baseline_config, sweep_grids, random_suffix=my_suffix, tpu_type = 'v5litepod-256')
