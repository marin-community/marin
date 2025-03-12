from optimizer_sweep.template import template



if __name__ == "__main__":
    sweep_grids = {
        "learning_rate": [5e-4, 1e-3, 2e-3, 4e-3, 8e-3],
        "weight_decay": [0.0, 0.5, 0.7, 0.9],
        "beta1": [0.9, 0.95, 0.98],
        "preconditioner_lr": [1e-1, 2e-1],
        "max_grad_norm": [0.0, 1.0, 2.0],
        "normalize_grads": [False, True],
        "partition_grads_into_blocks": [False, True],
        "block_size": [128, 256, 512],
        "preconditioner_update_probability": [0.05, 0.1],
        "update_prob_flat_start": [500, 1000, 2000],
        "warmup": [1000, 2000, 4000],
        'train_batch_size': [128, 256, 512, 1024]
    }

    baseline_config = {
        "learning_rate": 2e-3,
        "weight_decay": 0.9,
        "beta1": 0.98,
        "preconditioner_lr": 1e-1,
        "preconditioner_init_scale": 1.0,
        "max_grad_norm": 1.0,
        "normalize_grads": True,
        "partition_grads_into_blocks": True,
        "block_size": 256,
        "preconditioner_update_probability": 0.05,
        "update_prob_flat_start": 500,
        "warmup": 2000,
        "min_lr_ratio": 0,
        'train_batch_size': 128
    }
    model_size = "130m"
    target_chinchilla = 1 
    my_suffix = None 
    template(model_size, target_chinchilla, 'kron', baseline_config, sweep_grids, random_suffix=my_suffix)     

                    
        