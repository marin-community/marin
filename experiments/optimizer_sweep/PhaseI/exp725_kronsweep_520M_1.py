from marin.optimizer_sweep.template import template

if __name__ == "__main__":
    sweep_grids = {
        "learning_rate": [0.0005, 0.001, 0.002, 0.004, 0.008],
        "weight_decay": [0.0, 0.5, 0.7, 0.9],
        "beta1": [0.9, 0.95, 0.98],
        "preconditioner_lr": [0.1, 0.2],
        "max_grad_norm": [0.0, 1.0, 2.0],
        "normalize_grads": [False, True],
        "partition_grads_into_blocks": [True],
        "block_size": [128, 256, 512],
        "preconditioner_update_probability": [0.05, 0.1],
        "update_prob_flat_start": [500, 1000, 2000],
        "warmup": [1000, 2000, 4000],
        "train_batch_size": [128, 256, 512, 1024],
    }
    baseline_config = {
        "learning_rate": 0.001,
        "weight_decay": 0.7,
        "beta1": 0.95,
        "preconditioner_lr": 0.2,
        "preconditioner_init_scale": 1,
        "max_grad_norm": 1,
        "normalize_grads": True,
        "partition_grads_into_blocks": True,
        "block_size": 256,
        "preconditioner_update_probability": 0.1,
        "update_prob_flat_start": 2000,
        "warmup": 1000,
        "min_lr_ratio": 0,
        "train_batch_size": 128,
    }
    model_size = "520m"
    target_chinchilla = 1
    my_suffix = None
    template(model_size, target_chinchilla, "kron", baseline_config, sweep_grids, random_suffix=my_suffix)
