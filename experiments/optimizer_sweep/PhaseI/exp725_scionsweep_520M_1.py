from marin.optimizer_sweep.template import template

if __name__ == "__main__":
    sweep_grids = {
        "learning_rate": [0.008, 0.016, 0.032, 0.064, 0.128],
        "weight_decay": [0, 0.1, 0.2],
        "momentum": [0.8, 0.9, 0.95, 0.98],
        "beta1": [0.8, 0.9, 0.95, 0.98],
        "scion_epsilon": [1e-20, 1e-15, 1e-10, 1e-05],
        "max_grad_norm": [0, 1.0, 2.0],
        "decay": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "scion_to_signum_lr": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "train_batch_size": [128, 256, 512, 1024],
    }
    baseline_config = {
        "learning_rate": 0.008,
        "weight_decay": 0.1,
        "min_lr_ratio": 0,
        "warmup": 0,
        "momentum": 0.9,
        "beta1": 0.98,
        "scion_epsilon": 1e-05,
        "max_grad_norm": 2,
        "lr_schedule": "linear",
        "scion_to_signum_lr": 0.2,
        "decay": 0.8,
        "train_batch_size": 128,
    }
    model_size = "520m"
    target_chinchilla = 1
    my_suffix = None
    template(model_size, target_chinchilla, "scion", baseline_config, sweep_grids, random_suffix=my_suffix)
