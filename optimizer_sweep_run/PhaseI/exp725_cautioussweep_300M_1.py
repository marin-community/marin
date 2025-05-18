from optimizer_sweep.template import template

if __name__ == "__main__":
    sweep_grids = {
        "train_batch_size": [128, 256],
        "weight_decay": [0.1, 0.2, 0.0],
        "learning_rate": [0.004, 0.008],
        "warmup": [1000, 2000, 4000],
        "epsilon": [1e-10, 1e-15, 1e-25],
        "max_grad_norm": [0, 1, 2],
        "beta1": [0.9, 0.95, 0.98],
        "beta2": [0.9, 0.95, 0.98],
    }
    baseline_config = {
        "learning_rate": 0.008,
        "weight_decay": 0.1,
        "min_lr_ratio": 0,
        "warmup": 2000,
        "beta1": 0.98,
        "beta2": 0.98,
        "epsilon": 1e-25,
        "max_grad_norm": 1,
        "train_batch_size": 128,
    }
    model_size = "300m"
    target_chinchilla = 1
    my_suffix = None
    template(model_size, target_chinchilla, "cautious", baseline_config, sweep_grids, random_suffix=my_suffix)
