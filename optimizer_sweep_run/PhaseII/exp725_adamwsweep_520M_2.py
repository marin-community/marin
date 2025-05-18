from optimizer_sweep.template import template

if __name__ == "__main__":
    sweep_grids = {"learning_rate": [0.008, 0.004], "weight_decay": [0.1, 0.2], "train_batch_size": [128, 256]}
    baseline_config = {
        "learning_rate": 0.004,
        "weight_decay": 0.2,
        "min_lr_ratio": 0,
        "warmup": 1000,
        "beta1": 0.9,
        "beta2": 0.98,
        "epsilon": 1e-10,
        "max_grad_norm": 1,
        "nesterov": False,
        "train_batch_size": 256,
    }
    model_size = "520m"
    target_chinchilla = 2
    my_suffix = None
    template(model_size, target_chinchilla, "adamw", baseline_config, sweep_grids, random_suffix=my_suffix)
