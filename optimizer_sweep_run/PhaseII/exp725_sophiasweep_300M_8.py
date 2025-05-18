from optimizer_sweep.template import template

if __name__ == "__main__":
    sweep_grids = {"weight_decay": [0, 0.1, 0.2, 0.3], "beta1": [0.95, 0.9], "beta2": [0.9, 0.95, 0.98, 0.99]}
    baseline_config = {
        "learning_rate": 0.004,
        "weight_decay": 0.1,
        "min_lr_ratio": 0,
        "warmup": 4000,
        "beta1": 0.9,
        "beta2": 0.9,
        "gamma": 0.0125,
        "epsilon": 1e-07,
        "max_grad_norm": 1,
        "train_batch_size": 128,
    }
    model_size = "300m"
    target_chinchilla = 8
    my_suffix = None
    template(model_size, target_chinchilla, "sophia", baseline_config, sweep_grids, random_suffix=my_suffix)
