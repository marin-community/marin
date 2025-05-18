from optimizer_sweep.template import template

if __name__ == "__main__":
    sweep_grids = {
        "learning_rate": [0.002, 0.004, 0.008, 0.016, 0.032],
        "weight_decay": [0, 0.1, 0.2, 0.3],
        "warmup": [500, 1000, 2000, 4000],
        "beta1": [0.8, 0.9, 0.95, 0.98],
        "beta2": [0.9, 0.95, 0.98, 0.99, 0.995],
        "gamma": [0.00625, 0.0125, 0.025, 0.05],
        "epsilon": [1e-17, 1e-12, 1e-07],
        "train_batch_size": [128, 256, 512, 1024],
    }
    baseline_config = {
        "learning_rate": 0.004,
        "weight_decay": 0.2,
        "min_lr_ratio": 0,
        "warmup": 4000,
        "beta1": 0.95,
        "beta2": 0.99,
        "gamma": 0.0125,
        "epsilon": 1e-12,
        "max_grad_norm": 1,
        "train_batch_size": 128,
    }
    model_size = "130m"
    target_chinchilla = 4
    my_suffix = "m"
    template(model_size, target_chinchilla, "sophia", baseline_config, sweep_grids, random_suffix=my_suffix)
