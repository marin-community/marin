from optimizer_sweep.template import template

if __name__ == "__main__":
    sweep_grids = {
        "learning_rate": [0.008, 0.016, 0.032],
        "weight_decay": [0, 0.1, 0.2],
        "warmup": [500, 1000, 2000, 4000],
        "beta1": [0.8, 0.9, 0.95, 0.98],
        "beta2": [0.9, 0.95, 0.98],
        "epsilon": [1e-25, 1e-20, 1e-15, 1e-10],
        "max_grad_norm": [0, 1.0, 2.0],
        "train_batch_size": [128, 256, 512, 1024],
    }
    baseline_config = {
        "learning_rate": 0.008,
        "weight_decay": 0.1,
        "min_lr_ratio": 0,
        "warmup": 2000,
        "beta1": 0.95,
        "beta2": 0.98,
        "epsilon": 1e-10,
        "max_grad_norm": 1,
        "nesterov": True,
        "train_batch_size": 128,
    }
    model_size = "130m"
    target_chinchilla = 8
    my_suffix = "k"
    template(model_size, target_chinchilla, "nadamw", baseline_config, sweep_grids, random_suffix=my_suffix)
