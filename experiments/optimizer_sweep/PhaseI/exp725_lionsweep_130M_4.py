from marin.optimizer_sweep.template import template

if __name__ == "__main__":
    sweep_grids = {
        "learning_rate": [0.0005, 0.001, 0.002, 0.004, 0.008],
        "weight_decay": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        "warmup": [500, 1000, 2000, 4000],
        "beta1": [0.8, 0.9, 0.95, 0.98],
        "beta2": [0.9, 0.95, 0.98],
        "max_grad_norm": [0, 1.0, 2.0],
        "train_batch_size": [128, 256, 512, 1024],
    }
    baseline_config = {
        "learning_rate": 0.001,
        "weight_decay": 0.7,
        "min_lr_ratio": 0,
        "warmup": 2000,
        "beta1": 0.9,
        "beta2": 0.95,
        "max_grad_norm": 1,
        "train_batch_size": 128,
    }
    model_size = "130m"
    target_chinchilla = 4
    my_suffix = "p"
    template(model_size, target_chinchilla, "lion", baseline_config, sweep_grids, random_suffix=my_suffix)
