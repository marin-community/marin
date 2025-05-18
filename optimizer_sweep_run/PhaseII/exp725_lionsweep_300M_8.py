from optimizer_sweep.template import template

if __name__ == "__main__":
    sweep_grids = {"beta2": [0.95, 0.9, 0.98], "learning_rate": [0.0005, 0.001], "train_batch_size": [256, 128]}
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
    model_size = "300m"
    target_chinchilla = 8
    my_suffix = None
    template(model_size, target_chinchilla, "lion", baseline_config, sweep_grids, random_suffix=my_suffix)
