from optimizer_sweep.template import template

if __name__ == "__main__":
    sweep_grids = {"learning_rate": [0.004, 0.008], "train_batch_size": [256, 128]}
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
    model_size = "300m"
    target_chinchilla = 4
    my_suffix = None
    template(model_size, target_chinchilla, "nadamw", baseline_config, sweep_grids, random_suffix=my_suffix)
