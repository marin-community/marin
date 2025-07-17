from marin.optimizer_sweep.template import template

if __name__ == "__main__":
    sweep_grids = {
        "learning_rate": [0.008, 0.016, 0.032, 0.064, 0.128],
        "weight_decay": [0, 0.1, 0.2],
        "momentum": [0.8, 0.9, 0.95, 0.98],
        "beta1": [0.8, 0.9, 0.95, 0.98],
        "beta2": [0.9, 0.95, 0.98],
        "epsilon": [1e-25, 1e-20, 1e-15, 1e-10],
        "muon_epsilon": [1e-25, 1e-20, 1e-15, 1e-10, 1e-05],
        "max_grad_norm": [0, 1.0, 2.0],
        "decay": [0.2, 0.4, 0.6, 0.8, 1.0],
        "muon_to_adam_lr": [0.1, 0.2, 0.3],
        "train_batch_size": [128, 256, 512, 1024],
    }
    baseline_config = {
        "learning_rate": 0.008,
        "weight_decay": 0.1,
        "min_lr_ratio": 0,
        "warmup": 0,
        "momentum": 0.98,
        "beta1": 0.8,
        "beta2": 0.98,
        "epsilon": 1e-15,
        "muon_epsilon": 1e-05,
        "max_grad_norm": 1,
        "lr_schedule": "linear",
        "muon_to_adam_lr": 0.3,
        "decay": 0.8,
        "train_batch_size": 128,
    }
    model_size = "130m"
    target_chinchilla = 8
    my_suffix = None
    template(model_size, target_chinchilla, "muon", baseline_config, sweep_grids, random_suffix=my_suffix)
