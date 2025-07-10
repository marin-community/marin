from marin.optimizer_sweep.template import template

if __name__ == "__main__":
    sweep_grids = {}
    baseline_config = {
        "learning_rate": 0.004,
        "weight_decay": 0.1,
        "min_lr_ratio": 0.0,
        "warmup": 4000,
        "beta1": 0.98,
        "beta2": 0.98,
        "epsilon": 1e-10,
        "max_grad_norm": 1,
        "train_batch_size": 256,
        "nesterov": True,
    }
    model_size = "1.2b"
    target_chinchilla = 8
    my_suffix = None
    template(
        model_size,
        target_chinchilla,
        "nadamw",
        baseline_config,
        sweep_grids,
        random_suffix=my_suffix,
        tpu_type="v5litepod-256",
    )
