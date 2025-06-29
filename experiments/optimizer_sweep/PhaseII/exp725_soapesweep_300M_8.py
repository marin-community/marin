from marin.optimizer_sweep.template import template

if __name__ == "__main__":
    sweep_grids = {
        "learning_rate": [0.008, 0.004],
        "precondition_frequency": [10],
        "partition_grads_into_blocks": [True],
        "block_size": [256, 512, 128],
        "train_batch_size": [256, 128],
    }
    baseline_config = {
        "learning_rate": 0.008,
        "weight_decay": 0.1,
        "min_lr_ratio": 0,
        "warmup": 1000,
        "beta1": 0.95,
        "beta2": 0.99,
        "shampoo_beta": 0.9,
        "precondition_frequency": 10,
        "partition_grads_into_blocks": True,
        "block_size": 512,
        "epsilon": 1e-10,
        "max_grad_norm": 1,
        "train_batch_size": 128,
    }
    model_size = "300m"
    target_chinchilla = 8
    my_suffix = None
    template(model_size, target_chinchilla, "soape", baseline_config, sweep_grids, random_suffix=my_suffix)
