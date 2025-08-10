from marin.optimizer_sweep.template import template
from marin.resources import TpuPodConfig

if __name__ == "__main__":
    sweep_grids = {}
    baseline_config = {
        "learning_rate": 0.004,
        "weight_decay": 0.1,
        "min_lr_ratio": 0.0,
        "warmup": 1000,
        "beta1": 0.95,
        "beta2": 0.99,
        "shampoo_beta": 0.9,
        "precondition_frequency": 10,
        "block_size": 512,
        "epsilon": 1e-10,
        "max_grad_norm": 1,
        "train_batch_size": 256,
        "mu_dtype": "bfloat16",
        "precond_dtype": "bfloat16",
    }
    model_size = "1.2b"
    target_chinchilla = 4
    my_suffix = "weightf32"
    template(
        model_size,
        target_chinchilla,
        "soape",
        baseline_config,
        sweep_grids,
        random_suffix=my_suffix,
        tpu_type=TpuPodConfig(tpu_type="v4-256", slice_count=1),
    )
