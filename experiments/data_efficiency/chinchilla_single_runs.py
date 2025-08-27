from experiments.data_efficiency.train import DataEfficiencyConfig, data_efficiency_train_step
from marin.execution.executor import executor_main
from levanter.optim.cautious import CautiousConfig
import numpy as np

# 800 steps w/ batch size 64 ==> 200M tokens


def get_optim_config_builder(batch_size: int, hidden_dim: int):
    def build_cautious_config():
        return CautiousConfig(
            learning_rate=0.33 * np.sqrt(batch_size) / hidden_dim,
            weight_decay=0.1,
            min_lr_ratio=0.0,
            warmup=0.1,
            beta1=0.95,
            beta2=0.98 ** (batch_size / 128),
            epsilon=1e-15,
            max_grad_norm=1,
            adamc_weight_decay=True,
            lr_schedule="linear",
            decay=0.2,
        )

    return build_cautious_config


train_steps = []
batch_size = 8
lr = 0.33

for model_name, hidden_dim in [
    ("150m4k", 512),
    ("300m4k", 768),
    ("600m4k", 1024),
    ("1_4b4k", 2048),
]:
    train_config = DataEfficiencyConfig(
        data_name="dclm",
        epochs=1,
        base_train_steps=800 * 64 / batch_size,
        train_batch_size=batch_size,
        lr_schedule="cosine",
        lr=0.33,
        weight_decay=0.1,
        wandb_project_name="suhas-data-efficiency",
        wandb_additional_tags=["chinchilla-single-runs"],
        model_name=model_name,
        nametag=f"-bs{batch_size}",
        tpu_type="v5p-8",
    )
    train_config.build_optimizer_config = get_optim_config_builder(batch_size, hidden_dim)
    train_steps.append(data_efficiency_train_step(train_config))

if __name__ == "__main__":
    executor_main(
        steps=train_steps,
        description="Data scaling baseline",
    )
