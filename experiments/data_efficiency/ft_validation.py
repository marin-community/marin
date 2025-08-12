from experiments.data_efficiency.train import DataEfficiencyConfig, data_efficiency_train_step
from marin.execution.executor import executor_main

# 800 steps ==> 200M tokens

train_steps = [
    data_efficiency_train_step(
        DataEfficiencyConfig(
            data_name="code",
            epochs=epochs,
            base_train_steps=base_train_steps,
            train_batch_size=64,
            lr_schedule="cosine",
            lr=lr,
            weight_decay=0.0,
            wandb_project_name="suhas-ft-validation",
            wandb_additional_tags=["starcoder-validation"],
            model_name=model_name,
            nametag=nametag,
            initialize_from_hf=initialize_from_hf,
        )
    )
    for base_train_steps in [200]
    for lr in [1e-3, 3e-4, 1e-4]
    for epochs in [2, 4, 8]
    for initialize_from_hf, nametag in [
        # (None, ""),
        # (
        #     "gs://marin-us-central2/checkpoints/data_efficiency/300m4k-209Mx16-dclm-cos-lr0.0030-wd1.60-bs64/hf/step-12799",
        #     "-0d",
        # ),
        # (
        #     "gs://marin-us-central2/checkpoints/data_efficiency/300m4k-209Mx16-dclm+sd0715^0.5-cos-lr0.0030-wd0.80-bs64/hf/step-25599",
        #     "-1d",
        # ),
        (
            "gs://marin-us-central2/checkpoints/data_efficiency/300m4k-209Mx16-dclm+ens8x0730^0.95-cos-lr0.0030-wd0.01-bs64/hf/step-255998",
            "-8d",
        )
    ]
    for model_name in ["300m4k"]
]

if __name__ == "__main__":
    executor_main(
        steps=train_steps,
        description="Data scaling baseline",
    )
