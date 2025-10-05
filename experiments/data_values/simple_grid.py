from experiments.data_values.train import DatagradsConfig, datagrads_train_step
from marin.execution.executor import ExecutorMainConfig, executor_main

force_run_failed = True
"""Set to True to force previously failed steps to run again."""

train_steps = [
    datagrads_train_step(
        DatagradsConfig(
            lr=lr,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            train_batch_size=bs,
            num_train_steps=num_train_steps,
            schedule_steps=24000,
            epsilon_root=epsilon_root,
            metagrad_segment_size=metagrad_segment_size,
            wandb_project_name="levanter",
            # Optional: override a few defaults baked from the prior YAML
            gcs_bucket="data-values-us-central-1",
            tpu_type="v5p-8",
            use_comma_main_mixture=False,
            lr_schedule="cosine", # cosine
            exp_name="smoothness-v4",
        )
    )
    for num_train_steps in [24000]
    for max_grad_norm in [1.0] #, 10., None]
    for lr in [1e-4, 2e-4, 4e-4, 6e-4, 8e-4, 1e-3]
    for bs in [1024]
    for weight_decay in [0.1] #, 0.2] #, 0.2]
    for epsilon_root in [1e-8] #, 1e-8, 1e-7]
    for metagrad_segment_size in [100]
]


def chunked(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


if __name__ == "__main__":
    for batch in chunked(train_steps, 8):
        # The draccus wrapper around executor_main allows us to pass in a config
        # object with our desired settings.
        executor_main(ExecutorMainConfig(force_run_failed=force_run_failed), batch)


"""
if __name__ == "__main__":
    executor_main(
        steps=train_steps,
        description="Metagrads test grid",
    )
"""