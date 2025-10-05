from experiments.data_values.train import DatagradsConfig, datagrads_train_step
from marin.execution.executor import ExecutorMainConfig, executor_main

force_run_failed = True
"""Set to True to force previously failed steps to run again."""

initialize_from_checkpoint_path = (
    "gs://marin-us-central1/checkpoints/data_grads/smoothness-v4-bs1024-s24000-lr1.0e-03-wd0.10-gc1.0-er1.00e-08/checkpoints/step-23999"
)


def zero_out_single_dataset(mixture_train_weights, dataset_name):
    weights_ = mixture_train_weights.copy()
    weights_[dataset_name] = 0.0
    return weights_


train_steps = [
    datagrads_train_step(
        DatagradsConfig(
            lr=lr,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            train_batch_size=bs,
            num_train_steps=num_train_steps,
            schedule_steps=num_train_steps,
            epsilon_root=epsilon_root,
            metagrad_segment_size=metagrad_segment_size,
            wandb_project_name="levanter",
            # Optional: override a few defaults baked from the prior YAML
            gcs_bucket="data-values-us-central-1",
            tpu_type="v5p-8",
            use_comma_main_mixture=False,
            lr_schedule="linear", # cosine
            exp_name="rand-cfx-wikitext-v2-debug2-{}".format(cfx_seed),
            #zero_loss_datasets=[cfx_dataset_name],
            cfx_seed=cfx_seed,
            train_only=True,
            initialize_from_checkpoint_path=initialize_from_checkpoint_path,
            # Minimal: train on a single HF dataset by id
            hf_dataset_id="dlwh/wikitext_103_detokenized",
            #eval_task_spec=eval_task_spec,
            #eval_max_eval_length=eval_max_eval_length,
            #eval_max_examples=eval_max_examples,
            #eval_harness_steps=10000,
        )
    )
    for cfx_seed in range(1)
    for num_train_steps in [100]
    for max_grad_norm in [1.0] #, 10., None]
    for lr in [1e-4]
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
    for batch in chunked(train_steps, 20):
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