from experiments.data_values.train import DatagradsConfig, datagrads_train_step
from experiments.llama import llama_150m, llama3_tokenizer
from marin.execution.executor import executor_main

train_steps = [
    datagrads_train_step(
        DatagradsConfig(
            lr=lr,
            weight_decay=weight_decay,
            #max_grad_norm=max_grad_norm,
            train_batch_size=bs,
            num_train_steps=num_train_steps,
            epsilon_root=epsilon_root,
            metagrad_segment_size=metagrad_segment_size,
            wandb_project_name="levanter",
            # Optional: override a few defaults baked from the prior YAML
            tpu_type="v5p-8",
            use_comma_main_mixture=False,
            exp_name="comma-test-v2",
            # Switch to Llama model and tokenizer
            #model_config=llama_150m,
            tokenizer=llama3_tokenizer,
        )
    )
    for num_train_steps in [20000]
    #for max_grad_norm in [1.0] #, 10., None]
    for lr in [3e-3] #, 2e-4, 4e-4]
    for bs in [512]
    for weight_decay in [0.1] #, 0.2]
    for epsilon_root in [1e-8] #, 1e-8, 1e-7]
    for metagrad_segment_size in [100]
]


def chunked(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

if __name__ == "__main__":
    for batch in chunked(train_steps, 8):
         executor_main(batch)

'''
if __name__ == "__main__":
    executor_main(
        steps=train_steps,
        description="Metagrads test grid",
    )
'''