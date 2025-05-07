from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer, llama_nano
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import CpuOnlyConfig

nano_train_config = SimpleTrainConfig(
    # Here we define the hardware resources we need.
    resources=CpuOnlyConfig(num_cpus=1),
    train_batch_size=32,
    num_train_steps=100,
    # set hyperparameters
    learning_rate=6e-4,
    weight_decay=0.1,
)

tinystories_hf_reference = "roneneldan/TinyStories"
tinystories_tokenized = default_tokenize(
    name=f"tokenized/{tinystories_hf_reference}",
    dataset=tinystories_hf_reference,
    tokenizer=llama3_tokenizer,
)

nano_tinystories_model = default_train(
    name="llama-nano-tinystories",
    # Steps can depend on other steps: tinystories_tokenized depends on tinystories
    tokenized=tinystories_tokenized,
    model_config=llama_nano,
    train_config=nano_train_config,
    # wandb tags
    tags=["llama", "nano", "tinystories", "tutorial"],
    # We can run many [eval_harness](https://github.com/EleutherAI/lm-evaluation-harness) tasks in the loop
    # during training, but there's no point in running evals on such a tiny model
    eval_harness_tasks=[],
    use_default_validation=False,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            tinystories_tokenized,
            nano_tinystories_model,
        ]
    )
