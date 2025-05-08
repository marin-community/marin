from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer, llama_nano
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import CpuOnlyConfig

# 1. Choose a dataset
tinystories_hf_id = "roneneldan/TinyStories"

# 2. Tokenize the dataset
tinystories_tokenized = default_tokenize(
    name=tinystories_hf_id,  # path to write tokenized files (tokenized/ will be prepended)
    dataset=tinystories_hf_id,  # HF dataset id
    tokenizer=llama3_tokenizer,
)

# 3. Define training configuration
nano_train_config = SimpleTrainConfig(
    # Here we define the hardware resources we need.
    resources=CpuOnlyConfig(num_cpus=1),
    train_batch_size=4,
    num_train_steps=100,
    # set hyperparameters
    learning_rate=6e-4,
    weight_decay=0.1,
    # keep eval quick for tutorial
    max_eval_batches=4,
)

# 4. Train the model

nano_tinystories_model = default_train(
    name="marin-nano-tinystories",
    # Steps can depend on other steps: nano_tinystories_model depends on tinystories_tokenized
    tokenized=tinystories_tokenized,
    model_config=llama_nano,
    train_config=nano_train_config,
    # wandb tags
    tags=["llama", "nano", "tinystories", "tutorial"],
    # We can run many [eval_harness](https://github.com/EleutherAI/lm-evaluation-harness) tasks in the loop
    # during training, but there's no point in running evals on such a tiny model
    eval_harness_tasks=[],
    # to keep tutorial fast, skip default validation sets
    use_default_validation=False,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            nano_tinystories_model,
        ]
    )
