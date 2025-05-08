"""
This is a tutorial on how to train a tiny model on a small dataset.

This is designed to run on a single **GPU**
"""

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer, llama_nano
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import GpuConfig

wikitext_hf_id = "dlwh/wikitext_2_detokenized"
wikitext_tokenized = default_tokenize(
    name=wikitext_hf_id,
    dataset=wikitext_hf_id,
    tokenizer=llama3_tokenizer,
)


nano_train_config = SimpleTrainConfig(
    # Here we define the hardware resources we need.
    # TODO: AutoResources
    resources=GpuConfig(gpu_count=1),
    train_batch_size=32,
    num_train_steps=100,
    learning_rate=6e-4,
    weight_decay=0.1,
)

nano_wikitext_model = default_train(
    name="llama-nano-wikitext",
    # Steps can depend on other steps: wikitext_tokenized depends on wikitext
    tokenized=wikitext_tokenized,
    model_config=llama_nano,
    train_config=nano_train_config,
    tags=["llama", "nano", "wikitext", "tutorial"],
    # no point in running evals on such a tiny model
    eval_harness_tasks=[],
    use_default_validation=False,
)


if __name__ == "__main__":
    executor_main(
        steps=[
            wikitext_tokenized,
            nano_wikitext_model,
        ]
    )
