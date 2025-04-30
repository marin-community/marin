"""
This is a tutorial on how to train a tiny model on a small dataset.

This is designed to run on a single **GPU**
"""

from levanter.models.llama import LlamaConfig

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import GpuConfig

wikitext_hf_reference = "dlwh/wikitext_2_detokenized"
wikitext_tokenized = default_tokenize(
    name=f"tokenized/${wikitext_hf_reference}",
    dataset=wikitext_hf_reference,
    tokenizer=llama3_tokenizer,
)


# This is a tiny model. It will be very bad but it will train quickly
llama_nano_config = LlamaConfig(
    seq_len=512,
    hidden_dim=32,
    intermediate_dim=128,
    num_heads=2,
    num_kv_heads=2,
    num_layers=2,
)

# TODO: AutoResources

nano_train_config = SimpleTrainConfig(
    # Here we define the hardware resources we need.
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
    model_config=llama_nano_config,
    train_config=nano_train_config,
    tags=["llama", "nano", "wikitext", "tutorial"],
    # no point in running evals on such a tiny model
    eval_harness_tasks=[],
)


if __name__ == "__main__":
    executor_main(
        steps=[
            wikitext_tokenized,
            nano_wikitext_model,
        ]
    )
