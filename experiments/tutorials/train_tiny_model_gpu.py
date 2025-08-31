"""
This is a tutorial on how to train a tiny model on a small dataset using a GPU.

This script demonstrates how to:
1. Train a tiny model on Wikitext-2 using a single GPU
2. Use GPU-specific training configuration
3. Run a quick training experiment

For CPU training, see train_tiny_model_cpu.py
"""

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama_nano
from experiments.marin_models import marin_tokenizer
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import GpuConfig

wikitext_hf_id = "dlwh/wikitext_2_detokenized"
wikitext_tokenized = default_tokenize(
    name="wikitext_2_detokenized",  # path to store the tokenized data inside PREFIX. `tokenized/` will be prepended
    dataset=wikitext_hf_id,  # dataset can be a step for downloaded data, or a raw HF id
    tokenizer=marin_tokenizer,  # the marin tokenizer is the llama3 tokenizer with a custom chat template
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
