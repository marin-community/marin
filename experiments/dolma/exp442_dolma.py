"""
Train Dolma/OLMo models.
https://github.com/stanford-crfm/marin/issues/442
"""

from experiments.defaults import default_train
from experiments.dolma.tokenize_dolma import DOLMA_OLMO_MIXTURE_WEIGHTS, tokenize_dolma_steps
from experiments.llama import llama_1_4b, llama_1_4b_train_config
from levanter.models.llama import LlamaConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config
from experiments.simple_train_config import SimpleTrainConfig

EXPERIMENT_TAG = ["442_dolma"]

dolma_llama3_tokenized = lm_mixture_data_config(
    components=tokenize_dolma_steps(),
    weights=DOLMA_OLMO_MIXTURE_WEIGHTS,
)

dolma_1_4b = default_train(
    name="dolma-1.4b",
    tokenized=dolma_llama3_tokenized,
    model_config=llama_1_4b,
    train_config=llama_1_4b_train_config,
    tags=EXPERIMENT_TAG,
)

## olmo replications

# (neox is close enough to olmo tokenizer)
dolma_neox_tokenized = lm_mixture_data_config(
    components=tokenize_dolma_steps(tokenizer="EleutherAI/gpt-neox-20b"),
    weights=DOLMA_OLMO_MIXTURE_WEIGHTS,
)

# https://arxiv.org/pdf/2402.00838 page 3 (Table 1
olmoish_1b_config = LlamaConfig(
    num_layers=16,
    hidden_dim=2048,
    intermediate_dim=7168,
    num_heads=16,
    num_kv_heads=16,
    seq_len=2048,
    tie_word_embeddings=True,
    # they don't learn the layer norm weights
    use_layer_norm_weight=False,
)

olmoish_1b_train_config = SimpleTrainConfig(
    learning_rate=4e-4,
    warmup=2000,
    weight_decay=0.1,
    train_batch_size=2048,
    num_train_steps=500000,  # 2048 * 2048 * 500000 = 2.1T tokens
    tpu_type="v5litepod-256"
)

olmoish_1b = default_train(
    name="olmoish-1b",
    tokenized=dolma_neox_tokenized,
    model_config=olmoish_1b_config,
    train_config=olmoish_1b_train_config,
    tags=EXPERIMENT_TAG + ["olmoish", "1b"],
)


if __name__ == "__main__":
    executor_main(steps=[olmoish_1b, dolma_1_4b])
