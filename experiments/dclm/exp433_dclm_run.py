from experiments.dclm.tokenize_dclm import DCLM_BASELINE_ONLY_MIXTURE, DCLM_MIXTURE_WEIGHTS
from experiments.defaults import SimpleTrainConfig, default_tokenize, default_train
from experiments.evals.evals import default_eval
from experiments.evals.task_configs import CORE_TASKS_PLUS_MMLU
from experiments.llama import LlamaConfig
from experiments.pretraining_datasets import dclm_baseline, proofpile_2, starcoderdata
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_mixture_data_config

gpt_neox_tokenizer = "EleutherAI/gpt-neox-20b"

### Define the datasets and tokenization configurations

dclm_baseline_tokenized_neox_wrong = default_tokenize(
    name="dclm_baseline",
    dataset=dclm_baseline,
    tokenizer=gpt_neox_tokenizer,
)


dclm_baseline_neox_tokenized = default_tokenize(
    name="dclm_baseline",
    dataset=dclm_baseline,
    tokenizer=gpt_neox_tokenizer,
)

starcoderdata_neox_tokenized = default_tokenize(
    name="starcoderdata", dataset=starcoderdata, tokenizer=gpt_neox_tokenizer, text_key="content"
)

proofpile_2_neox_tokenized = default_tokenize(
    name="proofpile_2",
    dataset=proofpile_2,
    tokenizer=gpt_neox_tokenizer,
)

DCLM_MIXTURE_COMPONENTS_NEOX_WRONG = {
    "dclm_baseline": dclm_baseline_tokenized_neox_wrong,
    "starcoderdata": starcoderdata_neox_tokenized,
    "proofpile_2": proofpile_2_neox_tokenized,
}

### Define the mixtures of datasets and their weights

# weights are from page 11 of https://arxiv.org/abs/2406.11794. Sampling is done uniformly over tokens.
# the 7B model was trained on 4.1T tokens, and the 1.4B model's data mixture weights are scaled accordingly.

# Define a mixture that has only dclm_baseline data; set the weights of the other datasets to 0

dclm_mixture_config_wrong = lm_mixture_data_config(
    components=DCLM_MIXTURE_COMPONENTS_NEOX_WRONG, weights=DCLM_MIXTURE_WEIGHTS
)
dclm_baseline_only_config_wrong = lm_mixture_data_config(
    components=DCLM_MIXTURE_COMPONENTS_NEOX_WRONG, weights=DCLM_BASELINE_ONLY_MIXTURE
)

### Define the model and training configurations

# hyperparams and numbers below are chosen to replicate the numbers in https://arxiv.org/abs/2406.11794.
# Table 1 (page 5) has # model parameters and # training tokens. Table 11, page 43 has the hyperparameters.
llama_1_4b_dclm = LlamaConfig(
    seq_len=2048,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=16,
    num_kv_heads=16,
    num_layers=24,
    use_flash_attention=True,
)

NUM_TRAIN_TOKENS = int(28.8e9)  # 28.8 billion tokens
NUM_TRAIN_STEPS = NUM_TRAIN_TOKENS // (256 * 2048)  # 256 is the batch size, 2048 is the sequence length

training_config = SimpleTrainConfig(
    tpu_type="v4-128",
    train_batch_size=256,
    num_train_steps=NUM_TRAIN_STEPS,
    learning_rate=3e-3,
    weight_decay=0.033,
    min_lr_ratio=0.1,
    warmup=5000,
    z_loss_weight=1e-4,
)

### Define the experiments- one for the mixture of datasets, and one for the baseline dataset only

EXPERIMENT_TAG_MIXTURE = ["433_dclm_1b_1x"]
EXPERIMENT_TAG_BASELINE_ONLY = ["433_dclm_baseline_1b_1x"]

dclm_mixture_model = default_train(
    name="dclm_1b_1x_replication_eval_check",
    tokenized=dclm_mixture_config_wrong,
    model_config=llama_1_4b_dclm,
    train_config=training_config,
    tags=EXPERIMENT_TAG_MIXTURE,
)

dclm_baseline_only_model = default_train(
    name="dclm_baseline_1b_1x_replication_nov12",
    tokenized=dclm_baseline_only_config_wrong,
    model_config=llama_1_4b_dclm,
    train_config=training_config,
    tags=EXPERIMENT_TAG_BASELINE_ONLY,
)

dclm_mixture_eval = default_eval(step=dclm_mixture_model, evals=CORE_TASKS_PLUS_MMLU)

dclm_baseline_only_eval = default_eval(step=dclm_baseline_only_model)

if __name__ == "__main__":
    executor_main(
        steps=[
            dclm_baseline_tokenized_neox_wrong,
            starcoderdata_neox_tokenized,
            proofpile_2_neox_tokenized,
            dclm_mixture_model,
            dclm_baseline_only_model,
            dclm_mixture_eval,
            dclm_baseline_only_eval,
        ]
    )
