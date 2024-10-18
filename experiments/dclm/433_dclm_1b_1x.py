import draccus
from defaults import default_tokenize
from pretraining_datasets import dclm_baseline, proofpile_2, the_stack_dedup

from marin.execution.executor import executor_main
from marin.training import TrainLmOnPodConfig

gpt_neox_tokenizer = "EleutherAI/gpt-neox-20b"

dclm_baseline_tokenized = default_tokenize(
    name="dclm_baseline",
    dataset=dclm_baseline,
    tokenizer=gpt_neox_tokenizer,
)

the_stack_dedup_tokenized = default_tokenize(
    name="the_stack_dedup",
    dataset=the_stack_dedup,
    tokenizer=gpt_neox_tokenizer,
)

proofpile_2_tokenized = default_tokenize(
    name="proofpile_2",
    dataset=proofpile_2,
    tokenizer=gpt_neox_tokenizer,
)

DCLM_FULL_COMPONENTS = {
    "dclm-baseline": dclm_baseline_tokenized,
    "the-stack-dedup": the_stack_dedup_tokenized,
    "proofpile-2": proofpile_2_tokenized,
}

# DCLM_MIXTURE_WEIGHTS = {
#     "dclm-baseline": 0.5,
#     "the-stack-dedup": 0.3,
#     "proofpile-2": 0.2,
# }

EXPERIMENT_TAG = ["433_dclm_1b_1x"]

# mixture_config = lm_mixture_data_config(
#     components=DCLM_FULL_COMPONENTS,
#     weights=,
# )

training_config = draccus.load(TrainLmOnPodConfig, open("config/training/dclm_1b_1x.yaml"))
# model_config = draccus.load(, open("config/training/llama_1_4b.yaml"))

# train_step = default_train(
#     name="dclm_1b_1x",
#     tokenized=mixture_config,
#     model_config=llama_1_4b,
#     train_config=training_config,
#     tags=EXPERIMENT_TAG,
# )

# eval_step = default_eval(
# )

if __name__ == "__main__":
    executor_main(
        steps=[
            dclm_baseline_tokenized,
            the_stack_dedup_tokenized,
            proofpile_2_tokenized,
            # train_step,
        ]
    )
