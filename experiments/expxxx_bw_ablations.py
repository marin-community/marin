from marin.execution.executor import InputName, executor_main
from experiments.defaults import default_tokenize, default_train
from levanter.data.text import TextLmDatasetFormat
from experiments.llama import llama3_tokenizer, llama_3_2_1b
from experiments.simple_train_config import SimpleTrainConfig
from marin.resources import TpuPodConfig
from marin.processing.tokenize.data_configs import lm_mixture_data_config
from experiments.evals.task_configs import BEYOND_WEB_TASKS
from experiments.dclm.tokenize_dclm import dclm_components_llama3

synthetic_dclm_10B_subset = InputName.hardcoded(
    "gs://marin-us-central2/documents/synthetic-dclm-subset-10B-wrapqa-3b-29caa4"
)

# 10B tokens
dclm_10B_subset_text = default_tokenize(
    name="dclm_10B_subset_text/llama-3b/text",
    dataset=synthetic_dclm_10B_subset,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="text"),
)

# 10 B tokens
dclm_10B_subset_generated_text = default_tokenize(
    name="dclm_10B_subset/llama-3b/wrap-qa",
    dataset=synthetic_dclm_10B_subset,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="generated_text"),
)

_batch_size = 256
_seq_len = 4096
num_train_steps = int(20e9 // (_batch_size * _seq_len))

llama_3_2_1b_10B_text_10B_wrapqa = default_train(
    name="dclm_10B_subset/llama-3.2-1b-hf-dclm-50-wrapqa-50",
    tokenized=lm_mixture_data_config(
        components={
            "dclm": dclm_components_llama3["dclm_baseline"],
            "wrap-qa": dclm_10B_subset_generated_text,
        },
        weights={"dclm": 0.50, "wrap-qa": 0.50},
    ),
    model_config=llama_3_2_1b,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type="v4-64"),
        train_batch_size=_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=5e-4,
        weight_decay=1e-7,
        warmup=0.05,
        lr_schedule="cosine",
        steps_per_task_eval=1000,  # roughly 1B token
        initialize_from_hf="meta-llama/Llama-3.2-1B",
    ),
    tags=["llama-1b", "dclm-50", "wrapqa-50", "lr=5e-4", "wd=1e-7", "cosine", "10B"],
    eval_harness_tasks=BEYOND_WEB_TASKS,
)

llama_3_2_1b_20B_text = default_train(
    name="dclm_10B_subset/llama-3.2-1b-hf-dclm-50-text-50",
    tokenized=lm_mixture_data_config(
        components={
            "dclm": dclm_components_llama3["dclm_baseline"],
            "text": dclm_10B_subset_text,
        },
        weights={"dclm": 0.50, "text": 0.50},
    ),
    model_config=llama_3_2_1b,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type="v4-64"),
        train_batch_size=_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=5e-4,
        weight_decay=1e-7,
        warmup=0.05,
        lr_schedule="cosine",
        steps_per_task_eval=1000,
        initialize_from_hf="meta-llama/Llama-3.2-1B",
    ),
    tags=["llama-1b", "dclm-50", "text-50", "lr=5e-4", "wd=1e-7", "cosine", "20B"],
    eval_harness_tasks=BEYOND_WEB_TASKS,
)

llama_3_2_1b_10B_text_10B_wrapqa_lr_5e_5 = default_train(
    name="dclm_10B_subset/llama-3.2-1b-hf-dclm-50-wrapqa-50-lr=5e-5",
    tokenized=lm_mixture_data_config(
        components={
            "dclm": dclm_components_llama3["dclm_baseline"],
            "wrap-qa": dclm_10B_subset_generated_text,
        },
        weights={"dclm": 0.50, "wrap-qa": 0.50},
    ),
    model_config=llama_3_2_1b,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type="v4-64"),
        train_batch_size=_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=5e-5,
        weight_decay=1e-7,
        warmup=0.05,
        lr_schedule="cosine",
        steps_per_task_eval=1000,  # roughly 1B token
        initialize_from_hf="meta-llama/Llama-3.2-1B",
    ),
    tags=["llama-1b", "dclm-50", "wrapqa-50", "lr=5e-5", "wd=1e-7", "cosine", "10B"],
    eval_harness_tasks=BEYOND_WEB_TASKS,
)

llama_3_2_1b_20B_text_lr_5e_5 = default_train(
    name="dclm_10B_subset/llama-3.2-1b-hf-dclm-50-text-50-lr=5e-5",
    tokenized=lm_mixture_data_config(
        components={
            "dclm": dclm_components_llama3["dclm_baseline"],
            "text": dclm_10B_subset_text,
        },
        weights={"dclm": 0.50, "text": 0.50},
    ),
    model_config=llama_3_2_1b,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type="v4-64"),
        train_batch_size=_batch_size,
        num_train_steps=num_train_steps,
        learning_rate=5e-5,
        weight_decay=1e-7,
        warmup=0.05,
        lr_schedule="cosine",
        steps_per_task_eval=1000,
        initialize_from_hf="meta-llama/Llama-3.2-1B",
    ),
    tags=["llama-1b", "dclm-50", "text-50", "lr=5e-5", "wd=1e-7", "cosine", "20B"],
    eval_harness_tasks=BEYOND_WEB_TASKS,
)


if __name__ == "__main__":
    executor_main(
        steps=[
            # llama_3_2_1b_10B_text_10B_wrapqa,
            # llama_3_2_1b_20B_text,
            llama_3_2_1b_10B_text_10B_wrapqa_lr_5e_5,
            llama_3_2_1b_20B_text_lr_5e_5,
        ],
    )
