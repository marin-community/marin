# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

synthetic_dclm_10B_1b_subset = InputName.hardcoded(
    "gs://marin-us-central2/documents/synthetic-dclm-subset-10B-wrapqa-1b-b6aef2"
)

synthetic_dclm_3B_high_quality_subset = InputName.hardcoded(
    "gs://marin-us-central2/documents/synthetic-dclm-subset-5B-chunked-wrapqa-hq-3b-906438"
)

synthetic_dclm_3B_low_quality_subset = InputName.hardcoded(
    "gs://marin-us-central2/documents/synthetic-dclm-subset-5B-chunked-wrapqa-lq-3b-d0de5e"
)

dclm_3B_low_quality_subset_non_chunked = InputName.hardcoded(
    "gs://marin-us-central2/documents/quality_filtering/synthetic-dclm-25B-subset-fineweb-edu-bottom-20-d424ed"
)

dclm_3B_high_quality_subset_non_chunked = InputName.hardcoded(
    "gs://marin-us-central2/documents/quality_filtering/synthetic-dclm-25B-subset-fineweb-edu-top-20-c22b1b"
)

synthetic_dclm_10B_8B_subset = InputName.hardcoded(
    "gs://marin-us-central2/documents/synthetic-dclm-subset-10B-wrapqa-8b-e0b426"
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

dclm_10B_8B_subset_generated_text = default_tokenize(
    name="dclm_10B_subset/llama-8b/wrap-qa",
    dataset=synthetic_dclm_10B_8B_subset,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="generated_text"),
)

dclm_10B_1b_subset_generated_text = default_tokenize(
    name="dclm_10B_subset/llama-1b/wrap-qa",
    dataset=synthetic_dclm_10B_1b_subset,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="generated_text"),
)

# 3B tokens
dclm_3B_high_quality_subset_text = default_tokenize(
    name="dclm_3B_high_quality_subset/llama-3b/text",
    dataset=synthetic_dclm_3B_high_quality_subset,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="text"),
)

dclm_3B_high_quality_subset_generated_text = default_tokenize(
    name="dclm_3B_high_quality_subset/llama-3b/wrap-qa",
    dataset=synthetic_dclm_3B_high_quality_subset,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="generated_text"),
)

dclm_3B_low_quality_subset_text = default_tokenize(
    name="dclm_3B_low_quality_subset/llama-3b/text",
    dataset=synthetic_dclm_3B_low_quality_subset,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="text"),
)

dclm_3B_low_quality_subset_generated_text = default_tokenize(
    name="dclm_3B_low_quality_subset/llama-3b/wrap-qa",
    dataset=synthetic_dclm_3B_low_quality_subset,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="generated_text"),
)

dclm_3B_low_quality_subset_text_non_chunked = default_tokenize(
    name="dclm_3B_low_quality_subset_non_chunked/llama-3b/text",
    dataset=dclm_3B_low_quality_subset_non_chunked,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="text"),
)


dclm_3B_high_quality_subset_text_non_chunked = default_tokenize(
    name="dclm_3B_high_quality_subset_non_chunked/llama-3b/wrap-qa",
    dataset=dclm_3B_high_quality_subset_non_chunked,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="text"),
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

# Experiment on 10B tokens re-written by Llama-1B model
llama_3_2_1b_10B_wrapqa_lr_5e_5 = default_train(
    name="dclm_10B_subset/llama-3.2-1b-hf-dclm-50-wrapqa-50-1b-lr=5e-5",
    tokenized=lm_mixture_data_config(
        components={
            "dclm": dclm_components_llama3["dclm_baseline"],
            "wrap-qa": dclm_10B_1b_subset_generated_text,
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
    tags=["llama-1b", "dclm-50", "wrapqa-50", "lr=5e-5", "wd=1e-7", "cosine", "10B", "M=1b"],
    eval_harness_tasks=BEYOND_WEB_TASKS,
)

# Experiment on 10B tokens re-written by Llama-8B model
llama_3_2_8b_10B_wrapqa_lr_5e_5 = default_train(
    name="dclm_10B_subset/llama-3.2-1b-hf-dclm-50-wrapqa-50-8b-lr=5e-5",
    tokenized=lm_mixture_data_config(
        components={
            "dclm": dclm_components_llama3["dclm_baseline"],
            "wrap-qa": dclm_10B_8B_subset_generated_text,
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
    tags=["llama-1b", "dclm-50", "wrapqa-50", "lr=5e-5", "wd=1e-7", "cosine", "10B", "M=8b"],
    eval_harness_tasks=BEYOND_WEB_TASKS,
)


# Experiment on High Quality vs. Low Quality 3B Tokens
# 6B tokens
num_train_steps = 6e9 // (_batch_size * _seq_len)
llama_3_2_1b_low_web_high_web = default_train(
    name="llama-3.2-1b-hf-lqweb-50-hqweb-50-6B",
    tokenized=lm_mixture_data_config(
        components={
            "low_quality_text": dclm_3B_low_quality_subset_text,
            "high_quality_text": dclm_3B_high_quality_subset_text,
        },
        weights={"low_quality_text": 0.50, "high_quality_text": 0.50},
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
    tags=["llama-1b", "lqweb-50", "hqweb-50", "lr=5e-5", "wd=1e-7", "cosine", "6B", "M=3b"],
    eval_harness_tasks=BEYOND_WEB_TASKS,
)

llama_3_2_1b_low_web_high_web_non_chunked = default_train(
    name="llama-3.2-1b-hf-lqweb-50-hqweb-50-6B-non-chunked",
    tokenized=lm_mixture_data_config(
        components={
            "low_quality_text": dclm_3B_low_quality_subset_text_non_chunked,
            "high_quality_text": dclm_3B_high_quality_subset_text_non_chunked,
        },
        weights={"low_quality_text": 0.50, "high_quality_text": 0.50},
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
    tags=["llama-1b", "lqweb-50", "hqweb-50", "lr=5e-5", "wd=1e-7", "cosine", "6B", "M=3b", "non-chunked"],
    eval_harness_tasks=BEYOND_WEB_TASKS,
)

llama_3_2_1b_low_synth_high_web = default_train(
    name="llama-3.2-1b-hf-lqsynth-50-hqweb-50-6B",
    tokenized=lm_mixture_data_config(
        components={
            "low_quality_text": dclm_3B_low_quality_subset_generated_text,
            "high_quality_text": dclm_3B_high_quality_subset_text,
        },
        weights={"low_quality_text": 0.50, "high_quality_text": 0.50},
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
    tags=["llama-1b", "lqsynth-50", "hqweb-50", "lr=5e-5", "wd=1e-7", "cosine", "6B", "M=3b"],
    eval_harness_tasks=BEYOND_WEB_TASKS,
)

llama_3_2_1b_low_synth_high_web_non_chunked = default_train(
    name="llama-3.2-1b-hf-lqsynth-50-hqweb-50-6B-non-chunked",
    tokenized=lm_mixture_data_config(
        components={
            "low_quality_text": dclm_3B_low_quality_subset_generated_text,
            "high_quality_text": dclm_3B_high_quality_subset_text_non_chunked,
        },
        weights={"low_quality_text": 0.50, "high_quality_text": 0.50},
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
    tags=["llama-1b", "lqsynth-50", "hqweb-50", "lr=5e-5", "wd=1e-7", "cosine", "6B", "M=3b", "non-chunked"],
    eval_harness_tasks=BEYOND_WEB_TASKS,
)

llama_3_2_1b_high_synth_high_web = default_train(
    name="llama-3.2-1b-hf-hgsynth-50-hqweb-50-6B",
    tokenized=lm_mixture_data_config(
        components={
            "high_quality_synth": dclm_3B_high_quality_subset_generated_text,
            "high_quality_text": dclm_3B_high_quality_subset_text,
        },
        weights={"high_quality_synth": 0.50, "high_quality_text": 0.50},
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
    tags=["llama-1b", "hgsynth-50", "hqweb-50", "lr=5e-5", "wd=1e-7", "cosine", "6B", "M=3b"],
    eval_harness_tasks=BEYOND_WEB_TASKS,
)

llama_3_2_1b_high_synth_high_web_non_chunked = default_train(
    name="llama-3.2-1b-hf-hgsynth-50-hqweb-50-6B-non-chunked",
    tokenized=lm_mixture_data_config(
        components={
            "high_quality_synth": dclm_3B_high_quality_subset_generated_text,
            "high_quality_text": dclm_3B_high_quality_subset_text_non_chunked,
        },
        weights={"high_quality_synth": 0.50, "high_quality_text": 0.50},
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
    tags=["llama-1b", "hgsynth-50", "hqweb-50", "lr=5e-5", "wd=1e-7", "cosine", "6B", "M=3b", "non-chunked"],
    eval_harness_tasks=BEYOND_WEB_TASKS,
)


if __name__ == "__main__":
    executor_main(
        steps=[
            # llama_3_2_1b_10B_text_10B_wrapqa,
            # llama_3_2_1b_20B_text,
            # llama_3_2_1b_10B_text_10B_wrapqa_lr_5e_5,
            # llama_3_2_1b_20B_text_lr_5e_5,
            # llama_3_2_1b_10B_wrapqa_lr_5e_5,
            # llama_3_2_1b_low_web_high_web,
            # llama_3_2_1b_low_synth_high_web,
            # llama_3_2_1b_high_synth_high_web,
            # llama_3_2_1b_low_web_high_web_non_chunked,
            # llama_3_2_1b_low_synth_high_web_non_chunked,
            # llama_3_2_1b_high_synth_high_web_non_chunked,
            llama_3_2_8b_10B_wrapqa_lr_5e_5,
        ],
    )
