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

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer
from levanter.data.text import TextLmDatasetFormat
from marin.execution.executor import InputName, executor_main
from experiments.expxxx_regmix_synth import llama_130m, _make_train_config
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config
from levanter.trainer import BatchSchedule
from experiments.evals.task_configs import BEYOND_WEB_TASKS

# 2,670,773,940 tokens
synthetic_dclm_1b_regular_text = InputName.hardcoded(
    "gs://marin-us-central2/documents/datashop-datasets/synthetic-dclm-subset-2f6c8d"
)
synthetic_dclm_1b_regular_text_tokenized = default_tokenize(
    name="synthetic-dclm-subset-1b-data-1b-regular-text",
    dataset=synthetic_dclm_1b_regular_text,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="text"),
)
regular_text_tokens = 2_670_773_940

# 2,566,428,029 tokens
synthetic_dclm_1b_hs_teacher = InputName.hardcoded(
    "gs://marin-us-central2/documents/synthetic-dclm-subset-1b-data-1b-hs-teacher-b2557c"
)
synthetic_dclm_1b_hs_teacher_tokenized = default_tokenize(
    name="synthetic-dclm-subset-1b-data-1b-hs-teacher",
    dataset=synthetic_dclm_1b_hs_teacher,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="generated_text"),
)
hs_teacher_tokens = 2_566_428_029

# 2,295,019,340 tokens
synthetic_dclm_1b_food_enthusiast = InputName.hardcoded(
    "gs://marin-us-central2/documents/synthetic-dclm-subset-1b-data-1b-food-enthusiast-e9455f"
)
synthetic_dclm_1b_food_enthusiast_tokenized = default_tokenize(
    name="synthetic-dclm-subset-1b-data-1b-food-enthusiast",
    dataset=synthetic_dclm_1b_food_enthusiast,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="generated_text"),
)
food_enthusiast_tokens = 2_566_428_029

# 2,522,099,159 tokens
synthetic_dclm_1b_geologist = InputName.hardcoded(
    "gs://marin-us-central2/documents/synthetic-dclm-subset-1b-data-1b-geologist-c39515"
)
synthetic_dclm_1b_geologist_tokenized = default_tokenize(
    name="synthetic-dclm-subset-1b-data-1b-geologist",
    dataset=synthetic_dclm_1b_geologist,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="generated_text"),
)
geologist_tokens = 2_295_019_340

# 1,795,773,510 tokens
synthetic_dclm_1b_scholar = InputName.hardcoded(
    "gs://marin-us-central2/documents/synthetic-dclm-subset-1b-data-1b-manuscript-studies-scholar-b3ebe6"
)
synthetic_dclm_1b_scholar_tokenized = default_tokenize(
    name="synthetic-dclm-subset-1b-data-1b-manuscript-studies-scholar",
    dataset=synthetic_dclm_1b_scholar,
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(text_key="generated_text"),
)
scholar_tokens = 1_795_773_510

# fill in with actual number of tokens
llama_130m_train_config = _make_train_config(total_tokens=4 * regular_text_tokens)

llama_130m_persona_no_rewrite_4_epoch = default_train(
    name="llama-130m-persona-no-rewrite-4-epoch",
    tokenized=synthetic_dclm_1b_regular_text_tokenized,
    model_config=llama_130m,
    train_config=llama_130m_train_config,
    eval_harness_tasks=BEYOND_WEB_TASKS,
)

llama_130m_train_config = _make_train_config(total_tokens=4 * hs_teacher_tokens)
llama_130m_persona_one_rewrite_4_epoch = default_train(
    name="llama-130m-persona-one-rewrite-4-epoch",
    tokenized=synthetic_dclm_1b_hs_teacher_tokenized,
    model_config=llama_130m,
    train_config=llama_130m_train_config,
    eval_harness_tasks=BEYOND_WEB_TASKS,
)

_MIXTURE_BLOCK_SIZE = 2048
batch_schedule = BatchSchedule(llama_130m_train_config.train_batch_size)
num_first_stage_tokens = hs_teacher_tokens // (llama_130m.seq_len * llama_130m_train_config.train_batch_size)
target_offset = batch_schedule.global_data_offset_by_step(num_first_stage_tokens)
aligned_offset = (target_offset // _MIXTURE_BLOCK_SIZE) * _MIXTURE_BLOCK_SIZE
first_stage_step_offset = batch_schedule.find_step_containing_offset(aligned_offset)

num_second_stage_tokens = (hs_teacher_tokens + food_enthusiast_tokens) // (
    llama_130m.seq_len * llama_130m_train_config.train_batch_size
)
target_offset = batch_schedule.global_data_offset_by_step(num_second_stage_tokens)
aligned_offset = (target_offset // _MIXTURE_BLOCK_SIZE) * _MIXTURE_BLOCK_SIZE
second_stage_step_offset = batch_schedule.find_step_containing_offset(aligned_offset)

num_third_stage_tokens = (hs_teacher_tokens + food_enthusiast_tokens + geologist_tokens) // (
    llama_130m.seq_len * llama_130m_train_config.train_batch_size
)
target_offset = batch_schedule.global_data_offset_by_step(num_third_stage_tokens)
aligned_offset = (target_offset // _MIXTURE_BLOCK_SIZE) * _MIXTURE_BLOCK_SIZE
third_stage_step_offset = batch_schedule.find_step_containing_offset(aligned_offset)

llama_130m_train_config = _make_train_config(
    total_tokens=hs_teacher_tokens + food_enthusiast_tokens + geologist_tokens + scholar_tokens
)
llama_130m_persona_four_rewrites_one_epoch = default_train(
    name="llama-130m-persona-four-rewrites-one-epoch",
    tokenized=lm_varying_mixture_data_config(
        components={
            "hs_teacher": synthetic_dclm_1b_hs_teacher_tokenized,
            "food_enthusiast": synthetic_dclm_1b_food_enthusiast_tokenized,
            "geologist": synthetic_dclm_1b_geologist_tokenized,
            "scholar": synthetic_dclm_1b_scholar_tokenized,
        },
        weights_list=[
            (0, {"hs_teacher": 1.0}),
            (first_stage_step_offset, {"food_enthusiast": 1.0}),
            (second_stage_step_offset, {"geologist": 1.0}),
            (third_stage_step_offset, {"scholar": 1.0}),
        ],
    ),
    model_config=llama_130m,
    train_config=llama_130m_train_config,
    eval_harness_tasks=BEYOND_WEB_TASKS,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            llama_130m_persona_no_rewrite_4_epoch,
            llama_130m_persona_one_rewrite_4_epoch,
            llama_130m_persona_four_rewrites_one_epoch,
        ]
    )
