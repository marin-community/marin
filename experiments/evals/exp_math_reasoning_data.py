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

"""
#TBD: Math Scaling

Evaluates models log-likelihood on math reasoning traces.
"""

import os
import logging
from dataclasses import dataclass
from functools import lru_cache


from experiments.llama import llama3_tokenizer
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from marin.evaluation.log_probs import default_lm_log_probs
from marin.execution.executor import executor_main, ExecutorStep, output_path_of, this_output_path
from marin.processing.tokenize import TokenizeConfig
from marin.processing.tokenize.data_configs import mixture_for_evaluation, TokenizerStep
from experiments.defaults import default_tokenize
from experiments.models import ModelConfig as HFModelConfig, download_model_step
from levanter.data.text import TextLmDatasetFormat, ChatLmDatasetFormat
from marin.evaluation.visualize import VizLmConfig, visualize_lm_log_probs

from zephyr import Dataset, Backend
import json
import fsspec

logger = logging.getLogger(__name__)


DEFAULT_CHAT_TEMPLATE = "{{messages[0]['content']}}{% generation %} {{messages[1]['content']}}{% endgeneration %}"

@dataclass(frozen=True)
class ProcessDataConfig:
    input_path: str
    output_path: str
    name: str


def process_deepseek_r1(cfg: ProcessDataConfig):
    def processing_func(input_file_path: str):
        with fsspec.open(input_file_path, "rt", compression="infer") as f:
            i = 0
            for line in f:
                row = json.loads(line)
                processed_row = {
                    "text": row['prompt'] + "\n\n" + row['response'].lstrip(),
                    "id": i,
                    "source": "deepseek-r1",
                }
                i += 1
                yield processed_row
    
    pipeline = (
        Dataset.from_files(os.path.join(cfg.input_path, "**/*.jsonl.gz"))
        .flat_map(processing_func)
        .write_jsonl(os.path.join(cfg.output_path, "data-{shard:05d}-of-{total:05d}.jsonl.gz"))
    )
    list(Backend.execute(pipeline))


def process_tony_correct(cfg: ProcessDataConfig):
    def processing_func(input_file_path: str):
        with fsspec.open(input_file_path, "rt", compression="infer") as f:
            i = 0
            for line in f:
                row = json.loads(line)
                processed_row = {
                    "text": row['input'] + "\n\n" + row['picked'].lstrip(),
                    "id": i,
                    "source": "tony",
                }
                i += 1
                yield processed_row
    
    pipeline = (
        Dataset.from_files(os.path.join(cfg.input_path, "**/*.jsonl.gz"))
        .flat_map(processing_func)
        .write_jsonl(os.path.join(cfg.output_path, "data-{shard:05d}-of-{total:05d}.jsonl.gz"))
    )
    list(Backend.execute(pipeline))


def process_tony_incorrect(cfg: ProcessDataConfig):
    def processing_func(input_file_path: str):
        with fsspec.open(input_file_path, "rt", compression="infer") as f:
            i = 0
            for line in f:
                row = json.loads(line)
                processed_row = {
                    "text": row['input'] + "\n\n" + row['not_picked'].lstrip(),
                    "id": i,
                    "source": "tony",
                }
                i += 1
                yield processed_row
    
    pipeline = (
        Dataset.from_files(os.path.join(cfg.input_path, "**/*.jsonl.gz"))
        .flat_map(processing_func)
        .write_jsonl(os.path.join(cfg.output_path, "data-{shard:05d}-of-{total:05d}.jsonl.gz"))
    )
    list(Backend.execute(pipeline))


def process_tony_subquestions(cfg: ProcessDataConfig):
    def processing_func(input_file_path: str):
        with fsspec.open(input_file_path, "rt", compression="infer") as f:
            i = 0
            for line in f:
                row = json.loads(line)
                for subquestion in row['picked_subquestions']:
                    messages = [
                        {"role": "user", "content": subquestion["question"]},
                        {"role": "assistant", "content": subquestion["answer"]}
                    ]
                    
                    processed_row = {
                        "messages": messages,
                        "id": i,
                        "source": "tony",
                    }
                    i += 1
                    yield processed_row
    
    pipeline = (
        Dataset.from_files(os.path.join(cfg.input_path, "**/*.jsonl.gz"))
        .flat_map(processing_func)
        .write_jsonl(os.path.join(cfg.output_path, "data-{shard:05d}-of-{total:05d}.jsonl.gz"))
    )
    list(Backend.execute(pipeline))


def get_tokenized_data_steps(tokenizer: str = llama3_tokenizer) -> dict[str, TokenizerStep]:
    deepseek_r1_config = ProcessDataConfig(
        input_path="gs://marin-us-central1/raw/math500/deepseek-r1/",
        output_path=this_output_path(),
        name="deepseek-r1",
    )
    tony_correct_config = ProcessDataConfig(
        input_path="gs://marin-us-central1/raw/math500/tony/",
        output_path=this_output_path(),
        name="tony-correct",
    )
    tony_incorrect_config = ProcessDataConfig(
        input_path="gs://marin-us-central1/raw/math500/tony/",
        output_path=this_output_path(),
        name="tony-incorrect",
    )
    tony_subquestions_config = ProcessDataConfig(
        input_path="gs://marin-us-central1/raw/math500/subquestions/",
        output_path=this_output_path(),
        name="tony-subquestions",
    )
    dataset_configs = [
        deepseek_r1_config, 
        tony_correct_config, 
        tony_incorrect_config, 
        tony_subquestions_config
    ]
    tokenized_formats = [
        TextLmDatasetFormat(), 
        TextLmDatasetFormat(), 
        TextLmDatasetFormat(), 
        ChatLmDatasetFormat(chat_template=DEFAULT_CHAT_TEMPLATE)
    ]
    dataset_steps = [
        ExecutorStep(
            name=os.path.join("documents", "math_reasoning_scaling", deepseek_r1_config.name),
            fn=process_deepseek_r1,
            config=deepseek_r1_config,
        ),
        ExecutorStep(
            name=os.path.join("documents", "math_reasoning_scaling", tony_correct_config.name),
            fn=process_tony_correct,
            config=tony_correct_config,
        ),
        ExecutorStep(
            name=os.path.join("documents", "math_reasoning_scaling", tony_incorrect_config.name),
            fn=process_tony_incorrect,
            config=tony_incorrect_config,
        ),
        ExecutorStep(
            name=os.path.join("documents", "math_reasoning_scaling", tony_subquestions_config.name),
            fn=process_tony_subquestions,
            config=tony_subquestions_config,
        ),
    ]
    tokenized_data_steps: dict[str, ExecutorStep[TokenizeConfig]] = {}
    for dataset_step, dataset_config, tokenized_format in zip(
        dataset_steps, dataset_configs, tokenized_formats
    ):
        tokenized_data_steps[dataset_config.name] = default_tokenize(
            name=dataset_config.name,
            dataset=output_path_of(dataset_step),
            tokenizer=tokenizer,
            is_validation=True,
            format=tokenized_format,
        )
    return tokenized_data_steps