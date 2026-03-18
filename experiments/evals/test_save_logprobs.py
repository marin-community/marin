# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Debug script for save_logprobs: runs Llama 3.2 1B on a toy dataset
to verify the end-to-end pipeline works.
"""

import json
import os
from dataclasses import dataclass

import fsspec
from levanter.compat.hf_checkpoints import HFCheckpointConverter

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer
from experiments.models import ModelConfig as HFModelConfig, download_model_step
from fray.cluster import ResourceConfig
from marin.evaluation.save_logprobs import default_save_logprobs
from marin.execution.executor import ExecutorStep, executor_main, output_path_of, this_output_path
from marin.processing.tokenize.data_configs import mixture_for_evaluation

TOY_DOCUMENTS = (
    "The quick brown fox jumps over the lazy dog. " * 50,
    "In mathematics, a prime number is a natural number greater than 1 that is not a product of two smaller natural numbers. " * 30,
    "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. " * 30,
    "The Earth orbits the Sun at an average distance of about 150 million kilometers. " * 40,
    "Water is composed of two hydrogen atoms and one oxygen atom bonded together. " * 40,
)


@dataclass(frozen=True)
class WriteToyDataConfig:
    output_path: str
    documents: tuple[str, ...]


def write_toy_data(config: WriteToyDataConfig):
    output_file = os.path.join(config.output_path, "data.jsonl.gz")
    with fsspec.open(output_file, "wt", compression="gzip") as f:
        for doc in config.documents:
            f.write(json.dumps({"text": doc}) + "\n")


toy_data_step = ExecutorStep(
    name="rohith-debug/toy_data",
    fn=write_toy_data,
    config=WriteToyDataConfig(
        output_path=this_output_path(),
        documents=TOY_DOCUMENTS,
    ),
)

tokenize_step = default_tokenize(
    name="rohith-debug/toy_tokenized",
    dataset=toy_data_step,
    tokenizer=llama3_tokenizer,
    is_validation=True,
)

eval_data = mixture_for_evaluation({"toy": tokenize_step})

model_info = HFModelConfig(hf_repo_id="meta-llama/Llama-3.2-1B", hf_revision="main")
model_instance = download_model_step(model_info)
model_identifier = f"{model_info.hf_repo_id}@{model_info.hf_revision}"
hf_model_config = HFCheckpointConverter.from_hf(model_identifier).config_from_hf_checkpoint(model_identifier)

save_logprobs_step = default_save_logprobs(
    checkpoint=output_path_of(model_instance),
    model=hf_model_config,
    data=eval_data,
    resource_config=ResourceConfig.with_tpu("v5p-8"),
    checkpoint_is_hf=True,
    per_device_batch_size=2,
    top_k=10,
    name="debug-llama-3.2-1b-save-logprobs",
)


if __name__ == "__main__":
    executor_main(
        steps=[save_logprobs_step],
        description="Debug save_logprobs with Llama 3.2 1B on toy data.",
    )
