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

# nodryrun
"""Synthetic data generation to rewrite high quality science data into QA format.

We start with an LLM generated prompt from the MEDU process to filter out high quality science data.
Then, we use a QA rephrasing prompt to convert the data into a QA format. We then anneal on this data
to generate a high quality dataset. We observe that this dataset gives us +4 points on MMLU STEM categories.
Issue: https://github.com/marin-community/marin/issues/923
"""

from experiments.cooldown_quality import QualityAblationConfig, default_quality_ablation
from experiments.datashop.default_configs import default_engine_kwargs
from experiments.datashop.defaults import default_consolidate, default_synthetic_data_generation
from experiments.defaults import default_tokenize
from experiments.evals.resource_configs import SINGLE_TPU_V6E_8
from experiments.exp923_medu_mmlu import mmlu_science_pipeline
from experiments.llama import llama3_tokenizer
from marin.execution.executor import executor_main, output_path_of
from marin.resources import TpuPodConfig

REPHRASE_THE_WEB_QA_TEMPLATE = """
{example}\n\nConvert the context above into a conversational format between a user and an assistant
with multiple tags of "Question:" followed by "Answer:"
"""

mmlu_science_pipeline.attributes = mmlu_science_pipeline.attributes.with_output_path(
    "attributes/quality_filtering/medu/medu-dclm-pretraining-subset-mmlu-science-91cfaa"
)
# NOTE(chris): Since the executor hash has changed, we need to re-run the consolidate step with the attributes
# from an overriden output path here.
mmlu_science_pipeline.filtered_documents = default_consolidate(
    mmlu_science_pipeline.attributes,
    mmlu_science_pipeline.config.pretraining_data_path,
    mmlu_science_pipeline.config.pretraining_data_path_name,
    mmlu_science_pipeline.config.experiment_name,
    mmlu_science_pipeline.config.filter_config_kwargs,
    mmlu_science_pipeline.config.consolidate_config_kwargs,
).with_output_path("documents/quality_filtering/medu/medu-dclm-pretraining-subset-mmlu-science-217322")

single_tpu_inference_engine_kwargs = {
    **default_engine_kwargs,
    "tensor_parallel_size": 1,
}
mmlu_science_qa = default_synthetic_data_generation(
    input_path=output_path_of(mmlu_science_pipeline.filtered_documents),
    model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
    data_generation_template=REPHRASE_THE_WEB_QA_TEMPLATE,
    input_filetype="jsonl.zst",
    prompt_column="text",
    engine_kwargs=single_tpu_inference_engine_kwargs,
    output_path="documents/medu-mmlu-science-llama8b-qa-whole",
    resource_config=SINGLE_TPU_V6E_8,
).with_output_path("documents/medu-mmlu-science-llama8b-qa-whole-1a419d")

mmlu_science_qa_whole_shard_tokenized = default_tokenize(
    name="medu-candidate-mmlu-science-llama-8b-qa",
    dataset=output_path_of(mmlu_science_qa),
    tokenizer=llama3_tokenizer,
).with_output_path("tokenized/medu-candidate-mmlu-science-llama-8b-qa-c92546")

mmlu_science_qa_model = default_quality_ablation(
    candidate_tokenized=mmlu_science_qa_whole_shard_tokenized,
    config=QualityAblationConfig(
        mcq_component=mmlu_science_qa_whole_shard_tokenized,
        resources=TpuPodConfig(tpu_type="v6e-128"),
        mcq_weight=0.30,
        candidate_weight=0.0,
        num_anneal_tokens=50_000_000_000,
        model_name_prefix="8b-dclm-70-qa-30-50b",
        permutation_type="linear",
    ),
).with_output_path("checkpoints/8b-dclm-70-qa-30-50b-tokenized-medu-candidate-mmlu-science-ll-qa-7d2930")

steps = [mmlu_science_qa_model]

if __name__ == "__main__":
    executor_main(steps)
