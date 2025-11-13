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
This experiment is used to target the linguistic olympiad benchmark using MEDU,
a process that automatically looks for data that is similar to the targeted dataset.

The benchmark is from: https://huggingface.co/datasets/ambean/lingOly
"""

import copy

from marin.datashop.pipeline import CorpusContent
from marin.execution.executor import ExecutorStep, output_path_of, this_output_path
from marin.transform.lingoly.to_dolma import ConvertLingolyToDolmaConfig, convert_lingoly_to_dolma

from experiments.datashop.datashop_datasets import datashop_dclm_annotation_subset, datashop_dclm_pretraining_subset
from experiments.datashop.datashop_runner import DatashopRunner, DatashopRunnerConfig
from experiments.datashop.default_configs import default_medu_config_kwargs, default_text_generation_config_kwargs
from experiments.eval_datasets import lingoly
from experiments.models import get_model_local_path, llama_3_3_70b_instruct

lingoly_dolma = ExecutorStep(
    name="documents/lingoly/dolma",
    fn=convert_lingoly_to_dolma,
    config=ConvertLingolyToDolmaConfig(
        input_path=output_path_of(lingoly, "benchmark.zip"),
        output_path=this_output_path("/lingoly_preamble_context_questions_joined"),
    ),
)

# Customize the text generation config kwargs to allow for more tokens to be generated
# and a longer thought process.
medu_config_kwargs = copy.deepcopy(default_medu_config_kwargs)
medu_config_kwargs["generation_kwargs"]["max_tokens"] = 1024
medu_config_kwargs["generation_kwargs"]["truncate_prompt_tokens"] = 7168
text_generation_config_kwargs = default_text_generation_config_kwargs
text_generation_config_kwargs["generation_kwargs"] = medu_config_kwargs["generation_kwargs"]

lingoly_datashop_runner = DatashopRunner(
    DatashopRunnerConfig(
        experiment_name="lingoly",
        annotator_model_name=get_model_local_path(llama_3_3_70b_instruct),
        pretraining_data_path=datashop_dclm_pretraining_subset,
        annotator_data_path=datashop_dclm_annotation_subset,
        corpus_content_paths=[
            CorpusContent(
                content=lingoly_dolma,
                content_type="filepath",
                glob_pattern="**/*.jsonl",
            )
        ],
        medu_pipeline_config_kwargs=medu_config_kwargs,
        text_generation_inference_config_kwargs=text_generation_config_kwargs,
    )
)

if __name__ == "__main__":
    lingoly_datashop_runner.run_eval_cluster_steps()
