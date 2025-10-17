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

from experiments.datashop.datashop_datasets import dclm_baseline_global_shard_1_local_shard_1
from marin.classifiers.utils import create_dataset, CreateDatasetConfig
from marin.download.filesystem.transfer import TransferConfig, transfer_files
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, output_path_of
from marin.generation.chunk_utils import ChunkingConfig, chunk_with_config, ChunkStrategy
from marin.processing.classification.config.inference_config import InferenceConfig, RuntimeConfig, DatasetSchemaConfig
from marin.processing.classification.autoscaler import AutoscalingActorPoolConfig
from marin.processing.classification.consolidate import consolidate, ConsolidateConfig, FilterConfig
from marin.processing.classification.inference import run_inference

from experiments.pretraining_datasets import dclm_baseline, dolmino
from marin.transform.dolmino.filter_dolmino import FilterDolminoConfig, filter_dolmino

# Total files in this shard is 279. The total dataset is roughly 40B tokens. We want to sample
# around 20 files first which equates to roughly 7B tokens.
synthetic_dclm_annotation_subset = ExecutorStep(
    name="documents/datashop-datasets/synthetic-dclm-subset",
    fn=transfer_files,
    config=TransferConfig(
        input_path=dclm_baseline_global_shard_1_local_shard_1,
        output_path=this_output_path(),
        num_random_files=20,
    ),
)
dclm_data_chunked = ExecutorStep(
    name="documents/synthetic-dclm-subset-chunk-1024",
    fn=chunk_with_config,
    config=ChunkingConfig(
        input_path=synthetic_dclm_annotation_subset,
        output_path=this_output_path(),
        glob_pattern="**/*.jsonl.zst",
        chunk_strategy=ChunkStrategy.CHAR,
        chunk_size=1024,
    ),
)

generation_kwargs = {
    "max_tokens": 1024,
    "temperature": 0.7,
}
engine_kwargs = {
    "tensor_parallel_size": 1,
    "enforce_eager": False,
    "max_model_len": 8192,
}

WRAP_QA_REPHRASE_PROMPT = """
A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the questions.
USER: Convert the following paragraph into a conversational format with
multiple tags of "Question:" followed by "Answer:":

<example>
{example}
</example>

Just return the converted paragraph. Do not include any other text.
"""

# synthetic_dclm_data_chunked = default_synthetic_data_generation(
#     # datashop_runner.filtered_documents,
#     # output_path_of(datashop_runner.filtered_documents),
#     dclm_data_chunked,
#     "documents/synthetic-dclm-subset-chunk-qa-1024",
#     "/opt/gcsfuse_mount/models/meta-llama--Llama-3-2-3B-Instruct--0cb88a4",
#     WRAP_QA_REPHRASE_PROMPT,
#     "jsonl.zst",
#     "text",
#     checkpoint_id_column={"metadata": "WARC-Record-ID"},
#     engine_kwargs=engine_kwargs,
#     generation_kwargs=generation_kwargs,
#     resource_config=SINGLE_TPU_V5p_8,
#     chunk_strategy=ChunkStrategy.CHAR,
#     chunk_size=1024,
# )


synthetic_dclm_wrapqa = ExecutorStep(
    name="documents/synthetic-dclm-subset-chunk-qa-1024-wrapqa",
    fn=run_inference,
    config=InferenceConfig(
        input_path=dclm_data_chunked,
        output_path=this_output_path(),
        model_name="/opt/gcsfuse_mount/models/meta-llama--Llama-3-2-3B-Instruct--0cb88a4",
        model_type="vllm",
        attribute_name="wrap_qa_rephrase",
        filetype="jsonl.zst",
        batch_size=512,
        resume=True,
        runtime=RuntimeConfig(memory_limit_gb=16, resources={"TPU-v6e-8-head": 1}),
        classifier_kwargs={
            "template": WRAP_QA_REPHRASE_PROMPT,
            "score_extractor_fn": None,
            "engine_kwargs": engine_kwargs,
            "generation_kwargs": generation_kwargs,
            "save_original_generation": True,
        },
    ),
)

WRAP_MED_PROMPT = """
A chat between a curious user and an artificial intelligence assistant.
The assistant gives helpful, detailed, and polite answers to the questions.
USER: For the following paragraph give me a diverse paraphrase of the same
in high quality English language as in sentences on Wikipedia:

<example>
{example}
</example>

Just return the paraphrased paragraph. Do not include any other text.
"""

synthetic_dclm_wrapmed = ExecutorStep(
    name="documents/synthetic-dclm-subset-chunk-qa-1024-wrapmed",
    fn=run_inference,
    config=InferenceConfig(
        input_path=dclm_data_chunked,
        output_path=this_output_path(),
        model_name="/opt/gcsfuse_mount/models/meta-llama--Llama-3-2-3B-Instruct--0cb88a4",
        model_type="vllm",
        attribute_name="wrap_med_paraphrase",
        filetype="jsonl.zst",
        batch_size=2048,
        resume=True,
        runtime=RuntimeConfig(memory_limit_gb=16, resources={"TPU": 1}),
        classifier_kwargs={
            "template": WRAP_MED_PROMPT,
            "score_extractor_fn": None,
            "engine_kwargs": engine_kwargs,
            "generation_kwargs": generation_kwargs,
            "save_original_generation": True,
        },
    ),
)

DIVERSE_QA_PROMPT = """
Task: Read the text, ask questions and answer them.
Follow these instructions:
1. Ask diverse questions that require different cognitive skills or cover different aspects of the
text.
2. Ask questions in various forms such as:
- Yes/No questions that require determining whether a statement is true or false.
- Open-ended questions that begin with words like what, how, when, where, why and who.
- Multi-choice questions that offers two or more options to choose from. Include the options in the
question.
- Comparison questions that compare two quantities or objects and determine the relationship
between them.
- Reading comprehension questions that test the ability to understand and analyze the text.
- Problem-solving questions that test the ability to solve mathematical, physical, or logical
problems.
3. Focus on asking questions about factual information, important knowledge, or concrete details in
the text.
4. Write questions and answers using clear and concise language.
5. Use plain text. Do not use Markdown.
6. Each question and answer pair should be on a separate line. Tag the question with "Question:" and
the answer with "Answer:".
Text:
{example}
Task:
After reading the above text, ask up to 8 questions and provide the correct answers following the
instructions. Give your response in this format:
Here are the questions and answers based on the provided text:
- Question: [first question] Answer: [first answer]
- Question: [second question] Answer: [second answer]
....
"""

synthetic_dclm_diverse_qa = ExecutorStep(
    name="documents/synthetic-dclm-subset-chunk-qa-1024-nemo-qa",
    fn=run_inference,
    config=InferenceConfig(
        input_path=dclm_data_chunked,
        output_path=this_output_path(),
        model_name="/opt/gcsfuse_mount/models/meta-llama--Llama-3-2-3B-Instruct--0cb88a4",
        model_type="vllm",
        attribute_name="diverse_qa",
        filetype="jsonl.zst",
        batch_size=2048,
        resume=True,
        runtime=RuntimeConfig(memory_limit_gb=16, resources={"TPU": 1}),
        classifier_kwargs={
            "template": DIVERSE_QA_PROMPT,
            "score_extractor_fn": None,
            "engine_kwargs": engine_kwargs,
            "generation_kwargs": generation_kwargs,
            "save_original_generation": True,
        },
    ),
)

MCQ_STYLE_PROMPT = """
Convert the following paragraph into a multiple choice question with 4 options and the correct answer:
The format should be:
- Question: [first question]
A. [first answer]
B. [second answer]
C. [third answer]
D. [fourth answer]
Answer: [correct answer]
- Question: [second question]
A. [first answer]
B. [second answer]
C. [third answer]
D. [fourth answer]
Answer: [correct answer]
...

Text to convert:
<example>
{example}
</example>

Just return the multiple choice questions and answer. Do not include any other text.
"""

tpu_v4_engine_kwargs = {
    "tensor_parallel_size": 1,
    "enforce_eager": False,
    "max_model_len": 8192,
}
synthetic_dclm_mcq_style = ExecutorStep(
    name="documents/synthetic-dclm-subset-chunk-qa-1024-mcq",
    fn=run_inference,
    config=InferenceConfig(
        input_path=dclm_data_chunked,
        output_path=this_output_path(),
        model_name="/opt/gcsfuse_mount/models/meta-llama--Llama-3-2-3B-Instruct--0cb88a4",
        model_type="vllm",
        attribute_name="mcq_style",
        filetype="jsonl.zst",
        batch_size=2048,
        resume=True,
        runtime=RuntimeConfig(memory_limit_gb=16, resources={"TPU": 1}),
        classifier_kwargs={
            "template": MCQ_STYLE_PROMPT,
            "score_extractor_fn": None,
            "engine_kwargs": tpu_v4_engine_kwargs,
            "generation_kwargs": generation_kwargs,
            "save_original_generation": True,
        },
    ),
)

# tpu_v4_single_tpu_chip_engine_kwargs = {
#     "tensor_parallel_size": 1,
#     "enforce_eager": False,
#     "max_model_len": 4096,
# }
synthetic_dclm_wrapqa_1b = ExecutorStep(
    name="documents/synthetic-dclm-subset-chunk-qa-1024-wrapqa-1b",
    fn=run_inference,
    config=InferenceConfig(
        input_path=dclm_data_chunked,
        output_path=this_output_path(),
        model_name="/opt/gcsfuse_mount/models/meta-llama--Llama-3-2-1B-Instruct--c4219cc",
        model_type="vllm",
        attribute_name="wrap_qa_rephrase",
        filetype="jsonl.zst",
        batch_size=2048,
        resume=True,
        runtime=RuntimeConfig(memory_limit_gb=16, resources={"TPU": 1}),
        classifier_kwargs={
            "template": WRAP_QA_REPHRASE_PROMPT,
            "score_extractor_fn": None,
            "engine_kwargs": tpu_v4_engine_kwargs,
            "generation_kwargs": generation_kwargs,
            "save_original_generation": True,
        },
    ),
)

# Roughly 10B Tokens from entire DCLM dataset
synthetic_dclm_10B_subset = ExecutorStep(
    name="documents/dclm-baseline-10B-subset",
    fn=transfer_files,
    config=TransferConfig(
        input_path=dclm_baseline,
        output_path=this_output_path(),
        num_random_files=80,
    ),
)

synthetic_dclm_10B_subset_chunked = ExecutorStep(
    name="documents/dclm-baseline-10B-subset-chunked",
    fn=chunk_with_config,
    config=ChunkingConfig(
        input_path=synthetic_dclm_10B_subset,
        output_path=this_output_path(),
        glob_pattern="**/*.jsonl.zst",
        chunk_strategy=ChunkStrategy.PASSAGE,
        chunk_size=350,  # 350 tokens per passage
    ),
)

synthetic_dclm_10B_subset_wrapqa = ExecutorStep(
    name="documents/synthetic-dclm-subset-10B-wrapqa-3b",
    fn=run_inference,
    config=InferenceConfig(
        input_path=synthetic_dclm_10B_subset_chunked,
        output_path=this_output_path(),
        model_name="/opt/gcsfuse_mount/models/meta-llama--Llama-3-2-3B-Instruct--0cb88a4",
        model_type="vllm",
        attribute_name="wrap_qa_rephrase",
        filetype="jsonl.zst",
        batch_size=2048,
        resume=True,
        runtime=RuntimeConfig(memory_limit_gb=16, resources={"TPU": 1}),
        classifier_kwargs={
            "template": WRAP_QA_REPHRASE_PROMPT,
            "score_extractor_fn": None,
            "engine_kwargs": engine_kwargs,
            "generation_kwargs": generation_kwargs,
            "save_original_generation": True,
        },
    ),
)

synthetic_dclm_10B_subset_wrapqa_1b = ExecutorStep(
    name="documents/synthetic-dclm-subset-10B-wrapqa-1b",
    fn=run_inference,
    config=InferenceConfig(
        input_path=synthetic_dclm_10B_subset_chunked,
        output_path=this_output_path(),
        model_name="/opt/gcsfuse_mount/models/meta-llama--Llama-3-2-1B-Instruct--c4219cc",
        model_type="vllm",
        attribute_name="wrap_qa_rephrase",
        filetype="jsonl.zst",
        batch_size=2048,
        resume=True,
        runtime=RuntimeConfig(memory_limit_gb=16, resources={"TPU": 1}),
        classifier_kwargs={
            "template": WRAP_QA_REPHRASE_PROMPT,
            "score_extractor_fn": None,
            "engine_kwargs": engine_kwargs,
            "generation_kwargs": generation_kwargs,
            "save_original_generation": True,
        },
        # classifier_actor_options={"resources": {"TPU": 1}},
        # use_autoscaling_actor_pool=True,
    ),
)

synthetic_dclm_10B_subset_wrapqa_8b = ExecutorStep(
    name="documents/synthetic-dclm-subset-10B-wrapqa-8b",
    fn=run_inference,
    config=InferenceConfig(
        input_path=synthetic_dclm_10B_subset_chunked,
        output_path=this_output_path(),
        model_name="/opt/gcsfuse_mount/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f",
        model_type="vllm",
        attribute_name="wrap_qa_rephrase",
        filetype="jsonl.zst",
        batch_size=512,
        resume=True,
        runtime=RuntimeConfig(memory_limit_gb=16, resources={"TPU": 1}),
        classifier_kwargs={
            "template": WRAP_QA_REPHRASE_PROMPT,
            "score_extractor_fn": None,
            "engine_kwargs": engine_kwargs,
            "generation_kwargs": generation_kwargs,
            "save_original_generation": True,
        },
        # classifier_actor_options={"resources": {"TPU": 1}},
        # use_autoscaling_actor_pool=True,
    ),
)

# NOTE(chris): Does not work
# synthetic_dclm_10B_subset_wrapqa_1b_ray_data = default_synthetic_data_generation(
#     # datashop_runner.filtered_documents,
#     # output_path_of(datashop_runner.filtered_documents),
#     synthetic_dclm_10B_subset_chunked,
#     "documents/synthetic-dclm-subset-10B-wrapqa-1b-ray-data",
#     "/opt/gcsfuse_mount/models/meta-llama--Llama-3-2-1B-Instruct--c4219cc",
#     WRAP_QA_REPHRASE_PROMPT,
#     "jsonl.zst",
#     "text",
#     checkpoint_id_column={"metadata": "WARC-Record-ID"},
#     engine_kwargs=engine_kwargs,
#     generation_kwargs=generation_kwargs,
#     resource_config=SINGLE_TPU_V4_8,
# )


# BEGIN EXPERIMENT: Impact of high quality data rephrasing vs. low quality data rephrasing
# Roughly 25B tokens given the heuristic that each file is roughly 150M tokens
synthetic_dclm_25B_subset = ExecutorStep(
    name="documents/dclm-baseline-25B-subset",
    fn=transfer_files,
    config=TransferConfig(
        input_path=dclm_baseline,
        output_path=this_output_path(),
        num_random_files=170,
    ),
)

synthetic_dclm_25B_finewebedu_filtered = ExecutorStep(
    name="attributes/synthetic-dclm-25B-finewebedu",
    fn=run_inference,
    config=InferenceConfig(
        input_path=synthetic_dclm_25B_subset,
        output_path=this_output_path(),
        model_name="HuggingFaceFW/fineweb-edu-classifier",
        model_type="fineweb",
        attribute_name="fineweb-edu-quality",
        filetype="jsonl.zst",
        batch_size=512,
        resume=True,
        runtime=RuntimeConfig(memory_limit_gb=16, resources={"TPU": 1}),
    ),
)

synthetic_dclm_5B_high_quality_subset = ExecutorStep(
    name="documents/quality_filtering/synthetic-dclm-25B-subset-fineweb-edu-top-20",
    fn=consolidate,
    config=ConsolidateConfig(
        input_path=synthetic_dclm_25B_subset,
        output_path=this_output_path(),
        filters=[
            FilterConfig(
                attribute_path=output_path_of(synthetic_dclm_25B_finewebedu_filtered),
                name="fineweb-edu-quality",
                type="classify",
                label="score",
                keep_fraction=0.2,
            ),
        ],
        filetype="jsonl.zst",
        ray_memory_limit_gb=12,
    ),
    pip_dependency_groups=["ddsketch"],
)

synthetic_dclm_5B_low_quality_subset = ExecutorStep(
    name="documents/quality_filtering/synthetic-dclm-25B-subset-fineweb-edu-bottom-20",
    fn=consolidate,
    config=ConsolidateConfig(
        input_path=synthetic_dclm_25B_subset,
        output_path=this_output_path(),
        filters=[
            FilterConfig(
                attribute_path=output_path_of(synthetic_dclm_25B_finewebedu_filtered),
                name="fineweb-edu-quality",
                type="classify",
                label="score",
                keep_fraction=0.8,
                reverse=True,
            ),
        ],
        filetype="jsonl.zst",
        ray_memory_limit_gb=12,
    ),
    pip_dependency_groups=["ddsketch"],
)

synthetic_dclm_5B_low_quality_subset_chunked = ExecutorStep(
    name="documents/synthetic-dclm-subset-5B-low-quality-subset-chunked",
    fn=chunk_with_config,
    config=ChunkingConfig(
        input_path=synthetic_dclm_5B_low_quality_subset,
        output_path=this_output_path(),
        glob_pattern="**/*.jsonl.zst",
        chunk_strategy=ChunkStrategy.PASSAGE,
        chunk_size=350,  # 350 tokens per passage
    ),
)


synthetic_dclm_5B_high_quality_subset_chunked = ExecutorStep(
    name="documents/synthetic-dclm-subset-5B-high-quality-subset-chunked",
    fn=chunk_with_config,
    config=ChunkingConfig(
        input_path=synthetic_dclm_5B_high_quality_subset,
        output_path=this_output_path(),
        glob_pattern="**/*.jsonl.zst",
        chunk_strategy=ChunkStrategy.PASSAGE,
        chunk_size=350,  # 350 tokens per passage
    ),
)

synthetic_dclm_5B_high_quality_subset_wrapqa = ExecutorStep(
    name="documents/synthetic-dclm-subset-5B-chunked-wrapqa-hq-3b",
    fn=run_inference,
    config=InferenceConfig(
        input_path=synthetic_dclm_5B_high_quality_subset_chunked,
        output_path=this_output_path(),
        model_name="/opt/gcsfuse_mount/models/meta-llama--Llama-3-2-3B-Instruct--0cb88a4",
        model_type="vllm",
        attribute_name="wrap_qa_rephrase",
        filetype="jsonl.zst",
        batch_size=2048,
        resume=True,
        runtime=RuntimeConfig(memory_limit_gb=16, resources={"TPU": 1}),
        classifier_kwargs={
            "template": WRAP_QA_REPHRASE_PROMPT,
            "score_extractor_fn": None,
            "engine_kwargs": engine_kwargs,
            "generation_kwargs": generation_kwargs,
            "save_original_generation": True,
        },
    ),
)

synthetic_dclm_5B_low_quality_subset_wrapqa = ExecutorStep(
    name="documents/synthetic-dclm-subset-5B-chunked-wrapqa-lq-3b",
    fn=run_inference,
    config=InferenceConfig(
        input_path=synthetic_dclm_5B_low_quality_subset_chunked,
        output_path=this_output_path(),
        model_name="/opt/gcsfuse_mount/models/meta-llama--Llama-3-2-3B-Instruct--0cb88a4",
        model_type="vllm",
        attribute_name="wrap_qa_rephrase",
        filetype="jsonl.zst",
        batch_size=2048,
        resume=True,
        runtime=RuntimeConfig(memory_limit_gb=16, resources={"TPU": 1}),
        classifier_kwargs={
            "template": WRAP_QA_REPHRASE_PROMPT,
            "score_extractor_fn": None,
            "engine_kwargs": engine_kwargs,
            "generation_kwargs": generation_kwargs,
            "save_original_generation": True,
        },
    ),
)


## BEGIN PERSONA HUB STYLE DATA ABLATIONS
synthetic_dclm_annotation_subset_1b = ExecutorStep(
    name="documents/datashop-datasets/synthetic-dclm-subset",
    fn=transfer_files,
    config=TransferConfig(
        input_path=dclm_baseline_global_shard_1_local_shard_1,
        output_path=this_output_path(),
        num_random_files=20,
    ),
)

dclm_data_chunked_1b = ExecutorStep(
    name="documents/synthetic-dclm-subset-1b-chunk-350",
    fn=chunk_with_config,
    config=ChunkingConfig(
        input_path=synthetic_dclm_annotation_subset_1b,
        output_path=this_output_path(),
        glob_pattern="**/*.jsonl.zst",
        chunk_strategy=ChunkStrategy.PASSAGE,
        chunk_size=350,  # 350 tokens per passage
    ),
)

# From: https://huggingface.co/datasets/proj-persona/PersonaHub/viewer/knowledge?views%5B%5D=knowledge&row=2
personas = [
    (
        "hs-teacher",
        "A high school teacher of social studies preparing lessons on the \
            development of political parties in the United States.",
    ),
    (
        "food-enthusiast",
        "A food enthusiast who enjoys exploring regional French cuisine, with a \
            particular interest in cheeses from the Normandy and Picardy regions.",
    ),
    ("geologist", "A geologist studying sunken volcanoes and the geological history of Pacific islands."),
    (
        "manuscript-studies-scholar",
        "A manuscript studies scholar or paleographer with expertise in early medieval European \
        texts and handwriting. This expert would be intrigued by the ninth-century manuscript \
        that contains copies of Theuthild's letters.",
    ),
]

PERSONA_REWRITE_PROMPT = """
Assume you are the persona described as follows. You are tasked with rewriting a text document and
you will simulate the perspective and writing style of the persona given.

<persona>
{persona}
</persona>

<example>
{example}
</example>

Now rewrite the example above given the persona described. Just rewrite the text and don't output anything else.
"""

persona_steps = []
for persona_name, persona in personas:
    persona_rewrite_prompt = PERSONA_REWRITE_PROMPT.format(persona=persona, example="{example}")
    synthetic_dclm_annotation_subset_1b_persona = ExecutorStep(
        name=f"documents/synthetic-dclm-subset-1b-data-1b-{persona_name}",
        fn=run_inference,
        config=InferenceConfig(
            input_path=dclm_data_chunked_1b,
            output_path=this_output_path(),
            model_name="/opt/gcsfuse_mount/models/meta-llama--Llama-3-2-1B-Instruct--c4219cc",
            model_type="vllm",
            attribute_name="persona_rewrite",
            filetype="jsonl.zst",
            batch_size=2048,
            resume=True,
            runtime=RuntimeConfig(memory_limit_gb=16, resources={"TPU": 1}),
            classifier_kwargs={
                "template": persona_rewrite_prompt,
                "score_extractor_fn": None,
                "engine_kwargs": engine_kwargs,
                "generation_kwargs": generation_kwargs,
                "save_original_generation": True,
            },
        ),
    )
    persona_steps.append(synthetic_dclm_annotation_subset_1b_persona)


### Experiment with active reading
wiki_longer_than_256_chars = ExecutorStep(
    name="documents/wiki-longer-than-256-chars",
    fn=filter_dolmino,
    config=FilterDolminoConfig(
        input_path=dolmino,
        output_path=this_output_path(),
        split="wiki",
        min_length=256,
    ),
)

wikipedia_subset_5k = ExecutorStep(
    name="documents/wikipedia-subset-5k",
    fn=create_dataset,
    config=CreateDatasetConfig(
        input_doc_path=output_path_of(wiki_longer_than_256_chars),
        output_dataset_path=this_output_path(),
        max_sample_size=5_000,
        filetype="jsonl.gz",
        merge_dataset_shards=False,
        columns_to_keep=["text", "id"],
    ),
)


ACTIVE_READING_PROMPT = """
Please consider the following document.
What are some strategies specific to this document that I can use to help me learn and
remember all of the information it contains?
Use Markdown and prefix each strategy with ##.

<example>
{example}
</example>
"""

active_reading_steps = ExecutorStep(
    name="documents/wikipedia-subset-5k-active-reading-v2",
    fn=run_inference,
    config=InferenceConfig(
        input_path=wikipedia_subset_5k,
        output_path=this_output_path(),
        model_name="/opt/gcsfuse_mount/models/meta-llama--Llama-3-2-1B-Instruct--c4219cc",
        model_type="vllm",
        attribute_name="active_reading",
        filetype="jsonl.gz",
        batch_size=512,
        resume=True,
        runtime=RuntimeConfig(memory_limit_gb=16, resources={"TPU": 1}),
        classifier_kwargs={
            "template": ACTIVE_READING_PROMPT,
            "post_process_fn": "active_reading",
            "engine_kwargs": engine_kwargs,
            "generation_kwargs": generation_kwargs,
            "save_original_generation": True,
        },
        dataset_schema=DatasetSchemaConfig(
            input_columns=["text", "id"],
            output_columns=["id", "attributes", "generated_text", "text"],
        ),
    ),
)

ACTIVE_READING_REWRITE_PROMPT = """
Here's a learning strategy:
{attribute}

Apply this strategy to the following document:
<document>
{example}
</document>

Use this learning strategy to rewrite the document. Do not include any other text.
Do not just repeat the document -actively use the learning strategy to rewrite the document.
"""

active_reading_steps_1b_rewrite = ExecutorStep(
    name="documents/wikipedia-subset-5k-active-reading-1b-rewrite-v2",
    fn=run_inference,
    config=InferenceConfig(
        input_path=active_reading_steps,
        output_path=this_output_path(),
        model_name="/opt/gcsfuse_mount/models/meta-llama--Llama-3-2-1B-Instruct--c4219cc",
        model_type="vllm_with_docs_and_attributes",
        attribute_name="active_reading",
        filetype="jsonl.gz",
        batch_size=512,
        resume=True,
        runtime=RuntimeConfig(memory_limit_gb=16, resources={"TPU": 1}),
        classifier_kwargs={
            "template": ACTIVE_READING_REWRITE_PROMPT,
            "post_process_fn": None,
            "attribute_list_within_attributes_name": "strategies",
            "engine_kwargs": engine_kwargs,
            "generation_kwargs": generation_kwargs,
            "save_original_generation": True,
        },
        dataset_schema=DatasetSchemaConfig(
            input_columns=["text", "id", "attributes"],
            output_columns=["id", "attributes", "generated_text", "text"],
        ),
    ),
)

active_reading_steps_1b_rewrite_autoscale = ExecutorStep(
    name="documents/wikipedia-subset-5k-active-reading-1b-rewrite-v2-autoscale",
    fn=run_inference,
    config=InferenceConfig(
        input_path=active_reading_steps,
        output_path=this_output_path(),
        model_name="/opt/gcsfuse_mount/models/meta-llama--Llama-3-2-1B-Instruct--c4219cc",
        model_type="vllm_with_docs_and_attributes",
        attribute_name="active_reading",
        filetype="jsonl.gz",
        batch_size=512,
        resume=True,
        runtime=RuntimeConfig(memory_limit_gb=16),
        classifier_kwargs={
            "template": ACTIVE_READING_REWRITE_PROMPT,
            "post_process_fn": None,
            "attribute_list_within_attributes_name": "strategies",
            "engine_kwargs": engine_kwargs,
            "generation_kwargs": generation_kwargs,
            "save_original_generation": True,
        },
        dataset_schema=DatasetSchemaConfig(
            input_columns=["text", "id", "attributes"],
            output_columns=["id", "attributes", "generated_text", "text"],
        ),
        autoscaling_actor_pool_config=AutoscalingActorPoolConfig(
            min_actors=1,
            max_actors=32,
            scale_up_threshold=0.8,
            scale_down_threshold=0.2,
            scale_check_interval=1.0,
            actor_kwargs={},
            actor_options={"resources": {"TPU": 1}},
        ),
    ),
)

wikipedia_subset_1_5M = ExecutorStep(
    name="documents/wikipedia-subset-1p5m",
    fn=create_dataset,
    config=CreateDatasetConfig(
        input_doc_path=output_path_of(wiki_longer_than_256_chars),
        output_dataset_path=this_output_path(),
        max_sample_size=1_500_000,
        filetype="jsonl.gz",
        merge_dataset_shards=False,
        columns_to_keep=["text", "id"],
    ),
)

wikipedia_subset_1_5M_chunked = ExecutorStep(
    name="documents/wikipedia-subset-1p5m-chunked",
    fn=chunk_with_config,
    config=ChunkingConfig(
        input_path=wikipedia_subset_1_5M,
        output_path=this_output_path(),
        glob_pattern="**/*.jsonl.gz",
        chunk_strategy=ChunkStrategy.PASSAGE,
        chunk_size=350,  # 350 tokens per passage
    ),
)

active_reading_1_5M_with_attributes = ExecutorStep(
    name="documents/wikipedia-subset-1p5m-active-reading-v2",
    fn=run_inference,
    config=InferenceConfig(
        input_path=wikipedia_subset_1_5M_chunked,
        output_path=this_output_path(),
        model_name="/opt/gcsfuse_mount/models/meta-llama--Llama-3-2-1B-Instruct--c4219cc",
        model_type="vllm",
        attribute_name="active_reading",
        filetype="jsonl.gz",
        batch_size=512,
        resume=True,
        runtime=RuntimeConfig(memory_limit_gb=16, resources={"TPU": 1}),
        classifier_kwargs={
            "template": ACTIVE_READING_PROMPT,
            "post_process_fn": "active_reading",
            "engine_kwargs": engine_kwargs,
            "generation_kwargs": generation_kwargs,
            "save_original_generation": True,
        },
        dataset_schema=DatasetSchemaConfig(
            input_columns=["text", "id", "metadata"],
            output_columns=["id", "attributes", "generated_text", "text", "metadata"],
        ),
        autoscaling_actor_pool_config=AutoscalingActorPoolConfig(
            min_actors=1,
            max_actors=32,
            scale_up_threshold=0.8,
            scale_down_threshold=0.2,
            scale_check_interval=1.0,
            actor_kwargs={},
            actor_options={"resources": {"TPU": 1}},
        ),
    ),
)

active_reading_steps_1_5M_rewrite = ExecutorStep(
    name="documents/wikipedia-subset-1p5m-active-reading-1b-rewrite-v2",
    fn=run_inference,
    config=InferenceConfig(
        input_path=active_reading_1_5M_with_attributes,
        output_path=this_output_path(),
        model_name="/opt/gcsfuse_mount/models/meta-llama--Llama-3-2-1B-Instruct--c4219cc",
        model_type="vllm_with_docs_and_attributes",
        attribute_name="active_reading",
        filetype="jsonl.gz",
        batch_size=512,
        resume=True,
        runtime=RuntimeConfig(memory_limit_gb=16),
        classifier_kwargs={
            "template": ACTIVE_READING_REWRITE_PROMPT,
            "post_process_fn": None,
            "attribute_list_within_attributes_name": "strategies",
            "engine_kwargs": engine_kwargs,
            "generation_kwargs": generation_kwargs,
            "save_original_generation": True,
        },
        dataset_schema=DatasetSchemaConfig(
            input_columns=["text", "id", "attributes", "metadata"],
            output_columns=["id", "attributes", "generated_text", "text", "metadata"],
        ),
        autoscaling_actor_pool_config=AutoscalingActorPoolConfig(
            min_actors=1,
            max_actors=32,
            scale_up_threshold=0.8,
            scale_down_threshold=0.2,
            scale_check_interval=1.0,
            actor_kwargs={},
            actor_options={"resources": {"TPU": 1}},
        ),
    ),
)

active_reading_1_5M_qa = ExecutorStep(
    name="documents/wikipedia-subset-1p5m-qa",
    fn=run_inference,
    config=InferenceConfig(
        input_path=wikipedia_subset_1_5M_chunked,
        output_path=this_output_path(),
        model_name="/opt/gcsfuse_mount/models/meta-llama--Llama-3-2-1B-Instruct--c4219cc",
        model_type="vllm",
        attribute_name="wrap_qa_rephrase",
        filetype="jsonl.gz",
        batch_size=512,
        resume=True,
        runtime=RuntimeConfig(memory_limit_gb=16),
        classifier_kwargs={
            "template": WRAP_QA_REPHRASE_PROMPT,
            "post_process_fn": None,
            "engine_kwargs": engine_kwargs,
            "generation_kwargs": generation_kwargs,
            "save_original_generation": True,
        },
        dataset_schema=DatasetSchemaConfig(
            input_columns=["text", "id", "metadata"],
            output_columns=["id", "generated_text", "text", "metadata"],
        ),
        autoscaling_actor_pool_config=AutoscalingActorPoolConfig(
            min_actors=1,
            max_actors=32,
            scale_up_threshold=0.8,
            scale_down_threshold=0.2,
            scale_check_interval=1.0,
            actor_kwargs={},
            actor_options={"resources": {"TPU": 1}},
        ),
    ),
)

if __name__ == "__main__":
    # executor_main([synthetic_dclm_10B_subset_wrapqa])
    executor_main(
        [
            # synthetic_dclm_25B_finewebedu_filtered,
            # synthetic_dclm_5B_wrapqa,
            # synthetic_dclm_5B_low_quality_subset_wrapqa,
            # synthetic_dclm_5B_high_quality_subset_wrapqa,
            # synthetic_dclm_10B_subset_wrapqa_1b,
            # synthetic_dclm_10B_subset_wrapqa_8b,
            # *persona_steps,
            # active_reading_steps_1b_rewrite_autoscale,
            # wikipedia_subset_1_5M,
            # active_reading_1_5M_with_attributes,
            # active_reading_steps_1_5M_rewrite,
            active_reading_1_5M_qa,
        ]
    )
