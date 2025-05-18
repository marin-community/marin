"""Synthetic data generation for the GSM8K dataset in the style of MIND.

Inspiration from the Olmo-2 paper where they utilize the MIND rewrite technique to generate
synthetic math datasets from existing datasets.
"""

from experiments.cooldown_quality import QualityAblationConfig, default_quality_ablation
from experiments.datashop.default_configs import default_engine_kwargs
from experiments.datashop.defaults import default_synthetic_data_generation
from experiments.defaults import default_tokenize
from experiments.exp923_medu_mmlu import mmlu_science_pipeline
from experiments.llama import llama3_tokenizer
from marin.execution.executor import executor_main, output_path_of

REPHRASE_THE_WEB_QA_TEMPLATE = """
{example}\n\nConvert the context above into a conversational format between a user and an assistant
with multiple tags of "Question:" followed by "Answer:"
"""

single_tpu_inference_engine_kwargs = {
    **default_engine_kwargs,
    "tensor_parallel_size": 1,
}
mmlu_science_qa = default_synthetic_data_generation(
    input_path=mmlu_science_pipeline.filtered_documents,
    model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
    document_name="medu-mmlu-science-llama8b-mind-qa",
    template=REPHRASE_THE_WEB_QA_TEMPLATE,
    input_filetype="jsonl.zst",
    prompt_column="text",
    engine_kwargs=single_tpu_inference_engine_kwargs,
    output_path="documents/medu-mmlu-science-llama8b-qa-whole",
)

mmlu_science_qa_whole_shard_tokenized = default_tokenize(
    name="medu-candidate-mmlu-science-llama-8b-qa",
    dataset=output_path_of(mmlu_science_qa),
    tokenizer=llama3_tokenizer,
)

mmlu_science_og_tokenized = default_tokenize(
    name="medu-candidate-mmlu-science",
    dataset=output_path_of(mmlu_science_pipeline.filtered_documents),
    tokenizer=llama3_tokenizer,
)

mmlu_science_qa_model = default_quality_ablation(
    candidate_tokenized=mmlu_science_qa_whole_shard_tokenized,
    config=QualityAblationConfig(
        mcq_component=mmlu_science_qa_whole_shard_tokenized,
        tpu_type="v6e-128",
        mcq_weight=0.30,
        candidate_weight=0.0,
        num_anneal_tokens=50_000_000_000,
        model_name_prefix="8b-dclm-70-qa-30-50b",
    ),
)

steps = [
    mmlu_science_qa_model,
]

if __name__ == "__main__":
    executor_main(steps)
