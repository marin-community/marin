"""Synthetic data generation for the GSM8K dataset in the style of MIND.

Inspiration from the Olmo-2 paper where they utilize the MIND rewrite technique to generate
synthetic math datasets from existing datasets.
"""

from transformers import AutoTokenizer

from experiments.models import get_model_local_path, llama_3_3_70b_instruct
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.generation.inference import TextGenerationInferenceConfig, run_inference

# huggingface_dataset_id = "openai/gsm8k"
model_name = "meta-llama/Llama-3.3-70B-Instruct"
tensor_parallel_size = 8

# dataset_name = get_directory_friendly_name(huggingface_dataset_id)
# gsm8k = ExecutorStep(
#     name=f"raw/{dataset_name}",
#     fn=download_hf,
#     config=DownloadConfig(
#         hf_dataset_id=huggingface_dataset_id,
#         revision=versioned("e53f048"),
#         gcs_output_path=this_output_path(),
#         wait_for_completion=True,
#     ),
# ).cd("main/train-00000-of-00001.parquet")

ECONOMIC_TEST_DESCRIPTION = """
## Test Type
This test appears to be a comprehensive assessment of economic knowledge, covering microeconomic and macroeconomic concepts, as well as statistical analysis and modeling, likely from an advanced academic or professional setting, such as a college-level economics course or a certification exam for economists.

## Required Languages, Skills, and Knowledge
The language model would need to understand economic terminology, concepts, and theories, including market structures, government intervention, and industry characteristics. It should possess strong problem-solving skills, with the ability to apply economic and statistical concepts to real-world scenarios, and be proficient in English, including grammar, syntax, and vocabulary related to economics and statistics. Additionally, it should be familiar with statistical analysis, including regression, hypothesis testing, and time series modeling.

## Ideal Training Data
The ideal training data for this evaluation would consist of a diverse and comprehensive corpus of texts, including economics and statistics textbooks, academic articles, research papers, practice problems, exams, and study guides. It should also include news articles, online resources, and policy documents discussing economic issues and trends, as well as diverse data sets, such as economic indicators, financial data, and statistical models, to help the model generalize its knowledge and apply it to different scenarios.
"""  # noqa: E501


def get_medu_prompt(test_description: str) -> str:
    """Returns the MEDU prompt with the test description filled in.

    Args:
        test_description: The test description to use in the prompt

    Returns:
        The MEDU prompt with the test description filled in, requiring only example to be formatted
    """
    return """
        The following document is being considered as training data for a
        Large Language Model.
        Provide a concise description of the document and an assessment of
        the quality of the text or code in the document.
        Key Attributes to Mention
        - Languages contained in the document
        - The coherence of the document
        - The skills the document demonstrates
        - The topics the document contains facts and information about
        Document:
        '''
        {example}
        '''
        Based on your previous reasoning, give me a concrete decision
        about the utility of the document as training data for the
        following benchmark.
        {test_description}
        Output your decision about the utility of the data as one of the
        following single words Great/Good/Okay/Poor/Useless without
        formatting.
        """.format(
        test_description=test_description, example="{example}"
    )


# wikipedia = "gs://marin-us-central2/documents/wiki-longer-than-256-chars-329843/data/wiki/wiki-0000/000_00000.jsonl.gz"
fineweb_dump = (
    "gs://marin-us-central2/documents/fineweb-small-resiliparse-preserve-formatting-e8c6ec/md/CC-MAIN-2024-18/000_00000"
)


tokenizer = AutoTokenizer.from_pretrained(model_name)
medu_inference = ExecutorStep(
    name="documents/medu-economics-score-llama-70b/fineweb-test",
    fn=run_inference,
    config=TextGenerationInferenceConfig(
        input_path=fineweb_dump,
        output_path=this_output_path(),
        model_name=get_model_local_path(llama_3_3_70b_instruct),
        engine_kwargs={
            "max_model_len": 8192,
            "enforce_eager": False,
            "tensor_parallel_size": tensor_parallel_size,
        },
        generation_kwargs={
            "temperature": 0.2,
            "max_tokens": 1024,
            "stop_token_ids": [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
        },
        template=get_medu_prompt(ECONOMIC_TEST_DESCRIPTION),
        tensor_parallel_size=tensor_parallel_size,
        prompt_column="text",
        filetype="jsonl.gz",
        one_to_one_input_output_mapping=False,
        num_instances=(1, 16),
    ),
)

if __name__ == "__main__":
    executor_main(steps=[medu_inference])
