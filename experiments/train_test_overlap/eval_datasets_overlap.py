from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from operations.download.huggingface.download import DownloadConfig
from operations.download.huggingface.download_hf import download_hf
from operations.raw2json.huggingface.qa.raw2json import DatasetConversionConfig, OutputFormatOptions, raw2json

"""
This script downloads HF datasets and converts them to dolma "text" format for decontamination.
We use the decontamination format to identify potential overlaps between pre-training data and evaluation data.

The script follows the pattern from eval_datasets.py but focuses only on the decontamination conversion.
"""

############################################################
# Download datasets
############################################################

# mmlu raw
mmlu_raw = ExecutorStep(
    name="raw/cais/mmlu_raw",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="cais/mmlu",
        revision=versioned("c30699e"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.md"],
    ),
    override_output_path="raw/cais/mmlu_raw",
)

# download gsm8k dataset
gsm8k_raw = ExecutorStep(
    name="raw/gsm8k",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="openai/gsm8k",
        revision=versioned("6e925c6"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.md"],
    ),
    override_output_path="raw/gsm8k/mainhf",
)

# download math dataset
math_raw = ExecutorStep(
    name="raw/hendrycks_math",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="EleutherAI/hendrycks_math",
        revision=versioned("21a5633"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.md"],
    ),
    override_output_path="raw/hendrycks/mathhf",
)

# download truthful_qa dataset
truthful_qa_raw = ExecutorStep(
    name="raw/truthful_qa",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="truthfulqa/truthful_qa",
        revision=versioned("741b827"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.md"],
    ),
    override_output_path="raw/truthful_qa/multiple_choicehf",
)

# download bbh dataset
bbh_raw = ExecutorStep(
    name="raw/bbh",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="SaylorTwift/bbh",
        revision=versioned("b5306be"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.md"],
    ),
    override_output_path="raw/SaylorTwift/bbhhf",
)

############################################################
# Convert datasets to dolma format for decontamination
############################################################

# Convert gsm8k to dolma format
gsm8k_convert_dolma = ExecutorStep(
    name="decontamination/gsm8k-dolma",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="gsm8k/main",
        subsets=["*"],
        splits=["test"],
        input_path=gsm8k_raw,
        hf_path="gsm8k/main",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="question",
        answer_text_key="answer",
    ),
)

# Convert math dataset to dolma format
math_convert_dolma = ExecutorStep(
    name="decontamination/math-dolma",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="hendrycks/math",
        subsets=["*"],
        splits=["test"],
        input_path=math_raw,
        hf_path="hendrycks/math",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="problem",
        answer_text_key="solution",
    ),
)

# Convert truthful_qa to dolma format
# columns are: question (string), best_answer (string), correct_answers (List[string])
truthful_qa_convert_dolma = ExecutorStep(
    name="decontamination/truthful_qa-dolma",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="truthful_qa/truthful_qa",
        subsets=["generation"],
        splits=["validation"],
        input_path=truthful_qa_raw,
        hf_path="truthful_qa/truthful_qa",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="question",
        answer_text_key="best_answer",
        options_key="correct_answers",
    ),
)

# Convert bbh to dolma format
bbh_convert_dolma = ExecutorStep(
    name="decontamination/bbh-dolma",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="SaylorTwift/bbh",
        subsets=["*"],
        splits=["test"],
        input_path=bbh_raw,
        hf_path="SaylorTwift/bbh",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="input",
        answer_text_key="target",
    ),
)


mmlu_convert_dolma = ExecutorStep(
    name="decontamination/mmlu",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="cais/mmlu",
        subsets=["*"],
        splits=["test"],
        input_path=mmlu_raw,
        hf_path="cais/mmlu",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="question",
        options_key="choices",
        answer_idx_key="answer",
        answer_labels=["A", "B", "C", "D"],
    ),
)
############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            # Download steps
            gsm8k_raw,
            math_raw,
            truthful_qa_raw,
            bbh_raw,
            mmlu_raw,
            # Decontamination conversion steps
            gsm8k_convert_dolma,
            math_convert_dolma,
            truthful_qa_convert_dolma,
            bbh_convert_dolma,
            mmlu_convert_dolma,
        ]
    )
