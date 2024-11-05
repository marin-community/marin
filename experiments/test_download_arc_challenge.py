from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from operations.download.huggingface.download import DownloadConfig
from operations.download.huggingface.download_ray_hf import download_ray_hf
from operations.raw2json.huggingface.qa.raw2json import DatasetConversionConfig, OutputFormatOptions, raw2json

# This script edits exp412_download_and_raw2json_hf_qa.py

"""
Downloads the following datasets
- ARC (Challenge)
"""
############################################################
# download ARC dataset
arc_challenge_download_step = ExecutorStep(
    name="raw/allenai_ai2_arc_temp",
    fn=download_ray_hf,
    config=DownloadConfig(
        hf_dataset_id=versioned("allenai/ai2_arc"),
        revision=versioned("210d026"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_url_glob= versioned("ARC-Challenge/*")
    ),
)


# We need to convert this to JSON
# Converts raw to JSON for:
# - ARC (Challenge)
# """
############################################################
# Convert data to evaluation format (i.e. JSON with "prompt", "response" fields)
# This is the input for internal evaluation which measures PPL model gives to correct responses to prompts

# This creates a JSON file representing the auxiliary training data subset of MMLU
arch_challenge_convert_eval = ExecutorStep(
    name="evaluation/allenai_ai2_arc_eval_train",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="allenai/ai2_arc",
        subsets=["*"], # There should be no subsets, check here first if script crashes 
        splits=["train"], # We test with the train split
        input_path=arc_challenge_download_step,
        hf_path="allenai/ai2_arc",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("evaluation"),
        prompt_key="question",
        options_key="choices",
        answer_idx_key="answerKey",
        answer_labels=["A", "B", "C", "D"],
    ),
)

# ############################################################
# # Convert mmlu to dolma format (i.e. JSON with "text" field)
# # This is used as input to the decontamination pipeline so documents with MMLU content are removed
# mmlu_convert_dolma = ExecutorStep(
#     name="decontamination/mmlu-dolma",
#     fn=raw2json,
#     config=DatasetConversionConfig(
#         dataset_name="cais/mmlu",
#         subsets=["all"],
#         splits=["dev", "test", "validation"],
#         input_path=mmlu_download_step,
#         hf_path="cais/mmlu",
#         output_path=this_output_path(),
#         output_format=OutputFormatOptions("decontamination"),
#         prompt_key="question",
#         options_key="choices",
#         answer_idx_key="answer",
#         answer_labels=["A", "B", "C", "D"],
#     ),
# )

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            arc_challenge_download_step,
            arch_challenge_convert_eval,
        ]
    )
