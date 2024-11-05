from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from operations.download.huggingface.download import DownloadConfig
from operations.download.huggingface.download import download_hf_dataset
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
    fn=download_hf_dataset,
    config=DownloadConfig(
        hf_dataset_id=versioned("allenai/ai2_arc"),
        revision=versioned("210d026"),
        hf_url_glob=versioned("ARC-Challenge/*"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
)


# We need to convert this to JSON
# Converts raw to JSON for:
# - ARC (Challenge)
# """
############################################################
# Convert data to evaluation format (i.e. JSON with "prompt", "response" fields)
# This is the input for internal evaluation which measures PPL model gives to correct responses to prompts

# This creates a JSON file representing the TRAIN set in Arc (Challenge)
# Sample of the original data:
# {
#     "answerKey": "B",
#     "choices": {
#         "label": ["A", "B", "C", "D"],
#         "text": ["Shady areas increased.", "Food sources increased.", "Oxygen levels increased.", "Available water increased."]
#     },
#     "id": "Mercury_SC_405487",
#     "question": "One year, the oak trees in a park began producing more acorns than usual. The next year, the population of chipmunks in the park also increased. Which best explains why there were more chipmunks the next year?"
# }

arch_challenge_convert_eval = ExecutorStep(
    name="evaluation/allenai_ai2_arc_eval_train",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="allenai/ai2_arc",
        subsets=["*"],  # There should be no subsets, check here first if script crashes
        splits=["train"],  # We test with the train split
        input_path=arc_challenge_download_step,
        hf_path="allenai/ai2_arc",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("evaluation"),
        prompt_key="question",
        options_key="choices.text",
        answer_idx_key="answerKey",
        answer_labels=["A", "B", "C", "D"],
    ),
)

############################################################
# Convert data to dolma format (i.e. JSON with "text" field)
# This is used as input to the decontamination pipeline so documents with dataset content are removed
# We use all the splits when decontamination
arc_challenge_convert_dolma = ExecutorStep(
    name="decontamination/arc_challenge",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="allenai/ai2_arc",
        subsets=["*"],
        splits=["train", "test", "validation"],
        input_path=arc_challenge_download_step,
        hf_path="allenai/ai2_arc",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="question",
        options_key="choices.text",
        answer_idx_key="answerKey",
        answer_labels=["A", "B", "C", "D"],
    ),
)

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            arc_challenge_download_step,
            arch_challenge_convert_eval,
            arc_challenge_convert_dolma,
        ]
    )
