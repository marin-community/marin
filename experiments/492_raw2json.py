from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned, output_path_of
from operations.download.huggingface.download import DownloadConfig, download
from operations.raw2json.huggingface.qa.raw2json import DatasetConversionConfig, OutputFormatOptions, raw2json
from levanter.data.text import LMSupervisedDatasetConfig
from marin.processing.tokenize import TokenizeConfig, tokenize, levanter_tokenize_supervised
from experiments.defaults import default_train
from experiments.dolma.tokenize_dolma import DOLMA_OLMO_MIXTURE_WEIGHTS, tokenize_dolma_steps
from experiments.llama import llama_150m, llama_1_4b_train_config, llama3_tokenizer
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config


"""
Downloads the following datasets
- mmlu
"""
############################################################
# download mmlu dataset
mmlu_download_step = ExecutorStep(
    name="raw/cais/mmlu",
    fn=download,
    config=DownloadConfig(
        hf_dataset_id="cais/mmlu",
        revision=versioned("c30699e"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
    override_output_path="gs://marin-us-central2/raw/cais/mmlu",
).cd("c30699e/huggingface.co/datasets/cais/mmlu/resolve/c30699e")

"""
Converts raw to JSON for:
- mmlu
"""
############################################################
# Convert mmlu to evaluation format (i.e. JSON with "prompt", "response" fields)
# This is the input for internal evaluation which measures PPL model gives to correct responses to prompts

# This creates a JSON file representing the auxiliary training data subset of MMLU
mmlu_convert_eval_aux = ExecutorStep(
    name="evaluation/mmlu-eval-aux",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="cais/mmlu",
        subsets=["all"],
        splits=["auxiliary_train"],
        input_path=mmlu_download_step,
        hf_path="cais/mmlu",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("evaluation"),
        prompt_key="question",
        options_key="choices",
        answer_idx_key="answer",
        answer_labels=["A", "B", "C", "D"],
    ),
)

# This creates one file per subject from MMLU, excluding the all and auxiliary training subsets
mmlu_convert_eval_subject = ExecutorStep(
    name="evaluation/mmlu-eval-subject",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="cais/mmlu",
        subsets=["*"],
        splits=["dev", "validation"],
        input_path=mmlu_download_step,
        hf_path="cais/mmlu",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("evaluation"),
        prompt_key="question",
        options_key="choices",
        answer_idx_key="answer",
        answer_labels=["A", "B", "C", "D"],
        exclude_subsets=["all", "auxiliary_train"],
    ),
)
############################################################
# Convert mmlu to dolma format (i.e. JSON with "text" field)
# This is used as input to the decontamination pipeline so documents with MMLU content are removed
mmlu_convert_dolma = ExecutorStep(
    name="decontamination/mmlu-dolma",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="cais/mmlu",
        subsets=["all"],
        splits=["dev", "test", "validation"],
        input_path=mmlu_download_step,
        hf_path="cais/mmlu",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="question",
        options_key="choices",
        answer_idx_key="answer",
        answer_labels=["A", "B", "C", "D"],
    ),
)

supervised_data_config = LMSupervisedDatasetConfig(
        validation_urls=[
            output_path_of(mmlu_convert_eval_aux),
            output_path_of(mmlu_convert_eval_subject),
        ],
        cache_dir=this_output_path(),
        input_field="prompt",
        output_field="response",
    )

supervised_data_cache = ExecutorStep(
    name="supervised/mmlu-cache",
    fn=levanter_tokenize_supervised,
    config=supervised_data_config
)

EXPERIMENT_TAG = ["456-llama-eval"]

mixture_config = lm_mixture_data_config(
    components=tokenize_dolma_steps(),
    weights=DOLMA_OLMO_MIXTURE_WEIGHTS,
)

supervised_data_config = LMSupervisedDatasetConfig(
    validation_urls=[
        output_path_of(mmlu_convert_eval_aux),
        output_path_of(mmlu_convert_eval_subject),
    ],
    cache_dir=output_path_of(supervised_data_cache),
    input_field="prompt",
    output_field="response",
)


train_step = default_train(
    name="llama-150m-supervised-eval",
    tokenized=mixture_config,
    model_config=llama_150m,
    train_config=llama_1_4b_train_config,
    tags=EXPERIMENT_TAG,
    supervised_data=supervised_data_config,
)


############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            mmlu_download_step,
            mmlu_convert_eval_aux,
            mmlu_convert_eval_subject,
            mmlu_convert_dolma,
            supervised_data_cache,
            *tokenize_dolma_steps().values(),
            train_step
        ]
    )
