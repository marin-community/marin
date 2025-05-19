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

# download humaneval dataset
humaneval_raw = ExecutorStep(
    name="raw/humaneval",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="openai/openai_humaneval",
        revision=versioned("7dce605"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.md"],
    ),
    override_output_path="raw/openai/openai_humanevalhf",
)


# download musr dataset
musr_raw = ExecutorStep(
    name="raw/musr",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="WillHeld/MuSRDecontam",
        revision=versioned("39b4f56"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.parquet"],
    ),
    override_output_path="raw/WillHeld/MuSRDecontamhf",
)


mmlu_pro_raw = ExecutorStep(
    name="raw/mmlu_pro",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="TIGER-Lab/MMLU-Pro",
        revision=versioned("475d58b"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.md"],
    ),
    override_output_path="raw/TIGER-Lab/MMLU-Prohf",
)

instruction_following_raw = ExecutorStep(
    name="raw/instruction_following_eval",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="wis-k/instruction-following-eval",
        revision=versioned("5a5661c"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.jsonl", "*.jsonl"],
    ),
    override_output_path="raw/wis-k/instruction-following-evalhf",
)

gpqa_raw = ExecutorStep(
    name="raw/gpqa",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="Idavidrein/gpqa",
        revision=versioned("90b8e5b"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.csv", "*.csv"],
    ),
    override_output_path="raw/Idavidrein/gpqa",
)

# Download HellaSwag dataset
hellaswag_raw = ExecutorStep(
    name="raw/hellaswag",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="Rowan/hellaswag",
        revision=versioned("6002345"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.jsonl", "*.jsonl"],
    ),
    override_output_path="raw/Rowan/hellaswaghf",
)

# download ai2_arc dataset
ai2_arc_raw = ExecutorStep(
    name="raw/ai2_arc",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="allenai/ai2_arc",
        revision=versioned("210d026"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.parquet"],
    ),
    override_output_path="raw/allenai/ai2_archf",
)

# download boolq dataset
boolq_raw = ExecutorStep(
    name="raw/boolq",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="google/boolq",
        revision=versioned("35b264d"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.jsonl", "*.jsonl"],
    ),
    override_output_path="raw/google/boolqhf",
)

# Insert Tau Commonsense QA download step
commonsense_qa_raw = ExecutorStep(
    name="raw/commonsense_qa",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="tau/commonsense_qa",
        revision=versioned("94630fe"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.parquet"],
    ),
    override_output_path="raw/tau/commonsense_qahf",
)

# download Lambada OpenAI dataset
lambada_openai_raw = ExecutorStep(
    name="raw/lambada_openai",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="EleutherAI/lambada_openai",
        revision=versioned("879e19a"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.jsonl", "*.jsonl"],
    ),
    override_output_path="raw/EleutherAI/lambada_openaihf",
)

# download AllenAI OpenBookQA dataset
openbookqa_raw = ExecutorStep(
    name="raw/openbookqa",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="allenai/openbookqa",
        revision=versioned("388097e"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.parquet"],
    ),
    override_output_path="raw/allenai/openbookqahf",
)

# download PIQA dataset
piqa_raw = ExecutorStep(
    name="raw/piqa",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="baber/piqa",
        revision=versioned("142f6d7"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.parquet"],
    ),
    override_output_path="raw/baber/piqahf",
)

# download Winograd WSC dataset
winograd_wsc_raw = ExecutorStep(
    name="raw/winograd_wsc",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="marcov/winograd_wsc_wsc273_promptsource",
        revision=versioned("63befd8"),
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
        hf_urls_glob=["**/*.parquet", "*.parquet"],
    ),
    override_output_path="raw/marcov/winograd_wsc_wsc273_promptsourcehf",
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

# Convert humaneval to dolma format
humaneval_convert_dolma = ExecutorStep(
    name="decontamination/humaneval",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="openai/openai_humaneval",
        subsets=["*"],
        splits=["test"],
        input_path=humaneval_raw,
        hf_path="openai/openai_humaneval",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="prompt",
        answer_text_key="canonical_solution",
    ),
)

# Convert instruction_following to dolma format (load remotely, no answers)
instruction_following_convert_dolma = ExecutorStep(
    name="decontamination/instruction_following",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="wis-k/instruction-following-eval",
        subsets=["*"],
        splits=["train"],
        input_path="wis-k/instruction-following-eval",
        hf_path="wis-k/instruction-following-eval",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="prompt",
        options_key="instruction_id_list",
        answer_text_ignore=True,
    ),
)

# Convert gpqa to dolma format (load from HF hub, single split)
gpqa_convert_dolma = ExecutorStep(
    name="decontamination/gpqa",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="Idavidrein/gpqa",
        subsets=["gpqa_main", "gpqa_extended", "gpqa_diamond"],
        splits=["train"],
        input_path="Idavidrein/gpqa",
        hf_path="Idavidrein/gpqa",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="Question",
        answer_text_key="Correct Answer",
        options_keys=["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"],
    ),
)


mmlu_pro_convert_dolma = ExecutorStep(
    name="decontamination/mmlu_pro",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="TIGER-Lab/MMLU-Pro",
        subsets=["*"],
        splits=["test"],
        input_path=mmlu_pro_raw,
        hf_path="TIGER-Lab/MMLU-Pro",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="question",
        options_key="options",
        answer_idx_key="answer_index",
        answer_labels=["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
    ),
)

# Convert musr to dolma format
musr_convert_dolma = ExecutorStep(
    name="decontamination/musr",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="WillHeld/MuSRDecontam",
        subsets=[""],
        splits=["test"],
        input_path=musr_raw,
        hf_path="WillHeld/MuSRDecontam",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="narrative",
        options_key="choices",
        answer_idx_key="answer_index",
        answer_text_key="answer_choice",
    ),
)

# Convert HellaSwag to dolma format
hellaswag_convert_dolma = ExecutorStep(
    name="decontamination/hellaswag",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="Rowan/hellaswag",
        subsets=["*"],
        splits=["test"],
        input_path=hellaswag_raw,
        hf_path="Rowan/hellaswag",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="ctx",
        answer_text_ignore=True,
    ),
)

# Convert AI2-ARC to dolma format
ai2_arc_convert_dolma = ExecutorStep(
    name="decontamination/ai2_arc",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="allenai/ai2_arc",
        subsets=["*"],
        splits=["test"],
        input_path=ai2_arc_raw,
        hf_path="allenai/ai2_arc",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="question",
        options_key="choices.text",
        answer_label_key="answerKey",
        answer_labels=["A", "B", "C", "D"],
        answer_text_ignore=True,
    ),
)

# Convert BoolQ to dolma format
boolq_convert_dolma = ExecutorStep(
    name="decontamination/boolq",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="google/boolq",
        subsets=["*"],
        splits=["validation"],
        input_path=boolq_raw,
        hf_path="google/boolq",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="question",
        answer_text_ignore=True,
    ),
)

# Insert Tau Commonsense QA conversion step
commonsense_qa_convert_dolma = ExecutorStep(
    name="decontamination/commonsense_qa",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="tau/commonsense_qa",
        subsets=["*"],
        splits=["validation"],
        input_path=commonsense_qa_raw,
        hf_path="tau/commonsense_qa",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="question",
        options_key="choices.text",
        answer_label_key="answerKey",
        answer_labels=["A", "B", "C", "D", "E"],
        answer_text_ignore=True,
    ),
)

# Convert Lambada OpenAI to dolma format
lambada_openai_convert_dolma = ExecutorStep(
    name="decontamination/lambada_openai",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="EleutherAI/lambada_openai",
        subsets=["*"],
        splits=["test"],
        input_path=lambada_openai_raw,
        hf_path="EleutherAI/lambada_openai",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="text",
        answer_text_ignore=True,
    ),
)

# Convert AllenAI OpenBookQA to dolma format
openbookqa_convert_dolma = ExecutorStep(
    name="decontamination/openbookqa",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="allenai/openbookqa",
        subsets=["*"],
        splits=["test"],
        input_path=openbookqa_raw,
        hf_path="allenai/openbookqa",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="question_stem",
        options_key="choices.text",
        answer_label_key="answerKey",
        answer_labels=["A", "B", "C", "D"],
        answer_text_ignore=True,
    ),
)

# Convert PIQA to dolma format
piqa_convert_dolma = ExecutorStep(
    name="decontamination/piqa",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="baber/piqa",
        subsets=["*"],
        splits=["test"],
        input_path=piqa_raw,
        hf_path="baber/piqa",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="goal",
        options_keys=["sol1", "sol2"],
        answer_label_key="label",
        answer_labels=["0", "1"],
        answer_text_ignore=True,
    ),
)

# Convert Winograd WSC to dolma format
winograd_wsc_convert_dolma = ExecutorStep(
    name="decontamination/winograd_wsc",
    fn=raw2json,
    config=DatasetConversionConfig(
        dataset_name="marcov/winograd_wsc_wsc273_promptsource",
        subsets=["*"],
        splits=["test"],
        input_path=winograd_wsc_raw,
        hf_path="marcov/winograd_wsc_wsc273_promptsource",
        output_path=this_output_path(),
        output_format=OutputFormatOptions("decontamination"),
        prompt_key="rendered_input",
        options_key="options",
        answer_label_key="label",
        answer_labels=["0", "1"],
        answer_text_ignore=True,
    ),
)

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            gsm8k_raw,
            math_raw,
            truthful_qa_raw,
            bbh_raw,
            mmlu_raw,
            humaneval_raw,
            gpqa_raw,
            instruction_following_raw,
            musr_raw,
            mmlu_pro_raw,
            boolq_raw,
            ai2_arc_raw,
            hellaswag_raw,
            piqa_raw,
            winograd_wsc_raw,
            commonsense_qa_raw,
            lambada_openai_raw,
            openbookqa_raw,
            gsm8k_convert_dolma,
            math_convert_dolma,
            truthful_qa_convert_dolma,
            bbh_convert_dolma,
            mmlu_convert_dolma,
            humaneval_convert_dolma,
            instruction_following_convert_dolma,
            gpqa_convert_dolma,
            musr_convert_dolma,
            mmlu_pro_convert_dolma,
            hellaswag_convert_dolma,
            ai2_arc_convert_dolma,
            boolq_convert_dolma,
            commonsense_qa_convert_dolma,
            lambada_openai_convert_dolma,
            openbookqa_convert_dolma,
            piqa_convert_dolma,
            winograd_wsc_convert_dolma,
        ]
    )
