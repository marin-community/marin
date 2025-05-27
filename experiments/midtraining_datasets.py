from experiments.defaults import default_download, default_tokenize
from experiments.llama import llama3_tokenizer
from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, this_output_path

finemath_commit_hash = "8f233cf"
finemath = ExecutorStep(
    name="raw/finemath",
    fn=download_hf,
    config=DownloadConfig(
        hf_dataset_id="HuggingFaceTB/finemath",
        revision=finemath_commit_hash,
        gcs_output_path=this_output_path(),
        wait_for_completion=True,
    ),
)


finemath_3_plus = finemath.cd("finemath-3plus")
finemath_3_plus_tokenized = default_tokenize(
    name="finemath_3_plus",
    dataset=finemath_3_plus,
    tokenizer=llama3_tokenizer,
).with_output_path("tokenized/finemath_3_plus-a26b0f/")

lavita_medical_qa_datasets = default_download(
    name="raw/lavita_medical_qa",
    hf_dataset_id="lavita/medical-qa-datasets",
    revision="59d48e2",
    override_output_path="raw/lavita_medical_qa",
)

openlifescienceai_medmcqqa = default_download(
    name="raw/openlifescienceai_medmcqqa",
    hf_dataset_id="openlifescienceai/medmcqa",
    revision="91c6572",
    override_output_path="raw/openlifescienceai_medmcqqa",
)
