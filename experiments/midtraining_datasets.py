from experiments.defaults import default_download, default_tokenize
from experiments.llama import llama3_tokenizer
from marin.download.huggingface.download_hf import DownloadConfig, download_hf
from marin.execution.executor import ExecutorStep, this_output_path
import datasets
from marin.transforms.lavita_transform import transform_lavita_dataset
from marin.transforms.openlifescienceai_transform import transform_openlifescienceai_dataset
from marin.experiments.midtraining_anneal import run_midtraining_anneal

BASE_MODEL_CHECKPOINT_PATH = "gs://path/to/your/base/model/checkpoint/step-XXXXXX"  # TODO: Update this path

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

def _transform_lavita_medical_qa_fn(input_path: str) -> None:
    dataset = datasets.load_from_disk(input_path)
    transformed_dataset = transform_lavita_dataset(dataset)
    transformed_dataset.save_to_disk(this_output_path())

lavita_medical_qa_transformed = ExecutorStep(
    name="transformed/lavita_medical_qa",
    fn=_transform_lavita_medical_qa_fn,
    dependencies={"input_path": lavita_medical_qa_datasets},
    override_output_path="transformed/lavita_medical_qa",
)

openlifescienceai_medmcqqa = default_download(
    name="raw/openlifescienceai_medmcqqa",
    hf_dataset_id="openlifescienceai/medmcqa",
    revision="91c6572",
    override_output_path="raw/openlifescienceai_medmcqqa",
)

def _transform_openlifescienceai_medmcqa_fn(input_path: str) -> None:
    dataset = datasets.load_from_disk(input_path)
    transformed_dataset = transform_openlifescienceai_dataset(dataset)
    transformed_dataset.save_to_disk(this_output_path())

openlifescienceai_medmcqqa_transformed = ExecutorStep(
    name="transformed/openlifescienceai_medmcqqa",
    fn=_transform_openlifescienceai_medmcqa_fn,
    dependencies={"input_path": openlifescienceai_medmcqqa},
    override_output_path="transformed/openlifescienceai_medmcqqa",
)

# Define the list of transformed datasets to be passed to the anneal function
# Each element is a tuple: (name_suffix, ExecutorStep_object_for_transformed_dataset)
datasets_for_annealing = [
    ("lavita", lavita_medical_qa_transformed),
    ("openlife", openlifescienceai_medmcqqa_transformed),
]

# Call the annealing function
# This will define multiple training executor steps (one for each dataset and one for combined)
midtraining_anneal_runs = run_midtraining_anneal(
    name_prefix="midtraining",
    transformed_datasets=datasets_for_annealing,
    base_model_checkpoint_path=BASE_MODEL_CHECKPOINT_PATH,
    tokenizer_path=llama3_tokenizer.path  # Assuming llama3_tokenizer is already defined and suitable
)
