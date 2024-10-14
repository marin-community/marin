import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urlparse

import draccus
import fsspec
from datasets import Dataset, get_dataset_config_names, load_dataset
from google.cloud import storage

from marin.core.data import QAExample, QAExampleMetadata
from marin.utilities.dataclass_utils import asdict_without_nones


class OutputFormatOptions(str, Enum):
    decontamination = "decontamination"
    evaluation = "evaluation"


@dataclass
class DatasetConversionConfig:
    """
    Configuration class for converting Hugging Face datasets to dolma format.

    Attributes:
        dataset_name (str): Name of the Hugging Face dataset to convert
        subsets (list[str]): List of subsets of dataset to convert
        splits (list[str]): List of splits of dataset to convert
        input_path (str): HF Hub, local, or GCP path where Hugging Face repo is stored
        hf_path (str): HF Hub path (e.g. cais/mmlu) for provenance
        output_path (str): where to store output of dolma processing
        output_format (str): format of output JSON from {decontaminaton, evaluation}
        prompt_key (str): key in HF data object for the prompt
        answer_text_key (str): key in HF data object for the answer text (e.g. "Paris")
        answer_idx_key (str): key in HF data object for the idx of the correct answer (e.g. 0)
        answer_label_key (str): key in HF data object for the label of the correct answer (e.g. "A")
        options_key (str): key in HF data object for the options (e.g. ["Rome", "London", "Berlin", "Paris"])
        answer_labels (list[str]): list of labels for an example (e.g. ["A", "B", "C", "D"])
        exclude_subsets (list[str]): list of subsets to exclude
        token (str): HF Hub token when authentication required, "env" means look at $HF_TOKEN
        trust_remote_code (str): allow load_dataset to use remote code to build dataset
    """

    dataset_name: str
    subsets: list[str]
    splits: list[str]
    input_path: str
    hf_path: str
    output_path: str
    output_format: OutputFormatOptions
    prompt_key: str
    answer_text_key: str = ""
    answer_idx_key: str = ""
    answer_label_key: str = ""
    options_key: str = ""
    answer_labels: list[str] = field(default_factory=list)
    exclude_subsets: list[str] = field(default_factory=list)
    token: str | bool = False
    trust_remote_code: bool = False


@dataclass
class DatasetWithMetaData:
    dataset: Dataset
    subset: str
    split: str


def download_directory_from_gcs(bucket_name: str, gcs_directory_path: str, local_directory_path: str) -> None:
    """
    Download an entire directory from a GCS bucket to a local directory.

    Args:
        bucket_name (str): The name of the GCS bucket.
        gcs_directory_path (str): The path to the directory in GCS (excluding the bucket name).
        local_directory_path (str): The local directory path where the files will be saved.
    """
    # Make download dir
    if not os.path.exists(local_directory_path):
        os.makedirs(local_directory_path)
    # Initialize the client
    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # List all the blobs (files) with the specified prefix
    blobs = bucket.list_blobs(prefix=gcs_directory_path)

    # Download each blob to the local directory
    for blob in blobs:
        # Construct the relative path of the file
        relative_path = os.path.relpath(blob.name, gcs_directory_path)
        local_file_path = os.path.join(local_directory_path, relative_path)

        # Create local directories if they do not exist
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Download the blob to the local file path
        blob.download_to_filename(local_file_path)
        print(f"Downloaded {blob.name} to {local_file_path}")


def load_datasets(config: DatasetConversionConfig) -> list[DatasetWithMetaData]:
    """
    Load the dataset from Hugging Face.
    This function returns all data for the requested subsets and splits.
    A separate dataset is produced for each subset,split pair.
    Downstream one file per subset,split pair will be created.

    Args:
        config (DatasetConversionConfig): The configuration for loading datasets.

    Returns:
        List[Dataset]: A list of Hugging Face datasets loaded according to the given configuration.
    """
    # set up input path which can be GCP path, HF Hub path, or local path
    # handle case of gs:// path which requires downloading resource from GCP to local for processing
    if config.input_path.startswith("gs://"):
        # parse gs://my-bucket/path/to/mmlu into "my-bucket", "path/to/mmlu", and "mmlu"
        parsed_url = urlparse(config.input_path)
        bucket = parsed_url.netloc
        gcp_path = parsed_url.path.lstrip("/")
        dir_name = os.path.basename(gcp_path)
        # download the repo from GCP path into local directory which is basename of provided path (e.g. mmlu)
        download_directory_from_gcs(bucket, gcp_path, dir_name)
        input_path = dir_name
    else:
        # for now other handled input paths such as local paths and HF Hub paths do not require special processing
        input_path = config.input_path
    datasets = []
    if not config.token and os.environ.get("HF_TOKEN", None):
        # if config.token is not set and there is a token in the standard environment variable use it
        token = os.environ["HF_TOKEN"]
    else:
        token = config.token
    # get subsets
    subsets = []
    for subset in config.subsets:
        if subset == "*":
            # "*" - is special for you want all subsets
            subsets += get_dataset_config_names(input_path)
        else:
            # add a typical subset in the iteration
            subsets.append(subset)
    subsets = [subset for subset in subsets if subset not in config.exclude_subsets]

    for subset in subsets:
        for split in config.splits:
            try:
                dataset_w_metadata = DatasetWithMetaData(
                    load_dataset(
                        input_path, subset, split=split, token=token, trust_remote_code=config.trust_remote_code
                    ),
                    subset,
                    split,
                )
                datasets.append(dataset_w_metadata)
            except Exception as e:
                print(f"Failed to load subset '{subset}' and split '{split}': {e}")

    return datasets


def get_nested_item(data: dict[str, Any], key: str, default_item: Any = None) -> Any:
    """
    Retrieve a nested item from a dictionary using a dot notation key.

    Args:
        data (dict): The dictionary from which to retrieve the nested item.
        key (str): A string representing the key path, with keys separated by dots.
                   For example, 'a.b.c' will retrieve data['a']['b']['c'].
        default_item (Any, optional): The value to return if the key path does not exist.
                                      Defaults to None.

    Returns:
        Any: The value at the specified key path, or the default_item if any key in the path
             does not exist or if the input data is not a dictionary.
    """
    keys = key.split(".")
    result = data
    try:
        for k in keys:
            result = result[k]
        return result
    except (KeyError, TypeError):
        return default_item


def is_kv_list(lst: list) -> bool:
    """
    Check if the given list is a list of key-value dictionaries.

    This function verifies if the input is a list where each element is a dictionary
    containing both "key" and "value" fields.

    Parameters:
    lst (list): The list to check.

    Returns:
    bool: True if the list is a list of dictionaries with "key" and "value" fields, False otherwise.

    Example:
    >>> is_kv_list([{"key": "A", "value": "Option 1"}, {"key": "B", "value": "Option 2"}])
    True

    >>> is_kv_list([{"key": "A"}, {"key": "B", "value": "Option 2"}])
    False

    >>> is_kv_list("Not a list")
    False
    """
    if isinstance(lst, list):
        return all(isinstance(item, dict) and "key" in item and "value" in item for item in lst)
    return False


def standardize_options(options: list | dict) -> list:
    """
    Standardize multiple choice options for LLM benchmarks.

    The function accepts various formats of multiple choice options and converts them
    into a standard list of answer values. The supported input formats are:

    1. A list of dictionaries with 'key' and 'value' pairs, which is sorted by the 'key'.
    2. A simple list of answer options.
    3. A dictionary mapping keys to answer values, which is sorted by the keys.

    Parameters:
    options (list or dict): The multiple choice options, which can be:
                            - A list of dictionaries with 'key' and 'value' fields.
                            - A list of answer values.
                            - A dictionary mapping keys to answer values.

    Returns:
    list: A list of answer values, sorted if the input was a dictionary or key-value list.

    Example:
    >>> options_dict = {"B": "Canada", "A": "France", "D": "United Kingdom", "C": "United States"}
    >>> standardize_options(options_dict)
    ['France', 'Canada', 'United States', 'United Kingdom']

    >>> options_list = ["France", "Canada", "United States", "United Kingdom"]
    >>> standardize_options(options_list)
    ['France', 'Canada', 'United States', 'United Kingdom']

    >>> options_kv_list = [{"key": "B", "value": "Canada"}, {"key": "A", "value": "France"},
    ...                    {"key": "D", "value": "United Kingdom"}, {"key": "C", "value": "United States"}]
    >>> standardize_options(options_kv_list)
    ['France', 'Canada', 'United States', 'United Kingdom']
    """
    if is_kv_list(options):
        sorted_values = [x["value"] for x in sorted(options, key=lambda x: x["key"])]
        return sorted_values
    elif isinstance(options, list):
        return options
    elif isinstance(options, dict):
        sorted_keys = sorted(list(options.keys()))
        sorted_values = [options[key] for key in sorted_keys]
        return sorted_values


def format_prompt_response(
    question_text: str, options: list[str], labels: list[str], answer_idx: str, answer_text: str
) -> tuple[str, str]:
    """
    Produce a fully formatted input string from a question, set of options, and labels

    Args:
        question_text (str): The text of the question
        options (list[str]): The list of options for the multiple choice question
        labels (list[str]): The list of labels
        answer_idx (str): The index of the correct answer
        answer_text (str): The text of the correct answer

    Returns:
        str, str: The formatted prompt and the formatted response
    """
    prompt = (
        question_text.strip()
        + "\n"
        + "\n".join([f"{labels[i]}. {choice}" for i, choice in enumerate(options)])
        + "\nAnswer:"
    )
    response = f"{labels[answer_idx]}. {answer_text}"
    return prompt, response


def raw2json(cfg: DatasetConversionConfig) -> None:

    # Load config parameters
    datasets = load_datasets(cfg)

    # go through (subset,split) pairs and upload file for that (subset,split) producing output JSON specified in config
    for dataset in datasets:
        output_path = os.path.join(
            cfg.output_path, f"{cfg.dataset_name}-{dataset.subset}-{dataset.split}-{cfg.output_format.value}.jsonl.gz"
        )
        with fsspec.open(output_path, "wt", compression="gzip") as dolma_file:
            for idx, example in enumerate(dataset.dataset):
                # create base document
                document = QAExample(
                    id=f"{cfg.dataset_name}-{dataset.subset}-{dataset.split}-{cfg.output_format.value}-{idx}",
                    source=cfg.dataset_name,
                    metadata=QAExampleMetadata(
                        subset=dataset.subset,
                        split=dataset.split,
                        provenance=f"https://huggingface.co/datasets/{cfg.hf_path}",
                    ),
                )
                # get the question text
                question_text = get_nested_item(example, cfg.prompt_key)
                # get the list of options in standardized form (list of options)
                options = standardize_options(get_nested_item(example, cfg.options_key, []))
                # first pass attempt to populate answer_text, answer_idx, answer_label
                # if there is a direct key to answer text, use this
                answer_text = get_nested_item(example, cfg.answer_text_key) if cfg.answer_text_key else ""
                # if there is a direct key for the idx into choices of correct answer, use this
                answer_idx = get_nested_item(example, cfg.answer_idx_key) if cfg.answer_idx_key else None
                # if there is a direct key for the label of the correct answer, use this
                answer_label = get_nested_item(example, cfg.answer_label_key) if cfg.answer_label_key else ""
                # try to populate answer_text, answer_idx, answer_label based on initial retrieved values
                if not answer_idx:
                    if answer_label and cfg.answer_labels:
                        # infer answer_idx (e.g. 0) from answer_label and list of potential labels
                        answer_idx = cfg.answer_labels.index(answer_label)
                    elif answer_text and options:
                        # infer answer_idx (e.g. 0) from answer_text and options list
                        answer_idx = cfg.answer_labels.index(answer_text)
                if not answer_label:
                    if answer_idx is not None and isinstance(answer_idx, int) and cfg.answer_labels:
                        # infer answer_label (e.g. A) from answer_label and list of potential labels
                        answer_label = cfg.answer_labels[answer_idx]
                if not answer_text:
                    if answer_idx is not None and isinstance(answer_idx, int) and options:
                        # infer answer text (e.g. Paris) from answer_idx and options list
                        answer_text = options[answer_idx]
                    else:
                        raise ValueError("No answer text was found. Please review config.")
                # set various metadata
                if options:
                    # list of potential answers
                    document.metadata.options = options
                if answer_idx:
                    # index into list of potential answers of correct answer
                    document.metadata.answer_idx = answer_idx
                if answer_label:
                    # label of correct answer (e.g. "A")
                    document.metadata.answer_label = answer_label
                if answer_text:
                    # answer text of correct answer
                    document.metadata.answer = answer_text
                if cfg.answer_labels:
                    # list of potential labels (e.g. ["A", "B", "C", 'D"])
                    document.metadata.answer_labels = cfg.answer_labels
                if cfg.output_format.value == "decontamination":
                    # decontamination output format is dolma with text as the key
                    document.text = question_text
                elif cfg.output_format.value == "evaluation":
                    # evaluation format wants prompt, response
                    if cfg.answer_labels and options:
                        prompt, response = format_prompt_response(
                            question_text, options, cfg.answer_labels, answer_idx, answer_text
                        )
                    else:
                        prompt = question_text + "\n\n"
                        response = answer_text
                    document.prompt = prompt
                    document.response = response
                # write json to output file
                print(json.dumps(asdict_without_nones(document)), file=dolma_file)


@draccus.wrap()
def main(cfg: DatasetConversionConfig) -> None:
    raw2json(cfg)


if __name__ == "__main__":
    main()
