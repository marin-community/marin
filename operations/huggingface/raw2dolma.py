import json
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import draccus
import fsspec
from datasets import Dataset, load_dataset, get_dataset_config_names


class OutputFormatOptions(str, Enum):
    decontamination = "decontamination"
    evaluation = "evaluation"


@dataclass
class DatasetConversionConfig:
    dataset_name: str
    subsets: list[str]
    splits: list[str]
    input_path: str
    hf_path: str
    output_prefix: str
    output_format: OutputFormatOptions
    prompt_key: str
    doc_input_format: str = ""
    subject_key: str = ""
    answer_text_key: str = ""
    answer_idx_key: str = ""
    answer_label_key: str = ""
    output_labels: list[str] = field(default_factory=list)
    options_key: str = ""
    token: str | bool = False
    trust_remote_code: bool = False


@dataclass
class DatasetWithMetaData:
    dataset: Dataset
    subset: str
    split: str


def load_datasets(config: DatasetConversionConfig) -> list[DatasetWithMetaData]:
    """
    Load the dataset from Hugging Face.

    This function returns all data for the requested subsets and splits.

    A separate dataset is produced for each subset,split pair. Downstream one file per subset,split pair will be created.

    Args:
        config (DatasetConversionConfig): The configuration for loading datasets.

    Returns:
        List[Dataset]: A list of Hugging Face datasets loaded according to the given configuration.
    """
    datasets = []
    if config.token == "env":
        # Setting token to 'env' indicates a request to retrieve the token from the $HF_TOKEN environment variable
        token = os.environ["HF_TOKEN"]
    else:
        token = config.token
    # get subsets
    subsets = []
    excludes = []
    for subset in config.subsets:
        if subset == "all-subsets":
            # * - is special for you want all
            subsets += get_dataset_config_names(config.input_path)
        elif subset.startswith("exclude:"):
            # you might want to exclude something e.g. in mmlu exclude:all
            excludes.append(subset.split("exclude:")[1])
        else:
            # add a typical subset in the iteration
            subsets.append(subset)
    subsets = [subset for subset in subsets if subset not in excludes]

    for subset in config.subsets:
        for split in config.splits:
            try:
                dataset_w_meta = DatasetWithMetaData(
                    load_dataset(
                        config.input_path, subset, split=split, token=token, trust_remote_code=config.trust_remote_code
                    ),
                    subset,
                    split,
                )
                datasets.append(dataset_w_meta)
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


def is_kv_list(lst):
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


def standardize_options(options):
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


@draccus.wrap()
def main(cfg: DatasetConversionConfig):

    # Load config parameters
    datasets = load_datasets(cfg)

    # go through (subset,split) pairs and upload file for that (subset,split) producing output JSON specified in config
    for dataset in datasets:
        output_path = os.path.join(
            cfg.output_prefix, f"{cfg.dataset_name}-{dataset.subset}-{dataset.split}-{cfg.output_format.value}.jsonl.gz"
        )
        with fsspec.open(output_path, "wt", compression="gzip") as dolma_file:
            for idx, example in enumerate(dataset.dataset):
                dolma_json = {
                    "id": f"{cfg.dataset_name}-{dataset.subset}-{dataset.split}-{cfg.output_format.value}-{idx}",
                    "source": cfg.dataset_name,
                    "metadata": {
                        "subset": dataset.subset,
                        "split": dataset.split,
                        "provenance": f"https://huggingface.co/datasets/{cfg.hf_path}",
                    },
                }
                # get the question text
                question_text = get_nested_item(example, cfg.prompt_key)
                # get the list of options in standardized form (list of options
                choices = standardize_options(get_nested_item(example, cfg.options_key, []))
                # if there is a direct key to answer text, use this
                answer_text = get_nested_item(example, cfg.answer_text_key) if cfg.answer_text_key else ""
                # if there is a direct key for the idx into choices of correct answer, use this
                answer_idx = get_nested_item(example, cfg.answer_idx_key) if cfg.answer_idx_key else -1
                # if there is a direct key for the label of the correct answer, use this
                answer_label = get_nested_item(example, cfg.answer_label_key) if cfg.answer_label_key else ""
                # check if you need to get the answer text by using the answer idx
                if not answer_text:
                    if answer_label and choices and cfg.output_choices:
                        answer_idx = cfg.output_choices.index(answer_label)
                    if isinstance(answer_idx, int) and answer_idx != -1 and choices:
                        answer_text = choices[answer_idx]
                    else:
                        raise ValueError("No answer text was found. Please review config.")
                if choices:
                    # list of potential answers
                    dolma_json["metadata"]["options"] = choices
                if answer_idx:
                    # index into list of potential answers of correct answer
                    dolma_json["metadata"]["answer_idx"] = answer_idx
                if answer_text:
                    # answer text of correct answer
                    dolma_json["metadata"]["answer"] = answer_text
                if answer_label:
                    # label of correct answer (e.g. "A")
                    dolma_json["metadata"]["answer_label"] = answer_label
                if cfg.output_labels:
                    # list of potential labels (e.g. ["A", "B", "C", 'D"])
                    dolma_json["metadata"]["output_labels"] = cfg.output_labels
                if cfg.output_format.value == "decontamination":
                    dolma_json["text"] = question_text
                elif cfg.output_format.value == "evaluation":
                    if cfg.output_labels and choices:
                        question_input = (
                            question_text.strip()
                            + "\n"
                            + "\n".join([f"{cfg.output_labels[i]}. {choice}" for i, choice in enumerate(choices)])
                            + "\nAnswer:"
                        )
                        answer_output = f"{cfg.output_labels[answer_idx]}. {answer_text}"
                    else:
                        question_input = question_text + "\n\n"
                        answer_output = answer_text
                    dolma_json["input"] = question_input
                    dolma_json["output"] = answer_output
                dolma_file.write(json.dumps(dolma_json) + "\n")


if __name__ == "__main__":
    main()
