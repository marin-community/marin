import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

import draccus
import fsspec
from datasets import load_dataset


class OutputFormatOptions(str, Enum):
    decontamination = "decontamination"
    evaluation = "evaluation"


@dataclass
class DatasetConversionConfig:
    dataset_name: str
    subset: str
    splits: list[str]
    input_path: str
    output_prefix: str
    output_format: OutputFormatOptions
    doc_input_format: str = ""
    subject_key: str = ""
    prompt_key: str = ""
    answer_text_key: str = ""
    answer_idx_key: str = ""
    output_choices: list[str] = field(default_factory=list)
    options_key: str = ""
    token: str | bool = True
    trust_remote_code: bool = False


def load_datasets(config: DatasetConversionConfig) -> List[Dataset]:
    """
    Load the dataset from Hugging Face.

    This function returns all data for the given split (rather than subject-specific data).

    Args:
        config (DatasetConversionConfig): The configuration for loading datasets.

    Returns:
        List[Dataset]: A list of Hugging Face datasets loaded according to the given configuration.
    """
    datasets = []
    input_path = config.input_path
    subset = config.subset
    if config.token == "env":
        # if user specifies look for token in environment
        token = os.environ["HF_TOKEN"]
    else:
        token = config.token
    for split in config.splits:
        datasets.append(
            load_dataset(input_path, subset, split=split, token=token, trust_remote_code=config.trust_remote_code)
        )
    return datasets


def get_nested_item(data: Dict[str, Any], key: str, default_item: Optional[Any] = None) -> Any:
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

    # Process dataset examples into selected output format {decontamination, evaluation}
    if cfg.output_format.value == "decontamination":
        # decontamination format is dolma format, expects "text" key for text to be decontaminated
        for dataset, split in zip(datasets, cfg.splits, strict=False):
            output_path = os.path.join(cfg.output_prefix, f"{cfg.dataset_name}-{split}-decontamination.jsonl.gz")
            with fsspec.open(output_path, "wt", compression="gzip") as dolma_file:
                for idx, example in enumerate(dataset):
                    subject = example.get(cfg.subject_key, "")
                    if cfg.answer_text_key:
                        answer = get_nested_item(example, cfg.answer_text_key)
                    elif cfg.answer_idx_key:
                        answer_idx = int(get_nested_item(example, cfg.answer_idx_key))
                        answer = cfg.output_choices[answer_idx]
                    else:
                        raise ValueError("Please specify either answer_text_key or answer_idx_key.")

                    dolma_json = {
                        "id": f"{cfg.dataset_name}-{split}-{subset}-{idx}",
                        "text": get_nested_item(example, cfg.prompt_key),
                        "source": cfg.dataset_name,
                        "metadata": {
                            "options": standardize_options(get_nested_item(example, cfg.options_key, [])),
                            "answer": answer,
                            "split": split,
                            "provenance": f"https://huggingface.co/datasets/{cfg.input_path}",
                            "hf_path": cfg.hf_path,
                        },
                    }
                    dolma_file.write(json.dumps(dolma_json) + "\n")
    elif cfg.output_format.value == "evaluation":
        # evaluation format expects "input" and "output" keys, this is used to determine PPL of expected output given the input
        for dataset, split in zip(datasets, cfg.splits, strict=False):
            # Storing the data in a dictionary with the subject as the key
            subject_files = defaultdict(str)

            for example in dataset:
                question = get_nested_item(example, cfg.prompt_key)

                choices = standardize_options(get_nested_item(example, cfg.options_key, []))

                question_input = (
                    question.strip()
                    + "\n"
                    + "\n".join([f"{cfg.output_choices[i]}. {choice}" for i, choice in enumerate(choices)])
                    + "\nAnswer:"
                )
                if cfg.answer_text_key:
                    answer = get_nested_item(example, cfg.answer_text_key)
                elif cfg.answer_idx_key:
                    answer_idx = int(get_nested_item(example, cfg.answer_idx_key))
                    answer = cfg.output_choices[answer_idx]
                else:
                    raise ValueError("Please specify either answer_text_key or answer_idx_key.")

                subject = get_nested_item(example, cfg.subject_key, "")

                subject_files[subject] += json.dumps({"input": question_input, "output": answer}) + "\n"

            # Writing from subject dict to corresponding files for each subject
            for subject in subject_files:
                output_path = os.path.join(
                    cfg.output_prefix, f"{cfg.dataset_name}-{subject}-{data_file}-evaluation.jsonl.gz"
                )
                with fsspec.open(output_path, "wt", compression="gzip") as f:
                    f.write(subject_files[subject])
    else:
        raise ValueError("Please specify either decontamination or evaluation for output_format.")


if __name__ == "__main__":
    main()
