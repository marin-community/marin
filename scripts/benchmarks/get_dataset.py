import json
import argparse
import os
from collections import defaultdict
from enum import Enum
from typing import List
from dataclasses import dataclass, field

import yaml
import fsspec
from datasets import load_dataset
import draccus


class OutputFormatOptions(str, Enum):
    decontamination = "decontamination"
    evaluation = "evaluation"


@dataclass
class DatasetConfig:
    dataset_name: str
    file_names: List[str]
    file_type: str
    path: str
    hf_path: str
    output_prefix: str
    output_format: OutputFormatOptions = OutputFormatOptions.decontamination
    doc_input_format: str = ""
    subject_key: str = ""
    prompt_key: str = ""
    answer_text_key: str = ""
    answer_idx_key: str = ""
    output_choices: List[str] = field(default_factory=list)
    options_key: str = ""
    token: str | bool = True


def load_datasets(config):
    """Load the dataset from huggingface.

    This function returns all data for the given split (rather than subject specific data).
    """
    datasets = []
    path = config.path
    hf_path = config.hf_path
    if config.token == "env":
        # if user specifies look for token in environment
        token = os.environ["HUGGINGFACE_TOKEN"]
    else:
        token = config.token
    for file_name in config.file_names:
        datasets.append(load_dataset(hf_path, path, split=file_name, token=token))
    return datasets

def get_nested_item(data, key, default_item=None):
    keys = key.split(".")
    result = data
    try:
        for k in keys:
            result = result[k]
        return result
    except (KeyError, TypeError):
        return default_item

@draccus.wrap()
def main(cfg: DatasetConfig):

    # Load config parameters
    datasets = load_datasets(cfg)

    # Load dataset from huggingface dataset
    if cfg.output_format.value == "decontamination":
        for dataset, file_name in zip(datasets, cfg.file_names):
            output_path = os.path.join(cfg.output_prefix, f"{cfg.dataset_name}-{file_name}-decontamination.jsonl.gz")
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
                        "id": f"{cfg.dataset_name}-{file_name}-{subject}-{idx}",
                        "text": get_nested_item(example, cfg.prompt_key),
                        "source": cfg.dataset_name,
                        "metadata": {
                            "options": get_nested_item(example, cfg.options_key, []),
                            "answer": answer,
                            "split": file_name,
                            "provenance": f"https://huggingface.co/datasets/{cfg.hf_path}",
                            "hf_path": cfg.hf_path,
                        },
                    }
                    dolma_file.write(json.dumps(dolma_json) + "\n")
    elif cfg.output_format.value == "evaluation":
        for dataset, data_file in zip(datasets, cfg.file_names):
            # Storing the data in a dictionary with the subject as the key
            subject_files = defaultdict(lambda: "")

            for example in dataset:
                question = get_nested_item(example, cfg.prompt_key)

                choices = get_nested_item(example, cfg.options_key, [])

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
                print(output_path)
                with fsspec.open(output_path, "wt", compression="gzip") as f:
                    f.write(subject_files[subject])
    else:
        raise ValueError("Please specify either decontamination or evaluation for output_format.")


if __name__ == "__main__":
    main()
