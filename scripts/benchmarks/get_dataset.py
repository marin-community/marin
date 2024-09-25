from datasets import load_dataset
import json
import argparse
import os
import yaml
import fsspec
from collections import defaultdict
from typing import List

from dataclasses import dataclass, field
import draccus

@dataclass
class DatasetConfig:
    dataset_name: str
    file_names: list
    file_type: str
    path: str
    hf_path: str
    output_prefix: str
    subject_key: str = ""
    prompt_key: str = ""
    answer_text_key: str = ""
    answer_idx_key: str = ""
    output_choices: List[str] = field(default_factory=list)
    options_key: str = ""
    output_format: str = ""
    

def load_datasets(config):
    """Load the dataset from huggingface cais.
    
    This function returns all data for the given split (rather than subject specific data).
    """
    datasets = []
    path = config.path
    hf_path = config.hf_path
    for file_name in config.file_names:
        datasets.append(load_dataset(hf_path, path, split=file_name))
    return datasets
    
@draccus.wrap()
def main(cfg: DatasetConfig):

    # Load config parameters
    datasets = load_datasets(cfg)
    
    # Load dataset from huggingface dataset
    if cfg.output_format == "decontamination":
        for dataset, file_name in zip(datasets, cfg.file_names):
            output_path = os.path.join(cfg.output_prefix, f"{cfg.dataset_name}-{file_name}-decontamination.jsonl.gz")
            with fsspec.open(output_path, "wt", compression='gzip') as dolma_file:
                for idx, example in enumerate(dataset):
                    subject = example.get(cfg.subject_key, "")
                    if cfg.answer_text_key:
                        answer = example[cfg.answer_text_key]
                    elif cfg.answer_idx_key:
                        answer_idx = int(example[cfg.answer_idx_key])
                        answer = cfg.output_choices[answer_idx]
                    else:
                        raise ValueError("Please specify either answer_text_key or answer_idx_key.")

                    dolma_json = {
                        "id": f"{cfg.dataset_name}-{file_name}-{subject}-{idx}",
                        "text": example[cfg.prompt_key],
                        "source": cfg.dataset_name,
                        "metadata": {
                            "options": example.get(cfg.options_key, []),
                            "answer": answer,
                            "split": file_name,
                            "provenance": f"https://huggingface.co/datasets/{cfg.hf_path}",
                            "hf_path": cfg.hf_path,
                        },
                    }
                    dolma_file.write(json.dumps(dolma_json) + "\n")
    elif cfg.output_format == "evaluation":
        for dataset, data_file in zip(datasets, cfg.file_names):
            # Storing the data in a dictionary with the subject as the key
            subject_files = defaultdict(lambda: '')
            
            for example in dataset:
                question = example[cfg.prompt_key]
                
                choices = example.get(cfg.options_key, [])
                
                input = question.strip() + "\n" + "\n".join([f"{cfg.output_choices[i]}. {choice}" for i, choice in enumerate(choices)]) + "\nAnswer:"
                if cfg.answer_text_key:
                    answer = example[cfg.answer_text_key]
                elif cfg.answer_idx_key:
                    answer_idx = int(example[cfg.answer_idx_key])
                    answer = cfg.output_choices[answer_idx]
                else:
                    raise ValueError("Please specify either answer_text_key or answer_idx_key.")
                
                subject = example.get(cfg.subject_key, "")
                
                subject_files[subject] += (json.dumps({"input": input, "output": answer}) + "\n")
            
            # Writing from subject dict to corresponding files for each subject
            for subject in subject_files:
                output_path = os.path.join(cfg.output_prefix, f"{cfg.dataset_name}-{subject}-{data_file}-evaluation.jsonl.gz")
                print(output_path)
                with fsspec.open(output_path, "wt", compression="gzip") as f:
                    f.write(subject_files[subject])
    else:
        raise ValueError("Please specify either decontamination or evaluation for output_format.")
    
    
if __name__ == "__main__":
    main()
