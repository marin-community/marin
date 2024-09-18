from datasets import load_dataset
import json
import argparse
import os
import glob
import yaml
import pathlib
import fsspec
from collections import defaultdict

def load_datasets(config):
    """Load the dataset from huggingface cais.
    
    This function returns all data for the given split (rather than subject specific data).
    """
    datasets = []
    path = config["path"]
    hf_path = config["hf_path"]
    for file_name in config["file_names"]:
        datasets.append(load_dataset(hf_path, path, split=file_name))
    return datasets
    

def main(args):
    with open(args.yaml, "r") as file:
        config = yaml.safe_load(file)

    # Load config parameters
    dataset_name = config["name"]
    file_names = config["file_names"]
    hf_path = config["hf_path"]
    datasets = load_datasets(config)
    output_prefix = pathlib.Path(args.output)
    subject_key = config.get("subject_key", "")
    prompt_key = config.get("prompt_key", "")
    answer_text_key = config.get("answer_text_key", "")
    answer_idx_key = config.get("answer_idx_key", "")
    output_choices = config.get("doc_output_choice", ["(A)", "(B)", "(C)", "(D)"])
    options_key = config.get("options_key", "")
    
    # Load dataset from huggingface dataset
    if args.decontamination:
        for dataset, file_name in zip(datasets, file_names):
            output_path = output_prefix / f"{dataset_name}-{file_name}-decontamination.jsonl.gz"
            with fsspec.open(output_path, "wt", compression='gzip') as dolma_file:
                for idx, example in enumerate(dataset):
                    subject = example.get(subject_key, "")
                    if answer_text_key:
                        answer = example[answer_text_key]
                    elif answer_idx_key:
                        answer_idx = int(example[answer_idx_key])
                        answer = output_choices[answer_idx]
                    else:
                        raise ValueError("Please specify either answer_text_key or answer_idx_key.")

                    dolma_json = {
                        "id": f"{dataset_name}-{file_name}-{subject}-{idx}",
                        "text": example[prompt_key],
                        "source": dataset_name,
                        "metadata": {
                            "options": example.get(options_key, []),
                            "answer": answer,
                            "split": file_name,
                            "provenance": f"https://huggingface.co/datasets/{hf_path}",
                            "hf_path": hf_path,
                        },
                    }
                    dolma_file.write(json.dumps(dolma_json) + "\n")
    elif args.evaluation:
        for dataset, data_file in zip(datasets, file_names):
            # Storing the data in a dictionary with the subject as the key
            subject_files = defaultdict(lambda: '')
            
            for example in dataset:
                question = example[prompt_key]
                
                choices = example.get(options_key, [])
                
                input = question.strip() + "\n" + "\n".join([f"{output_choices[i]}. {choice}" for i, choice in enumerate(choices)]) + "\nAnswer:"
                if answer_text_key:
                    answer = example[answer_text_key]
                elif answer_idx_key:
                    answer_idx = int(example[answer_idx_key])
                    answer = output_choices[answer_idx]
                else:
                    raise ValueError("Please specify either answer_text_key or answer_idx_key.")
                
                subject = example.get(subject_key, "")
                
                subject_files[subject] += (json.dumps({"input": input, "output": answer}) + "\n")
            
            # Writing from subject dict to corresponding files for each subject
            for subject in subject_files:
                output_path = output_prefix / f"{dataset_name}-{subject}-{data_file}-evaluation.jsonl.gz"
                with fsspec.open(output_path, "wt", compression="gzip") as f:
                    f.write(subject_files[subject])
    else:
        raise ValueError("Please specify either decontamination or evaluation mode.")
    
    
if __name__ == "__main__":
    # Make a flag for either internal evaluation or decontamination mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--decontamination", action="store_true")
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--output", type=str, default=".")
    parser.add_argument("--yaml", type=str)
    args = parser.parse_args()
    main(args)
    