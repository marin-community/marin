from datasets import load_dataset
import json
import argparse
import os
import glob

DATASET  = "mmlu"   # Name of the dataset
DATASET_REPO = "cais/mmlu"  # Huggingface dataset repo

def load_dataset_mmlu(split):
    """Load the MMLU dataset from huggingface cais.
    
    This function returns all data for the given split (rather than subject specific data).
    """
    dataset = load_dataset(DATASET_REPO, "all", split=split)
    return dataset

def generate_decontamination_dataset():
    data_files = ["dev", "validation", "test"]
    datasets = [load_dataset_mmlu(data_file) for data_file in data_files]
    for dataset, data_file in zip(datasets, data_files):
        with open(f"{DATASET}-{data_file}-decontamination.jsonl", "w") as dolma_file:
            for idx, example in enumerate(dataset):
                subject = example["subject"]
                dolma_json = {
                    "id": f"{DATASET}-{data_file}-{subject}-{idx}",
                    "text": example["question"],
                    "source": DATASET,
                    "metadata": {
                        "options": example["choices"],
                        "answer": example["answer"],
                        "split": data_file,
                        "provenance": "https://huggingface.co/datasets/cais/mmlu",
                        "hf_path": "cais/mmlu"
                    },
                }
                dolma_file.write(json.dumps(dolma_json) + "\n")
    
def generate_evaluation_dataset():
    # Remove old files since we are using append mode
    for eval_file in glob.glob("*_evaluation.jsonl"):
        os.remove(eval_file)
    data_files = ["validation", "auxiliary_train"]
    datasets = [load_dataset_mmlu(data_file) for data_file in data_files]
    for dataset, data_file in zip(datasets, data_files):
        with open(f"{DATASET}-{data_file}.jsonl", "w") as dolma_file:
            for idx, example in enumerate(dataset):
                question = example["question"]
                
                choices = example["choices"]
                
                input_format = f"Q: {question.strip()}\n(A) {choices[0]} (B) {choices[1]} (C) {choices[2]} (D) {choices[3]}\nA:"
                
                answer = example["answer"]
                
                output_format = ['(A)', '(B)', '(C)', '(D)'][answer]
                
                subject = example["subject"] if data_file == "validation" else "auxiliary_train"
                with open(f"{DATASET}-{subject}-{data_file}-evaluation.jsonl", "a") as f:
                    f.write(json.dumps({"input": input_format, "output": output_format}) + "\n")
    

def main(args):
    if args.decontamination:
        generate_decontamination_dataset()
    elif args.evaluation:
        generate_evaluation_dataset()
    
    
if __name__ == "__main__":
    # Make a flag for either internal evaluation or decontamination mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--decontamination", action="store_true")
    parser.add_argument("--evaluation", action="store_true")
    args = parser.parse_args()
    main(args)
    