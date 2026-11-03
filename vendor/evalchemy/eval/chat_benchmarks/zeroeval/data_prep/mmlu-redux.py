from datasets import load_dataset, Dataset
from datasets import get_dataset_config_names
from huggingface_hub import HfApi
import os

data_name = "edinburgh-dawg/mmlu-redux"

# list all configs for this dataset
configs = get_dataset_config_names(data_name)

reformatted_data = []

for subset in configs:
    subset_data = load_dataset(data_name, subset, split="test")
    for index, item in enumerate(subset_data):
        correct_answer = None
        if item["error_type"] == "ok":
            correct_answer = item["choices"][int(item["answer"])]
        elif item["error_type"] == "wrong_groundtruth" and item["correct_answer"] in list("ABCDEF"):
            correct_answer = item["choices"]["ABCDEF".index(item["correct_answer"])]
        else:
            # multiple answers, bad questions, etc.
            continue
        reformatted_data.append(
            {
                "id": f"mmlu-redux-{subset}-#{index}",
                "question": item["question"],
                "choices": item["choices"],
                "correct_answer": correct_answer,
                "source": data_name,
                "config": subset,
                "task_type": "multiple_choice",
            }
        )

print(len(reformatted_data))

# Upload to HF
target_dataset_name = "yuchenlin/zero-eval"
target_dataset_config = "mmlu-redux"
target_dataset_split = "test"

dataset = Dataset.from_list(reformatted_data)
dataset.push_to_hub(
    repo_id=target_dataset_name,
    config_name=target_dataset_config,
    split=target_dataset_split,
    token=os.environ.get("HUGGINGFACE_TOKEN"),
    commit_message=f"Update the merged results.",
)
