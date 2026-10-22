from datasets import load_dataset, Dataset
from datasets import get_dataset_config_names
from huggingface_hub import HfApi
import os

data_name = "openai/gsm8k"

# list all configs for this dataset
subset = "main"
split = "test"
reformatted_data = []

subset_data = load_dataset(data_name, subset, split=split)
for index, item in enumerate(subset_data):
    reformatted_data.append(
        {
            "id": f"gsm8k-{subset}-{split}-#{index}",
            "question": item["question"],
            "answer": item["answer"].split("###")[1].strip(),
            "source": data_name,
            "config": subset,
            "task_type": "qa",
        }
    )

print(len(reformatted_data))

# Upload to HF
target_dataset_name = "yuchenlin/zero-eval"
target_dataset_config = "gsm"
target_dataset_split = "test"

dataset = Dataset.from_list(reformatted_data)
dataset.push_to_hub(
    repo_id=target_dataset_name,
    config_name=target_dataset_config,
    split=target_dataset_split,
    token=os.environ.get("HUGGINGFACE_TOKEN"),
    commit_message=f"Update the merged results.",
)
