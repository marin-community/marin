from datasets import load_dataset, Dataset
from datasets import get_dataset_config_names
from huggingface_hub import HfApi
import os

data_name = "cruxeval-org/cruxeval"

# list all configs for this dataset
split = "test"
reformatted_data = []


def make_direct_output_prompt(code, input):
    return f"""You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. 

[PYTHON]
{code}
assert f({input}) == ??
[/PYTHON]
"""


subset_data = load_dataset(data_name, split=split)
for index, item in enumerate(subset_data):
    reformatted_data.append(
        {
            "id": f"crux-{split}-#{index}",
            "question": make_direct_output_prompt(item["code"], item["input"]).strip(),
            "answer": item["output"],
            "source": data_name,
            "config": split,
            "task_type": "qa",
        }
    )

print(len(reformatted_data))

# Upload to HF
target_dataset_name = "flydust/zero-eval"
target_dataset_config = "crux"
target_dataset_split = "test"

dataset = Dataset.from_list(reformatted_data)
dataset.push_to_hub(
    repo_id=target_dataset_name,
    config_name=target_dataset_config,
    split=target_dataset_split,
    token=os.environ.get("HUGGINGFACE_TOKEN"),
    commit_message=f"Update the merged results.",
)
