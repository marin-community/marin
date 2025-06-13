from instruction_datasets import INSTRUCTION_DATASET_NAME_TO_CONFIG, download_dataset_step, transform_dataset_step
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)




def main():
    return


if __name__ == "__main__":
    dataset_names = ["open-thoughts/OpenThoughts3-1.2M"]
    all_steps = []
    for dataset_name in dataset_names:
        config = INSTRUCTION_DATASET_NAME_TO_CONFIG[dataset_name]
        downloaded_dataset = download_dataset_step(config)
        all_steps.append(downloaded_dataset)
        transformed_dataset = transform_dataset_step(config, downloaded_dataset)
        all_steps.append(transformed_dataset)

    executor_main(steps=all_steps)
