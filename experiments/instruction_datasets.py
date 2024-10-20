from dataclasses import dataclass

from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    output_path_of,
    this_output_path,
)
from operations.download.huggingface.download import DownloadConfig, download
from operations.transform.conversation.transform_conversation import TransformSFTDatasetConfig, transform_dataset


@dataclass
class InstructionDatasetConfig:
    hf_dataset_id: str
    revision: str
    wait_for_completion: bool
    metadata_columns: list[str]
    filetype: str


INSTRUCTION_DATASETS = [
    InstructionDatasetConfig(
        hf_dataset_id="meta-math/MetaMathQA",
        revision="aa4f34d",
        wait_for_completion=True,
        metadata_columns=["type"],
        filetype="json",
    ),
    InstructionDatasetConfig(
        hf_dataset_id="allenai/tulu-v2-sft-mixture",
        revision="6248b17",
        wait_for_completion=True,
        metadata_columns=["dataset", "id"],
        filetype="parquet",
    ),
    InstructionDatasetConfig(
        hf_dataset_id="openbmb/UltraInteract_sft",
        revision="2b102e4",
        wait_for_completion=True,
        metadata_columns=["task", "dataset"],
        filetype="parquet",
    ),
    InstructionDatasetConfig(
        hf_dataset_id="teknium/OpenHermes-2.5",
        revision="b820378",
        wait_for_completion=True,
        metadata_columns=["id"],
        filetype="json",
    ),
]


def create_steps():
    steps = []
    for dataset in INSTRUCTION_DATASETS:
        dataset_name = dataset.hf_dataset_id.replace("/", "--")
        download_step = ExecutorStep(
            name=f"raw/{dataset_name}",
            fn=download,
            config=DownloadConfig(
                hf_dataset_id=dataset.hf_dataset_id,
                revision=dataset.revision,
                gcs_output_path=this_output_path(),
                wait_for_completion=dataset.wait_for_completion,
            ),
        )
        download_data = output_path_of(
            download_step,
            f"{dataset.revision}/huggingface.co/datasets/{dataset.hf_dataset_id}/resolve/{dataset.revision}",
        )

        transform_step = ExecutorStep(
            name=f"documents/{dataset_name}",
            fn=transform_dataset,
            config=TransformSFTDatasetConfig(
                input_path=download_data,
                output_path=this_output_path(),
                shard_size=5000,
                metadata_columns=dataset.metadata_columns,
                filetype=dataset.filetype,
                source=dataset.hf_dataset_id,
            ),
        )

        steps.append(download_step)
        steps.append(transform_step)

    return steps


if __name__ == "__main__":
    steps = create_steps()
    executor_main(steps)
