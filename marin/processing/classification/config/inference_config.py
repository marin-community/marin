import yaml
import dataclasses

from marin.core.runtime import TaskConfig


class StorageConfig:
    def __init__(self, gcs_bucket_name: str, gcs_blob_name: str, hf_repo_id: str, hf_filename: str, local_filepath: str):
        self.gcs_bucket_name = gcs_bucket_name
        self.gcs_blob_name = gcs_blob_name
        self.hf_repo_id = hf_repo_id
        self.hf_filename = hf_filename
        self.local_filepath = local_filepath


class RuntimeConfig:
    def __init__(self, requirements_filepath: str, memory_limit_gb: int, tpu_resources_per_task: int):
        self.requirements_filepath = requirements_filepath
        self.memory_limit_gb = memory_limit_gb
        self.tpu_resources_per_task = self.set_tpu_resources(tpu_resources_per_task)

    def set_tpu_resources(self, tpu_resources_per_task: int):
        if tpu_resources_per_task > 0:
            resources = {"TPU": tpu_resources_per_task}
        else:
            resources = {}

        return resources


class InferenceConfig:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        model_name: str,
        attribute_name: str,
        storage: StorageConfig,
        runtime: RuntimeConfig,
        task: TaskConfig,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_name = model_name
        self.attribute_name = attribute_name
        self.storage = storage
        self.runtime = runtime
        self.task = task

    @staticmethod
    def from_yaml(path):
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if "storage" in data:
            storage = StorageConfig(**data["storage"])
        else:
            storage = None

        if "runtime" in data:
            runtime = RuntimeConfig(**data["runtime"])
        else:
            runtime = None

        if "task" in data:
            task = TaskConfig(**data["task"])
        else:
            task = None

        return InferenceConfig(
            input_dir=data["input_dir"],
            output_dir=data["output_dir"],
            model_name=data["model_name"],
            attribute_name=data["attribute_name"],
            storage=storage,
            runtime=runtime,
            task=task,
        )
