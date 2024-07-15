import yaml
import dataclasses


class StorageConfig:
    def __init__(self, gcs_bucket_name, gcs_blob_name, hf_repo_id, hf_filename, local_filepath):
        self.gcs_bucket_name = gcs_bucket_name
        self.gcs_blob_name = gcs_blob_name
        self.hf_repo_id = hf_repo_id
        self.hf_filename = hf_filename
        self.local_filepath = local_filepath


class RuntimeConfig:
    def __init__(self, requirements_filepath, memory_limit_gb, tpu_resources_per_task):
        self.requirements_filepath = requirements_filepath
        self.memory_limit_gb = memory_limit_gb
        self.tpu_resources_per_task = tpu_resources_per_task


class InferenceConfig:
    def __init__(self, input_dir, output_dir, model_name, storage, runtime, use_ray_data):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.model_name = model_name
        self.storage = storage
        self.runtime = runtime
        self.use_ray_data = use_ray_data

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

        return InferenceConfig(
            input_dir=data["input_dir"],
            output_dir=data["output_dir"],
            model_name=data["model_name"],
            storage=storage,
            runtime=runtime,
            use_ray_data=data["use_ray_data"],
        )
