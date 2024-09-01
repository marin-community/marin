from dataclasses import dataclass, field

from marin.core.runtime import TaskConfig


@dataclass
class RuntimeConfig:
    requirements_filepath: str
    memory_limit_gb: int
    tpu_resources_per_task: int

    @property
    def ray_resources(self):
        if self.tpu_resources_per_task > 0:
            resources = {"TPU": self.tpu_resources_per_task}
        else:
            resources = {}

        return resources


@dataclass
class InferenceConfig:
    input_dir: str
    output_dir: str
    model_name: str
    attribute_name: str
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
