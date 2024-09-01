from dataclasses import dataclass, field

from marin.core.runtime import TaskConfig


@dataclass
class RuntimeConfig:
    requirements_filepath: str
    memory_limit_gb: int
    resources: dict = field(default_factory=dict)


@dataclass
class InferenceConfig:
    input_dir: str
    output_dir: str
    model_name: str
    attribute_name: str
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
