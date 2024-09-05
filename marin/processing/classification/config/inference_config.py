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
    """A path to a model or the name of a model. 
    If it doesn't have the classifier type in its name, you need to specify the model_type."""
    attribute_name: str
    model_type: str | None = None
    """The type of the model. Currently: fasttext, fineweb, or None."""
    runtime: RuntimeConfig = field(default_factory=lambda : RuntimeConfig(requirements_filepath="marin/processing/classification/config/dclm_fasttext_requirements.txt", memory_limit_gb=12))
    task: TaskConfig = field(default_factory=TaskConfig)
