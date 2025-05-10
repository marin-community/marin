from dataclasses import dataclass, field

from marin.core.runtime import TaskConfig


@dataclass
class RuntimeConfig:
    memory_limit_gb: int
    resources: dict = field(default_factory=dict)


@dataclass
class InferenceConfig:
    input_path: str

    # A path to a model or the name of a model. I f it doesn't have the classifier type in its name, you need to
    # specify the model_type.
    model_name: str
    attribute_name: str

    # The type of the model. Currently: fasttext, fineweb, or None.
    model_type: str | None = None
    output_path: str | None = None

    # Ray runtime config.
    runtime: RuntimeConfig = field(
        default_factory=lambda: RuntimeConfig(
            memory_limit_gb=0.1,
        )
    )

    # Ray task config.
    task: TaskConfig = field(default_factory=TaskConfig)

    # The filetype of the input data.
    filetype: str = "jsonl.gz"

    classifier_kwargs: dict = field(default_factory=dict)
