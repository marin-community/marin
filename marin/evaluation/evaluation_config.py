from dataclasses import dataclass, field

from experiments.evals.resource_configs import ResourceConfig


@dataclass(frozen=True)
class EvalTaskConfig:
    name: str
    """Name of the evaluation task."""

    num_fewshot: int
    """Number of few-shot examples to evaluate on."""

    task_alias: str | None = None
    """Alias for the task name."""


@dataclass(frozen=True)
class EvaluationConfig:
    evaluator: str
    """Name of the evaluator to run."""

    model_name: str | None
    """
    Can be a name of the model in Hugging Face (e.g, google/gemma-2b) or
    a name given to the model checkpoint (e.g., $RUN/$CHECKPOINT).

    If None, the model_path should be provided and the name will be imputed from the path,
     using Levanter's path conventions. (i.e. $RUN/hf/step-$STEP --> $RUN-$STEP)
    """

    evaluation_path: str = "tmp/output"
    """
    Where to write results to. Can be a local path (e.g., /path/to/output) or
    a path on GCS (e.g., gs://bucket/path/to/output).
    """

    evals: list[EvalTaskConfig] = field(default_factory=list)
    """
    List of specific evals within an evaluation harness to run. This would be a list of
    tasks in for EleutherAI's lm-evaluation-harness or a list of evals from HELM (e.g., mmlu, lite, etc.).
    See https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/presentation, or
    https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks
    for the full list.
    """

    model_path: str | None = None
    """
    Optional: Path to the model. Can be a path on GCS.
    """

    discover_latest_checkpoint: bool = False
    """
    Whether to discover the latest HF checkpoint in the model path.
    """

    launch_with_ray: bool = True
    """
    Whether to launch the evaluation run with Ray.
    """

    max_eval_instances: int | None = None
    """
    Maximum number of evaluation instances to run.
    """

    engine_kwargs: dict | None = None
    """
    Additional keyword arguments to pass to the vLLM engine.
    """

    resource_config: ResourceConfig | None = None
    """
    Additional keyword arguments to pass to the Ray resources.
    """
