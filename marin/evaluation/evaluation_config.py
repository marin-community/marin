from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvaluationConfig:
    evaluator: str
    """Name of the evaluator to run."""

    model_name: str
    """
    Can be a name of the model in Hugging Face (e.g, google/gemma-2b) or
    a name given to the model checkpoint (e.g., $RUN/$CHECKPOINT).
    """

    evaluation_path: str = "tmp/output"
    """
    Where to write results to. Can be a local path (e.g., /path/to/output) or
    a path on GCS (e.g., gs://bucket/path/to/output).
    """

    evals: list[str] = field(default_factory=list)
    """
    Which specific evals within an evaluation harness to run. This would be a list of
    tasks in for EleutherAI's lm-evaluation-harness or a list of evals from HELM (e.g., mmlu, lite, etc.).
    See https://github.com/stanford-crfm/helm/tree/main/src/helm/benchmark/presentation for the full list.
    """

    model_path: str | None = None
    """
    Optional: Path to the model. Can be a path on GCS.
    """

    max_eval_instances: int | None = None
    """
    Maximum number of instances to evaluate on.
    """

    launch_with_ray: bool = True
    """
    Whether to launch the evaluation run with Ray.
    """
