import os

from experiments.metrics.gcp_related import NUM_RESTART_CONFIG, get_number_of_restarts
from experiments.metrics.github_related import (
    GITHUB_API_CONFIG,
    GITHUB_ISSUE_CONFIG,
    get_average_duration_for_all_workflows,
    get_closed_issues_with_label,
)
from marin.execution.executor import ExecutorMainConfig, ExecutorStep, executor_main, output_path_of, this_output_path
from marin.utilities.metrics_utils import MergeConfig, merge

average_duration = ExecutorStep(
    name=os.path.join("metrics", "github", "average_duration"),
    fn=get_average_duration_for_all_workflows,
    config=GITHUB_API_CONFIG(GITHUB_TOKEN=os.getenv("GITHUB_TOKEN"), output_path=this_output_path()),
)

closed_issues = ExecutorStep(
    name=os.path.join("metrics", "github", "closed_issues"),
    fn=get_closed_issues_with_label,
    config=GITHUB_ISSUE_CONFIG(
        GITHUB_TOKEN=os.getenv("GITHUB_TOKEN"),
        output_path=this_output_path(),
        LABEL="experiments",
    ),
)

number_of_restarts = ExecutorStep(
    name=os.path.join("metrics", "gcp", "number_of_restarts"),
    fn=get_number_of_restarts,
    config=NUM_RESTART_CONFIG(output_path=this_output_path()),
    pip_dependency_groups=["google-cloud-logging"],
)

merge = ExecutorStep(
    name=os.path.join("metrics", "merge"),
    fn=merge,
    config=MergeConfig(
        output_path=this_output_path(),
        merge_paths=[
            output_path_of(average_duration),
            output_path_of(closed_issues),
            output_path_of(number_of_restarts),
        ],
    ),
)
if __name__ == "__main__":
    executor_main(
        ExecutorMainConfig(force_run=[closed_issues.name, number_of_restarts.name, merge.name, average_duration.name]),
        [average_duration, closed_issues, number_of_restarts, merge],
    )
