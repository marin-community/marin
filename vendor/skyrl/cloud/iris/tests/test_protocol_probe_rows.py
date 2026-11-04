"""A dataset's canonical rows take the turn limit and task machine the launch configuration gives it."""

from pathlib import Path

import yaml
from rolloutengine.models import TaskRolloutSpec
from rolloutengine.test_support import arithmetic_task
from skyrl_train.dataset.tasks import TASK_ROLLOUT_COLUMN, TASK_SPEC_COLUMN, _canonical_task_prompt
from taskcompendium.examples.protocol_probes import (
    PROBE_DATASET,
    PROBE_TURNS,
    SHELLSIM_MACHINE,
    lock_grader_environment,
    probe_check,
    probe_tasks,
)

from cloud.iris.launch_config import LaunchTopology
from cloud.iris.rl_config_translation import compose_skyrl_config, parse_rl_config

PROBE_SETTINGS = {
    "task_spec": {
        "max_turns_by_dataset": {PROBE_DATASET: PROBE_TURNS},
        "task_machine_by_dataset": {PROBE_DATASET: SHELLSIM_MACHINE},
    }
}


def task_sessions(tmp_path: Path):
    recipe = Path(__file__).resolve().parents[1] / "configs/qwen_megatron_smoke.yaml"
    raw = yaml.safe_load(recipe.read_text())
    raw["environment"] = {"task_sessions": PROBE_SETTINGS}
    path = tmp_path / "recipe.yaml"
    path.write_text(yaml.safe_dump(raw))
    topology = LaunchTopology(num_nodes=1, gpus_per_node=8, gpu_variant="H100")
    return compose_skyrl_config(parse_rl_config(str(path)), {}, topology).config.environment.task_sessions


def lowered(task, settings) -> TaskRolloutSpec:
    row = _canonical_task_prompt({TASK_SPEC_COLUMN: task.model_dump_json()}, 0, environment_configs=settings)
    return TaskRolloutSpec.model_validate_json(row[TASK_ROLLOUT_COLUMN])


def test_probe_rows_get_their_turns_and_a_machine_where_needed_while_other_rows_keep_the_defaults(tmp_path):
    settings = task_sessions(tmp_path)
    probes = {
        probe_check(task): lowered(task, settings)
        for task in probe_tasks(grader_environment=lock_grader_environment("/locks/grader.lock"), variants=1)
    }

    assert {kind: spec.limits.max_turns for kind, spec in probes.items()} == dict.fromkeys(probes, PROBE_TURNS)
    machines = {kind: None if spec.task_machine is None else spec.task_machine.backend for kind, spec in probes.items()}
    assert machines == {
        "required_file": "shellsim",
        "tool_execution": "shellsim",
        "thinking_disabled": None,
        "url_loading": None,
    }
    assert probes["required_file"].verifier_machine is not None
    assert probes["required_file"].verifier_machine.backend == settings.task_spec.package_verifier.backend

    plain = lowered(arithmetic_task(), settings)
    assert (plain.limits.max_turns, plain.task_machine) == (1, None)
