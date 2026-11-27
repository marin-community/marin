# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lower frozen Micro210 packages into native Pi rollouts without changing task membership.

The lowering keeps each package's provisioning, verifier, and grading semantics, binds the
runtime judge transport where a frozen grading plan consults the judge, and writes the derived
native task table with its binding receipt.
"""

import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
import tomli_w
from rolloutengine.agents.pi.agent import PI_HARNESS
from rolloutengine.models import MachineRuntimeSpec, TaskRolloutSpec
from taskcompendium.importers.harbor.config import HarborTaskConfig, read_task_config
from taskcompendium.importers.harbor.package import HarborTask, JobSettings, PackageTests, lower_package
from taskcompendium.models import ScriptGrader, TaskSpec
from taskcompendium.runtime.resources import resource_bytes
from verifyit.spec import JudgeSpec

JUDGE_CREDENTIAL_ENV = "OPENAI_API_KEY"
"""The frozen verifier env variable whose presence declares a judge connection."""

JUDGE_CONNECTION_ENV = (JUDGE_CREDENTIAL_ENV, "OPENAI_BASE_URL", "MODEL_NAME")
"""Judge connection variables a frozen verifier environment declares; the runtime strips them
from ``task.toml`` and re-supplies them through ``JUDGE_COMMAND``."""

JUDGE_ENV_BINDINGS = (
    ("JUDGE_BASE_URL", "VERIFYIT_JUDGE_BASE_URL"),
    ("JUDGE_API_KEY", "VERIFYIT_JUDGE_API_KEY"),
    ("OPENAI_BASE_URL", "VERIFYIT_JUDGE_BASE_URL"),
    ("OPENAI_API_KEY", "VERIFYIT_JUDGE_API_KEY"),
    ("MODEL_NAME", "VERIFYIT_JUDGE_MODEL"),
)
"""The judge connection names frozen scripts read, mapped to the runtime transport variables."""

JUDGE_COMMAND = (
    "sh",
    "-c",
    "export " + " ".join(f'{name}="${source}"' for name, source in JUDGE_ENV_BINDINGS) + '; exec "$@"',
    "micro210-judge",
)


def bind_script_judge(grader: ScriptGrader, model: str, request_timeout: float) -> ScriptGrader:
    """Declare the judge transport; the frozen script keeps its own rubric and connection reading."""
    return grader.model_copy(
        update={
            "argv": (*JUDGE_COMMAND, *grader.argv),
            "judge": JudgeSpec(model=model, rubric="", request_timeout=request_timeout),
        }
    )


@dataclass(frozen=True)
class NativePackage:
    lowered: HarborTask
    settings: HarborTaskConfig
    judge_required: bool


JUDGED_GRADER_NAMES = frozenset({"equivalence_judge", "equivalence_label", "judge_failed"})
"""The sealed grader registry names whose grade plan consults the judge."""


def frozen_plan_uses_judge(task: TaskSpec) -> bool:
    """Whether the task's frozen Evalchemy grading plan consults the judge.

    The frozen driver reads the judge connection only for a judged grade plan, and the sealed
    spec records that selection: a judged grader name, or the math equivalence grader's
    reference when the frozen source escalates a mismatched answer to the judge.
    """
    config = next((resource for resource in task.resources.verifier if resource.path == "config.json"), None)
    if config is None:
        return False
    grader = json.loads(resource_bytes(config))["spec"]["grader"]
    return grader["name"] in JUDGED_GRADER_NAMES or bool((grader.get("reference") or {}).get("judge"))


def lower_frozen_package(task: TaskSpec, agent_timeout: float, image_digests: Mapping[str, str]) -> NativePackage:
    """Stage the archived package and replace its host credential references with a runtime judge binding."""
    with TemporaryDirectory(prefix="micro210-native-") as directory:
        package = Path(directory)
        for resource in task.resources.verifier:
            path = package / resource.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(resource_bytes(resource))
            if resource.mode is not None:
                path.chmod(int(resource.mode, 8))
        config_path = package / "task.toml"
        settings = read_task_config(package)
        config = tomllib.loads(config_path.read_text())
        judge_env = config["verifier"].get("env", {})
        needs_judge = JUDGE_CREDENTIAL_ENV in judge_env
        if needs_judge:
            config["verifier"]["env"] = {
                key: value for key, value in judge_env.items() if key not in JUDGE_CONNECTION_ENV
            }
            config_path.write_text(tomli_w.dumps(config))
        lowered = lower_package(
            package,
            task_id=task.id,
            source=task.source,
            rule=PackageTests(),
            image_digests=image_digests,
            job=JobSettings(agent_timeout=agent_timeout),
        )
    return NativePackage(lowered, settings, needs_judge)


def native_rollout(rollout: TaskRolloutSpec, policy: dict, image_digests: Mapping[str, str]) -> TaskRolloutSpec:
    """Lower a frozen package without selecting new tasks or changing its verifier."""
    task = rollout.task
    if rollout.harness != "harbor_trial":
        grader = task.grader
        assert isinstance(grader, ScriptGrader)
        # Only judged grading plans read the judge connection; exact-match and code-execution
        # graders run without provider credentials.
        if frozen_plan_uses_judge(task):
            grader = bind_script_judge(
                grader, policy["evaluation"]["judge"]["model"], policy["evaluation"]["judge"]["request_timeout"]
            )
            task = task.model_copy(update={"grader": grader})
        return rollout.model_copy(update={"task": task})

    harbor = policy["evaluation"]["config"]["terminal_bench"]["harbor"]
    package = lower_frozen_package(task, harbor["override_timeout_sec"], image_digests)
    lowered, settings = package.lowered, package.settings
    native, episode = lowered.task, lowered.episode
    grader = native.grader
    assert isinstance(grader, ScriptGrader)
    if package.judge_required:
        grader = bind_script_judge(
            grader, policy["evaluation"]["judge"]["model"], policy["evaluation"]["judge"]["bixbench_request_timeout"]
        )
    native = native.model_copy(
        update={
            "tags": task.tags,
            "labels": task.labels,
            "grader": grader,
            # The sealed Harbor protocol scores the workspace an agent leaves when it stops at a limit.
            "verifies_stopped_workspace": True,
        }
    )
    machine = MachineRuntimeSpec(
        backend="daytona",
        network=episode.network,
        cpus=episode.cpus,
        memory_mb=episode.memory_mb,
        storage_mb=episode.storage_mb,
        gpus=episode.gpus,
        user=None if settings.agent.user is None else str(settings.agent.user),
        startup_timeout=settings.environment.build_timeout_sec,
        cleanup_timeout=60,
    )
    limits = rollout.limits.model_copy(
        update={
            "total_turn_timeout": episode.agent_timeout,
            "attempt_timeout": harbor["trial_attempt_timeout_sec"],
            "verifier_timeout": episode.verifier_timeout,
        }
    )
    return rollout.model_copy(
        update={
            "task": native,
            "harness": PI_HARNESS,
            "task_machine": machine,
            "limits": limits,
        }
    )


def native_inputs(
    source: str, destination: str, policy: dict, image_digests: dict[str, str], *, rows: tuple[int, ...] = ()
) -> None:
    """Write the native binding of the sealed table in its original row order."""
    table = pq.read_table(source)
    if rows:
        table = table.take(pa.array(rows))
    native_rows = []
    for row in table.to_pylist():
        rollout = native_rollout(TaskRolloutSpec.model_validate_json(row["task_rollout_spec"]), policy, image_digests)
        native_rows.append(
            {**row, "task_spec": rollout.task.model_dump_json(), "task_rollout_spec": rollout.model_dump_json()}
        )
    pq.write_table(pa.Table.from_pylist(native_rows, schema=table.schema), destination)
