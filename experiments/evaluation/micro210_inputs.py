# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind frozen Micro210 task resources to a new judge without changing task membership."""

import json
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
import tomli_w
from rolloutengine.agents.pi.agent import PI_HARNESS
from rolloutengine.models import MachineRuntimeSpec, TaskRolloutSpec
from taskcompendium.importers.harbor.config import HarborTaskConfig, read_task_config
from taskcompendium.importers.harbor.package import HarborTask, JobSettings, PackageTests, lower_package
from taskcompendium.models import ScriptGrader, TaskResource, TaskSpec
from taskcompendium.runtime.resources import inline_resource, resource_bytes
from verifyit.spec import JudgeSpec

from experiments.evaluation.judge_config import JudgeConfig

RESOURCE_DIRECTORY = Path(__file__).parent

JUDGE_COMMAND = (
    "sh",
    "-c",
    'export JUDGE_BASE_URL="$VERIFYIT_JUDGE_BASE_URL" JUDGE_API_KEY="$VERIFYIT_JUDGE_API_KEY" '
    'OPENAI_BASE_URL="$VERIFYIT_JUDGE_BASE_URL" OPENAI_API_KEY="$VERIFYIT_JUDGE_API_KEY" '
    'MODEL_NAME="$VERIFYIT_JUDGE_MODEL"; exec "$@"',
    "micro210-judge",
)


TASK_OWNED_JUDGE_REFERENCE = "task-owned rubric"


def bind_script_judge(grader: ScriptGrader, model: str, request_timeout: float) -> ScriptGrader:
    """Declare the judge transport while the frozen script retains its rubric."""
    return grader.model_copy(
        update={
            "argv": (*JUDGE_COMMAND, *grader.argv),
            "judge": JudgeSpec(model=model, references=(TASK_OWNED_JUDGE_REFERENCE,), request_timeout=request_timeout),
        }
    )


@dataclass(frozen=True)
class NativePackage:
    lowered: HarborTask
    settings: HarborTaskConfig
    judge_required: bool


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
        needs_judge = "OPENAI_API_KEY" in judge_env
        if needs_judge:
            config["verifier"]["env"] = {
                key: value
                for key, value in judge_env.items()
                if key not in {"OPENAI_API_KEY", "OPENAI_BASE_URL", "MODEL_NAME"}
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
        # Only equivalence-judged rows need a provider connection. The task's script retains its rubric.
        if any(resource.path == "config.json" for resource in task.resources.verifier):
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
    native = native.model_copy(update={"tags": task.tags, "labels": task.labels, "grader": grader})
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


def evalchemy_judge_resources(files: dict[str, TaskResource], judge: JudgeConfig) -> dict[str, bytes]:
    """Bind the Evalchemy task's transport and model while retaining its rubric."""
    config = json.loads(resource_bytes(files["config.json"]))
    config.update(judge_model=judge.model, judge=asdict(judge))
    source = resource_bytes(files["grade.py"]).decode()
    bindings = {
        "from eval.execution.judge import JudgeBackend": (
            "from micro210_together_judge import TogetherJudgeBackend as JudgeBackend"
        ),
        "JudgeBackend(judge, session)": 'JudgeBackend(judge, session, policy=config["judge"])',
    }
    for original, replacement in bindings.items():
        if source.count(original) != 1:
            raise ValueError(f"Frozen Micro210 grade.py changed its judge binding: {original}")
        source = source.replace(original, replacement)
    return {
        "config.json": json.dumps(config).encode(),
        "grade.py": source.encode(),
        "micro210_together_judge.py": (RESOURCE_DIRECTORY / "together_judge.py").read_bytes(),
        "micro210_judge_config.py": (RESOURCE_DIRECTORY / "judge_config.py").read_bytes(),
    }


def bixbench_judge_resources(files: dict[str, TaskResource], judge: JudgeConfig) -> dict[str, bytes]:
    """Bind BixBench's JSON-schema client to the same judge and explicit deadlines."""
    config = tomllib.loads(resource_bytes(files["task.toml"]).decode())
    config["verifier"]["env"].update(
        OPENAI_API_KEY="${OPENAI_API_KEY}", OPENAI_BASE_URL=judge.base_url, MODEL_NAME=judge.model
    )
    source = resource_bytes(files["tests/llm_judge.py"]).decode()
    bindings = {
        'OpenAI(api_key=os.getenv("OPENAI_API_KEY"))': (
            f'OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout={judge.bixbench_request_timeout!r}, '
            f"max_retries={judge.bixbench_transport_attempts - 1})"
        ),
        "max_tokens=1024,": (
            f"max_tokens={judge.bixbench_max_tokens}, temperature={judge.bixbench_temperature!r}, "
            f"top_p={judge.top_p!r}, "
            f"reasoning_effort={str(judge.reasoning_effort)!r},"
        ),
    }
    for original, replacement in bindings.items():
        if source.count(original) != 1:
            raise ValueError(f"Frozen BixBench client changed its judge binding: {original}")
        source = source.replace(original, replacement)
    return {"task.toml": tomli_w.dumps(config).encode(), "tests/llm_judge.py": source.encode()}


def task_with_judge(task: TaskSpec, judge: JudgeConfig) -> TaskSpec:
    """Replace only judge configuration and transport resources in a task package."""
    files = {resource.path: resource for resource in task.resources.verifier}
    replacements = {}
    if "config.json" in files:
        replacements.update(evalchemy_judge_resources(files, judge))
    if "bixbench" in task.id:
        replacements.update(bixbench_judge_resources(files, judge))
    if not replacements:
        return task
    resources = tuple(resource for resource in task.resources.verifier if resource.path not in replacements) + tuple(
        inline_resource(path, content) for path, content in replacements.items()
    )
    return task.model_copy(update={"resources": task.resources.model_copy(update={"verifier": resources})})


def rebind_judge_inputs(source: str, destination: str, judge: JudgeConfig) -> None:
    """Write a derived table with the same rows, prompts, answers and rollout settings."""
    table = pq.read_table(source)
    rows = []
    for row in table.to_pylist():
        rollout = TaskRolloutSpec.model_validate_json(row["task_rollout_spec"])
        task = task_with_judge(rollout.task, judge)
        rollout = rollout.model_copy(update={"task": task})
        rows.append({**row, "task_spec": task.model_dump_json(), "task_rollout_spec": rollout.model_dump_json()})
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), destination)
