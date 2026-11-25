"""Load canonical task rows into the rollout worker's execution contract."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marinskyrl.agent_harnesses import PI_HARNESS
from omegaconf import OmegaConf
from rolloutengine.harness.assistant_prefix import ASSISTANT_PREFIX_HARNESS
from rolloutengine.validation import validate_spec
from shellbox.backends.daytona.machine import DaytonaMachineFactory
from shellbox.backends.local.machine import check_local_environment
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, ExecutionRequirements, Machine, MachineSpec
from skyrl_gym.rollout_specs import is_shell_harness
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.tasks import SourceTaskDataset
from skyrl_train.task_inputs import TaskInput, task_messages
from taskcompendium.convert.structure_closure import structure_closure_task
from taskcompendium.models import (
    GraderMachine,
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    ResourceGroups,
    ScriptGrader,
    Source,
    TextMessage,
)
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.resources import inline_resource
from tests.cpu.task_specs import NATIVE_ENVIRONMENT, SHELL_ENVIRONMENT, machine_runtime, text_task
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast


class LocalAdmissionFactory:
    backend = Backend.LOCAL

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        check_local_environment(requirements, spec)

    async def create(self, spec: MachineSpec) -> Machine:
        raise AssertionError(f"Admission must not acquire a machine: {spec}")


def _load(tmp_path: Path, tasks, **task_spec_overrides) -> list[TaskInput]:
    path = tmp_path / "tasks.parquet"
    pq.write_table(pa.Table.from_pylist([{"task_spec": task.model_dump_json()} for task in tasks]), path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
        chat_template="{% for message in messages %}{{ message.content }} {% endfor %}",
    )
    configs = OmegaConf.to_container(get_default_config().environment.task_sessions, resolve=True)
    configs["task_spec"].update(task_spec_overrides)
    dataset = SourceTaskDataset([str(path)], tokenizer, 128, num_workers=1, environment_configs=configs)
    return [row["input"] for row in dataset.collate_fn(list(dataset))]


FACTORIES = {
    "daytona": DaytonaMachineFactory(client_factory=lambda: pytest.fail("No sandbox may be created")),
    "local": LocalAdmissionFactory(),
}


def test_a_dataset_machine_attaches_only_to_rows_that_need_one(tmp_path: Path):
    shell = text_task("shell-task", source="probes").model_copy(update={"environment_requirements": SHELL_ENVIRONMENT})
    output = text_task("output-task", source="probes").model_copy(update={"output_paths": ("/workspace/answer.txt",)})
    reply = text_task("reply-task", source="probes")
    runtime = machine_runtime("shellsim")
    inputs = _load(tmp_path, (shell, output, reply), task_machine_by_dataset={"probes": runtime.model_dump()})
    loaded = [item.spec for item in inputs]
    assert [spec.task_machine for spec in loaded] == [runtime, runtime, None]
    for spec in (loaded[0], loaded[2]):
        validate_spec(spec, {"shellsim": ShellSimMachineFactory()})


def test_command_semantics_select_the_machine_for_a_mixed_dataset(tmp_path: Path):
    simulated = text_task("simulated-task", source="mixed").model_copy(
        update={"environment_requirements": SHELL_ENVIRONMENT}
    )
    linux = text_task("linux-task", source="mixed").model_copy(update={"environment_requirements": NATIVE_ENVIRONMENT})
    reply = text_task("reply-task", source="mixed")
    shellsim, daytona = machine_runtime("shellsim"), machine_runtime("daytona")
    inputs = _load(
        tmp_path,
        (simulated, linux, reply),
        task_machine_by_command_semantics={
            CommandSemantics.SHELL_SIMULATOR.value: shellsim.model_dump(),
            CommandSemantics.LINUX_PROCESS.value: daytona.model_dump(),
        },
    )
    loaded = [item.spec for item in inputs]
    assert [spec.task_machine for spec in loaded] == [shellsim, daytona, None]
    for spec in loaded:
        validate_spec(spec, {"shellsim": ShellSimMachineFactory(), **FACTORIES})


def test_dataset_harness_binding_keeps_agent_and_assistant_prefix_execution(tmp_path: Path):
    agent = text_task("agent-task", source="agent-source")
    ordinary = text_task("ordinary-task", source="ordinary-source")
    prefix = agent.model_copy(
        update={
            "id": "prefix-task",
            "context": ConversationInput(
                events=(TextMessage(role="user", content="Complete"), TextMessage(role="assistant", content='{"a":['))
            ),
        }
    )

    inputs = _load(tmp_path, (agent, ordinary, prefix), harness_by_dataset={"agent-source": PI_HARNESS})
    loaded = [item.spec for item in inputs]

    assert loaded[0].harness == PI_HARNESS
    assert is_shell_harness(loaded[1].harness)
    assert loaded[2].harness == ASSISTANT_PREFIX_HARNESS


def test_canonical_rows_lower_at_load_and_admit_image_and_package_graders(tmp_path: Path):
    image = text_task("image-task", source="nvidia/Nemotron-RL-Ultra-Training-Blends").model_copy(
        update={
            "labels": {"subset_id": "competitive-coding"},
            "grader": ScriptGrader(
                argv=("sh", "/tests/grade.sh"),
                placement=GraderMachine(
                    environment=EnvironmentRequirements(
                        command_semantics=CommandSemantics.LINUX_PROCESS,
                        docker_image="fixture@sha256:" + "0" * 64,
                    )
                ),
            ),
        }
    )
    prefix = text_task("prefix-task", source="structure-closure").model_copy(
        update={
            "context": ConversationInput(
                events=(TextMessage(role="user", content="Complete"), TextMessage(role="assistant", content='{"a":['))
            ),
            "grader": ScriptGrader(
                argv=("python", "/tests/grade.py"),
                placement=GraderMachine(
                    environment=EnvironmentRequirements(
                        command_semantics=CommandSemantics.LINUX_PROCESS,
                        packages_lock="s3://fixture/grader.lock",
                    )
                ),
            ),
        }
    )
    inputs = _load(tmp_path, (image, prefix))
    assert [(item.metadata.task_id, item.metadata.subset_id) for item in inputs] == [
        ("image-task", "competitive-coding"),
        ("prefix-task", "structure-closure"),
    ]
    loaded = [item.spec for item in inputs]
    assert is_shell_harness(loaded[0].harness)
    assert loaded[1].harness == ASSISTANT_PREFIX_HARNESS
    for spec in loaded:
        validate_spec(spec, FACTORIES)


def test_curated_structure_closure_row_keeps_its_prefix_last(tmp_path: Path):
    extras = {"prefix": '{"a":[1', "expected": '{"a":[1,2]}', "format": "json"}
    config = inline_resource("config.json", json.dumps({"extras": extras}).encode())
    native = text_task("native-id", source="structure-closure").model_copy(
        update={"resources": ResourceGroups(verifier=(config,))}
    )
    payload = {
        "task": {**native.model_dump(mode="json"), "grader": {"kind": "session"}},
        "runtime": {"task_machine": None, "verifier_machine": None},
        "session": {"task_session": "structure_closure", "max_turns": 1},
    }
    # Curation sets source.dataset to the source repository, so it is not "structure-closure".
    release = Source(dataset="published/release", revision="b" * 40, row="train.parquet:7", importer_revision="1")
    curated = structure_closure_task(RawRow("curated-id", release, {"lowered_task_spec": json.dumps(payload)})).task

    spec = _load(tmp_path, (curated,))[0].spec

    assert task_messages(spec)[-1] == {"role": "assistant", "content": '{"a":[1'}
    assert spec.harness == ASSISTANT_PREFIX_HARNESS
    validate_spec(spec, {})
