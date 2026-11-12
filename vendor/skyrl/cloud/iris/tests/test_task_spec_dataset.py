"""Load canonical task rows into the rollout worker's execution contract."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf
from rolloutengine.harness.assistant_prefix import ASSISTANT_PREFIX_HARNESS
from rolloutengine.validation import validate_spec
from shellbox.backends.daytona.machine import DaytonaMachineFactory
from shellbox.backends.local.machine import check_local_environment
from shellbox.machine import Backend, ExecutionRequirements, Machine, MachineSpec
from skyrl_gym.rollout_specs import is_shell_harness
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.tasks import SourceTaskDataset
from skyrl_train.task_inputs import task_messages
from taskcompendium.convert.structure_closure import structure_closure_task
from taskcompendium.models import (
    CommandSemantics,
    ConversationInput,
    EnvironmentRequirements,
    ResourceGroups,
    ScriptGrader,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.models import RawRow
from taskcompendium.runtime.resources import inline_resource
from tests.cpu.task_specs import text_task
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast


class LocalAdmissionFactory:
    backend = Backend.LOCAL

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        check_local_environment(requirements, spec)

    async def create(self, spec: MachineSpec) -> Machine:
        raise AssertionError(f"Admission must not acquire a machine: {spec}")


def _loaded_specs(tmp_path: Path, tasks: list[TaskSpec]) -> list:
    path = tmp_path / "tasks.parquet"
    pq.write_table(pa.Table.from_pylist([{"task_spec": task.model_dump_json()} for task in tasks]), path)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
        chat_template="{% for message in messages %}{{ message.content }} {% endfor %}",
    )
    config = get_default_config()
    dataset = SourceTaskDataset(
        [str(path)],
        tokenizer,
        128,
        num_workers=1,
        environment_configs=OmegaConf.to_container(config.environment.task_sessions, resolve=True),
    )
    return [row["input"].spec for row in dataset.collate_fn(list(dataset))]


def test_canonical_rows_lower_at_load_and_admit_image_and_package_graders(tmp_path: Path):
    image = text_task("image-task").model_copy(
        update={
            "grader": ScriptGrader(
                argv=("sh", "/tests/grade.sh"),
                environment=EnvironmentRequirements(
                    command_semantics=CommandSemantics.LINUX_PROCESS,
                    docker_image="fixture@sha256:" + "0" * 64,
                ),
            )
        }
    )
    prefix = text_task("prefix-task").model_copy(
        update={
            "context": ConversationInput(
                events=(TextMessage(role="user", content="Complete"), TextMessage(role="assistant", content='{"a":['))
            ),
            "grader": ScriptGrader(
                argv=("python", "/tests/grade.py"),
                environment=EnvironmentRequirements(
                    command_semantics=CommandSemantics.LINUX_PROCESS,
                    packages_lock="s3://fixture/grader.lock",
                ),
            ),
        }
    )
    loaded = _loaded_specs(tmp_path, [image, prefix])
    assert [spec.task.id for spec in loaded] == ["image-task", "prefix-task"]
    assert is_shell_harness(loaded[0].harness)
    assert loaded[1].harness == ASSISTANT_PREFIX_HARNESS
    factories = {
        "daytona": DaytonaMachineFactory(client_factory=lambda: pytest.fail("No sandbox may be created")),
        "local": LocalAdmissionFactory(),
    }
    for spec in loaded:
        validate_spec(spec, factories)


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

    spec = _loaded_specs(tmp_path, [curated])[0]

    assert task_messages(spec)[-1] == {"role": "assistant", "content": '{"a":[1'}
    assert spec.harness == ASSISTANT_PREFIX_HARNESS
    validate_spec(spec, {})
