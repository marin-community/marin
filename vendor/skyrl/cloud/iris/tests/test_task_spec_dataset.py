"""Load canonical task rows into the rollout worker's execution contract."""

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
from taskcompendium.models import CommandSemantics, ConversationInput, EnvironmentRequirements, ScriptGrader, TextMessage
from tests.cpu.task_specs import text_task
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast


class LocalAdmissionFactory:
    backend = Backend.LOCAL

    def check_environment(self, requirements: ExecutionRequirements, spec: MachineSpec) -> None:
        check_local_environment(requirements, spec)

    async def create(self, spec: MachineSpec) -> Machine:
        raise AssertionError(f"Admission must not acquire a machine: {spec}")


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
    prefix = text_task("prefix-task", source="structure-closure").model_copy(
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
    path = tmp_path / "tasks.parquet"
    pq.write_table(pa.Table.from_pylist([{"task_spec": task.model_dump_json()} for task in (image, prefix)]), path)
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
    loaded = [row["input"].spec for row in dataset.collate_fn(list(dataset))]
    assert [spec.task.id for spec in loaded] == ["image-task", "prefix-task"]
    assert is_shell_harness(loaded[0].harness)
    assert loaded[1].harness == ASSISTANT_PREFIX_HARNESS
    factories = {
        "daytona": DaytonaMachineFactory(client_factory=lambda: pytest.fail("No sandbox may be created")),
        "local": LocalAdmissionFactory(),
    }
    for spec in loaded:
        validate_spec(spec, factories)
