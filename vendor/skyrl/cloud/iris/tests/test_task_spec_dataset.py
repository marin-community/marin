"""Load canonical task rows into the rollout worker's execution contract."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marinskyrl.agent_harnesses import PI_HARNESS
from marinskyrl.task_sources import PreparedTaskCacheSource
from omegaconf import OmegaConf
from rolloutengine.harness.assistant_prefix import ASSISTANT_PREFIX_HARNESS
from rolloutengine.validation import validate_spec
from shellbox.backends.daytona.machine import DaytonaMachineFactory, DaytonaNetworkMode, DaytonaNetworkPolicy
from shellbox.backends.docker.machine import DockerMachineFactory
from shellbox.backends.local.support import check_local_environment
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Backend, Machine, MachineSpec, NetworkPolicy, PackageSpec, TaskRequirements
from skyrl_gym.rollout_specs import is_shell_harness
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.tasks import SourceTaskDataset
from skyrl_train.dataset.machine_routing import tier_machine
from skyrl_train.task_inputs import TaskInput, task_messages
from taskcompendium.runtime.requirements import ROUTE_BACKENDS_LABEL
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
from tests.cpu.task_specs import NATIVE_ENVIRONMENT, SHELL_ENVIRONMENT, machine_runtime, text_task, write_selection_panel
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from experiments.post_training.rl_agentic_recipe import TIER_MACHINES


class LocalAdmissionFactory:
    backend = Backend.LOCAL

    def check_environment(self, requirements: TaskRequirements, spec: MachineSpec) -> None:
        check_local_environment(requirements, spec)

    async def create(self, spec: MachineSpec) -> Machine:
        raise AssertionError(f"Admission must not acquire a machine: {spec}")


def tokenizer():
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
        chat_template="{% for message in messages %}{{ message.content }} {% endfor %}",
    )


def _load(tmp_path: Path, tasks, *, machine_factories=None, **task_spec_overrides) -> list[TaskInput]:
    path = tmp_path / "tasks.parquet"
    pq.write_table(pa.Table.from_pylist([{"task_spec": task.model_dump_json()} for task in tasks]), path)
    configs = OmegaConf.to_container(get_default_config().environment.task_sessions, resolve=True)
    configs["task_spec"].update(task_spec_overrides)
    dataset = SourceTaskDataset(
        [str(path)],
        tokenizer(),
        128,
        num_workers=1,
        environment_configs=configs,
        machine_factories=FACTORIES if machine_factories is None else machine_factories,
    )
    return [row["input"] for row in dataset.collate_fn(list(dataset))]


FACTORIES = {
    "shellsim": ShellSimMachineFactory(),
    "daytona": DaytonaMachineFactory(client_factory=lambda: pytest.fail("No sandbox may be created")),
    "local": LocalAdmissionFactory(),
}


def test_tier_routing_obeys_the_configured_factory_network_policy(tmp_path):
    task = text_task("networked", source="mixed").model_copy(
        update={"environment_requirements": NATIVE_ENVIRONMENT.model_copy(update={"capabilities": ("network",)})}
    )
    factories = {
        "daytona": DaytonaMachineFactory(network_policy=DaytonaNetworkPolicy(DaytonaNetworkMode.BLOCK_ALL)),
        "docker": DockerMachineFactory(),
    }
    loaded = _load(
        tmp_path,
        (task,),
        machine_factories=factories,
        task_machine_by_tier={backend: machine_runtime(backend).model_dump() for backend in factories},
    )
    assert loaded[0].spec.task_machine.backend == "docker"
    validate_spec(loaded[0].spec, factories)


def test_offline_tier_admission_does_not_require_provider_credentials(monkeypatch):
    monkeypatch.delenv("SHELLBOX_OFFLINE_TEST_API_KEY", raising=False)
    runtime = machine_runtime("daytona").model_dump()
    selected = tier_machine(
        NATIVE_ENVIRONMENT,
        {"daytona": runtime},
        (),
        {"daytona": {"api_key_env": "SHELLBOX_OFFLINE_TEST_API_KEY", "ttl_minutes": 60}},
        None,
    )
    assert selected == runtime


def test_prepared_routes_follow_changed_provider_configuration(tmp_path):
    selection = tmp_path / "selection"
    selection.mkdir()
    task = text_task("fixture", source="fixture").model_copy(update={"environment_requirements": NATIVE_ENVIRONMENT})
    write_selection_panel(selection, [task], "fixture", "1", lambda _: {"family": "math"})
    cache = PreparedTaskCacheSource(str(selection / "validation.parquet"), "fixture@1", str(tmp_path / "prepared"))
    configs = OmegaConf.to_container(get_default_config().environment.task_sessions, resolve=True)
    configs["task_spec"]["task_machine_by_tier"] = {"daytona": machine_runtime("daytona").model_dump(mode="json")}
    configs["daytona"] = {"api_key_env": "SHELLBOX_OFFLINE_TEST_API_KEY", "ttl_minutes": 60}
    first = SourceTaskDataset(
        [cache.path], tokenizer(), 128, num_workers=1, environment_configs=configs, prepared_cache=cache
    )
    before = first.collate_fn(list(first))[0]["input"].spec.task_machine
    assert list(Path(cache.cache_dir).glob("*.parquet"))
    configs["daytona"] = None
    second = SourceTaskDataset(
        [cache.path], tokenizer(), 128, num_workers=1, environment_configs=configs, prepared_cache=cache
    )
    after = second.collate_fn(list(second))[0]["input"].spec.task_machine
    assert before.backend == "daytona"
    assert after is None


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
        task_machine_by_tier={
            Backend.SHELLSIM.value: shellsim.model_dump(),
            Backend.DAYTONA.value: daytona.model_dump(),
        },
    )
    loaded = [item.spec for item in inputs]
    assert [spec.task_machine for spec in loaded] == [shellsim, daytona, None]
    for spec in loaded:
        validate_spec(spec, {"shellsim": ShellSimMachineFactory(), **FACTORIES})


def test_tier_routing_fails_up_to_the_next_registered_tier(tmp_path: Path):
    """A simulator task that needs the network skips ShellSim; an unregistered cheaper tier is skipped too."""
    networked_simulator = text_task("networked-simulator", source="mixed").model_copy(
        update={
            "environment_requirements": EnvironmentRequirements(
                command_semantics=CommandSemantics.SHELL_SIMULATOR,
                capabilities=("shell", "filesystem", "network"),
                package_spec=PackageSpec(pypi_packages=("httpx==0.28.0",)),
            )
        }
    )
    native = text_task("native-task", source="mixed").model_copy(
        update={
            "environment_requirements": NATIVE_ENVIRONMENT.model_copy(update={"capabilities": ("shell", "filesystem")})
        }
    )
    shellsim, daytona, local = (machine_runtime(backend) for backend in ("shellsim", "daytona", "local"))
    # The simulator's ladder skips ShellSim (it serves no ALLOW network) and binds local with the
    # network its declared capability needs.
    simulated_load = _load(
        tmp_path,
        (networked_simulator,),
        task_machine_by_tier={
            Backend.SHELLSIM.value: shellsim.model_dump(),
            Backend.LOCAL.value: local.model_dump(),
        },
    )
    # The native image task's ladder starts at local, which this deployment does not register,
    # so Daytona serves.
    native_load = _load(
        tmp_path,
        (native,),
        task_machine_by_tier={
            Backend.SHELLSIM.value: shellsim.model_dump(),
            Backend.DAYTONA.value: daytona.model_dump(),
        },
    )
    networked_local = local.model_copy(update={"network": NetworkPolicy.ALLOW})
    assert [spec.spec.task_machine for spec in simulated_load] == [networked_local]
    assert [spec.spec.task_machine for spec in native_load] == [daytona]
    for spec in (*simulated_load, *native_load):
        validate_spec(spec.spec, {"shellsim": ShellSimMachineFactory(), **FACTORIES})


def test_tier_routing_keeps_the_template_network_unless_the_task_declares_network(tmp_path: Path):
    """A tier template keeps its own network policy; a declared network capability binds ALLOW."""
    plain = text_task("plain", source="mixed").model_copy(
        update={
            "environment_requirements": NATIVE_ENVIRONMENT.model_copy(update={"capabilities": ("shell", "filesystem")})
        }
    )
    networked = text_task("networked", source="mixed").model_copy(
        update={
            "environment_requirements": NATIVE_ENVIRONMENT.model_copy(
                update={"capabilities": ("shell", "filesystem", "network")}
            )
        }
    )
    local = machine_runtime("local")
    loaded = _load(
        tmp_path,
        (plain, networked),
        task_machine_by_tier={Backend.LOCAL.value: local.model_dump()},
    )
    assert [spec.spec.task_machine.network for spec in loaded] == [NetworkPolicy.DENY, NetworkPolicy.ALLOW]


def test_tier_routing_binds_no_machine_when_no_registered_tier_satisfies(tmp_path: Path):
    """A lock-sourced Linux task is a local-tier task; without a registered local tier it binds none."""
    lock_task = text_task("lock-task", source="mixed").model_copy(
        update={
            "environment_requirements": EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS,
                capabilities=("shell", "filesystem"),
                packages_lock="s3://bucket/lock",
            )
        }
    )
    loaded = _load(
        tmp_path,
        (lock_task,),
        task_machine_by_tier={
            Backend.SHELLSIM.value: machine_runtime("shellsim").model_dump(),
            Backend.DAYTONA.value: machine_runtime("daytona").model_dump(),
        },
    )
    assert [spec.spec.task_machine for spec in loaded] == [None]


@pytest.mark.parametrize("registered, expected", [("local", "local"), ("daytona", None)])
def test_recorded_routes_require_a_registered_candidate(tmp_path: Path, registered, expected):
    task = text_task("routed", source="mixed").model_copy(
        update={
            "environment_requirements": EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS,
                capabilities=("shell", "filesystem"),
                docker_image="fixture@sha256:" + "0" * 64,
            ),
            "labels": {ROUTE_BACKENDS_LABEL: json.dumps([Backend.LOCAL.value])},
        }
    )
    loaded = _load(tmp_path, (task,), task_machine_by_tier={registered: machine_runtime(registered).model_dump()})
    machine = loaded[0].spec.task_machine
    assert (None if machine is None else machine.backend) == expected


def test_shipped_recipe_preserves_machine_binding_after_local_verification(tmp_path):
    task = text_task("locally-verified", source="mixed").model_copy(
        update={
            "environment_requirements": NATIVE_ENVIRONMENT,
            "labels": {ROUTE_BACKENDS_LABEL: json.dumps(["local", "daytona"])},
        }
    )
    loaded = _load(
        tmp_path,
        (task,),
        task_machine_by_tier={machine.backend: machine.model_dump(mode="json") for machine in TIER_MACHINES},
    )
    assert loaded[0].spec.task_machine.backend == "daytona"
    validate_spec(loaded[0].spec, FACTORIES)


@pytest.mark.parametrize("explicit_backend, expected", [(None, "local"), ("shellsim", None), ("local", "local")])
def test_recorded_candidates_exclude_the_cheaper_backend_that_failed_controls(tmp_path, explicit_backend, expected):
    task = text_task("failed-simulator", source="mixed").model_copy(
        update={
            "environment_requirements": SHELL_ENVIRONMENT,
            "labels": {ROUTE_BACKENDS_LABEL: json.dumps(["local"])},
        }
    )
    loaded = _load(
        tmp_path,
        (task,),
        task_machine_by_tier={backend: machine_runtime(backend).model_dump() for backend in ("shellsim", "local")},
        task_machine_by_dataset=(
            {} if explicit_backend is None else {"mixed": machine_runtime(explicit_backend).model_dump()}
        ),
    )
    machine = loaded[0].spec.task_machine
    assert (None if machine is None else machine.backend) == expected
    if machine is None:
        with pytest.raises(ValueError):
            validate_spec(loaded[0].spec, FACTORIES)
    else:
        validate_spec(loaded[0].spec, FACTORIES)


@pytest.mark.parametrize("unsupported", [{"cpus": 1}, {"network": NetworkPolicy.ALLOW}])
def test_tier_routing_skips_unsupported_deployment_templates(tmp_path: Path, unsupported):
    task = text_task("simulator", source="mixed").model_copy(update={"environment_requirements": SHELL_ENVIRONMENT})
    simulated = machine_runtime("shellsim").model_copy(update=unsupported)
    local = machine_runtime("local")
    loaded = _load(
        tmp_path, (task,), task_machine_by_tier={"shellsim": simulated.model_dump(), "local": local.model_dump()}
    )
    assert loaded[0].spec.task_machine == local
    validate_spec(loaded[0].spec, FACTORIES)


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
