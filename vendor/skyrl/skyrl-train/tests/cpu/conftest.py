import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

# CPU tests already run in the locked uv environment. Ray's uv hook would package
# this checkout and create another environment for every local Ray session.
os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import ray  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402
import zstandard  # noqa: E402
from marinskyrl.environment_contract import TrainingType  # noqa: E402
from skyrl_train import telemetry as training_telemetry  # noqa: E402
from skyrl_train.distillation import ChosenTokenTeacherEvidence  # noqa: E402
from skyrl_train.trajectory_runners.types import TrajectoryID, VerifierTestCollection  # noqa: E402
from tests.cpu.task_specs import lowered_task  # noqa: E402
from tests.cpu.util import stub_megatron_modules  # noqa: E402


@pytest.fixture
def task_lowering():
    return lowered_task


# A slow test starts its own Ray cluster of about 4 GiB, and four workers running the rest of the suite fill most
# of a 16 GiB CI runner, so at most one slow test runs per 12 GiB of host memory. A second one on that runner
# pushes it past Ray's memory monitor, which then kills actors.
# pytest-xdist sets this in each worker process.
XDIST_WORKER_COUNT_ENV = "PYTEST_XDIST_WORKER_COUNT"
HOST_MEMORY_PER_SLOW_TEST_BYTES = 12 * 2**30


def pytest_configure(config: pytest.Config) -> None:
    """Split the host's cores between pytest-xdist workers.

    Torch defaults every process to one intra-op thread per physical core, so N workers would each spin that many
    threads. OMP_NUM_THREADS carries the same limit into subprocesses that tests spawn.
    """
    worker_count = os.environ.get(XDIST_WORKER_COUNT_ENV)
    if worker_count is None:
        return
    threads = max(1, os.cpu_count() // int(worker_count))
    torch.set_num_threads(threads)
    os.environ["OMP_NUM_THREADS"] = str(threads)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Start tests marked ``slow`` first, spread over as many pytest-xdist workers as host memory allows.

    Under ``--dist loadgroup``, xdist runs each ``xdist_group`` on one worker, hands out the largest groups first,
    and keeps the collection order among the rest. Moving slow tests to the front starts them before the short
    ones; grouping them bounds how many run at once. This hook runs before xdist reads the group markers.
    """
    slow = [item for item in items if item.get_closest_marker("slow") is not None]
    items[:] = slow + [item for item in items if item.get_closest_marker("slow") is None]
    worker_count = os.environ.get(XDIST_WORKER_COUNT_ENV)
    if worker_count is None:
        return
    host_memory = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    group_count = min(int(worker_count), max(1, round(host_memory / HOST_MEMORY_PER_SLOW_TEST_BYTES)))
    for index, item in enumerate(slow):
        item.add_marker(pytest.mark.xdist_group(f"slow-{index % group_count}"))


@pytest.fixture
def megatron_wrapper(monkeypatch):
    """Construct replay wrappers with single-rank CPU parallel state."""
    stub_megatron_modules()
    # Megatron is optional in CPU CI; load the wrapper after installing its stubs.
    from skyrl_train.workers.megatron import megatron_model_wrapper as mmw

    monkeypatch.setattr(mmw, "get_model_config", lambda model: model.config)
    monkeypatch.setattr(mmw.mpu, "is_pipeline_first_stage", lambda **kwargs: True, raising=False)
    monkeypatch.setattr(mmw.mpu, "is_pipeline_last_stage", lambda **kwargs: True, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_context_parallel_world_size", lambda: 1, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_context_parallel_rank", lambda: 0, raising=False)
    monkeypatch.setattr(mmw.mpu, "get_tensor_model_parallel_world_size", lambda: 1, raising=False)

    def build(*, packing, controller, num_experts):
        config = OmegaConf.create({"trainer": {"use_sample_packing": packing}})
        actor = SimpleNamespace(config=SimpleNamespace(num_moe_experts=num_experts))
        wrapper = mmw.MegatronModelWrapper(config, [actor], logprob_chunk_size=None)
        wrapper.router_replay = controller
        return wrapper

    return build


@pytest.fixture
def chosen_teacher_evidence() -> ChosenTokenTeacherEvidence:
    return ChosenTokenTeacherEvidence(
        trajectory_ids=("math_0", "swe_0"),
        route_ids=("math", "swe"),
        teacher_id="teacher-a",
        teacher_revision="teacher-revision",
        plan_version="mopd-v1",
        valid_mask=torch.tensor([[True, True, False], [True, False, False]]),
        chosen_logprobs=torch.tensor([[-0.5, -2.0, torch.nan], [-0.25, torch.nan, torch.nan]]),
    )


@pytest.fixture
def local_distillation_config():
    """Attach the supported single-teacher distillation plan to a test config."""

    def configure(config):
        OmegaConf.set_struct(config, False)
        config.trainer.algorithm.distillation = {
            "objective": "sampled_reverse_kl",
            "routing_plan": "opd",
            "coefficient": 0.5,
            "reward_mode": "add",
        }
        config.teachers = {
            "primary": {
                "source": "local_inference",
                "placement": "pinned",
                "model": {"path": "Qwen/teacher", "revision": "teacher-revision"},
                "backend": "vllm",
                "evidence": "chosen_token",
                "resources": {
                    "num_nodes": 1,
                    "gpus_per_node": 1,
                    "tensor_parallel_size": 1,
                    "colocation_group": "teacher",
                },
            }
        }
        config.teacher_routing = {
            "opd": {
                "revision": "route-revision",
                "routes": {"default": {"teacher": "primary", "weight": 1.0}},
            }
        }
        return config

    return configure


@pytest.fixture
def verifier_test_collection_factory():
    def build(trial: int, outcomes: dict[str, str], *, complete: bool = True) -> VerifierTestCollection:
        return {
            "parser": "test",
            "complete": complete,
            "tests": [
                {
                    "record_id": f"trial-{trial}:{test_id}",
                    "trial_id": TrajectoryID(instance_id="task", repetition_id=trial),
                    "test_id": test_id,
                    "outcome": outcome,
                    "output": f"{test_id}: {outcome}",
                }
                for test_id, outcome in outcomes.items()
            ],
        }

    return build


def _kill_registry_actors() -> None:
    registry_module = sys.modules.get("skyrl_train.utils.function_registry")
    if registry_module is None:
        return
    for registry in registry_module.BaseFunctionRegistry.__subclasses__():
        registry.shutdown_actor()


@contextmanager
def _local_ray_session() -> Iterator[None]:
    if not ray.is_initialized():
        # No CPU test reads the dashboard; skipping it saves each concurrent session its start-up time and memory.
        ray.init(include_dashboard=False)
    try:
        yield
    finally:
        if ray.is_initialized():
            _kill_registry_actors()
            ray.shutdown()


@pytest.fixture
def ray_init() -> Iterator[None]:
    """Run one Ray-dependent CPU test in a local session."""
    with _local_ray_session():
        yield


@pytest.fixture(scope="module")
def ray_module() -> Iterator[None]:
    """Share a local Ray session across an actor-heavy test module."""
    with _local_ray_session():
        yield


@pytest.fixture(scope="module")
def single_rank_group(tmp_path_factory):
    """A world-size-1 gloo process group so distributed collectives (TP
    all-reduces, broadcast_object_list) run as no-ops on one CPU process."""
    created = False
    if not dist.is_initialized():
        # A file rendezvous keeps concurrent pytest-xdist workers from contending for one TCP port.
        rendezvous = tmp_path_factory.mktemp("single_rank_group") / "store"
        dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
        created = True
    try:
        yield dist.group.WORLD
    finally:
        if created:
            dist.destroy_process_group()


@dataclass
class DeliveredTelemetry:
    """Records the real rigging exporter posted, each with its batch's resource attributes."""

    rows: list[dict] = field(default_factory=list)

    def flush(self) -> list[dict]:
        assert training_telemetry.telemetry.flush(timeout=5)
        return self.rows

    def select(self, name: str, **attributes: str) -> list[dict]:
        return [
            row
            for row in self.flush()
            if row["name"] == name and all(row["attributes"].get(key) == value for key, value in attributes.items())
        ]

    def values(self, name: str, **attributes: str) -> list[float]:
        return [row["value"] for row in self.select(name, **attributes)]


@pytest.fixture
def telemetry_endpoint(monkeypatch) -> Iterator[DeliveredTelemetry]:
    """Point process telemetry at a fake Finelog endpoint that accepts every batch."""
    exporter = training_telemetry.telemetry
    exporter.shutdown(timeout=0)
    delivered = DeliveredTelemetry()

    def post(session, endpoint, *, data, headers, timeout):
        if headers.get("Content-Encoding") == "zstd":
            data = zstandard.ZstdDecompressor().decompress(data)
        envelope = json.loads(data)
        resource = envelope["resource"]["attributes"]
        delivered.rows.extend({**row, "resource": resource} for row in envelope["records"])
        return SimpleNamespace(
            status_code=200, headers={}, json=lambda: {"batch_id": envelope["batch_id"], "status": "accepted"}
        )

    monkeypatch.setattr(exporter.requests.Session, "post", post)
    monkeypatch.setenv("SKYRL_TELEMETRY_ENDPOINT", "http://finelog.test/v1/ingest")
    monkeypatch.setenv("SKYRL_RUN_ID", "telemetry-test")
    monkeypatch.setenv("SKYRL_EXECUTION_UID", "test-attempt")
    monkeypatch.setenv("SKYRL_TRAINING_TYPE", TrainingType.ASYNC.value)
    yield delivered
    exporter.shutdown(timeout=0)


@pytest.fixture
def delivered_telemetry(telemetry_endpoint) -> Iterator[DeliveredTelemetry]:
    """Own trainer-role process telemetry for one test and yield what it delivers."""
    with training_telemetry.process_telemetry(training_telemetry.TRAINER_ROLE):
        yield telemetry_endpoint


@dataclass
class FakeCuda:
    """The torch.cuda allocator surface the learner memory recorder reads."""

    allocated: int = 100
    reserved: int = 160
    peak_allocated: int = 900
    peak_reserved: int = 960
    failure: str | None = None
    backend: str = "native"

    def current_device(self):
        if self.failure == "identity":
            raise RuntimeError("CUDA context unavailable")
        return 2

    def get_allocator_backend(self):
        return self.backend

    def get_device_properties(self, device):
        return SimpleNamespace(uuid="GPU-physical-two")

    def reset_peak_memory_stats(self, device):
        if self.failure == "reset":
            raise RuntimeError("CUDA peak reset unavailable")
        self.peak_allocated, self.peak_reserved = self.allocated, self.reserved

    def use_memory(self, allocated, reserved):
        self.allocated, self.reserved = allocated, reserved
        self.peak_allocated = max(self.peak_allocated, allocated)
        self.peak_reserved = max(self.peak_reserved, reserved)

    def memory_stats(self, device):
        if self.failure == "sample":
            raise RuntimeError("CUDA memory sample unavailable")
        return {
            "allocated_bytes.all.current": self.allocated,
            "reserved_bytes.all.current": self.reserved,
            "allocated_bytes.all.peak": self.peak_allocated,
            "reserved_bytes.all.peak": self.peak_reserved,
        }

    def mem_get_info(self, device):
        return 2000, 4096


@pytest.fixture
def fake_cuda(monkeypatch) -> FakeCuda:
    cuda = FakeCuda()
    for name in (
        "current_device",
        "get_allocator_backend",
        "get_device_properties",
        "reset_peak_memory_stats",
        "memory_stats",
        "mem_get_info",
    ):
        monkeypatch.setattr(torch.cuda, name, getattr(cuda, name))
    return cuda
