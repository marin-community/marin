# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import random
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from threading import Thread

import pytest
from draccus.utils import Dataclass
from marin.execution import THIS_OUTPUT_PATH
from marin.execution.executor import (
    Executor,
    ExecutorStep,
    InputName,
    _get_info_path,
    collect_dependencies_and_version,
    output_path_of,
    this_output_path,
    versioned,
)
from marin.execution.executor_step_status import (
    STATUS_SUCCESS,
    StatusFile,
)


@dataclass(frozen=True)
class MyConfig:
    input_path: str
    output_path: str
    n: int
    m: int


# Different Ray processes running `ExecutorStep`s cannot share variables, so use filesystem.
# Helper functions


def create_log():
    # Note that different steps cannot share variables
    with tempfile.NamedTemporaryFile(prefix="executor-log-") as f:
        return f.name


def append_log(path: str, obj: dataclass):
    with open(path, "a") as f:
        print(json.dumps(asdict(obj) if obj else None), file=f)


def read_log(path: str):
    with open(path) as f:
        return list(map(json.loads, f.readlines()))


def cleanup_log(path: str):
    os.unlink(path)


def create_executor(temp_dir: str):
    """Create an Executor that lives in a temporary directory."""
    return Executor(prefix=temp_dir, executor_info_base_path=temp_dir)


def test_executor():
    """Test basic executor functionality."""
    log = create_log()

    def fn(config: MyConfig | None):
        append_log(log, config)

    a = ExecutorStep(name="a", fn=fn, config=None)

    b = ExecutorStep(
        name="b",
        fn=fn,
        config=MyConfig(
            input_path=output_path_of(a, "sub"),
            output_path=this_output_path(),
            n=versioned(3),
            m=4,
        ),
    )

    with tempfile.TemporaryDirectory(prefix="executor-") as temp_dir:
        executor = create_executor(temp_dir)
        executor.run(steps=[b])

        assert len(executor.steps) == 2
        assert executor.output_paths[a].startswith(executor.prefix + "/a-")
        assert executor.output_paths[b].startswith(executor.prefix + "/b-")

        # Check the results
        results = read_log(log)
        assert len(results) == 2
        assert results[0] is None
        assert re.match(executor.prefix + r"/a-(\w+)/sub", results[1]["input_path"])
        assert re.match(executor.prefix + r"/b-(\w+)", results[1]["output_path"])
        assert results[1]["n"] == 3
        assert results[1]["m"] == 4

        def asdict_optional(obj):
            return asdict(obj) if obj else None

        def check_info(step_info: dict, step: ExecutorStep):
            assert step_info["name"] == step.name
            assert step_info["output_path"] == executor.output_paths[step]
            assert step_info["config"] == asdict_optional(executor.configs[step])
            assert step_info["version"] == executor.versions[step]

        # Check the status and info files
        with open(executor.executor_info_path) as f:
            info = json.load(f)
            assert info["prefix"] == executor.prefix
            for step_info, step in zip(info["steps"], executor.steps, strict=True):
                check_info(step_info, step)

        for step in executor.steps:
            status_file = StatusFile(executor.output_paths[step], worker_id="check")
            assert status_file.status == STATUS_SUCCESS
            info_path = _get_info_path(executor.output_paths[step])
            with open(info_path) as f:
                step_info = json.load(f)
                check_info(step_info, step)

    cleanup_log(log)


def test_status_file_reads_legacy_format(tmp_path):
    output_dir = tmp_path / "step"
    output_dir.mkdir()
    status_path = output_dir / ".executor_status"

    events = [
        {"date": "2024-09-28T13:29:20.780705", "status": "WAITING", "message": None},
        {"date": "2024-09-28T13:29:21.091470", "status": "RUNNING", "message": None},
        {"date": "2024-09-28T13:29:47.559614", "status": "SUCCESS", "message": None},
    ]

    with open(status_path, "w") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")

    status_file = StatusFile(str(output_dir), worker_id="legacy-reader")
    assert status_file.status == "SUCCESS"


def test_force_run_failed():
    log = create_log()

    temp_file_to_mark_failure = tempfile.NamedTemporaryFile(prefix="executor-fail-", delete=False)
    # make sure it exists
    temp_file_to_mark_failure.write(b"hello")
    temp_file_to_mark_failure.close()

    path = temp_file_to_mark_failure.name
    assert os.path.exists(path)

    def fn(config: MyConfig | None):
        print(config.input_path, os.path.exists(config.input_path), flush=True)
        if os.path.exists(config.input_path):
            raise Exception("Failed")
        else:
            append_log(log, config)

    def fn_pass(config: MyConfig | None):
        append_log(log, config)

    b = ExecutorStep(
        name="b",
        fn=fn,
        config=MyConfig(
            input_path=path,
            output_path=this_output_path(),
            n=1,
            m=1,
        ),
    )

    a = ExecutorStep(
        name="a",
        fn=fn_pass,
        config=MyConfig(
            input_path=output_path_of(b, "sub"),
            output_path=this_output_path(),
            n=2,
            m=2,
        ),
    )

    with tempfile.TemporaryDirectory(prefix="executor-") as temp_dir:
        executor_initial = Executor(prefix=temp_dir, executor_info_base_path=temp_dir)
        with pytest.raises(Exception, match="Failed"):
            executor_initial.run(steps=[a])

        with pytest.raises(FileNotFoundError):
            read_log(log)

        # remove the file to say we're allowed to run
        os.unlink(temp_file_to_mark_failure.name)

        # Re-run without force_run
        executor_non_force = Executor(prefix=temp_dir, executor_info_base_path=temp_dir)

        with pytest.raises(Exception, match=r".*failed previously.*"):
            executor_non_force.run(steps=[a])

        # should still be failed
        with pytest.raises(FileNotFoundError):
            read_log(log)

        # Rerun with force_run_failed
        executor_force = Executor(prefix=temp_dir, executor_info_base_path=temp_dir)
        executor_force.run(steps=[a], force_run_failed=True)
        results = read_log(log)
        assert len(results) == 2

    cleanup_log(log)


def test_status_actor_one_executor_waiting_for_another():
    # Test when 2 experiments have a step in common and one waits for another to finish
    with tempfile.NamedTemporaryFile() as file:
        with open(file.name, "w") as f:
            f.write("0")

        @dataclass
        class Config:
            number: int
            path: str
            wait: int
            input_path: str

        def fn(config: Config):
            time.sleep(config.wait)
            with open(config.path, "r") as f:
                number = int(f.read())
            with open(config.path, "w") as f:
                f.write(str(number + config.number))

        a = ExecutorStep(name="a", fn=fn, config=Config(versioned(1), file.name, 2, ""))
        b = ExecutorStep(name="b", fn=fn, config=Config(versioned(2), file.name, 0, output_path_of(a)))

        with tempfile.TemporaryDirectory(prefix="executor-") as temp_dir:
            executor1 = create_executor(temp_dir)
            executor2 = create_executor(temp_dir)

            run1 = Thread(target=executor1.run, args=([a],))
            run2 = Thread(target=executor2.run, args=([a, b],))

            run1.start()
            run2.start()

            run1.join()
            run2.join()

            with open(file.name, "r") as f:
                assert int(f.read()) == 3


def test_status_actor_multiple_steps_race_condition():
    # Test when there are many steps trying to run simultaneously.
    # Open a temp dir, make a step that write a random file in that temp dir. Make 10 of these steps and run them
    # in parallel. Check that only one of them runs
    with tempfile.TemporaryDirectory(prefix="output_path") as output_path:

        @dataclass
        class Config:
            path: str

        def fn(config: Config):
            random_str = str(random.randint(0, 1000))
            time.sleep(2)
            with open(os.path.join(config.path, random_str), "w") as f:
                f.write("1")

        with tempfile.TemporaryDirectory(prefix="executor-") as temp_dir:
            executor_refs = []
            for _ in range(10):
                executor = create_executor(temp_dir)
                thread = Thread(
                    target=executor.run, args=([ExecutorStep(name="step", fn=fn, config=Config(output_path))],)
                )
                thread.start()
                executor_refs.append(thread)

            for executor_ref in executor_refs:
                executor_ref.join()

            files = os.listdir(output_path)
            print(files)
            assert len(files) == 1
            os.unlink(os.path.join(output_path, files[0]))


@pytest.mark.skipif(
    lambda: int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "0")) > 1,
    reason="Overloaded cluster makes this test flaky.",
)
def test_parallelism():
    """Make sure things that parallel execution is possible."""
    log = create_log()

    # Note that due to parallelism, total wall-clock time should be `run_time` +
    # overhead, as long as all the jobs can get scheduled.
    run_time = 5
    parallelism = 6

    def fn(config: MyConfig):
        append_log(log, config)
        time.sleep(run_time)

    bs = [
        ExecutorStep(name=f"b{i}", fn=fn, config=MyConfig(input_path="/", output_path=this_output_path(), n=1, m=1))
        for i in range(parallelism)
    ]
    with tempfile.TemporaryDirectory(prefix="executor-") as temp_dir:
        executor = create_executor(temp_dir)
        start_time = time.time()
        executor.run(steps=bs)
        end_time = time.time()

        results = read_log(log)
        assert len(results) == parallelism
        for i in range(parallelism):
            assert results[i]["output_path"].startswith(executor.prefix + "/b")

        serial_duration = run_time * parallelism
        actual_duration = end_time - start_time
        print(f"Duration: {actual_duration:.2f}s")
        assert (
            actual_duration < serial_duration * 0.75
        ), f"""Expected parallel execution to be at least 25% faster than serial.
            Actual: {actual_duration:.2f}s, Serial: {serial_duration:.2f}s"""

    cleanup_log(log)


def test_versioning():
    """Make sure that versions (output paths) are computed properly based on
    upstream dependencies and only the versioned fields."""
    with tempfile.TemporaryDirectory(prefix="executor-") as temp_dir:

        def fn(config: MyConfig):
            pass

        def get_output_path(a_input_path: str, a_n: int, a_m: int, name: str, b_n: int, b_m: int):
            """Make steps [a -> b] with the given arguments, and return the output_path of `b`."""
            a = ExecutorStep(
                name="a",
                fn=fn,
                config=MyConfig(
                    input_path=versioned(a_input_path), output_path=this_output_path(), n=versioned(a_n), m=a_m
                ),
            )
            b = ExecutorStep(
                name="b",
                fn=fn,
                config=MyConfig(
                    input_path=output_path_of(a, name), output_path=this_output_path(), n=versioned(b_n), m=b_m
                ),
            )
            executor = create_executor(temp_dir)
            executor.run(steps=[b])
            output_path = executor.output_paths[b]
            return output_path

        defaults = dict(a_input_path="a", a_n=1, a_m=1, name="foo", b_n=1, b_m=1)
        default_output_path = get_output_path(**defaults)

        def assert_same_version(**kwargs):
            output_path = get_output_path(**(defaults | kwargs))
            assert output_path == default_output_path

        def assert_diff_version(**kwargs):
            output_path = get_output_path(**(defaults | kwargs))
            assert output_path != default_output_path

        # Changing some of the fields should affect the output path, but not all
        assert_same_version()
        assert_diff_version(a_input_path="aa")
        assert_diff_version(a_n=2)
        assert_same_version(a_m=2)
        assert_diff_version(name="bar")
        assert_diff_version(b_n=2)
        assert_same_version(b_m=2)


def test_dedup_version():
    """Make sure that two `ExecutorStep`s resolve to the same."""

    def fn(config: MyConfig | None):
        pass

    def create_step():
        a = ExecutorStep(name="a", fn=fn, config=None)
        b = ExecutorStep(
            name="b",
            fn=fn,
            config=MyConfig(
                input_path=output_path_of(a, "sub"),
                output_path=this_output_path(),
                n=versioned(3),
                m=4,
            ),
        )
        return b

    b1 = create_step()
    b2 = create_step()

    with tempfile.TemporaryDirectory(prefix="executor-") as temp_dir:
        executor = create_executor(temp_dir)
        executor.run(steps=[b1, b2])
        assert len(executor.steps) == 2


def test_run_only_some_steps():
    """Make sure that only some steps are run."""
    log = create_log()

    def fn(config: Dataclass | None):
        append_log(log, config)

    @dataclass(frozen=True)
    class CConfig:
        m: 10

    a = ExecutorStep(name="a", fn=fn, config=None)
    c = ExecutorStep(name="c", fn=fn, config=CConfig(m=10))
    b = ExecutorStep(
        name="b",
        fn=fn,
        config=MyConfig(
            input_path=output_path_of(a, "sub"),
            output_path=this_output_path(),
            n=versioned(3),
            m=4,
        ),
    )

    with tempfile.TemporaryDirectory(prefix="executor-") as temp_dir:
        executor = create_executor(temp_dir)
        executor.run(steps=[b, c], run_only=["^b$"])

        results = read_log(log)
        assert len(results) == 2
        assert results[0] is None
        assert results[1]["m"] == 4

    cleanup_log(log)

    with tempfile.TemporaryDirectory(prefix="executor-") as temp_dir:
        executor = create_executor(temp_dir)
        executor.run(steps=[a, b, c], run_only=["a", "c"])

        # these can execute in any order
        results = read_log(log)
        assert len(results) == 2
        assert (results[0] is None and results[1]["m"] == 10) or (results[1] is None and results[0]["m"] == 10)


@dataclass(frozen=True)
class DummyCfg:
    x: int = 0
    input_path: str | None = None
    output_path: str = THIS_OUTPUT_PATH


def dummy_fn(cfg: DummyCfg):
    # write one tiny file so the step "does something"
    out_path = os.path.join(cfg.output_path, "dummy")
    os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(out_path, "done.txt"), "w") as f:
        f.write(str(cfg.x))
    return cfg.x


def shouldnt_run_fn(cfg: DummyCfg):
    raise RuntimeError("This function should not run.")


# ----------------------------------------------------------------------
#  Unit tests for collect_dependencies_and_version
# ----------------------------------------------------------------------


def test_collect_deps_skip_vs_block():
    parent = ExecutorStep(name="parent", fn=dummy_fn, config=DummyCfg(x=1))

    # ----- skip parent -------------------------------------------------
    inp_skip = InputName(step=parent, name="ckpt.pt").nonblocking()
    computed_deps = collect_dependencies_and_version(inp_skip)
    deps = computed_deps.dependencies
    ver = computed_deps.version
    pseudo = computed_deps.pseudo_dependencies

    assert parent in pseudo and parent not in deps
    # Placeholder looks like "DEP[0]/ckpt.pt"
    assert ver == {"": "DEP[0]/ckpt.pt"}

    # ----- require parent (default) ------------------------------------
    inp_block = InputName(step=parent, name="ckpt.pt")  # no .skip_parent()
    computed_deps = collect_dependencies_and_version(inp_block)
    deps = computed_deps.dependencies
    ver = computed_deps.version
    pseudo = computed_deps.pseudo_dependencies

    assert parent in deps and parent not in pseudo
    assert ver == {"": "DEP[0]/ckpt.pt"}  # same placeholder, but in deps


# ----------------------------------------------------------------------
#  Parent-version should still affect child hash
# ----------------------------------------------------------------------


def test_parent_version_bubbles_into_skip_child():
    """
    Change parent's config ➜ child's version must change even if parent
    is only a pseudo-dependency.
    """
    with tempfile.TemporaryDirectory(prefix="executor-") as temp_dir:
        # First parent/child pair (parent.x = 1)
        parent1 = ExecutorStep(name="parent", fn=dummy_fn, config=DummyCfg(x=versioned(1)))
        child1_cfg = DummyCfg(0, input_path=parent1.cd("dummy").nonblocking())
        child1 = ExecutorStep(
            name="child",
            fn=dummy_fn,
            config=child1_cfg,
        )

        executor = create_executor(temp_dir)
        executor.run(steps=[child1])
        version1 = executor.version_strs[child1]
        executor = create_executor(temp_dir)

        # Second pair - identical except parent.x = 2
        parent2 = ExecutorStep(name="parent2", fn=dummy_fn, config=DummyCfg(x=versioned(2)))
        child2 = ExecutorStep(
            name="child",
            fn=dummy_fn,
            config=DummyCfg(x=0, input_path=parent2.cd("dummy").nonblocking()),
        )
        executor.run(steps=[child2])
        version2 = executor.version_strs[child2]

        # Hashes should differ
        assert version1 != version2


def test_parent_doesnt_run_on_skip_parent():
    """
    Parent should not run if child is a skip-parent.
    """
    with tempfile.TemporaryDirectory(prefix="executor-") as temp_dir:
        parent = ExecutorStep(name="parent", fn=shouldnt_run_fn, config=DummyCfg(x=1))
        child = ExecutorStep(
            name="child",
            fn=dummy_fn,
            config=DummyCfg(input_path=parent.cd("dummy").nonblocking()),
        )

        executor = create_executor(temp_dir)
        executor.run(steps=[child])


def test_skippable_parent_will_run_if_asked():
    """
    Parent should run if child is a skip-parent and we ask it to.
    """
    with tempfile.TemporaryDirectory(prefix="executor-") as temp_dir:
        parent = ExecutorStep(name="parent", fn=dummy_fn, config=DummyCfg(x=1))
        child = ExecutorStep(
            name="child",
            fn=dummy_fn,
            config=DummyCfg(input_path=parent.cd("dummy").nonblocking()),
        )

        executor = create_executor(temp_dir)
        executor.run(steps=[child], run_only=["parent"])

        # make sure parent ran
        assert os.path.exists(os.path.join(executor.output_paths[parent], "dummy", "done.txt"))


def test_parent_will_run_if_some_child_is_not_skippable():
    """
    Parent should run if child is a skip-parent and we ask it to.
    """
    with tempfile.TemporaryDirectory(prefix="executor-") as temp_dir:
        parent = ExecutorStep(name="parent", fn=dummy_fn, config=DummyCfg(x=1))
        child = ExecutorStep(
            name="child",
            fn=dummy_fn,
            config=DummyCfg(input_path=parent.cd("dummy").nonblocking()),
        )

        child2 = ExecutorStep(
            name="child2",
            fn=dummy_fn,
            config=DummyCfg(input_path=parent.cd("dummy")),  # no skip
        )

        executor = create_executor(temp_dir)
        executor.run(steps=[child, child2])

        # make sure parent ran
        assert os.path.exists(os.path.join(executor.output_paths[parent], "dummy", "done.txt"))


def test_status_file_takeover_stale_lock_then_refresh(tmp_path):
    """Test taking over a stale lock from a dead worker and then refreshing it."""
    from marin.execution.executor_step_status import HEARTBEAT_TIMEOUT, Lease

    # Simulate worker A creating a stale lock (as if it died)
    dead_worker = StatusFile(tmp_path, worker_id="dead-worker")
    dead_worker.try_acquire_lock()

    # Manually backdate the lock to make it stale
    generation, _ = dead_worker._read_lock_with_generation()
    stale_lease = Lease(worker_id="dead-worker", timestamp=time.time() - HEARTBEAT_TIMEOUT - 10)
    dead_worker._write_lock(stale_lease, if_generation_match=generation)

    # Worker B comes along and takes over
    live_worker = StatusFile(tmp_path, worker_id="live-worker")

    # Verify the lock is stale
    _, lease = live_worker._read_lock_with_generation()
    assert lease is not None
    assert lease.is_stale()

    # Take over the stale lock
    assert live_worker.try_acquire_lock()

    # Verify we now own the lock
    _, lease_after_takeover = live_worker._read_lock_with_generation()
    assert lease_after_takeover.worker_id == "live-worker"

    # Now try to refresh
    time.sleep(0.1)
    live_worker.refresh_lock()

    _, lease_after_refresh = live_worker._read_lock_with_generation()
    assert lease_after_refresh.worker_id == "live-worker"
    assert lease_after_refresh.timestamp > lease_after_takeover.timestamp
