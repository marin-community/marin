# TODO: pytest doesn't work with Ray because these functions cannot be found
import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass

import pytest
import ray

from marin.execution.executor import Executor, ExecutorStep, get_info_path, output_path_of, this_output_path, versioned
from marin.execution.executor_step_status import STATUS_SUCCESS, get_current_status, get_status_path, read_events


@pytest.fixture
def ray_start():
    # (nothing to do for setup)
    yield
    ray.shutdown()  # teardown


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


def create_temp_dir():
    with tempfile.TemporaryDirectory(prefix="executor-", delete=False) as temp_dir:
        return temp_dir


def create_executor(temp_dir: str):
    """Create an Executor that lives in a temporary directory."""
    return Executor(prefix=temp_dir, executor_info_base_path=temp_dir)


def cleanup_executor(executor: Executor):
    """Deletes the info and status files for all the steps."""
    for step in executor.steps:
        output_path = executor.output_paths[step]
        os.unlink(get_status_path(output_path))
        os.unlink(get_info_path(output_path))
        os.rmdir(output_path)
    os.unlink(executor.executor_info_path)
    os.rmdir(executor.prefix)


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

    executor = create_executor(create_temp_dir())
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
        status_path = get_status_path(executor.output_paths[step])
        events = read_events(status_path)
        assert get_current_status(events) == STATUS_SUCCESS
        info_path = get_info_path(executor.output_paths[step])
        with open(info_path) as f:
            step_info = json.load(f)
            check_info(step_info, step)

    cleanup_log(log)
    cleanup_executor(executor)


def test_forcerun():
    """Test force run functionality."""

    def get_b_list(log, m):
        def fn(config: MyConfig | None):
            append_log(log, config)

        b = ExecutorStep(
            name="b",
            fn=fn,
            config=MyConfig(
                input_path="/",
                output_path=this_output_path(),
                n=versioned(3),
                m=m,
            ),
        )

        return [b]

    log = create_log()
    dir_temp = create_temp_dir()
    executor_initial = Executor(prefix=dir_temp, executor_info_base_path=dir_temp)
    executor_initial.run(steps=get_b_list(log, 4))

    # Re run without force_run
    executor_non_force = Executor(prefix=dir_temp, executor_info_base_path=dir_temp)
    executor_non_force.run(steps=get_b_list(log, 5))  # This would not run b again so m would be 4
    # Check the results
    results = read_log(log)
    assert len(results) == 1
    assert results[0]["n"] == 3
    assert results[0]["m"] == 4

    log = create_log()
    executor_initial = Executor(prefix=dir_temp, executor_info_base_path=dir_temp)
    executor_initial.run(steps=get_b_list(log, 4))

    # Re run without force_run
    executor_force = Executor(prefix=dir_temp, executor_info_base_path=dir_temp, force_run=["b"])
    executor_force.run(steps=get_b_list(log, 5))  # This would run b again so m would be 5
    results = read_log(log)
    assert len(results) == 1
    assert results[0]["n"] == 3
    assert results[0]["m"] == 5

    cleanup_log(log)


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
    executor = create_executor(create_temp_dir())
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
    cleanup_executor(executor)


def test_versioning():
    """Make sure that versions (output paths) are computed properly based on
    upstream dependencies and only the versioned fields."""
    temp_dir = create_temp_dir()  # Make sure we use the same one for all executors for reproducibility

    def fn(config: MyConfig):
        pass

    def get_output_path(a_input_path: str, a_n: int, a_m: int, name: str, b_n: int, b_m: int):
        """Make steps [a -> b] with the given arguments, and return the output_path of `b`."""
        a = ExecutorStep(
            name="a",
            fn=fn,
            config=MyConfig(input_path=versioned(a_input_path), output_path=this_output_path(), n=versioned(a_n), m=a_m),
        )
        b = ExecutorStep(
            name="b",
            fn=fn,
            config=MyConfig(input_path=output_path_of(a, name), output_path=this_output_path(), n=versioned(b_n), m=b_m),
        )
        executor = create_executor(temp_dir)
        executor.run(steps=[b])
        output_path = executor.output_paths[b]
        cleanup_executor(executor)
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
