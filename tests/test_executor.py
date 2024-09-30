# TODO: pytest doesn't work with Ray because these functions cannot be found
import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass

import pytest
import ray

from marin.execution.executor import Executor, ExecutorStep, output_path_of, this_output_path, versioned


@pytest.fixture
def ray_start():
    # nothing to do for setup

    yield

    # teardown
    ray.shutdown()


@dataclass(frozen=True)
class MyConfig:
    input_path: str
    output_path: str
    n: int
    m: int


# Different Ray processes running `ExecutorStep`s cannot share variables, so use filesystem.
# Helper functions
def create_temp():
    # Note that different steps cannot share variables
    with tempfile.NamedTemporaryFile(delete=False) as f:
        return f.name


def append_temp(path: str, obj: dataclass):
    with open(path, "a") as f:
        print(json.dumps(asdict(obj) if obj else None), file=f)


def read_temp(path: str):
    with open(path) as f:
        return list(map(json.loads, f.readlines()))


def cleanup_temp(path: str):
    os.unlink(path)


def test_executor():
    """Test basic executor functionality."""
    temp = create_temp()

    def fn(config: MyConfig | None):
        append_temp(temp, config)

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

    executor = Executor(prefix="/tmp")
    executor.run(steps=[b])

    assert len(executor.steps) == 2
    assert executor.output_paths[a].startswith("/tmp/a-")
    assert executor.output_paths[b].startswith("/tmp/b-")

    # Check the results
    results = read_temp(temp)
    assert len(results) == 2
    assert results[0] is None
    assert re.match(r"/tmp/a-(\w+)/sub", results[1]["input_path"])
    assert re.match(r"/tmp/b-(\w+)", results[1]["output_path"])
    assert results[1]["n"] == 3
    assert results[1]["m"] == 4

    cleanup_temp(temp)


def test_parallelism():
    """Make sure things that parallel execution is possible."""
    temp = create_temp()

    # Note that due to parallelism, total wall-clock time should be `run_time` +
    # overhead, as long as all the jobs can get scheduled.
    run_time = 5
    parallelism = 6

    def fn(config: MyConfig):
        append_temp(temp, config)
        time.sleep(run_time)

    bs = [
        ExecutorStep(name=f"b{i}", fn=fn, config=MyConfig(input_path="/", output_path=this_output_path(), n=1, m=1))
        for i in range(parallelism)
    ]
    executor = Executor(prefix="/tmp")
    start_time = time.time()
    executor.run(steps=bs)
    end_time = time.time()

    results = read_temp(temp)
    assert len(results) == parallelism
    for i in range(parallelism):
        assert results[i]["output_path"].startswith("/tmp/b")

    serial_duration = run_time * parallelism
    actual_duration = end_time - start_time
    print(f"Duration: {actual_duration:.2f}s")
    assert (
        actual_duration < serial_duration * 0.75
    ), f"""Expected parallel execution to be at least 25% faster than serial.
        Actual: {actual_duration:.2f}s, Serial: {serial_duration:.2f}s"""

    cleanup_temp(temp)


def test_versioning():
    """Make sure that versions (output paths) are computed properly based on
    upstream dependencies and only the versioned fields."""

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
        executor = Executor(prefix="/tmp")
        executor.run(steps=[b])
        return executor.output_paths[b]

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

