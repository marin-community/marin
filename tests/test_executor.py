import json
import os
import random
import re
import tempfile
import time
from dataclasses import asdict, dataclass

import pytest
import ray
from draccus.utils import Dataclass

from marin.execution.executor import Executor, ExecutorStep, get_info_path, output_path_of, this_output_path, versioned
from marin.execution.executor_step_status import (
    STATUS_SUCCESS,
    get_current_status,
    get_status_path,
    read_events,
)


@pytest.fixture(scope="module", autouse=True)
def ray_start():
    ray.init(namespace="marin", ignore_reinit_error=True)
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
            status_path = get_status_path(executor.output_paths[step])
            events = read_events(status_path)
            assert get_current_status(events) == STATUS_SUCCESS
            info_path = get_info_path(executor.output_paths[step])
            with open(info_path) as f:
                step_info = json.load(f)
                check_info(step_info, step)

    cleanup_log(log)


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

        with pytest.raises(Exception, match=".*failed previously.*"):
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


def test_status_actor():
    """Test the status actor that keeps track of statuses."""

    def test_one_executor_waiting_for_another():
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

            @ray.remote
            def run_fn(executor, steps):
                executor.run(steps=steps)

            with tempfile.TemporaryDirectory(prefix="executor-") as temp_dir:
                executor1 = create_executor(temp_dir)
                executor2 = create_executor(temp_dir)

                run1 = run_fn.remote(executor1, [a])
                run2 = run_fn.remote(executor2, [a, b])

                ray.get([run1, run2])

                with open(file.name, "r") as f:
                    assert int(f.read()) == 3

    test_one_executor_waiting_for_another()

    def test_multiple_steps_race_condition():
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

            @ray.remote
            def run_fn(executor, steps):
                executor.run(steps=steps)

            with tempfile.TemporaryDirectory(prefix="executor-") as temp_dir:

                executor_refs = []
                for _ in range(10):
                    executor = create_executor(temp_dir)
                    executor_refs.append(
                        run_fn.remote(executor, [ExecutorStep(name="step", fn=fn, config=Config(output_path))])
                    )

                ray.get(executor_refs)

                files = os.listdir(output_path)
                print(files)
                assert len(files) == 1
                os.unlink(os.path.join(output_path, files[0]))

    test_multiple_steps_race_condition()


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
