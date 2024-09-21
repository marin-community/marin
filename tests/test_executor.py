import re
from dataclasses import dataclass
from marin.execution.executor import ExecutorStep, Executor, get_input, get_output, versioned

@dataclass(frozen=True)
class MyConfig:
    input_path: str
    output_path: str
    n: int
    m: int

def test_executor():
    """Test basic executor functionality."""
    results: list[MyConfig | None] = []

    def fn(config: MyConfig):
        results.append(config)

    a = ExecutorStep(name="a", fn=fn, config=None)
    b = ExecutorStep(name="b", fn=fn, config=MyConfig(input_path=get_input(a, "sub"), output_path=get_output(), n=versioned(3), m=4))

    executor = Executor(prefix="/tmp")
    executor.run(steps=[b])

    assert re.match(r"/tmp/a-(\w+)/sub", results[1].input_path)
    assert re.match(r"/tmp/b-(\w+)", results[1].output_path)
    assert results[1].n == 3

    assert len(results) == 2

    assert len(executor.steps) == 2
    assert executor.output_paths[a].startswith("/tmp/a-")
    assert executor.output_paths[b].startswith("/tmp/b-")

def test_versioning():
    """Make sure that versions (output paths) are computed properly based on upstream dependencies and only the versioned fields."""
    def fn(config: MyConfig):
        pass

    def get_output_path(a_input_path: str, a_n: int, a_m:int, name: str, b_n: int, b_m: int):
        """Make steps [a -> b] with the given arguments, and return the output_path of `b`."""
        a = ExecutorStep(name="a", fn=fn, config=MyConfig(input_path=versioned(a_input_path), output_path=get_output(), n=versioned(a_n), m=a_m))
        b = ExecutorStep(name="b", fn=fn, config=MyConfig(input_path=get_input(a, name), output_path=get_output(), n=versioned(b_n), m=b_m))
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