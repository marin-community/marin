# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import random
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from threading import Event, Thread

import marin.execution.executor_step_status as executor_step_status
import pytest
from draccus.utils import Dataclass
from fray.types import ResourceConfig
from marin.evaluation.perplexity_gap import (
    GapFinderModelConfig,
    model_perplexity_scores,
    raw_text_dataset,
)
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    instantiate_config,
    mirrored,
    output_path_of,
    this_output_path,
    upstream_steps,
    versioned,
    Executor,
    _get_info_path,
    collect_dependencies_and_version,
    compute_output_path,
    materialize,
)
from marin.execution.executor_step_status import (
    STATUS_SUCCESS,
    StatusFile,
    step_lock,
)


@dataclass(frozen=True)
class MyConfig:
    input_path: str
    output_path: str
    n: int
    m: int


# Different processes running `ExecutorStep`s cannot share variables, so use filesystem.
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


def test_step_lock_terminal_status_does_not_race_heartbeat(tmp_path, monkeypatch):
    terminal_status_written = Event()
    heartbeat_waiting = Event()
    heartbeat_after_terminal_status = Event()
    original_refresh_lock = StatusFile.refresh_lock

    def refresh_lock(status_file: StatusFile) -> None:
        heartbeat_waiting.set()
        assert terminal_status_written.wait(timeout=1)
        try:
            original_refresh_lock(status_file)
        finally:
            heartbeat_after_terminal_status.set()

    monkeypatch.setattr(executor_step_status, "HEARTBEAT_INTERVAL", 0)
    monkeypatch.setattr(StatusFile, "refresh_lock", refresh_lock)

    with step_lock(str(tmp_path), "step") as status_file:
        assert heartbeat_waiting.wait(timeout=1)
        status_file.write_status(STATUS_SUCCESS)
        terminal_status_written.set()
        assert heartbeat_after_terminal_status.wait(timeout=1)
    assert not StatusFile(str(tmp_path), "check").has_active_lock()


def test_model_perplexity_score_step_hash_changes_when_tokenizer_changes():
    base_kwargs = dict(
        name="marin-score",
        datasets={"eval": raw_text_dataset("gs://example-bucket/eval.jsonl")},
        resource_config=ResourceConfig.with_tpu("v5p-8", regions=["us-central1"]),
    )
    step_a = model_perplexity_scores(
        **base_kwargs,
        model=GapFinderModelConfig(
            checkpoint_path="marin-community/marin-8b-base",
            checkpoint_is_hf=True,
            tokenizer="meta-llama/Llama-3.1-8B",
        ),
    )
    step_b = model_perplexity_scores(
        **base_kwargs,
        model=GapFinderModelConfig(
            checkpoint_path="marin-community/marin-8b-base",
            checkpoint_is_hf=True,
            tokenizer="marin-community/marin-tokenizer",
        ),
    )

    with tempfile.TemporaryDirectory(prefix="executor-") as temp_dir:
        executor = create_executor(temp_dir)
        executor.compute_version(step_a, is_pseudo_dep=False)
        executor.compute_version(step_b, is_pseudo_dep=False)

        assert executor.output_paths[step_a] != executor.output_paths[step_b]


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
        with pytest.raises(RuntimeError, match=r"1 step\(s\) failed"):
            executor_initial.run(steps=[a])

        with pytest.raises(FileNotFoundError):
            read_log(log)

        # remove the file to say we're allowed to run
        os.unlink(temp_file_to_mark_failure.name)

        # Re-run with force_run_failed=False
        executor_non_force = Executor(prefix=temp_dir, executor_info_base_path=temp_dir)

        with pytest.raises(Exception, match=r".*failed previously.*"):
            executor_non_force.run(steps=[a], force_run_failed=False)

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


def test_executor_version_stable_across_prefixes():
    """Regression for marin-community/marin#5216 (legacy ``Executor`` path).

    Identity hashes for ``ExecutorStep`` chains must not depend on the Marin
    bucket prefix. The same logical pipeline resolved under
    ``gs://marin-us-central1`` vs ``gs://marin-us-east5`` was producing
    distinct hashes once the chain exceeded ``Executor._MAX_INLINE_DEPTH``,
    because ``_dep_version`` falls back to the deep dep's absolute
    ``output_paths[dep]`` which carries the prefix.

    The chain below has depth 6 at the leaf, which exceeds the default
    ``_MAX_INLINE_DEPTH`` of 4 and forces the fallback for every step from
    ``f`` onward.
    """

    def fn(config):
        pass

    def build_chain_leaf():
        prev = ExecutorStep(name="a", fn=fn, config=None)
        for name in ("b", "c", "d", "e", "f", "g"):
            prev = ExecutorStep(
                name=name,
                fn=fn,
                config=MyConfig(
                    input_path=output_path_of(prev),
                    output_path=this_output_path(),
                    n=1,
                    m=1,
                ),
            )
        return prev  # leaf "g" has depth 6

    leaf_central = build_chain_leaf()
    leaf_east = build_chain_leaf()

    central = Executor(prefix="gs://marin-us-central1", executor_info_base_path="gs://marin-us-central1")
    central.compute_version(leaf_central, is_pseudo_dep=False)

    east = Executor(prefix="gs://marin-us-east5", executor_info_base_path="gs://marin-us-east5")
    east.compute_version(leaf_east, is_pseudo_dep=False)

    # Sanity: deep-dep fallback must actually be active, otherwise the test
    # would silently pass even with the bug present.
    assert central._dep_depth(leaf_central) > Executor._MAX_INLINE_DEPTH

    # Per-step hashes must match across regions.
    central_steps = sorted(central.steps, key=lambda s: s.name)
    east_steps = sorted(east.steps, key=lambda s: s.name)
    for c, e in zip(central_steps, east_steps, strict=True):
        assert c.name == e.name
        c_hash = central.output_paths[c].rsplit("-", 1)[-1]
        e_hash = east.output_paths[e].rsplit("-", 1)[-1]
        assert c_hash == e_hash, f"{c.name} hash flipped across prefixes: {c_hash} vs {e_hash}"

    # Output paths themselves must still differ — that's where the prefix lives.
    assert central.output_paths[leaf_central].startswith("gs://marin-us-central1/")
    assert east.output_paths[leaf_east].startswith("gs://marin-us-east5/")


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


def test_mirrored_versioning():
    """MirroredValue wrapping VersionedValue should version the inner value."""

    @dataclass(frozen=True)
    class Cfg:
        input_path: str
        output_path: str

    deps = collect_dependencies_and_version(
        Cfg(input_path=mirrored(versioned("some/path"), budget_gb=50), output_path="out")
    )
    assert deps.version == {"input_path": "some/path"}


def test_mirrored_instantiate_config():
    """MirroredValue should resolve to mirror:// path."""

    @dataclass(frozen=True)
    class Cfg:
        input_path: str
        output_path: str

    cfg = Cfg(input_path=mirrored(versioned("documents/data"), budget_gb=10), output_path="out")
    resolved = instantiate_config(cfg, output_path="/out", output_paths={}, prefix="/bucket")
    assert resolved.input_path == "mirror://documents/data"


def test_mirrored_nesting_raises():
    with pytest.raises(ValueError, match="nest"):
        mirrored(mirrored("x"))


def test_mirrored_changes_version():
    """Changing the path inside mirrored() should change the version hash."""
    deps1 = collect_dependencies_and_version(
        MyConfig(input_path=mirrored(versioned("data/v1")), output_path="out", n=versioned(1), m=1)
    )
    deps2 = collect_dependencies_and_version(
        MyConfig(input_path=mirrored(versioned("data/v2")), output_path="out", n=versioned(1), m=1)
    )
    assert deps1.version != deps2.version


def test_status_file_takeover_stale_lock_then_refresh(tmp_path):
    """Test taking over a stale lock from a dead worker and then refreshing it."""
    from rigging.distributed_lock import HEARTBEAT_TIMEOUT, Lease

    # Simulate worker A creating a stale lock (as if it died)
    dead_worker = StatusFile(tmp_path, worker_id="dead-worker")
    dead_worker.try_acquire_lock()

    # Manually backdate the lock to make it stale via the underlying lease
    lock = dead_worker._lock
    generation, _ = lock._read_with_generation()
    stale_lease = Lease(worker_id="dead-worker", timestamp=time.time() - HEARTBEAT_TIMEOUT - 10)
    lock._write(stale_lease, if_generation_match=generation)

    # Worker B comes along and takes over
    live_worker = StatusFile(tmp_path, worker_id="live-worker")

    # Verify the lock is stale
    _, lease = live_worker._lock._read_with_generation()
    assert lease is not None
    assert lease.is_stale()

    # Take over the stale lock
    assert live_worker.try_acquire_lock()

    # Verify we now own the lock
    _, lease_after_takeover = live_worker._lock._read_with_generation()
    assert lease_after_takeover.worker_id == "live-worker"

    # Now try to refresh
    time.sleep(0.1)
    live_worker.refresh_lock()

    _, lease_after_refresh = live_worker._lock._read_with_generation()
    assert lease_after_refresh.worker_id == "live-worker"
    assert lease_after_refresh.timestamp > lease_after_takeover.timestamp


# ---------------------------------------------------------------------------
# Placeholder traversal and materialization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Cfg:
    output_path: str
    x: int = 0


def _step(name: str = "step", x: int = 0) -> ExecutorStep:
    """Build a throwaway ExecutorStep. The fn is never invoked in these tests."""
    return ExecutorStep(name=name, fn=lambda c: None, config=_Cfg(output_path="", x=x))


@pytest.mark.parametrize("value", [None, 0, "", "abc", 3.14, True])
def test_primitive_returns_empty(value):
    assert upstream_steps(value) == []


@pytest.mark.parametrize("empty", [[], {}, (), set()], ids=["list", "dict", "tuple", "set"])
def test_empty_collections_return_empty(empty):
    assert upstream_steps(empty) == []


def test_single_executor_step():
    step = _step()
    assert upstream_steps(step) == [step]


@dataclass
class _OuterWithStep:
    inner: ExecutorStep
    other: int = 0


@pytest.fixture
def step():
    return _step()


@pytest.mark.parametrize(
    "make_container",
    [
        pytest.param(lambda s: _OuterWithStep(inner=s), id="dataclass"),
        pytest.param(lambda s: {"a": s, "b": 1}, id="dict"),
        pytest.param(lambda s: [1, s, "x"], id="list"),
        pytest.param(lambda s: (1, s, "x"), id="tuple"),
        # ExecutorStep is hashable (id-based), so it can live in a set.
        pytest.param(lambda s: {s}, id="set"),
    ],
)
def test_container_with_embedded_step(step, make_container):
    assert upstream_steps(make_container(step)) == [step]


def test_duplicate_step_returned_once(step):
    @dataclass
    class Cfg:
        a: ExecutorStep
        b: ExecutorStep
        c: list

    result = upstream_steps(Cfg(a=step, b=step, c=[step, step]))
    assert result == [step]


def test_input_name_with_step_included(step):
    inp = InputName(step=step, name="ckpt.pt")
    assert upstream_steps(inp) == [step]


def test_input_name_without_step_returns_empty():
    # Hardcoded path: no step reference.
    inp = InputName.hardcoded("some/path")
    assert upstream_steps(inp) == []


def test_input_name_nonblocking_still_included(step):
    inp = InputName(step=step, name="ckpt.pt").nonblocking()
    assert upstream_steps(inp) == [step]


def test_versioned_value_is_leaf():
    # versioned() rejects InputName; its inner value is a primitive for dep purposes.
    assert upstream_steps(versioned(42)) == []


def test_mirrored_value_unwrapped():
    # MirroredValue wraps a path string; no embedded steps, but the walker
    # must descend without raising.
    assert upstream_steps(mirrored(versioned("data/v1"))) == []


def test_determinism_repeated_walk():
    a, b, c = _step("a"), _step("b"), _step("c")

    @dataclass
    class Cfg:
        items: list
        extra: dict

    cfg = Cfg(items=[a, b], extra={"k1": c, "k2": a})
    first = upstream_steps(cfg)
    second = upstream_steps(cfg)
    assert first == second
    assert first == [a, b, c]


def test_depth_first_field_declaration_order():
    """Walk must be depth-first in field-declaration order, not BFS or sorted."""
    leaf_a = _step("leaf_a")
    leaf_b = _step("leaf_b")
    leaf_c = _step("leaf_c")

    @dataclass
    class Inner:
        # field order matters: leaf_a before leaf_b
        x: ExecutorStep
        y: ExecutorStep

    @dataclass
    class Outer:
        # depth-first: descend into `first` (yielding leaf_a, leaf_b) BEFORE
        # visiting `second` (leaf_c). BFS would give [leaf_c, leaf_a, leaf_b].
        first: Inner
        second: ExecutorStep

    cfg = Outer(first=Inner(x=leaf_a, y=leaf_b), second=leaf_c)
    assert upstream_steps(cfg) == [leaf_a, leaf_b, leaf_c]


def test_order_distinguishes_from_sorted():
    """Order is traversal-driven, not name-sorted."""
    z = _step("z_first")
    a = _step("a_second")

    @dataclass
    class Cfg:
        head: ExecutorStep  # 'z_first' name but encountered first
        tail: ExecutorStep

    assert upstream_steps(Cfg(head=z, tail=a)) == [z, a]


def test_dict_value_order_preserved():
    # Python dicts preserve insertion order (3.7+); the walker must follow that.
    a, b, c = _step("a"), _step("b"), _step("c")
    d = {"second": b, "first": a, "third": c}
    assert upstream_steps(d) == [b, a, c]


def test_nested_dataclass_dict_list():
    """Mixed-container traversal stays depth-first in declaration order."""
    s1, s2, s3, s4 = _step("s1"), _step("s2"), _step("s3"), _step("s4")

    @dataclass
    class Cfg:
        a: dict = field(default_factory=dict)
        b: list = field(default_factory=list)

    cfg = Cfg(a={"k1": s1, "k2": [s2, s3]}, b=[s4])
    # Expected: dict values in insertion order, then list elements, depth-first.
    # `a` first, descend: s1, then [s2, s3] -> s2, s3; then `b`: s4.
    assert upstream_steps(cfg) == [s1, s2, s3, s4]


def test_input_name_inside_dataclass():
    parent = _step("parent")

    @dataclass
    class Cfg:
        ckpt: InputName

    assert upstream_steps(Cfg(ckpt=InputName(step=parent, name="ckpt.pt"))) == [parent]


def test_does_not_walk_into_step_config():
    """The walker must not descend into a returned step's own config."""
    grandparent = _step("grandparent")

    @dataclass
    class ParentCfg:
        output_path: str
        upstream: ExecutorStep

    parent = ExecutorStep(
        name="parent",
        fn=lambda c: None,
        config=ParentCfg(output_path="", upstream=grandparent),
    )

    # Walking `parent` directly returns only `parent`, not its embedded
    # `grandparent`. Transitive deps are the executor's job.
    assert upstream_steps(parent) == [parent]


# ---------------------------------------------------------------------------
# materialize
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LeafCfg:
    """Tokenize-style step config — no embedded ExecutorSteps."""

    output_path: str
    n: int = 1


@dataclass(frozen=True)
class _TrainCfg:
    """Training-style config that references an upstream tokenize step."""

    output_path: str
    tokens: str  # holds an InputName placeholder until materialize substitutes it
    n: int = 0


def test_materialize_idempotent_for_placeholder_free_config(tmp_path):
    """A config with no embedded ExecutorSteps round-trips unchanged."""
    cfg = _TrainCfg(output_path=str(tmp_path / "train"), tokens="/already/concrete", n=7)

    # Sanity check the precondition: no embedded steps to materialize.
    assert upstream_steps(cfg) == []

    result = materialize(cfg, prefix=str(tmp_path))

    assert result == cfg
    # And no executor metadata should have been written — `materialize` must
    # short-circuit when there is nothing to do.
    assert not (tmp_path / "experiments").exists()


def test_materialize_substitutes_upstream_step_path(tmp_path):
    """A config with one InputName referencing an upstream step gets a concrete path."""
    call_count = {"n": 0}
    seen_output_paths: list[str] = []

    def tokenize_fn(cfg: _LeafCfg) -> None:
        call_count["n"] += 1
        seen_output_paths.append(cfg.output_path)
        # Simulate writing some output so the directory exists.
        os.makedirs(cfg.output_path, exist_ok=True)

    tok_step = ExecutorStep(
        name="tok",
        fn=tokenize_fn,
        config=_LeafCfg(output_path=this_output_path(), n=versioned(1)),
    )

    # The training config references the upstream step's output.
    train_output_path = str(tmp_path / "train-run")
    train_cfg = _TrainCfg(
        output_path=train_output_path,
        tokens=output_path_of(tok_step, "tokens.bin"),
    )

    result = materialize(train_cfg, prefix=str(tmp_path))

    # The fake fn ran exactly once.
    assert call_count["n"] == 1
    # The placeholder was replaced with a concrete path under tmp_path.
    assert isinstance(result.tokens, str)
    assert result.tokens.startswith(str(tmp_path) + "/tok-")
    assert result.tokens.endswith("/tokens.bin")
    # And it is the same path the fn saw.
    assert result.tokens.startswith(seen_output_paths[0])
    # The training output_path passes through unchanged (already concrete).
    assert result.output_path == train_output_path


def test_materialize_uses_provided_prefix(tmp_path):
    """Sub-step output paths land under the provided prefix, not marin_prefix()."""

    def tokenize_fn(cfg: _LeafCfg) -> None:
        os.makedirs(cfg.output_path, exist_ok=True)

    tok_step = ExecutorStep(
        name="tok",
        fn=tokenize_fn,
        config=_LeafCfg(output_path=this_output_path()),
    )
    train_cfg = _TrainCfg(
        output_path=str(tmp_path / "train"),
        tokens=output_path_of(tok_step, "tokens.bin"),
    )

    result = materialize(train_cfg, prefix=str(tmp_path))

    # The substituted path must be under the explicit prefix we passed.
    assert result.tokens.startswith(str(tmp_path) + "/")


def test_materialize_rejects_unresolved_output_path_placeholder(tmp_path):
    """If the caller forgot to resolve config.output_path, fail loudly."""
    cfg = _TrainCfg(
        output_path=this_output_path(),  # OutputName placeholder, not a concrete str
        tokens="/already/concrete",
    )
    with pytest.raises(TypeError, match="OutputName"):
        materialize(cfg, prefix=str(tmp_path))


def test_materialize_concurrent_callers_run_step_once(tmp_path):
    """Two callers materializing configs that share an upstream step
    must run that step exactly once across them."""

    @dataclass(frozen=True)
    class SlowLeafCfg:
        output_path: str
        marker_dir: str

    call_count = {"n": 0}

    def tokenize_fn(cfg: SlowLeafCfg) -> None:
        # The step_lock + status protocol guarantees only one caller actually
        # invokes fn; sibling callers see STATUS_SUCCESS and skip.
        call_count["n"] += 1
        os.makedirs(cfg.output_path, exist_ok=True)
        # Write a marker so we can verify completion.
        with open(os.path.join(cfg.marker_dir, f"call-{call_count['n']}"), "w") as f:
            f.write("done")

    marker_dir = tempfile.mkdtemp(prefix="materialize-marker-")

    tok_step = ExecutorStep(
        name="shared-tok",
        fn=tokenize_fn,
        config=SlowLeafCfg(output_path=this_output_path(), marker_dir=marker_dir),
    )

    # Both training configs reference the same upstream step.
    train_cfg_a = _TrainCfg(
        output_path=str(tmp_path / "train-a"),
        tokens=output_path_of(tok_step, "tokens.bin"),
    )
    train_cfg_b = _TrainCfg(
        output_path=str(tmp_path / "train-b"),
        tokens=output_path_of(tok_step, "tokens.bin"),
    )

    results: dict[str, _TrainCfg] = {}

    def run(label: str, cfg: _TrainCfg) -> None:
        results[label] = materialize(cfg, prefix=str(tmp_path))

    threads = [
        Thread(target=run, args=("a", train_cfg_a)),
        Thread(target=run, args=("b", train_cfg_b)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exactly one caller ran the upstream step.
    assert call_count["n"] == 1
    # Both substituted paths point at the same upstream output.
    assert results["a"].tokens == results["b"].tokens
    assert results["a"].tokens.startswith(str(tmp_path) + "/shared-tok-")


# ---------------------------------------------------------------------------
# compute_output_path
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SubmitCfg:
    """Launcher-style config: own output_path is OutputName, plus an
    upstream InputName placeholder."""

    output_path: object  # str at runtime, but starts as OutputName(None)
    tokenized: object = None  # may hold an InputName referencing upstream
    extra: object = None  # may hold a nested OutputName(name="X")


def _noop_fn(config) -> None:
    return None


def test_compute_output_path_returns_concrete_path(tmp_path):
    """``compute_output_path`` produces a concrete path under the given prefix
    without running anything or touching GCS."""
    config = _SubmitCfg(output_path=this_output_path())

    path = compute_output_path("my-step", config, prefix=str(tmp_path))

    assert isinstance(path, str)
    assert path.startswith(str(tmp_path) + "/my-step-")
    # No GCS / disk side-effects: compute_version only walks in-memory state.
    assert not (tmp_path / "experiments").exists()


def test_compute_output_path_honors_override(tmp_path):
    """``override_output_path`` short-circuits hash derivation and returns the
    override verbatim (after prefix resolution)."""
    config = _SubmitCfg(output_path=this_output_path())
    override = "gs://some-bucket/exact/path"

    path = compute_output_path("my-step", config, override_output_path=override, prefix=str(tmp_path))

    assert path == override


def test_compute_output_path_is_deterministic_and_versioned(tmp_path):
    """Same (name, config) -> same path across calls; different versioned
    values -> different paths."""
    cfg_a = _SubmitCfg(output_path=this_output_path(), extra=versioned("a"))
    cfg_b = _SubmitCfg(output_path=this_output_path(), extra=versioned("b"))

    path_a1 = compute_output_path("step", cfg_a, prefix=str(tmp_path))
    path_a2 = compute_output_path("step", cfg_a, prefix=str(tmp_path))
    path_b = compute_output_path("step", cfg_b, prefix=str(tmp_path))

    assert path_a1 == path_a2
    assert path_a1 != path_b
