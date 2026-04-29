# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact, PathMetadata
from marin.execution.executor import Executor, ExecutorStep, _dag_tpu_regions, resolve_executor_step
from marin.execution.remote import RemoteCallable, remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem import MARIN_CROSS_REGION_OVERRIDE_ENV

# ---------------------------------------------------------------------------
# Artifact types
# ---------------------------------------------------------------------------


@dataclass
class TokenizeMetadata:
    path: str
    num_tokens: int


@dataclass
class TrainMetadata:
    tokens_seen: int
    checkpoint_path: str


@dataclass
class NestedMetadata:
    path: str
    resources: ResourceConfig


# ---------------------------------------------------------------------------
# Pipeline functions: download → tokenize → train
#
# Each function accepts artifact instances as inputs and returns an artifact
# describing its output.
# ---------------------------------------------------------------------------


def download_raw_data(output_path: str, source_url: str) -> PathMetadata:
    """Download raw data shards to output_path."""
    data_dir = os.path.join(output_path, "data")
    os.makedirs(data_dir, exist_ok=True)
    for shard in range(3):
        with open(os.path.join(data_dir, f"shard-{shard}.jsonl"), "w") as f:
            for i in range(10):
                json.dump({"id": shard * 10 + i, "text": f"doc {shard * 10 + i}", "src": source_url}, f)
                f.write("\n")
    return PathMetadata(path=data_dir)


def tokenize_data(output_path: str, raw_data: PathMetadata, tokenizer: str) -> TokenizeMetadata:
    """Tokenize documents from the raw data artifact."""
    data_dir = os.path.join(output_path, "data")
    os.makedirs(data_dir, exist_ok=True)

    total_tokens = 0
    out_docs = []
    for fname in sorted(os.listdir(raw_data.path)):
        with open(os.path.join(raw_data.path, fname)) as f:
            for line in f:
                doc = json.loads(line)
                tokens = doc["text"].split()
                total_tokens += len(tokens)
                out_docs.append({"id": doc["id"], "tokens": tokens, "tokenizer": tokenizer})

    with open(os.path.join(data_dir, "tokenized.jsonl"), "w") as f:
        for doc in out_docs:
            json.dump(doc, f)
            f.write("\n")

    return TokenizeMetadata(path=data_dir, num_tokens=total_tokens)


def train_on_tokenized_data(output_path: str, tokenized: TokenizeMetadata) -> TrainMetadata:
    """Train a model on the tokenized data artifact."""
    os.makedirs(output_path, exist_ok=True)
    total_tokens = 0
    for fname in sorted(os.listdir(tokenized.path)):
        with open(os.path.join(tokenized.path, fname)) as f:
            for line in f:
                doc = json.loads(line)
                total_tokens += len(doc["tokens"])

    ckpt_path = os.path.join(output_path, "ckpt")
    with open(ckpt_path, "w") as f:
        f.write(f"model trained on {total_tokens} tokens\n")

    return TrainMetadata(tokens_seen=total_tokens, checkpoint_path=ckpt_path)


# ---------------------------------------------------------------------------
# Artifact tests
# ---------------------------------------------------------------------------


def test_artifact_save_and_load_typed(tmp_path: Path):
    artifact = PathMetadata(path="/data/shards")
    Artifact.save(artifact, tmp_path.as_posix())

    loaded = Artifact.load(tmp_path.as_posix(), PathMetadata)
    assert loaded == artifact
    assert loaded.path == "/data/shards"


def test_artifact_save_and_load_untyped(tmp_path: Path):
    artifact = TokenizeMetadata(path="/tokenized", num_tokens=42)
    Artifact.save(artifact, tmp_path.as_posix())

    loaded = Artifact.load(tmp_path.as_posix())
    assert isinstance(loaded, dict)
    assert loaded["path"] == "/tokenized"
    assert loaded["num_tokens"] == 42


def test_artifact_save_nested_dataclass(tmp_path: Path):
    artifact = NestedMetadata(path="/nested", resources=ResourceConfig(cpu=2, ram="4g"))
    Artifact.save(artifact, tmp_path.as_posix())

    loaded = Artifact.load(tmp_path.as_posix())
    assert isinstance(loaded, dict)
    assert loaded["path"] == "/nested"
    assert loaded["resources"]["cpu"] == 2
    assert loaded["resources"]["ram"] == "4g"


def test_artifact_roundtrip_through_pipeline(tmp_path: Path):
    """Save an artifact in one step, load it in the next — the core handoff pattern."""
    step1_out = (tmp_path / "step1").as_posix()
    step2_out = (tmp_path / "step2").as_posix()

    # Step 1: download
    raw = download_raw_data(step1_out, "http://example.com")
    Artifact.save(raw, step1_out)

    # Step 2: tokenize — load upstream artifact, run, save
    loaded_raw = Artifact.load(step1_out, PathMetadata)
    tokenized = tokenize_data(step2_out, loaded_raw, "word")
    Artifact.save(tokenized, step2_out)

    assert isinstance(tokenized, TokenizeMetadata)
    assert tokenized.num_tokens == 60  # 30 docs * 2 words each

    # Both artifacts are loadable from their respective output paths
    assert Artifact.load(step1_out, PathMetadata) == raw
    assert Artifact.load(step2_out, TokenizeMetadata) == tokenized


# ---------------------------------------------------------------------------
# resolve_executor_step tests
# ---------------------------------------------------------------------------


def test_resolve_executor_step_binds_config():
    """resolve_executor_step should produce a zero-arg callable with config bound."""
    received = {}

    def my_fn(config):
        received["config"] = config

    step = ExecutorStep(name="download", fn=my_fn, config=None)
    resolved = resolve_executor_step(step, config={"url": "http://example.com"}, output_path="/out/download-abc123")

    assert resolved.output_path == "/out/download-abc123"
    assert resolved.deps == []

    # Call the resolved fn — it should invoke my_fn with the config
    resolved.fn("/tmp/foobar")
    assert received["config"] == {"url": "http://example.com"}


def test_runner_saves_artifact_automatically(tmp_path):
    """The runner should auto-save BaseModel results to output_path."""
    out = tmp_path.as_posix()

    step = StepSpec(
        name="test_save",
        override_output_path=out,
        fn=lambda output_path: PathMetadata(path=output_path),
    )

    runner = StepRunner()
    runner.run([step])

    loaded = Artifact.load(out, PathMetadata)
    assert loaded.path == out


def test_resolve_executor_step_preserves_deps():
    step = ExecutorStep(name="train", fn=lambda c: None, config=None)
    dep1 = StepSpec(name="download", override_output_path="/out/download-abc123")
    dep2 = StepSpec(name="tokenize", override_output_path="/out/tokenize-def456")
    resolved = resolve_executor_step(
        step,
        config={},
        output_path="/out/train-abc123",
        deps=[dep1, dep2],
    )
    assert resolved.dep_paths == ["/out/download-abc123", "/out/tokenize-def456"]


def test_step_spec_as_executor_step_round_trip():
    """StepSpec -> ExecutorStep -> StepSpec should preserve identity."""
    prefix = "gs://test-bucket"
    dep = StepSpec(
        name="download",
        output_path_prefix=prefix,
        hash_attrs={"source": "web"},
        fn=lambda output_path: output_path,
    )
    step = StepSpec(
        name="tokenize",
        output_path_prefix=prefix,
        hash_attrs={"tokenizer": "llama3"},
        deps=[dep],
        fn=lambda output_path: output_path,
    )

    executor_step = step.as_executor_step()
    # override_output_path should be the computed path (prefix/name_hash)
    assert executor_step.override_output_path == step.output_path
    assert step.output_path.startswith(f"{prefix}/tokenize_")

    dep_spec = StepSpec(name="download", override_output_path=dep.output_path)
    resolved = resolve_executor_step(
        executor_step,
        config={},
        output_path=step.output_path,
        deps=[dep_spec],
    )

    assert resolved.name == step.name
    assert resolved.hash_attrs == step.hash_attrs
    assert resolved.fn is step.fn
    assert resolved.output_path == step.output_path
    assert resolved.output_path_prefix == prefix
    assert resolved.dep_paths == [dep.output_path]


def _build_three_level_dag(prefix: str) -> tuple[StepSpec, StepSpec, StepSpec]:
    """download → normalize → tokenize, all rooted at ``prefix``."""
    download = StepSpec(
        name="download",
        output_path_prefix=prefix,
        hash_attrs={"source": "fineweb-edu", "revision": "87f0914"},
    )
    normalize = StepSpec(
        name="normalize",
        output_path_prefix=prefix,
        deps=[download],
        hash_attrs={"text_field": "text", "relative_input_path": "sample/10BT"},
    )
    tokenize = StepSpec(
        name="tokenize",
        output_path_prefix=prefix,
        deps=[normalize],
        hash_attrs={"tokenizer": "gpt2"},
    )
    return download, normalize, tokenize


def test_step_spec_hash_id_stable_across_prefixes():
    """Identity hashes must not depend on the Marin bucket prefix.

    Regression for marin-community/marin#5216: the same logical pipeline
    resolved under different ``MARIN_PREFIX`` values (e.g. region failover
    from ``gs://marin-us-central1`` to ``gs://marin-us-east5``) was producing
    distinct hashes, changing output paths, checkpoint ids, and W&B run ids.
    """
    central = _build_three_level_dag("gs://marin-us-central1")
    east = _build_three_level_dag("gs://marin-us-east5")

    for c, e in zip(central, east, strict=True):
        assert c.hash_id == e.hash_id, f"{c.name} hash flipped across prefixes: {c.hash_id} vs {e.hash_id}"
        assert c.name_with_hash == e.name_with_hash

    # Output paths must still differ — that's where the prefix lives.
    for c, e in zip(central, east, strict=True):
        assert c.output_path != e.output_path
        assert c.output_path.startswith("gs://marin-us-central1/")
        assert e.output_path.startswith("gs://marin-us-east5/")


def test_step_spec_hash_id_via_marin_prefix_env(monkeypatch):
    """Same as above, but driven by the ``MARIN_PREFIX`` env var path."""
    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-central1")
    central = [
        StepSpec(name="download", hash_attrs={"source": "fineweb-edu"}),
    ]
    central.append(StepSpec(name="normalize", deps=[central[0]], hash_attrs={"text_field": "text"}))
    central.append(StepSpec(name="tokenize", deps=[central[1]], hash_attrs={"tokenizer": "gpt2"}))
    central_paths = [s.output_path for s in central]  # force prefix resolution into cached_property

    monkeypatch.setenv("MARIN_PREFIX", "gs://marin-us-east5")
    east = [
        StepSpec(name="download", hash_attrs={"source": "fineweb-edu"}),
    ]
    east.append(StepSpec(name="normalize", deps=[east[0]], hash_attrs={"text_field": "text"}))
    east.append(StepSpec(name="tokenize", deps=[east[1]], hash_attrs={"tokenizer": "gpt2"}))
    east_paths = [s.output_path for s in east]

    for c, e in zip(central, east, strict=True):
        assert c.hash_id == e.hash_id

    assert all(p.startswith("gs://marin-us-central1/") for p in central_paths)
    assert all(p.startswith("gs://marin-us-east5/") for p in east_paths)


# ---------------------------------------------------------------------------
# StepRunner tests: three-step pipeline
# ---------------------------------------------------------------------------


def _build_pipeline(tmp_path: Path) -> list[StepSpec]:
    """Build download → tokenize → train as StepSpecs.

    Each step function returns an artifact.  The runner auto-saves any
    BaseModel result to the step's output_path.  Inter-step data flows
    through ``Artifact.load`` — deferred to execution time via lambdas.
    """

    tmp_path_posix = tmp_path.as_posix()

    source_url = "http://data.example.com/raw.tar"
    download_step = StepSpec(
        name="download",
        output_path_prefix=tmp_path_posix,
        hash_attrs={"source_url": source_url},
        fn=lambda output_path: download_raw_data(output_path, source_url),
    )

    # Artifact.load must be deferred to execution time (upstream hasn't run yet)
    tokenizer = "word"
    tokenize_step = StepSpec(
        name="tokenize",
        output_path_prefix=tmp_path_posix,
        hash_attrs={"tokenizer": tokenizer},
        deps=[download_step],
        fn=lambda output_path: tokenize_data(
            output_path,
            Artifact.load(download_step.output_path, PathMetadata),
            tokenizer,
        ),
    )
    train_step = StepSpec(
        name="train",
        output_path_prefix=tmp_path_posix,
        deps=[tokenize_step],
        fn=lambda output_path: train_on_tokenized_data(
            output_path, Artifact.load(tokenize_step.output_path, TokenizeMetadata)
        ),
    )
    return [download_step, tokenize_step, train_step]


def test_runner_executes_pipeline(tmp_path: Path):
    """The runner should execute download → tokenize → train in order."""
    steps = _build_pipeline(tmp_path)
    runner = StepRunner()
    runner.run(steps)

    download_path = steps[0].output_path
    tokenize_path = steps[1].output_path
    train_path = steps[2].output_path

    # Download produced shards
    raw_artifact = Artifact.load(download_path, PathMetadata)
    assert os.path.isdir(raw_artifact.path)
    assert len(os.listdir(raw_artifact.path)) == 3

    # Tokenize produced output with correct token count
    tokenize_artifact = Artifact.load(tokenize_path, TokenizeMetadata)
    assert tokenize_artifact.num_tokens == 60  # 30 docs * 2 words each

    # Train produced a checkpoint
    train_artifact = Artifact.load(train_path, TrainMetadata)
    assert train_artifact.tokens_seen > 0
    assert os.path.exists(train_artifact.checkpoint_path)


def test_runner_skips_completed_steps(tmp_path: Path):
    """Running the same pipeline twice should skip already-succeeded steps."""
    steps = _build_pipeline(tmp_path)

    runner1 = StepRunner()
    runner1.run(steps)

    # Record modification times
    tokenize_artifact_path = os.path.join(steps[1].output_path, ".artifact")
    mtime_before = os.path.getmtime(tokenize_artifact_path)

    # Re-run — all steps should be skipped
    runner2 = StepRunner()
    runner2.run(steps)

    mtime_after = os.path.getmtime(tokenize_artifact_path)
    assert mtime_before == mtime_after, "Tokenize artifact should not have been rewritten"


def test_runner_dry_run(tmp_path: Path):
    """Dry run should not create any output directories."""
    steps = _build_pipeline(tmp_path)
    runner = StepRunner()
    runner.run(steps, dry_run=True)

    for step in steps:
        out = step.output_path
        assert not os.path.exists(out), f"{out} should not exist after dry run"


def test_runner_respects_dependency_order(tmp_path: Path):
    """Steps should execute in dependency order even if given out of order."""
    steps = _build_pipeline(tmp_path)
    reversed_steps = list(reversed(steps))

    runner = StepRunner()
    runner.run(reversed_steps)

    train_artifact = Artifact.load(steps[2].output_path, TrainMetadata)
    assert train_artifact.tokens_seen > 0


def test_runner_max_concurrent(tmp_path: Path):
    """max_concurrent=1 should still complete the full pipeline."""
    steps = _build_pipeline(tmp_path)
    runner = StepRunner()
    runner.run(steps, max_concurrent=1)

    train_artifact = Artifact.load(steps[2].output_path, TrainMetadata)
    assert train_artifact.tokens_seen > 0


def test_runner_walks_transitive_deps(tmp_path: Path):
    """Passing only terminal steps should cause the runner to walk and run transitive deps."""
    executed: list[str] = []

    def record(name: str):
        def _fn(output_path: str) -> PathMetadata:
            executed.append(name)
            return PathMetadata(path=output_path)

        return _fn

    dep = StepSpec(
        name="dep",
        override_output_path=(tmp_path / "dep").as_posix(),
        fn=record("dep"),
    )
    mid = StepSpec(
        name="mid",
        override_output_path=(tmp_path / "mid").as_posix(),
        deps=[dep],
        fn=record("mid"),
    )
    terminal = StepSpec(
        name="terminal",
        override_output_path=(tmp_path / "terminal").as_posix(),
        deps=[mid],
        fn=record("terminal"),
    )

    StepRunner().run([terminal])

    assert executed == ["dep", "mid", "terminal"]


def test_runner_walks_transitive_deps_with_cache_hit(tmp_path: Path):
    """Deps already succeeded on disk must be recognized via cache-hit during the walk."""
    dep = StepSpec(
        name="dep",
        override_output_path=(tmp_path / "dep").as_posix(),
        fn=lambda output_path: PathMetadata(path=output_path),
    )
    downstream_ran: list[str] = []

    def run_downstream(output_path: str) -> PathMetadata:
        downstream_ran.append(output_path)
        return PathMetadata(path=output_path)

    downstream = StepSpec(
        name="downstream",
        override_output_path=(tmp_path / "downstream").as_posix(),
        deps=[dep],
        fn=run_downstream,
    )

    # Prime the cache for ``dep`` only.
    StepRunner().run([dep])
    assert downstream_ran == []

    # Pass only ``downstream``; the runner walks deps and cache-hits ``dep``.
    StepRunner().run([downstream])
    assert downstream_ran == [(tmp_path / "downstream").as_posix()]


def test_runner_consumes_unbounded_iterator(tmp_path: Path):
    """The runner must not pre-consume the iterable — it must support unbounded generators.

    The generator yields forever unless ``stop`` is set; we set it from inside
    a terminal's function after N terminals have executed. A batch-flatten
    implementation would try to exhaust the generator before running any step
    and hang (caught by the per-test timeout).
    """
    import threading

    stop = threading.Event()
    executed: list[str] = []
    lock = threading.Lock()
    n_terminals = 3

    def on_execute(name: str):
        def _fn(output_path: str) -> PathMetadata:
            with lock:
                executed.append(name)
                # Count terminals executed; signal the generator to stop once
                # we've run enough.
                terminal_count = sum(1 for e in executed if e.startswith("t_"))
            if terminal_count >= n_terminals:
                stop.set()
            return PathMetadata(path=output_path)

        return _fn

    dep = StepSpec(
        name="shared_dep",
        override_output_path=(tmp_path / "shared_dep").as_posix(),
        fn=on_execute("dep"),
    )

    def unbounded_generator():
        i = 0
        while not stop.is_set():
            name = f"t_{i}"
            yield StepSpec(
                name=name,
                override_output_path=(tmp_path / name).as_posix(),
                deps=[dep],
                fn=on_execute(name),
            )
            i += 1

    StepRunner().run(unbounded_generator())

    assert "dep" in executed
    terminals = [e for e in executed if e.startswith("t_")]
    assert len(terminals) >= n_terminals


def test_runner_dedups_shared_deps(tmp_path: Path):
    """A dep shared by multiple terminals must be executed exactly once."""
    dep_runs: list[str] = []

    def run_dep(output_path: str) -> PathMetadata:
        dep_runs.append(output_path)
        return PathMetadata(path=output_path)

    dep = StepSpec(
        name="shared_dep",
        override_output_path=(tmp_path / "shared_dep").as_posix(),
        fn=run_dep,
    )
    a = StepSpec(
        name="a",
        override_output_path=(tmp_path / "a").as_posix(),
        deps=[dep],
        fn=lambda output_path: PathMetadata(path=output_path),
    )
    b = StepSpec(
        name="b",
        override_output_path=(tmp_path / "b").as_posix(),
        deps=[dep],
        fn=lambda output_path: PathMetadata(path=output_path),
    )

    StepRunner().run([a, b])

    assert dep_runs == [(tmp_path / "shared_dep").as_posix()]


def test_runner_preserves_underlying_step_exception(tmp_path: Path):
    """The top-level runner error should retain the original failing exception as a cause."""

    def failing_step(_output_path: str) -> None:
        raise ValueError("sentinel step failure")

    step = StepSpec(
        name="failing_step",
        override_output_path=(tmp_path / "failing_step").as_posix(),
        fn=failing_step,
    )

    runner = StepRunner()
    with pytest.raises(RuntimeError, match=r"1 step\(s\) failed") as exc_info:
        runner.run([step])

    step_failure = exc_info.value.__cause__
    assert isinstance(step_failure, RuntimeError)
    assert "Step failed: failing_step" in str(step_failure)
    assert isinstance(step_failure.__cause__, ValueError)
    assert "sentinel step failure" in str(step_failure.__cause__)


# ---------------------------------------------------------------------------
# Local vs Fray execution tests
# ---------------------------------------------------------------------------


def test_step_with_remote_fn_uses_fray(tmp_path: Path):
    """A RemoteCallable fn should go through RemoteCallable.submit."""

    @remote
    def my_step(output_path):
        return PathMetadata(path=output_path)

    step = StepSpec(
        name="fray_step",
        override_output_path=tmp_path.as_posix(),
        fn=my_step,
    )

    runner = StepRunner()
    runner.run([step])

    loaded = Artifact.load(tmp_path.as_posix(), PathMetadata)
    assert loaded.path == tmp_path.as_posix()


# ---------------------------------------------------------------------------
# StepSpec.resources dispatch tests
# ---------------------------------------------------------------------------


class _SubmitSpy:
    """Wraps a fray client and captures every ``submit`` call's request."""

    def __init__(self, inner):
        self._inner = inner
        self.requests = []

    def submit(self, request, adopt_existing: bool = True):
        self.requests.append(request)
        return self._inner.submit(request, adopt_existing=adopt_existing)

    def __getattr__(self, item):
        return getattr(self._inner, item)


def test_plain_fn_runs_inline_without_resources(tmp_path: Path, fray_client):
    """A plain callable with no ``resources`` field runs inline — no Fray submit."""
    spy = _SubmitSpy(fray_client)
    from fray.client import set_current_client

    def my_step(output_path: str) -> PathMetadata:
        return PathMetadata(path=output_path)

    step = StepSpec(
        name="inline_step",
        override_output_path=tmp_path.as_posix(),
        fn=my_step,
    )

    with set_current_client(spy):
        StepRunner().run([step])

    assert spy.requests == []
    loaded = Artifact.load(tmp_path.as_posix(), PathMetadata)
    assert loaded.path == tmp_path.as_posix()


def test_step_resources_dispatches_via_fray(tmp_path: Path, fray_client):
    """Setting ``resources`` on a StepSpec submits ``fn`` as a Fray job."""
    spy = _SubmitSpy(fray_client)
    from fray.client import set_current_client

    custom = ResourceConfig.with_cpu(cpu=2, ram="8g")

    def my_step(output_path: str) -> PathMetadata:
        return PathMetadata(path=output_path)

    step = StepSpec(
        name="resourced_step",
        override_output_path=tmp_path.as_posix(),
        fn=my_step,
        resources=custom,
    )

    with set_current_client(spy):
        StepRunner().run([step])

    assert len(spy.requests) == 1
    assert spy.requests[0].resources == custom
    loaded = Artifact.load(tmp_path.as_posix(), PathMetadata)
    assert loaded.path == tmp_path.as_posix()


def test_executor_step_resources_propagates_to_step_spec(tmp_path: Path):
    """``ExecutorStep.resources`` flows through ``resolve_executor_step``."""
    custom = ResourceConfig.with_cpu(cpu=4, ram="16g")

    def my_fn(config):
        return None

    step = ExecutorStep(name="explicit", fn=my_fn, config=None, resources=custom)
    resolved = resolve_executor_step(step, config={}, output_path="/out/explicit")

    assert resolved.resources == custom
    # fn is wrapped to ignore output_path; it should NOT be a RemoteCallable.
    assert not isinstance(resolved.fn, RemoteCallable)


def test_step_resources_wins_over_remote_callable(tmp_path: Path, fray_client):
    """When both ``step.resources`` and a ``RemoteCallable`` ``fn`` are set,
    ``step.resources`` is used and the inner callable is unwrapped."""
    spy = _SubmitSpy(fray_client)
    from fray.client import set_current_client

    explicit = ResourceConfig.with_cpu(cpu=8, ram="32g")
    legacy = ResourceConfig.with_cpu(cpu=1, ram="2g")

    @remote(resources=legacy)
    def my_step(output_path):
        return PathMetadata(path=output_path)

    step = StepSpec(
        name="dual_step",
        override_output_path=tmp_path.as_posix(),
        fn=my_step,
        resources=explicit,
    )

    with set_current_client(spy):
        StepRunner().run([step])

    assert len(spy.requests) == 1
    assert spy.requests[0].resources == explicit, "step.resources must take precedence"
    loaded = Artifact.load(tmp_path.as_posix(), PathMetadata)
    assert loaded.path == tmp_path.as_posix()


# ---------------------------------------------------------------------------
# @remote decorator tests
# ---------------------------------------------------------------------------


def test_remote_decorator_returns_remote_callable():
    """@remote should return a RemoteCallable with default CPU resources."""

    def original_fn(config):
        pass

    wrapped = remote(original_fn)

    assert isinstance(wrapped, RemoteCallable)
    assert wrapped.resources == ResourceConfig.with_cpu()
    assert not isinstance(original_fn, RemoteCallable)


def test_remote_decorator_with_custom_resources():
    """@remote(resources=...) should use the specified resources."""
    custom = ResourceConfig.with_cpu(cpu=4, ram="16g")

    @remote(resources=custom)
    def my_fn(config):
        pass

    assert isinstance(my_fn, RemoteCallable)
    assert my_fn.resources == custom


def test_resolve_executor_step_picks_up_remote_decorator():
    """resolve_executor_step should propagate @remote resources to the resolved fn."""

    @remote
    def my_fn(config):
        pass

    step = ExecutorStep(name="test", fn=my_fn, config=None)
    resolved = resolve_executor_step(step, config={}, output_path="/out/test-abc")

    assert isinstance(resolved.fn, RemoteCallable)
    assert resolved.fn.resources == ResourceConfig.with_cpu()


def test_remote_decorator_resources_are_preserved():
    """@remote decorator resources should be preserved through resolve_executor_step."""
    custom = ResourceConfig.with_cpu(cpu=8, ram="32g")

    @remote(resources=custom)
    def my_fn(config):
        pass

    step = ExecutorStep(name="test", fn=my_fn, config=None)
    resolved = resolve_executor_step(step, config={}, output_path="/out/test-abc")

    assert isinstance(resolved.fn, RemoteCallable)
    assert resolved.fn.resources == custom


def test_resolve_executor_step_infers_region_for_iris_without_pin():
    @remote
    def my_fn(config):
        pass

    step = ExecutorStep(name="test", fn=my_fn, config=None)
    with (patch("marin.execution.executor._iris_backend_is_active", return_value=True),):
        resolved = resolve_executor_step(
            step,
            config={"input_path": "gs://marin-us-central2/data/input"},
            output_path="/out/test-abc",
        )

    assert isinstance(resolved.fn, RemoteCallable)
    assert resolved.fn.resources.regions == ["us-central2"]


def test_resolve_executor_step_preserves_explicit_empty_regions():
    @remote(resources=ResourceConfig.with_cpu(regions=[]))
    def my_fn(config):
        pass

    step = ExecutorStep(name="test", fn=my_fn, config=None)
    with (patch("marin.execution.executor._iris_backend_is_active", return_value=True),):
        resolved = resolve_executor_step(
            step,
            config={"input_path": "gs://marin-us-central2/data/input"},
            output_path="/out/test-abc",
        )

    assert isinstance(resolved.fn, RemoteCallable)
    assert resolved.fn.resources.regions == []


def test_resolve_executor_step_infers_region_from_dependencies():
    @remote
    def my_fn(config):
        pass

    step = ExecutorStep(name="test", fn=my_fn, config=None)
    dep = StepSpec(name="dep", override_output_path="gs://marin-us-east1/dependency/output")
    with (patch("marin.execution.executor._iris_backend_is_active", return_value=True),):
        resolved = resolve_executor_step(
            step,
            config={"local_only": "/tmp/foo"},
            output_path="/out/test-abc",
            deps=[dep],
        )

    assert isinstance(resolved.fn, RemoteCallable)
    assert resolved.fn.resources.regions == ["us-east1"]


def test_resolve_executor_step_raises_on_cross_region_inputs_without_pin():
    @remote
    def my_fn(config):
        pass

    step = ExecutorStep(name="test", fn=my_fn, config=None)
    with (patch("marin.execution.executor._iris_backend_is_active", return_value=True),):
        with pytest.raises(ValueError, match="cross-region GCS dependencies"):
            resolve_executor_step(
                step,
                config={"input_path": "gs://marin-us-central2/data/input"},
                output_path="gs://marin-us-east1/data/output",
            )


def test_resolve_executor_step_raises_on_cross_region_even_with_override_env(monkeypatch):
    @remote
    def my_fn(config):
        pass

    monkeypatch.setenv(MARIN_CROSS_REGION_OVERRIDE_ENV, "1")
    step = ExecutorStep(name="test", fn=my_fn, config=None)
    with (patch("marin.execution.executor._iris_backend_is_active", return_value=True),):
        with pytest.raises(ValueError, match="cross-region GCS dependencies"):
            resolve_executor_step(
                step,
                config={"input_path": "gs://marin-us-central2/data/input"},
                output_path="gs://marin-us-east1/data/output",
            )


def test_resolve_executor_step_uses_dag_tpu_regions_without_gcs_inputs():
    @remote
    def my_fn(config):
        pass

    step = ExecutorStep(name="test", fn=my_fn, config=None)
    with (patch("marin.execution.executor._iris_backend_is_active", return_value=True),):
        resolved = resolve_executor_step(
            step,
            config={"local_only": "/tmp/foo"},
            output_path="/out/test-abc",
            dag_tpu_regions=["us-west4", "us-central2"],
        )

    assert isinstance(resolved.fn, RemoteCallable)
    assert resolved.fn.resources.regions == ["us-central2", "us-west4"]


def test_resolve_executor_step_intersects_gcs_and_dag_tpu_regions():
    @remote
    def my_fn(config):
        pass

    step = ExecutorStep(name="test", fn=my_fn, config=None)
    with (patch("marin.execution.executor._iris_backend_is_active", return_value=True),):
        resolved = resolve_executor_step(
            step,
            config={"input_path": "gs://marin-us-central2/data/input"},
            output_path="/out/test-abc",
            dag_tpu_regions=["us-west4", "us-central2"],
        )

    assert isinstance(resolved.fn, RemoteCallable)
    assert resolved.fn.resources.regions == ["us-central2"]


def test_resolve_executor_step_raises_on_disjoint_gcs_and_dag_tpu_regions():
    @remote
    def my_fn(config):
        pass

    step = ExecutorStep(name="test", fn=my_fn, config=None)
    with (patch("marin.execution.executor._iris_backend_is_active", return_value=True),):
        with pytest.raises(ValueError, match="no overlap between GCS regions"):
            resolve_executor_step(
                step,
                config={"input_path": "gs://marin-us-east1/data/input"},
                output_path="/out/test-abc",
                dag_tpu_regions=["us-central2"],
            )


def test_resolve_executor_step_raises_on_disjoint_gcs_and_dag_tpu_regions_even_with_override_env(monkeypatch):
    @remote
    def my_fn(config):
        pass

    monkeypatch.setenv(MARIN_CROSS_REGION_OVERRIDE_ENV, "1")
    step = ExecutorStep(name="test", fn=my_fn, config=None)
    with (patch("marin.execution.executor._iris_backend_is_active", return_value=True),):
        with pytest.raises(ValueError, match="no overlap between GCS regions"):
            resolve_executor_step(
                step,
                config={"input_path": "gs://marin-us-east1/data/input"},
                output_path="/out/test-abc",
                dag_tpu_regions=["us-central2"],
            )


def test_resolve_executor_step_raises_for_dual_region_bucket_location():
    @remote
    def my_fn(config):
        pass

    step = ExecutorStep(name="test", fn=my_fn, config=None)
    with (
        patch("marin.execution.executor._iris_backend_is_active", return_value=True),
        patch("marin.execution.executor.get_bucket_location", return_value="NAM4"),
    ):
        with pytest.raises(ValueError, match="non-regional bucket location"):
            resolve_executor_step(
                step,
                config={"input_path": "gs://external-bucket/path/to/data"},
                output_path="/out/test-abc",
            )


def test_resolve_executor_step_skips_bucket_location_permission_failures():
    class Forbidden(Exception):
        pass

    @remote
    def my_fn(config):
        pass

    step = ExecutorStep(name="test", fn=my_fn, config=None)
    with (
        patch("marin.execution.executor._iris_backend_is_active", return_value=True),
        patch("marin.execution.executor.get_bucket_location", side_effect=Forbidden("no bucket metadata access")),
    ):
        resolved = resolve_executor_step(
            step,
            config={"input_path": "gs://external-bucket/path/to/data"},
            output_path="/out/test-abc",
        )

    assert isinstance(resolved.fn, RemoteCallable)
    assert resolved.fn.resources.regions is None


def test_executor_resolve_steps_infers_region_from_dependency_output_path():
    @remote
    def dep_fn(_config):
        pass

    @remote
    def my_fn(_config):
        pass

    dep = ExecutorStep(name="dep", fn=dep_fn, config=None)
    step = ExecutorStep(name="test", fn=my_fn, config=None)
    executor = Executor(prefix="/tmp/executor", executor_info_base_path="/tmp/executor-info")
    executor.configs = {dep: {}, step: {"local_only": "/tmp/foo"}}
    executor.dependencies = {dep: [], step: [dep]}
    executor.output_paths = {dep: "gs://marin-us-east1/dependency/output", step: "/tmp/test-output"}

    with (patch("marin.execution.executor._iris_backend_is_active", return_value=True),):
        resolved_dep, resolved_step = executor._resolve_steps([dep, step])

    assert isinstance(resolved_dep.fn, RemoteCallable)
    assert isinstance(resolved_step.fn, RemoteCallable)
    assert resolved_step.fn.resources.regions == ["us-east1"]


def test_executor_resolve_steps_uses_downstream_tpu_regions_for_upstream_steps():
    @remote
    def prep_fn(_config):
        pass

    @remote(resources=ResourceConfig.with_tpu("v5p-8", regions=["us-central2"]))
    def train_fn(_config):
        pass

    prep = ExecutorStep(name="prep", fn=prep_fn, config=None)
    train = ExecutorStep(name="train", fn=train_fn, config=None)
    executor = Executor(prefix="/tmp/executor", executor_info_base_path="/tmp/executor-info")
    executor.configs = {prep: {"local_only": "/tmp/foo"}, train: {"local_only": "/tmp/bar"}}
    executor.dependencies = {prep: [], train: [prep]}
    executor.output_paths = {prep: "/tmp/prep-output", train: "/tmp/train-output"}

    with (patch("marin.execution.executor._iris_backend_is_active", return_value=True),):
        resolved_prep, resolved_train = executor._resolve_steps([prep, train])

    assert isinstance(resolved_prep.fn, RemoteCallable)
    assert isinstance(resolved_train.fn, RemoteCallable)
    assert resolved_prep.fn.resources.regions == ["us-central2"]
    assert resolved_train.fn.resources.regions == ["us-central2"]


def test_executor_resolve_steps_picks_one_region_for_multi_region_tpu_component():
    @remote
    def prep_fn(_config):
        pass

    @remote(resources=ResourceConfig.with_tpu("v5p-8", regions=["us-west4", "us-central2"]))
    def train_fn(_config):
        pass

    prep = ExecutorStep(name="prep", fn=prep_fn, config=None)
    train = ExecutorStep(name="train", fn=train_fn, config=None)
    executor = Executor(prefix="/tmp/executor", executor_info_base_path="/tmp/executor-info")
    executor.configs = {prep: {"local_only": "/tmp/foo"}, train: {"local_only": "/tmp/bar"}}
    executor.dependencies = {prep: [], train: [prep]}
    executor.output_paths = {prep: "/tmp/prep-output", train: "/tmp/train-output"}

    with (patch("marin.execution.executor._iris_backend_is_active", return_value=True),):
        resolved_prep, resolved_train = executor._resolve_steps([prep, train])

    assert isinstance(resolved_prep.fn, RemoteCallable)
    assert isinstance(resolved_train.fn, RemoteCallable)
    assert resolved_prep.fn.resources.regions == ["us-central2"]
    assert resolved_train.fn.resources.regions == ["us-central2"]


def test_executor_resolve_steps_uses_component_gcs_region_to_pick_tpu_region():
    @remote
    def prep_fn(_config):
        pass

    @remote(resources=ResourceConfig.with_tpu("v5p-8", regions=["us-west4", "us-central2"]))
    def train_fn(_config):
        pass

    prep = ExecutorStep(name="prep", fn=prep_fn, config=None)
    train = ExecutorStep(name="train", fn=train_fn, config=None)
    executor = Executor(prefix="/tmp/executor", executor_info_base_path="/tmp/executor-info")
    executor.configs = {
        prep: {"input_path": "gs://marin-us-west4/data/input"},
        train: {"local_only": "/tmp/bar"},
    }
    executor.dependencies = {prep: [], train: [prep]}
    executor.output_paths = {prep: "/tmp/prep-output", train: "/tmp/train-output"}

    with (patch("marin.execution.executor._iris_backend_is_active", return_value=True),):
        resolved_prep, resolved_train = executor._resolve_steps([prep, train])

    assert isinstance(resolved_prep.fn, RemoteCallable)
    assert isinstance(resolved_train.fn, RemoteCallable)
    assert resolved_prep.fn.resources.regions == ["us-west4"]
    assert resolved_train.fn.resources.regions == ["us-west4"]


def test_executor_resolve_steps_does_not_apply_unrelated_tpu_regions():
    @remote
    def cpu_fn(_config):
        pass

    @remote(resources=ResourceConfig.with_tpu("v5p-8", regions=["us-central2"]))
    def tpu_fn(_config):
        pass

    cpu_step = ExecutorStep(name="cpu", fn=cpu_fn, config=None)
    tpu_step = ExecutorStep(name="tpu", fn=tpu_fn, config=None)
    executor = Executor(prefix="/tmp/executor", executor_info_base_path="/tmp/executor-info")
    executor.configs = {
        cpu_step: {"input_path": "gs://marin-us-east1/data/input"},
        tpu_step: {"local_only": "/tmp/bar"},
    }
    executor.dependencies = {cpu_step: [], tpu_step: []}
    executor.output_paths = {cpu_step: "/tmp/cpu-output", tpu_step: "/tmp/tpu-output"}

    with (patch("marin.execution.executor._iris_backend_is_active", return_value=True),):
        resolved_cpu, resolved_tpu = executor._resolve_steps([cpu_step, tpu_step])

    assert isinstance(resolved_cpu.fn, RemoteCallable)
    assert isinstance(resolved_tpu.fn, RemoteCallable)
    assert resolved_cpu.fn.resources.regions == ["us-east1"]
    assert resolved_tpu.fn.resources.regions == ["us-central2"]


def test_dag_tpu_regions_intersects_explicit_regions():
    @remote(resources=ResourceConfig.with_tpu("v5p-8", regions=["us-central2", "us-west4"]))
    def first(_config):
        pass

    @remote(resources=ResourceConfig.with_tpu("v5p-8", regions=["us-central2", "us-east1"]))
    def second(_config):
        pass

    steps = [
        ExecutorStep(name="first", fn=first, config=None),
        ExecutorStep(name="second", fn=second, config=None),
    ]

    assert _dag_tpu_regions(steps) == ["us-central2"]


def test_dag_tpu_regions_raises_on_disjoint_explicit_regions():
    @remote(resources=ResourceConfig.with_tpu("v5p-8", regions=["us-west4"]))
    def first(_config):
        pass

    @remote(resources=ResourceConfig.with_tpu("v5p-8", regions=["us-central2"]))
    def second(_config):
        pass

    steps = [
        ExecutorStep(name="first", fn=first, config=None),
        ExecutorStep(name="second", fn=second, config=None),
    ]

    with pytest.raises(ValueError, match="No common region satisfies all TPU steps"):
        _dag_tpu_regions(steps)


def test_dag_tpu_regions_raises_on_disjoint_explicit_regions_even_with_override_env(monkeypatch):
    @remote(resources=ResourceConfig.with_tpu("v5p-8", regions=["us-west4"]))
    def first(_config):
        pass

    @remote(resources=ResourceConfig.with_tpu("v5p-8", regions=["us-central2"]))
    def second(_config):
        pass

    monkeypatch.setenv(MARIN_CROSS_REGION_OVERRIDE_ENV, "1")
    steps = [
        ExecutorStep(name="first", fn=first, config=None),
        ExecutorStep(name="second", fn=second, config=None),
    ]

    with pytest.raises(ValueError, match="No common region satisfies all TPU steps"):
        _dag_tpu_regions(steps)


def test_dag_tpu_regions_uses_iris_variant_regions_when_not_pinned():
    @remote(resources=ResourceConfig.with_tpu("v5p-8"))
    def first(_config):
        pass

    @remote(resources=ResourceConfig.with_tpu("v5p-8"))
    def second(_config):
        pass

    steps = [
        ExecutorStep(name="first", fn=first, config=None),
        ExecutorStep(name="second", fn=second, config=None),
    ]

    with patch(
        "marin.execution.executor._regions_for_tpu_variant_from_iris",
        return_value={"us-central2", "us-west4"},
    ):
        assert _dag_tpu_regions(steps) == ["us-central2", "us-west4"]


def test_dag_tpu_regions_unions_device_alternative_regions():
    @remote(resources=ResourceConfig.with_tpu(["v5p-8", "v6e-4"]))
    def first(_config):
        pass

    step = ExecutorStep(name="first", fn=first, config=None)

    with patch(
        "marin.execution.executor._regions_for_tpu_variant_from_iris",
        side_effect=lambda variant: {
            "v5p-8": {"us-central2", "us-west4"},
            "v6e-4": {"us-east1"},
        }[variant],
    ):
        assert _dag_tpu_regions([step]) == ["us-central2", "us-east1", "us-west4"]


def test_step_without_remote_is_plain_fn():
    """A plain function with no @remote should not be RemoteCallable."""

    def my_fn(config):
        pass

    step = ExecutorStep(name="test", fn=my_fn, config=None)
    resolved = resolve_executor_step(step, config={}, output_path="/out/test-abc")

    assert not isinstance(resolved.fn, RemoteCallable)


def test_runner_propagates_context_vars(tmp_path):
    """StepRunner must propagate contextvars to worker threads.

    This ensures that fray's ``set_current_client`` is visible inside step
    functions dispatched by the thread pool, so ZephyrContext (and anything
    else that calls ``current_client()``) picks up the correct client.
    """
    import contextvars

    test_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("test_var", default=None)
    observed: list[str | None] = []

    def capture_ctx(output_path: str):
        observed.append(test_var.get())
        return PathMetadata(path=output_path)

    step = StepSpec(
        name="ctx_check",
        override_output_path=(tmp_path / "ctx_check").as_posix(),
        fn=capture_ctx,
    )

    test_var.set("from_parent")
    runner = StepRunner()
    runner.run([step])

    assert observed == ["from_parent"], f"Expected context var to propagate, got {observed}"


def test_runner_propagates_fray_client(tmp_path):
    """StepRunner explicitly propagates the fray client to worker threads.

    This tests the explicit client capture path (not just generic contextvars)
    to ensure current_client() returns the correct client inside step functions.
    """
    from fray.client import current_client, set_current_client

    class FakeClient:
        """Marker client to verify propagation."""

        pass

    observed_clients: list[type] = []

    def check_client(output_path: str):
        client = current_client()
        observed_clients.append(type(client))
        return PathMetadata(path=output_path)

    step = StepSpec(
        name="fray_check",
        override_output_path=(tmp_path / "fray_check").as_posix(),
        fn=check_client,
    )

    fake = FakeClient()
    with set_current_client(fake):
        runner = StepRunner()
        runner.run([step])

    assert observed_clients == [FakeClient], f"Expected FakeClient in worker thread, got {observed_clients}"
