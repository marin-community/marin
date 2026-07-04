# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import contextvars
import json
import os
import threading
from pathlib import Path

import marin.execution.step_runner as step_runner_module
import pytest
from fray.current_client import current_client, set_current_client
from fray.types import ResourceConfig
from marin.execution.artifact import FINGERPRINT_KEY, VERSION_KEY, Artifact, read_artifact, write_artifact
from marin.execution.remote import RemoteCallable, remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from marin.execution.step_status import STATUS_SUCCESS, StatusFile, worker_id
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Artifact types
# ---------------------------------------------------------------------------


class TokenizeMetadata(BaseModel):
    path: str
    num_tokens: int


class TrainMetadata(BaseModel):
    tokens_seen: int
    checkpoint_path: str


# ---------------------------------------------------------------------------
# Pipeline functions: download → tokenize → train
#
# Each function accepts artifact instances as inputs and returns an artifact
# describing its output.
# ---------------------------------------------------------------------------


def download_raw_data(output_path: str, source_url: str) -> Artifact:
    """Download raw data shards to output_path."""
    data_dir = os.path.join(output_path, "data")
    os.makedirs(data_dir, exist_ok=True)
    for shard in range(3):
        with open(os.path.join(data_dir, f"shard-{shard}.jsonl"), "w") as f:
            for i in range(10):
                json.dump({"id": shard * 10 + i, "text": f"doc {shard * 10 + i}", "src": source_url}, f)
                f.write("\n")
    return Artifact(path=data_dir)


def tokenize_data(output_path: str, raw_data: Artifact, tokenizer: str) -> TokenizeMetadata:
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
# Manual save/load API
# ---------------------------------------------------------------------------


def test_write_then_read_artifact_typed(tmp_path: Path):
    write_artifact(Artifact(path="/data/shards"), tmp_path.as_posix())

    loaded = read_artifact(tmp_path.as_posix(), Artifact)
    assert loaded.path == "/data/shards"


def test_artifact_roundtrip_through_pipeline(tmp_path: Path):
    """Save an artifact in one step, load it in the next — the core handoff pattern."""
    step1_out = (tmp_path / "step1").as_posix()
    step2_out = (tmp_path / "step2").as_posix()

    # Step 1: download
    raw = download_raw_data(step1_out, "http://example.com")
    write_artifact(raw, step1_out)

    # Step 2: tokenize — load upstream artifact, run, save
    loaded_raw = read_artifact(step1_out, Artifact)
    tokenized = tokenize_data(step2_out, loaded_raw, "word")
    write_artifact(tokenized, step2_out)

    assert isinstance(tokenized, TokenizeMetadata)
    assert tokenized.num_tokens == 60  # 30 docs * 2 words each

    # Both artifacts are loadable from their respective output paths
    assert read_artifact(step1_out, Artifact).path == raw.path
    assert read_artifact(step2_out, TokenizeMetadata) == tokenized


# ---------------------------------------------------------------------------
# StepSpec identity tests
# ---------------------------------------------------------------------------


def test_runner_saves_artifact_automatically(tmp_path):
    """The runner should auto-save BaseModel results to output_path."""
    out = tmp_path.as_posix()

    step = StepSpec(
        name="test_save",
        override_output_path=out,
        fn=lambda output_path: Artifact(path=output_path),
    )

    runner = StepRunner()
    runner.run([step])

    loaded = read_artifact(out, Artifact)
    assert loaded.path == out


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


def test_step_spec_output_path_trailing_slash_prefix():
    """Regression for marin-community/marin#6904: object-store keys are not normalized,
    so a doubled separator from a trailing-slash prefix is a *different* key than the
    one slash-collapsing readers resolve."""
    override = StepSpec(
        name="slimpajama-6b",
        output_path_prefix="s3://marin-na/marin/",
        override_output_path="slimpajama-6b/2026.06.28",
    )
    assert override.output_path == "s3://marin-na/marin/slimpajama-6b/2026.06.28"

    hashed = StepSpec(name="tokenize", output_path_prefix="s3://marin-na/marin/")
    assert hashed.output_path == f"s3://marin-na/marin/{hashed.name_with_hash}"


# ---------------------------------------------------------------------------
# StepRunner tests: three-step pipeline
# ---------------------------------------------------------------------------


def _build_pipeline(tmp_path: Path) -> list[StepSpec]:
    """Build download → tokenize → train as StepSpecs.

    Each step function returns an artifact. The runner auto-saves any result via
    ``write_artifact``. Inter-step data flows through ``read_artifact`` — deferred
    to execution time via lambdas.
    """

    tmp_path_posix = tmp_path.as_posix()

    source_url = "http://data.example.com/raw.tar"
    download_step = StepSpec(
        name="download",
        output_path_prefix=tmp_path_posix,
        hash_attrs={"source_url": source_url},
        fn=lambda output_path: download_raw_data(output_path, source_url),
    )

    # read_artifact must be deferred to execution time (upstream hasn't run yet)
    tokenizer = "word"
    tokenize_step = StepSpec(
        name="tokenize",
        output_path_prefix=tmp_path_posix,
        hash_attrs={"tokenizer": tokenizer},
        deps=[download_step],
        fn=lambda output_path: tokenize_data(
            output_path,
            read_artifact(download_step.output_path, Artifact),
            tokenizer,
        ),
    )
    train_step = StepSpec(
        name="train",
        output_path_prefix=tmp_path_posix,
        deps=[tokenize_step],
        fn=lambda output_path: train_on_tokenized_data(
            output_path, read_artifact(tokenize_step.output_path, TokenizeMetadata)
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
    raw_artifact = read_artifact(download_path, Artifact)
    assert os.path.isdir(raw_artifact.path)
    assert len(os.listdir(raw_artifact.path)) == 3

    # Tokenize produced output with correct token count
    tokenize_artifact = read_artifact(tokenize_path, TokenizeMetadata)
    assert tokenize_artifact.num_tokens == 60  # 30 docs * 2 words each

    # Train produced a checkpoint
    train_artifact = read_artifact(train_path, TrainMetadata)
    assert train_artifact.tokens_seen > 0
    assert os.path.exists(train_artifact.checkpoint_path)


def test_runner_skips_completed_steps(tmp_path: Path):
    """Running the same pipeline twice should skip already-succeeded steps."""
    steps = _build_pipeline(tmp_path)

    runner1 = StepRunner()
    runner1.run(steps)

    # Record modification times
    tokenize_artifact_path = os.path.join(steps[1].output_path, ".artifact.json")
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

    train_artifact = read_artifact(steps[2].output_path, TrainMetadata)
    assert train_artifact.tokens_seen > 0


def test_runner_max_concurrent(tmp_path: Path):
    """max_concurrent=1 should still complete the full pipeline."""
    steps = _build_pipeline(tmp_path)
    runner = StepRunner()
    runner.run(steps, max_concurrent=1)

    train_artifact = read_artifact(steps[2].output_path, TrainMetadata)
    assert train_artifact.tokens_seen > 0


def test_runner_walks_transitive_deps(tmp_path: Path):
    """Passing only terminal steps should cause the runner to walk and run transitive deps."""
    executed: list[str] = []

    def record(name: str):
        def _fn(output_path: str) -> Artifact:
            executed.append(name)
            return Artifact(path=output_path)

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
        fn=lambda output_path: Artifact(path=output_path),
    )
    downstream_ran: list[str] = []

    def run_downstream(output_path: str) -> Artifact:
        downstream_ran.append(output_path)
        return Artifact(path=output_path)

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


# ---------------------------------------------------------------------------
# Pruning already-built subtrees
# ---------------------------------------------------------------------------


def _recording_step(name: str, out: str, executed: list[str], deps: list[StepSpec] | None = None) -> StepSpec:
    """A StepSpec whose fn appends ``name`` to ``executed`` when (and only when) it runs."""

    def _fn(output_path: str) -> Artifact:
        executed.append(name)
        return Artifact(path=output_path)

    return StepSpec(name=name, override_output_path=out, deps=deps or [], fn=_fn)


def _download_normalize_tokenize(tmp_path: Path, executed: list[str]) -> tuple[StepSpec, StepSpec, StepSpec]:
    download = _recording_step("download", (tmp_path / "download").as_posix(), executed)
    normalize = _recording_step("normalize", (tmp_path / "normalize").as_posix(), executed, deps=[download])
    tokenize = _recording_step("tokenize", (tmp_path / "tokenize").as_posix(), executed, deps=[normalize])
    return download, normalize, tokenize


def test_runner_prunes_built_terminal_subtree(tmp_path: Path):
    """A cached terminal must not rebuild its unbuilt intermediate ancestors.

    Regression for #6783: ``download → normalize → tokenize`` with only ``tokenize``
    cached used to rebuild the unpinned ``normalize`` (and ``download``) anyway.
    """
    executed: list[str] = []
    _download, _normalize, tokenize = _download_normalize_tokenize(tmp_path, executed)

    # Prime ONLY the terminal as a cached SUCCESS; its ancestors stay unbuilt.
    StatusFile(tokenize.output_path, worker_id()).write_status(STATUS_SUCCESS)

    StepRunner().run([tokenize])

    # Terminal served from cache, ancestors pruned: nothing ran.
    assert executed == []


def test_runner_builds_unbuilt_ancestors(tmp_path: Path):
    """With nothing cached, the full chain runs in dependency order (no over-pruning)."""
    executed: list[str] = []
    _download, _normalize, tokenize = _download_normalize_tokenize(tmp_path, executed)

    StepRunner().run([tokenize])

    assert executed == ["download", "normalize", "tokenize"]


def test_runner_prune_keeps_unbuilt_sibling_branch(tmp_path: Path):
    """A shared dep feeding a cached branch and an unbuilt branch still runs once.

    Diamond ``a → {b, c}``, ``b → d``, ``c → d``: ``c`` is cached but its dep ``d``
    is not. Pruning ``c`` must not strand ``d`` — ``d`` is still needed by ``b``.
    """
    executed: list[str] = []
    d = _recording_step("d", (tmp_path / "d").as_posix(), executed)
    b = _recording_step("b", (tmp_path / "b").as_posix(), executed, deps=[d])
    c = _recording_step("c", (tmp_path / "c").as_posix(), executed, deps=[d])
    a = _recording_step("a", (tmp_path / "a").as_posix(), executed, deps=[b, c])

    # ``c`` is already built; its dep ``d`` is not.
    StatusFile(c.output_path, worker_id()).write_status(STATUS_SUCCESS)

    StepRunner().run([a])

    assert "c" not in executed  # served from cache, not rebuilt
    assert executed.count("d") == 1  # built once, for the unbuilt ``b`` branch
    assert sorted(executed) == ["a", "b", "d"]


def test_runner_does_not_prune_dev_node(tmp_path: Path):
    """A mutable (dev) artifact at SUCCESS always rebuilds, so its deps are not pruned."""
    executed: list[str] = []
    dep = StepSpec(
        name="dep",
        override_output_path=(tmp_path / "dep").as_posix(),
        hash_attrs={FINGERPRINT_KEY: "dep-fp", VERSION_KEY: "2026.01.01"},
        fn=lambda output_path: executed.append("dep") or Artifact(path=output_path),
    )
    dev = StepSpec(
        name="dev",
        override_output_path=(tmp_path / "dev").as_posix(),
        deps=[dep],
        hash_attrs={FINGERPRINT_KEY: "dev-fp", VERSION_KEY: "dev"},
        fn=lambda output_path: executed.append("dev") or Artifact(path=output_path),
    )

    # Even with a SUCCESS status on disk, a dev version is rebuilt — and pulls its dep.
    StatusFile(dev.output_path, worker_id()).write_status(STATUS_SUCCESS)

    StepRunner().run([dev])

    assert "dev" in executed
    assert "dep" in executed


def test_runner_prune_cache_vanished_fails(tmp_path: Path, monkeypatch):
    """If a pruned node's cached output is gone at confirm time, the run fails loudly
    rather than running the node with its inputs pruned away."""
    executed: list[str] = []
    dep = _recording_step("dep", (tmp_path / "dep").as_posix(), executed)
    terminal = _recording_step("terminal", (tmp_path / "terminal").as_posix(), executed, deps=[dep])

    # Treat ``terminal`` as built (pruning ``dep``) although nothing is on disk — the
    # state if its cached output vanished between scheduling and confirmation.
    monkeypatch.setattr(step_runner_module, "_is_built", lambda s: s.output_path == terminal.output_path)

    with pytest.raises(RuntimeError, match=r"1 step\(s\) failed"):
        StepRunner().run([terminal])

    # The pruned node is not run (its inputs were dropped), and neither is its dep.
    assert executed == []


def test_runner_consumes_unbounded_iterator(tmp_path: Path):
    """The runner must not pre-consume the iterable — it must support unbounded generators.

    The generator yields forever unless ``stop`` is set; we set it from inside
    a terminal's function after N terminals have executed. A batch-flatten
    implementation would try to exhaust the generator before running any step
    and hang (caught by the per-test timeout).
    """

    stop = threading.Event()
    executed: list[str] = []
    lock = threading.Lock()
    n_terminals = 3

    def on_execute(name: str):
        def _fn(output_path: str) -> Artifact:
            with lock:
                executed.append(name)
                # Count terminals executed; signal the generator to stop once
                # we've run enough.
                terminal_count = sum(1 for e in executed if e.startswith("t_"))
            if terminal_count >= n_terminals:
                stop.set()
            return Artifact(path=output_path)

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

    def run_dep(output_path: str) -> Artifact:
        dep_runs.append(output_path)
        return Artifact(path=output_path)

    dep = StepSpec(
        name="shared_dep",
        override_output_path=(tmp_path / "shared_dep").as_posix(),
        fn=run_dep,
    )
    a = StepSpec(
        name="a",
        override_output_path=(tmp_path / "a").as_posix(),
        deps=[dep],
        fn=lambda output_path: Artifact(path=output_path),
    )
    b = StepSpec(
        name="b",
        override_output_path=(tmp_path / "b").as_posix(),
        deps=[dep],
        fn=lambda output_path: Artifact(path=output_path),
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
        return Artifact(path=output_path)

    step = StepSpec(
        name="fray_step",
        override_output_path=tmp_path.as_posix(),
        fn=my_step,
    )

    runner = StepRunner()
    runner.run([step])

    loaded = read_artifact(tmp_path.as_posix(), Artifact)
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


def _run_step_with_submit_spy(step: StepSpec, fray_client) -> _SubmitSpy:
    spy = _SubmitSpy(fray_client)
    with set_current_client(spy):
        StepRunner().run([step])
    return spy


def _call_remote_with_submit_spy(fn, fray_client) -> _SubmitSpy:
    spy = _SubmitSpy(fray_client)
    with set_current_client(spy):
        fn()
    return spy


def _assert_single_submit_extras(spy: _SubmitSpy, expected: list[str]) -> None:
    assert len(spy.requests) == 1
    assert spy.requests[0].environment.extras == expected


def _assert_single_submit_env(spy: _SubmitSpy, expected: dict[str, str]) -> None:
    assert len(spy.requests) == 1
    for key, value in expected.items():
        assert spy.requests[0].environment.env_vars[key] == value


def test_step_resources_dispatches_via_fray(tmp_path: Path, fray_client):
    """Setting ``resources`` on a StepSpec submits ``fn`` as a Fray job."""
    spy = _SubmitSpy(fray_client)

    custom = ResourceConfig.with_cpu(cpu=2, ram="8g")

    def my_step(output_path: str) -> Artifact:
        return Artifact(path=output_path)

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
    loaded = read_artifact(tmp_path.as_posix(), Artifact)
    assert loaded.path == tmp_path.as_posix()


def test_step_resources_dispatch_uses_device_extra(tmp_path: Path, fray_client):
    resources = ResourceConfig.with_gpu("H100", count=8)

    def my_step(output_path: str) -> Artifact:
        return Artifact(path=output_path)

    step = StepSpec(
        name="gpu_resourced_step",
        override_output_path=tmp_path.as_posix(),
        fn=my_step,
        resources=resources,
    )

    _assert_single_submit_extras(_run_step_with_submit_spy(step, fray_client), ["gpu"])


def test_remote_resources_dispatch_uses_device_extra(tmp_path: Path, fray_client):
    resources = ResourceConfig.with_gpu("H100", count=8)

    @remote(resources=resources)
    def my_step(output_path: str) -> Artifact:
        return Artifact(path=output_path)

    step = StepSpec(
        name="remote_gpu_step",
        override_output_path=tmp_path.as_posix(),
        fn=my_step,
    )

    _assert_single_submit_extras(_run_step_with_submit_spy(step, fray_client), ["gpu"])


def test_remote_dependency_groups_can_override_device_extra(tmp_path: Path, fray_client):
    resources = ResourceConfig.with_gpu("H100", count=8)

    @remote(resources=resources, pip_dependency_groups=[])
    def my_step(output_path: str) -> Artifact:
        return Artifact(path=output_path)

    step = StepSpec(
        name="remote_gpu_step_without_extras",
        override_output_path=tmp_path.as_posix(),
        fn=my_step,
    )

    _assert_single_submit_extras(_run_step_with_submit_spy(step, fray_client), [])


def test_remote_vllm_tpu_dependency_group_sets_target_device(tmp_path: Path, fray_client):
    resources = ResourceConfig.with_tpu("v6e-4")

    @remote(resources=resources, pip_dependency_groups=["eval", "vllm"])
    def my_step(output_path: str) -> Artifact:
        return Artifact(path=output_path)

    step = StepSpec(
        name="remote_vllm_tpu_step",
        override_output_path=tmp_path.as_posix(),
        fn=my_step,
    )

    spy = _run_step_with_submit_spy(step, fray_client)

    _assert_single_submit_extras(spy, ["eval", "vllm"])
    _assert_single_submit_env(spy, {"VLLM_TARGET_DEVICE": "tpu"})


def test_remote_direct_call_uses_device_extra(fray_client):
    resources = ResourceConfig.with_gpu("H100", count=8)

    @remote(resources=resources)
    def my_step() -> None:
        return None

    _assert_single_submit_extras(_call_remote_with_submit_spy(my_step, fray_client), ["gpu"])


def test_remote_direct_call_dependency_groups_can_override_device_extra(fray_client):
    resources = ResourceConfig.with_gpu("H100", count=8)

    @remote(resources=resources, pip_dependency_groups=[])
    def my_step() -> None:
        return None

    _assert_single_submit_extras(_call_remote_with_submit_spy(my_step, fray_client), [])


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


def test_runner_propagates_context_vars(tmp_path):
    """StepRunner must propagate contextvars to worker threads.

    This ensures that fray's ``set_current_client`` is visible inside step
    functions dispatched by the thread pool, so ZephyrContext (and anything
    else that calls ``current_client()``) picks up the correct client.
    """

    test_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("test_var", default=None)
    observed: list[str | None] = []

    def capture_ctx(output_path: str):
        observed.append(test_var.get())
        return Artifact(path=output_path)

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

    class FakeClient:
        """Marker client to verify propagation."""

        pass

    observed_clients: list[type] = []

    def check_client(output_path: str):
        client = current_client()
        observed_clients.append(type(client))
        return Artifact(path=output_path)

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
