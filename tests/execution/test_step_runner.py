# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path

from dataclasses import dataclass

from marin.execution.artifact import Artifact, PathMetadata
from marin.execution.executor import ExecutorStep
from marin.execution.step_spec import StepSpec
from marin.execution.step_runner import StepRunner, resolve_executor_step

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
    resolved = resolve_executor_step(
        step,
        config={},
        output_path="/out/train-abc123",
        deps=["/out/download-abc123", "/out/tokenize-def456"],
    )
    assert resolved.deps == ["/out/download-abc123", "/out/tokenize-def456"]


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
        deps=[download_step.output_path],
        fn=lambda output_path: tokenize_data(
            output_path,
            Artifact.load(download_step.output_path, PathMetadata),
            tokenizer,
        ),
    )
    train_step = StepSpec(
        name="train",
        output_path_prefix=tmp_path_posix,
        deps=[tokenize_step.output_path],
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
