# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import cast

import pytest
from verifyit.adapters.harness_runtime import score_corpus
from verifyit.file_ops.read import MAX_ARTIFACT_BYTES
from verifyit.grade import Status


def producer(tmp_path: Path, verdict: dict) -> tuple[Path, Path]:
    root = tmp_path / "harness"
    package = root / "lm_eval"
    tasks = package / "tasks"
    tasks.mkdir(parents=True)
    config = tasks / "fixture.yaml"
    config.write_text("task: fixture\n")
    (package / "verifyit_runtime.py").write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        "request = json.loads(Path(sys.argv[1]).read_text())\n"
        f"payload = {verdict!r}\n"
        "observations = payload['detail'].pop('observations')\n"
        "batches = []\n"
        "if observations:\n"
        "    Path(request['observations_dir'], 'batch.json').write_text(json.dumps(observations))\n"
        "    batches.append('batch.json')\n"
        "payload['detail']['observation_batches'] = batches\n"
        "(Path(os.environ['VERIFYIT_LOGS_DIR']) / 'corpus-verdict.json').write_text(json.dumps(payload))\n"
    )
    return root, config


def test_failed_runtime_cannot_export_positive_source_metrics(tmp_path):
    root, config = producer(
        tmp_path,
        {"status": "infra_error", "reward": 0, "detail": {"observations": [{"acc": 1}], "aggregates": {"acc": 1}}},
    )
    result = score_corpus(root, config, [{"doc": {}, "responses": ["answer"]}])
    assert result.verdict.status == Status.INFRA_ERROR
    assert result.verdict.reward == 0
    assert result.observations == ()
    assert result.aggregates == {}


def test_runtime_preserves_unbounded_named_metrics_and_raw_observations(tmp_path):
    root, config = producer(
        tmp_path,
        {
            "status": "scored",
            "reward": 0,
            "detail": {"observations": [{"word_perplexity": [-9, 3]}], "aggregates": {"word_perplexity": 20.0855}},
        },
    )
    result = score_corpus(root, config, [{"doc": {"text": "one two three"}, "responses": [-9]}])
    assert result.verdict.status == Status.SCORED
    assert result.verdict.reward == 0
    assert result.observations == ({"word_perplexity": [-9, 3]},)
    assert result.aggregates == {"word_perplexity": 20.0855}


def test_runtime_rejects_nonfinite_json_and_missing_sample_observations(tmp_path):
    root, config = producer(
        tmp_path, {"status": "scored", "reward": 0, "detail": {"observations": [], "aggregates": {"bleu": 100}}}
    )
    invalid = score_corpus(root, config, [{"doc": {}, "responses": [float("nan")]}])
    assert invalid.verdict.status == Status.INVALID_TASK
    assert invalid.verdict.reward == 0
    assert not invalid.aggregates
    with pytest.raises(RuntimeError, match="omitted samples"):
        score_corpus(root, config, [{"doc": {}, "responses": ["answer"]}])


@pytest.mark.parametrize("seed", [10**1000, True, -1, 2**32])
def test_malformed_aggregation_seed_cannot_export_corpus_metrics(tmp_path, seed):
    root, config = producer(
        tmp_path, {"status": "scored", "reward": 0, "detail": {"observations": [{"acc": 1}], "aggregates": {"acc": 1}}}
    )
    result = score_corpus(root, config, [{"doc": {}, "responses": ["answer"]}], aggregation_seed=seed)
    assert result.verdict.status == Status.INVALID_TASK
    assert result.verdict.reward == 0
    assert result.observations == ()
    assert result.aggregates == {}


def test_unhashable_execution_stage_is_invalid_task(tmp_path):
    root, config = producer(
        tmp_path, {"status": "scored", "reward": 0, "detail": {"observations": [{"acc": 1}], "aggregates": {"acc": 1}}}
    )
    result = score_corpus(root, config, [{"doc": {}, "responses": ["answer"]}], stage=cast(str, []))
    assert result.verdict.status == Status.INVALID_TASK
    assert result.verdict.reward == 0
    assert not result.aggregates


def test_observation_stage_rejects_premature_point_metrics(tmp_path):
    root, config = producer(
        tmp_path, {"status": "scored", "reward": 0, "detail": {"observations": [{"acc": 1}], "aggregates": {"acc": 1}}}
    )
    with pytest.raises(RuntimeError, match="premature aggregates"):
        score_corpus(root, config, [{"doc": {}, "responses": ["answer"]}], stage="observations")


@pytest.mark.parametrize("stage", ["complete", "observations"])
def test_large_corpus_preserves_all_batched_observations_and_source_aggregation(tmp_path, stage):
    root, config = producer(
        tmp_path, {"status": "scored", "reward": 0, "detail": {"observations": [], "aggregates": {}}}
    )
    (root / "lm_eval" / "verifyit_runtime.py").write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        "request = json.loads(Path(sys.argv[1]).read_text())\n"
        "observations = [{'acc': sample['responses'][0]} for sample in request['samples']]\n"
        "batches = []\n"
        "for start in range(0, len(observations), 10000):\n"
        "    filename = f'batch-{start}.json'\n"
        "    Path(request['observations_dir'], filename).write_text(json.dumps(observations[start:start+10000]))\n"
        "    batches.append(filename)\n"
        "aggregates = {} if request['stage'] == 'observations' else "
        "{'acc': sum(item['acc'] for item in observations) / len(observations)}\n"
        "verdict = {'status': 'scored', 'reward': 0, "
        "'detail': {'observation_batches': batches, 'aggregates': aggregates}}\n"
        "Path(os.environ['VERIFYIT_LOGS_DIR'], 'corpus-verdict.json').write_text(json.dumps(verdict))\n"
    )
    samples = [{"doc": {}, "responses": [index % 2]} for index in range(100_000)]
    expected = tuple({"acc": sample["responses"][0]} for sample in samples)
    assert len(json.dumps(expected).encode()) > MAX_ARTIFACT_BYTES

    result = score_corpus(root, config, samples, stage=stage)

    assert (result.verdict.status, result.verdict.reward) == (Status.SCORED, 0)
    assert result.observations == expected
    assert result.aggregates == ({} if stage == "observations" else {"acc": 0.5})
    assert "observations" not in result.verdict.detail
    assert len(json.dumps(result.verdict.detail).encode()) < MAX_ARTIFACT_BYTES


@pytest.mark.parametrize(
    "replacement",
    [
        '[{"acc": NaN}]',
        '[{"acc": 1e10000}]',
        '[{"acc": 0, "acc": 1}]',
    ],
)
def test_invalid_batch_json_cannot_export_source_metrics(tmp_path, replacement):
    root, config = producer(
        tmp_path, {"status": "scored", "reward": 0, "detail": {"observations": [{"acc": 1}], "aggregates": {"acc": 1}}}
    )
    script = root / "lm_eval" / "verifyit_runtime.py"
    script.write_text(
        script.read_text() + f"Path(request['observations_dir'], 'batch.json').write_text({replacement!r})\n"
    )

    with pytest.raises(RuntimeError, match="unreadable observation batch"):
        score_corpus(root, config, [{"doc": {}, "responses": ["answer"]}])


@pytest.mark.parametrize("artifact", ["oversized", "symlink"])
def test_observation_batches_retain_artifact_size_and_regular_file_boundary(tmp_path, artifact):
    root, config = producer(
        tmp_path, {"status": "scored", "reward": 0, "detail": {"observations": [{"acc": 1}], "aggregates": {"acc": 1}}}
    )
    script = root / "lm_eval" / "verifyit_runtime.py"
    mutation = (
        f"batch.write_bytes(b'x' * {MAX_ARTIFACT_BYTES + 1})\n"
        if artifact == "oversized"
        else "target = batch.with_name('target.json')\nbatch.rename(target)\nbatch.symlink_to(target)\n"
    )
    script.write_text(script.read_text() + "batch = Path(request['observations_dir'], 'batch.json')\n" + mutation)

    with pytest.raises(RuntimeError, match="unreadable observation batch"):
        score_corpus(root, config, [{"doc": {}, "responses": ["answer"]}])


@pytest.mark.parametrize("filenames", [["../samples.json"], ["batch.json", "batch.json"]])
def test_batch_manifest_cannot_escape_owned_directory_or_reuse_samples(tmp_path, filenames):
    root, config = producer(
        tmp_path, {"status": "scored", "reward": 0, "detail": {"observations": [{"acc": 1}], "aggregates": {"acc": 1}}}
    )
    script = root / "lm_eval" / "verifyit_runtime.py"
    script.write_text(
        script.read_text()
        + f"payload['detail']['observation_batches'] = {filenames!r}\n"
        + "Path(os.environ['VERIFYIT_LOGS_DIR'], 'corpus-verdict.json').write_text(json.dumps(payload))\n"
    )

    with pytest.raises(RuntimeError, match="distinct files within the observations directory"):
        score_corpus(root, config, [{"doc": {}, "responses": ["answer"]}] * len(filenames))


def test_extra_batch_observation_cannot_export_source_metrics(tmp_path):
    root, config = producer(
        tmp_path,
        {"status": "scored", "reward": 0, "detail": {"observations": [{"acc": 1}] * 2, "aggregates": {"acc": 1}}},
    )

    with pytest.raises(RuntimeError, match="too many sample observations"):
        score_corpus(root, config, [{"doc": {}, "responses": ["answer"]}])
