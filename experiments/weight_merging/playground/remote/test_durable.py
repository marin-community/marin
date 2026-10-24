# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from experiments.weight_merging.playground.remote.durable import REFERENCE_FIELDS, CandidateStore, digest


def complete_run(root: Path) -> tuple[Path, dict, list[dict]]:
    run = root / "run"
    run.mkdir()
    panel = json.dumps({"items": [{"id": "item"}]}).encode()
    reference = dict.fromkeys(REFERENCE_FIELDS, "frozen")
    reference.update(panel_sha256=digest(panel), decode_seeds=[17, 29], models={})
    candidates = [{"name": "instruct", "source": "instruct"}]
    metadata = {**reference, "candidates": candidates}
    for name, data in (
        ("panel.json", panel),
        ("run-metadata.json", json.dumps(metadata).encode()),
        ("model-paths.json", b"{}"),
        ("model-sources.json", b"{}"),
        ("reload-checks.json", b"[]"),
    ):
        (run / name).write_bytes(data)
    rows = [
        {"candidate": "instruct", "item_id": "item", "mode": mode, "seed": seed, "tokens": [42]}
        for mode, seed in (("greedy", 0), ("sample", 17), ("sample", 29))
    ]
    (run / "outputs.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return run, reference, candidates


def test_interrupted_publication_never_becomes_done_and_retry_commits(tmp_path, monkeypatch):
    run, reference, candidates = complete_run(tmp_path)
    store = CandidateStore(str(tmp_path / "remote"), reference, candidates)
    store.initialize()
    put = store.fs.pipe_file

    def interrupted(path, data):
        if path.endswith("done.json"):
            raise OSError("preempted before done marker")
        return put(path, data)

    with monkeypatch.context() as patch:
        patch.setattr(store.fs, "pipe_file", interrupted)
        with pytest.raises(OSError, match="preempted"):
            store.publish(run, "instruct")
    assert not store.done("instruct")
    store.publish(run, "instruct")
    assert store.done("instruct")
    store.collect(tmp_path / "collected")
    assert (tmp_path / "collected/outputs.jsonl").read_bytes() == (run / "outputs.jsonl").read_bytes()
    (tmp_path / "remote/candidates/instruct/run/outputs.jsonl").write_text("corrupted")
    with pytest.raises(ValueError, match="object changed"):
        store.done("instruct")


def test_partial_candidate_and_changed_recipe_cannot_be_reused(tmp_path):
    run, reference, candidates = complete_run(tmp_path)
    store = CandidateStore(str(tmp_path / "remote"), reference, candidates)
    original = (run / "outputs.jsonl").read_text()
    (run / "outputs.jsonl").write_text(original.splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="incomplete"):
        store.publish(run, "instruct")
    assert not store.done("instruct")
    (run / "outputs.jsonl").write_text(original)
    store.publish(run, "instruct")
    changed = CandidateStore(str(tmp_path / "remote"), reference, [{**candidates[0], "rope_theta": 10000}])
    with pytest.raises(ValueError, match="contract changed"):
        changed.done("instruct")


def test_registered_checkpoint_resume_uses_hashes_across_pod_paths(tmp_path):
    run, reference, candidates = complete_run(tmp_path)
    source = {"path": "/old-pod/model", "weights_sha256": {"weights": "original"}, "config_sha256": "config"}
    reference["models"] = {"instruct": source}
    metadata = json.loads((run / "run-metadata.json").read_text())
    metadata["models"] = {"instruct": {**source, "path": "/new-pod/model"}}
    (run / "run-metadata.json").write_text(json.dumps(metadata))
    store = CandidateStore(str(tmp_path / "remote"), reference, candidates)
    store.publish(run, "instruct")
    assert store.done("instruct")
    source["weights_sha256"] = {"weights": "changed"}
    changed = CandidateStore(str(tmp_path / "remote"), reference, candidates)
    with pytest.raises(ValueError, match="contract changed"):
        changed.done("instruct")
