# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path

import fsspec
from marin.rl.skyrl import SkyRLRun
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.long_loop import rl_terminal_evaluation as terminal


def test_preserve_completed_micro210_reads_the_native_launch_envelope(tmp_path, monkeypatch):
    # Envelope and generation-only result contract from completed Mochi .129 Micro210.
    output = "s3://marin-us-east-08a/marin/evaluations/mochi129"
    temporary = "s3://marin-us-east-08a/tmp/ttl=14d/mochi129"
    filesystem = fsspec.filesystem("file")

    def resolve(url, **kwargs):
        return filesystem, str(tmp_path / str(url).removeprefix("s3://"))

    def open_url(url, mode="rb", **kwargs):
        fs, key = resolve(url)
        if "w" in mode:
            Path(key).parent.mkdir(parents=True, exist_ok=True)
        return fs.open(key, mode, **kwargs)

    monkeypatch.setattr("rigging.filesystem.storage_path._url_to_fs", resolve)
    monkeypatch.setattr("rigging.filesystem.storage_path._open_url", open_url)
    monkeypatch.setattr("marin.rl.skyrl_records.filesystem_for", resolve)
    monkeypatch.setattr("rigging.fsutil.verified_copy.filesystem_for", resolve)
    (StoragePath(output) / "resolved-launch.yaml").write_text(
        json.dumps(
            {
                "config": {"artifacts": {"attempts_root": f"{temporary}/attempts"}},
                "train_data_sources": [],
                "val_data_sources": ["frozen-micro210"],
            }
        )
    )
    archive = StoragePath(temporary) / "attempts/acc93c2f5dba/trial/result.json"
    archive.write_bytes(b'{"verifier_result":{"rewards":{"reward":1}}}')
    result = SkyRLRun(
        hf_model_uri=None,
        global_step=None,
        tokenizer_uri="marin-community/marin-tokenizer-tools",
        tokenizer_revision="3f6879acdaf3dae7874b2d48c7d11b9d8c6faf09",
        checkpoint_root=f"{temporary}/checkpoints",
        draft_checkpoint_root=None,
        terminal_manifest_uri=f"{output}/terminal.json",
        iris_job_id="/fixture/mochi129-micro210",
        run_id="long-loop/mochi-v5-exp3-rl/micro210-2026.11.25.129",
        attempt_id="acc93c2f5dba",
    )

    def finish(arguments):
        receipt = Path(arguments[arguments.index("--receipt") + 1])
        receipt.write_text(json.dumps({"scored": 421, "assigned": 434, "coverage_eligible": True}))

    monkeypatch.setattr(terminal, "native_command", finish)
    recovered = terminal.preserve_evaluation(terminal.EvaluationPreservation(result, output, f"{output}/preserve"))
    assert recovered == result
    permanent = f"{output}/trial-archives"
    assert (StoragePath(permanent) / "attempts/acc93c2f5dba/trial/result.json").read_bytes() == archive.read_bytes()
    coverage = json.loads((StoragePath(output) / "coverage-ready.json").read_bytes())
    manifest = StoragePath(coverage["archive_copy"]["manifest_uri"]).read_bytes()
    assert coverage["archive_copy"]["manifest_sha256"] == hashlib.sha256(manifest).hexdigest()
    assert coverage["archive_copy"]["files"] == 1
    assert coverage["scored"] == 421
    assert json.loads((StoragePath(output) / "preserve/preservation.json").read_bytes()) == {
        "source": output,
        "job_id": result.iris_job_id,
    }
