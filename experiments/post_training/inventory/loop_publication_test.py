# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from marin.execution.artifact import RECORD_FILENAME
from marin.execution.lazy import OUT, ArtifactStep, apply, resolve
from marin.rl.skyrl import SkyRLRun
from rigging.timing import Timestamp

from experiments.post_training import snowball_rl_gate as gate
from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training.inventory import loop_publication
from experiments.post_training.inventory.publication_test_support import result_record
from experiments.post_training.inventory.publication_test_support import write_json as _json

pytest_plugins = ("experiments.post_training.snowball_rl_test_fixtures",)


def _gate_result(output, verdict, interpretation):
    return gate.GateReport(
        path=output,
        preset="learning-medium",
        verdict=verdict,
        label=None,
        reason=interpretation,
        run=gate.RunMetadata(settings=(), tasks=None, init=None, replicates=1, requirements=()),
        control=gate.ControlMetadata(main_commit="control", revision=presets.PRESET_REVISION, runs=4),
        invariants=(),
        guards=(),
        reported=(),
        gaps=(),
    )


@pytest.mark.parametrize("mode", ["legacy", "snowball"])
def test_publication_retry_reuses_completed_training(tmp_path, monkeypatch, mode, recorded_runs, publication_transport):
    result = result_record()
    transport = publication_transport
    training_calls = []
    original = presets.PRESETS["learning-medium"]
    preset = replace(
        original, updates=2, measured=(1, 2), evaluations=(replace(original.evaluations[0], file="validation"),)
    )
    monkeypatch.setattr(presets, "PRESETS", dict(presets.PRESETS) | {preset.name: preset})
    start = 1794000000000
    job = SimpleNamespace(started_at=Timestamp.from_ms(start), finished_at=Timestamp.from_ms(start + 3600000))
    monkeypatch.setattr(
        loop_publication,
        "open_iris_client",
        lambda **kwargs: nullcontext(SimpleNamespace(job_status=lambda job_id: job)),
    )

    def write_native(output, attempt="native-1", kind="full"):
        record_path = output / RECORD_FILENAME
        saved = record_path.read_bytes() if record_path.exists() else None
        assert recorded_runs(preset, "test/training", initial_score=0) == output
        if saved is not None:
            record_path.write_bytes(saved)
        export = output / "exports"
        launch = json.loads((output / "resolved-launch.yaml").read_text())["config"]
        launch["run"] = {"id": "native-run", "attempt_id": attempt}
        launch["inputs"]["evaluation_data"]["full_panel"] = launch["inputs"]["validation_data"]
        launch["skyrl"]["data"] = {"expected_eval_rows": None, "val_data": launch["inputs"]["validation_data"]}
        destination = output / "terminal.json"
        if kind == "native-failed":
            destination.unlink(missing_ok=True)
            launch["artifacts"] = {"attempts_root": str(output / "attempts")}
            destination = output / "attempts" / f"{attempt}.json"
        _json(
            destination,
            {
                "config": launch,
                "result": {
                    "state": "failed" if kind == "native-failed" else "succeeded",
                    "iris_job_id": f"/tests/{attempt}",
                    "provenance": {"tree_hash": "native-tree", "base_commit": "native-base", "dirty": False},
                },
            },
        )
        resolved = json.loads(json.dumps(launch))
        resolved["skyrl"]["data"]["val_data"] = ["/native/evaluation.parquet"]
        (output / "resolved-launch.yaml").write_text(yaml.safe_dump({"config": resolved}))
        _json(export / "tracking.json", {"wandb_url": f"https://wandb.example/team/project/runs/{attempt}"})
        for step, windows in ((1, 2), (2, 3)):
            path = export / f"metrics/step-{step:06d}.json"
            _json(path, json.loads(path.read_text()) | {"attempt_id": attempt, "policy/policy_update_steps": windows})
        _json(
            export / "metrics/step-000003.json",
            json.loads((export / "metrics/step-000002.json").read_text())
            | {
                "step": 3,
                "trainer/global_step": 3,
                "attempt_id": "old-attempt",
                "policy/policy_update_steps": 999,
            },
        )
        for step in (0, 2):
            dump = export / f"dumped_evals/global_step_{step}_evals/panel.jsonl"
            count = int(kind) if kind in ("99", "98") else 100
            rows = [json.loads(line) for line in dump.read_text().splitlines()][:count]
            if step == 0:
                for index, row in enumerate(rows):
                    row["verification"]["score"] = float(index == 0)
            if kind == "foreign" and step == 2:
                rows[-1]["task_id"] = "foreign-task"
            if kind == "missing" and step == 0:
                dump.unlink()
            else:
                dump.write_text("".join(json.dumps(row) + "\n" for row in rows))
            path = export / f"metrics/step-{step:06d}.json"
            _json(
                path,
                json.loads(path.read_text())
                | {
                    "attempt_id": "old-attempt" if kind == "stale" and step == 0 else attempt,
                    "eval/all/num_attempted": len(rows),
                    "eval/all/num_scored": len(rows),
                },
            )
        decoy = export / "full_panel/dumped_evals/global_step_2_evals/panel.jsonl"
        decoy.parent.mkdir(parents=True, exist_ok=True)
        decoy.write_text(json.dumps(rows[0] | {"verification": rows[0]["verification"] | {"score": 1}}) + "\n")

    def train(config):
        output = Path(config["output"])
        output.mkdir(parents=True, exist_ok=True)
        training_calls.append(output)
        if mode == "legacy":
            _json(output / "exports/loop-result.json", result)
        else:
            write_native(output)
        return SkyRLRun(
            path=str(output),
            hf_model_uri=None,
            global_step=777,
            tokenizer_uri="test/model",
            tokenizer_revision="revision",
            checkpoint_root=str(output / "checkpoints"),
            draft_checkpoint_root=None,
            terminal_manifest_uri=str(output / "terminal.json"),
            iris_job_id="/tests/native-1",
        )

    training = ArtifactStep(
        name="test/training",
        version=presets.PRESET_REVISION,
        artifact_type=SkyRLRun,
        build_config=lambda ctx: {"output": ctx.output_path},
        run=train,
    )
    resolve(training)
    gate_step = None
    if mode == "snowball":
        gate_step = apply(
            "test/gate-native-1",
            _gate_result,
            version=presets.PRESET_REVISION,
            artifact_type=gate.GateReport,
            verdict="fail",
            interpretation="original",
            output=OUT,
        )
        resolve(gate_step)
        job.finished_at = None
        unavailable = loop_publication.publication_step(training, gate=gate_step)
        with pytest.raises(RuntimeError, match=r"step\(s\) failed"):
            resolve(unavailable)
        assert transport.rows == {}
        job.finished_at = Timestamp.from_ms(start + 3600000)
        metric = Path(training.path()) / "exports/metrics/step-000001.json"
        values = json.loads(metric.read_text())
        for field, value in (("policy/policy_update_steps", None), ("policy/raw_grad_norm", float("nan"))):
            _json(metric, values | {field: value})
            with pytest.raises(RuntimeError, match=r"step\(s\) failed"):
                resolve(unavailable)
            assert transport.rows == {}
        _json(metric, values)
    publication = loop_publication.publication_step(training, gate=gate_step)
    transport.fail_readback = True
    with pytest.raises(RuntimeError, match=r"step\(s\) failed"):
        resolve(publication)
    summary_path = (
        Path(publication.deps[0].path()) / "loop-result.json"
        if mode == "snowball"
        else Path(training.path()) / "exports/loop-result.json"
    )
    retained = summary_path.read_bytes()
    job.finished_at = Timestamp.from_ms(start + 7200000)
    published = resolve(publication)
    assert len(training_calls) == 1 and len(transport.rows) == 1
    assert summary_path.read_bytes() == retained
    stored = next(iter(transport.rows.values()))
    assert json.loads((Path(published.path) / "publication.json").read_text())["verified"]
    if mode == "legacy":
        assert stored == result
        return
    assert stored["status"] == "succeeded" and stored["verdict"] == "fail"
    assert stored["optimizer_steps"] == 5 and stored["elapsed_seconds"] == 3600
    assert stored["baseline"] == {"reward": 0.01, "attempted": 100, "scored": 100}
    assert stored["final"] == {"reward": 0.5, "attempted": 100, "scored": 100}
    assert stored["wandb_url"].endswith("/native-1")
    assert stored["evaluation"]["task_count"] == 100
    assert stored["evaluation"]["task_set_sha256"] is None and stored["evaluation"]["protocol_sha256"] is None
    assert stored["baseline_source"].endswith("global_step_0_evals")
    job.finished_at = Timestamp.from_ms(start + 3600000)
    identities = {published.attempt_id}
    for index, kind in enumerate(("99", "98", "missing", "stale", "foreign", "native-failed"), 2):
        write_native(Path(training.path()), f"native-{index}", kind)
        completed_gate = apply(
            f"test/gate-native-{index}",
            _gate_result,
            version=presets.PRESET_REVISION,
            artifact_type=gate.GateReport,
            verdict="invalid",
            interpretation=kind,
            output=OUT,
        )
        resolve(completed_gate)
        item = resolve(loop_publication.publication_step(training, gate=completed_gate))
        assert item.attempt_id not in identities
        identities.add(item.attempt_id)
        record = transport.rows[(item.run_id, item.attempt_id)]
        assert record["status"] == ("failed" if kind == "native-failed" else "succeeded")
        assert record["evaluation"]["task_count"] == 100
        if kind in ("99", "98"):
            assert all(
                record[phase]["scored"] == record[phase]["attempted"] == int(kind) for phase in ("baseline", "final")
            )
        if kind in ("missing", "stale"):
            assert record["baseline"] is None and record["final"] == {"reward": 0.5, "attempted": 100, "scored": 100}
        if kind == "foreign":
            assert record["baseline"] == {"reward": 0.01, "attempted": 100, "scored": 100} and record["final"] is None
    changed_gate = apply(
        "test/another-group-attempt",
        _gate_result,
        version=presets.PRESET_REVISION,
        artifact_type=gate.GateReport,
        verdict="invalid",
        interpretation="another replica changed",
        output=OUT,
    )
    resolve(changed_gate)
    another = resolve(loop_publication.publication_step(training, gate=changed_gate))
    assert another.attempt_id not in identities
    assert len(training_calls) == 1
    assert len(transport.rows) == 8
