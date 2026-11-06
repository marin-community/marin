# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snowball publication consumes real raw attempt records at a fake native edge."""

import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from marin.execution.artifact import RECORD_FILENAME
from marin.execution.lazy import OUT, ArtifactStep, apply, resolve
from marin.rl.skyrl import ResolvedModelLocator, SkyRLOutputPaths, SkyRLRun, SkyRLRunConfig
from rigging.timing import Timestamp

from experiments.post_training import snowball_rl_gate as gate
from experiments.post_training import snowball_rl_presets as presets
from experiments.post_training import snowball_rl_publication as publication
from experiments.post_training.inventory import loop_result
from experiments.post_training.snowball_rl_test_fixtures import write_json

pytest_plugins = ("experiments.post_training.snowball_rl_test_fixtures",)
pytestmark = pytest.mark.usefixtures("synthetic_current_controls", "source_provenance")


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


def test_snowball_publication_retries_reuse_native_attempt_and_retained_summary(
    tmp_path,
    monkeypatch,
    recorded_runs,
    publication_transport,
):
    original = presets.PRESETS["learning-medium"]
    preset = replace(
        original, updates=2, measured=(1, 2), evaluations=(replace(original.evaluations[0], file="validation"),)
    )
    monkeypatch.setattr(presets, "PRESETS", dict(presets.PRESETS) | {preset.name: preset})
    start = 1794000000000
    job = SimpleNamespace(started_at=Timestamp.from_ms(start), finished_at=Timestamp.from_ms(start + 3600000))
    monkeypatch.setattr(presets, "iris_client", lambda: nullcontext(SimpleNamespace(job_status=lambda name: job)))
    calls = []
    scenario = {"kind": "full"}

    def native(config):
        output = Path(config.output.export_root).parent
        record_path = output / RECORD_FILENAME
        saved = record_path.read_bytes() if record_path.exists() else None
        recorded_runs(preset, "test/training", initial_score=0, attempt_id=config.attempt_id)
        if saved is not None:
            record_path.write_bytes(saved)
        launch = json.loads((output / "resolved-launch.yaml").read_text())["config"]
        launch["run"] = {"id": config.run_id, "attempt_id": config.attempt_id}
        write_json(
            output / "terminal.json",
            {
                "config": launch,
                "result": {
                    "state": "succeeded",
                    "iris_job_id": "/tests/native",
                    "provenance": {},
                },
            },
        )
        (output / "resolved-launch.yaml").write_text(yaml.safe_dump({"config": launch}))
        export = output / "exports"
        write_json(
            export / "tracking.json",
            {"attempt_id": config.attempt_id, "wandb_url": "https://wandb.ai/team/project/runs/" + config.attempt_id},
        )
        for step, windows in ((1, 2), (2, 3)):
            path = export / f"metrics/step-{step:06d}.json"
            values = json.loads(path.read_text()) | {"policy/policy_update_steps": windows}
            if scenario["kind"] == "nonfinite" and step == 1:
                values["policy/raw_grad_norm"] = float("nan")
            write_json(path, values)
        write_json(
            export / "metrics/step-000003.json",
            {
                "step": 3,
                "trainer/global_step": 3,
                "timing/step_wall/unaccounted": 0,
                "attempt_id": "old-attempt",
                "policy/policy_update_steps": 999,
                "policy/policy_loss": 0.1,
                "policy/raw_grad_norm": 0.2,
            },
        )
        kind = scenario["kind"]
        for step in (0, 2):
            dump = export / f"dumped_evals/global_step_{step}_evals/panel.jsonl"
            rows = [json.loads(line) for line in dump.read_text().splitlines()][: 98 if kind == "98" else 100]
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
            write_json(
                path,
                json.loads(path.read_text())
                | {
                    "attempt_id": "old-attempt" if kind == "stale" and step == 0 else config.attempt_id,
                    "eval/all/num_attempted": len(rows),
                    "eval/all/num_scored": len(rows),
                },
            )
        decoy = export / "full_panel/dumped_evals/global_step_2_evals/panel.jsonl"
        write_json(decoy, rows[0] | {"verification": rows[0]["verification"] | {"score": 1}})
        return SkyRLRun(
            path=str(output),
            hf_model_uri=None,
            global_step=777,
            tokenizer_uri="model",
            tokenizer_revision="revision",
            checkpoint_root=str(output / "checkpoints"),
            draft_checkpoint_root=None,
            terminal_manifest_uri=str(output / "terminal.json"),
            iris_job_id="/tests/native",
        )

    monkeypatch.setattr(loop_result, "run_skyrl", native)

    def produce(output, attempt):
        config = SkyRLRunConfig(
            launch_config_yaml=yaml.safe_dump({"inputs": {"train_data": [{"identity": "training@1"}]}}),
            run_id="native-run",
            attempt_id=attempt,
            model=ResolvedModelLocator("model", "initializer@1", "model", "tokenizer", "revision"),
            output=SkyRLOutputPaths(
                str(output / "checkpoints"),
                str(output / "exports"),
                str(output / "attempts"),
                str(output / "resolved-launch.yaml"),
                str(output / "terminal.json"),
            ),
            export_hf=False,
            draft_checkpoint_root=None,
        )
        return loop_result.run_loop(
            loop_result.LoopRunConfig(
                config,
                preset.name,
                2,
                snowball=loop_result.SnowballAttempt("native-fingerprint", "p40"),
            )
        )

    def train(config):
        calls.append(config["output"])
        return produce(Path(config["output"]), "native-1")

    training = ArtifactStep(
        name="test/training",
        version=presets.PRESET_REVISION,
        artifact_type=SkyRLRun,
        build_config=lambda ctx: {"output": ctx.output_path},
        run=train,
    )
    resolve(training)

    def completed_gate(attempt, verdict):
        step = apply(
            f"test/gate-{attempt}",
            _gate_result,
            version=presets.PRESET_REVISION,
            artifact_type=gate.GateReport,
            verdict=verdict,
            interpretation=attempt,
            output=OUT,
        )
        resolve(step)
        return step

    job.finished_at = None
    produce(Path(training.path()), "timing-unavailable")
    unavailable = publication.publication_step(training, gate=completed_gate("timing-unavailable", "invalid"))
    with pytest.raises(RuntimeError) as caught:
        resolve(unavailable)
    cause = caught.value
    causes = []
    while cause is not None:
        causes.append(str(cause))
        cause = cause.__cause__
    assert any("positive elapsed time" in message for message in causes), causes
    assert publication_transport.rows == {}
    job.finished_at = Timestamp.from_ms(start + 3600000)
    produce(Path(training.path()), "native-1")
    source_gate = completed_gate("native-1", "fail")
    report = publication.publication_step(training, gate=source_gate)
    publication_transport.fail_readback = True
    with pytest.raises(RuntimeError, match=r"step\(s\) failed"):
        resolve(report)
    summary_path = Path(report.deps[0].path()) / "loop-result.json"
    retained = summary_path.read_bytes()
    job.finished_at = Timestamp.from_ms(start + 7200000)
    published = resolve(report)
    assert len(calls) == 1
    assert summary_path.read_bytes() == retained
    assert json.loads((Path(published.path) / "publication.json").read_text())["verified"]
    stored = publication_transport.rows[(published.run_id, published.attempt_id)]
    assert stored["status"] == "succeeded" and stored["verdict"] == "fail"
    assert stored["optimizer_steps"] == 5 and stored["elapsed_seconds"] == 3600
    assert stored["baseline"] == {"reward": 0.01, "attempted": 100, "scored": 100}
    assert stored["final"] == {"reward": 0.5, "attempted": 100, "scored": 100}
    assert stored["wandb_url"].endswith("/native-1")
    assert stored["evaluation"]["task_count"] == 100
    identities = {published.attempt_id}
    for index, kind in enumerate(("98", "missing", "stale", "foreign", "nonfinite"), 2):
        scenario["kind"] = kind
        produce(Path(training.path()), f"native-{index}")
        item = resolve(publication.publication_step(training, gate=completed_gate(f"native-{index}", "invalid")))
        assert item.attempt_id not in identities
        identities.add(item.attempt_id)
        record = publication_transport.rows[(item.run_id, item.attempt_id)]
        assert record["status"] == "succeeded" and record["verdict"] == "invalid"
        if kind == "98":
            assert all(record[phase]["scored"] == record[phase]["attempted"] == 98 for phase in ("baseline", "final"))
        if kind in ("missing", "stale"):
            assert record["baseline"] is None and record["final"] == {"reward": 0.5, "attempted": 100, "scored": 100}
        if kind == "foreign":
            assert record["baseline"] == {"reward": 0.01, "attempted": 100, "scored": 100} and record["final"] is None
        if kind == "nonfinite":
            assert record["optimizer_steps"] == 3
    another = resolve(publication.publication_step(training, gate=completed_gate("another-group", "invalid")))
    assert another.attempt_id not in identities
    assert len(calls) == 1
