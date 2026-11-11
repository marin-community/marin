# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from marin.execution.lazy import StepContext
from marin.experiment.cli import graph_handles
from rigging.timing import Duration

from experiments.post_training.long_loop import spec_pipeline
from experiments.post_training.long_loop.__main__ import validate_graph
from experiments.post_training.long_loop.execution_policy import DEFAULT_TASK_TIMEOUT_SECONDS
from experiments.post_training.long_loop.run_spec import allocation_peaks, resolve_spec, stage_blockers
from experiments.post_training.long_loop.spec_pipeline import build_spec_graph

SPEC = Path(__file__).parents[1] / "specs" / "round1.yaml"


def test_production_and_smoke_preserve_data_context_and_quota():
    production = resolve_spec(SPEC, smoke=False)
    smoke = resolve_spec(SPEC, smoke=True)
    assert smoke["datasets"] == production["datasets"]
    assert smoke["sft"]["data"] == production["sft"]["data"]
    actual = hashlib.sha256(
        json.dumps(production["sft"]["data"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert actual == production["datasets"]["sft"]["provenance"]["canonical_data_sha256"]
    assert smoke["sft"]["stop_after_steps"] < production["sft"]["stop_after_steps"]
    assert smoke["rl"]["config"]["trainer"]["max_steps"] < production["rl"]["config"]["trainer"]["max_steps"]
    assert smoke["sft"]["model"] == production["sft"]["model"]
    assert smoke["rl"]["config"]["context_budget"] == production["rl"]["config"]["context_budget"]
    for field in ("rollout_buffer", "policy", "ref"):
        assert smoke["rl"]["config"]["trainer"][field] == production["rl"]["config"]["trainer"][field]
    assert smoke["blockers"] == production["blockers"]


def test_changed_pin_and_semantic_smoke_override_fail_before_graph(tmp_path):
    root = tmp_path / "specs"
    shutil.copytree(SPEC.parent, root)
    path = root / SPEC.name
    raw = yaml.safe_load(path.read_text())
    raw["smoke_overrides"]["context_length"] = 2048
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="Smoke cannot change"):
        resolve_spec(path, smoke=True)
    shutil.copyfile(SPEC, path)
    data = root / "sft-data.yaml"
    data.write_text(data.read_text() + "\n# changed bytes\n")
    with pytest.raises(ValueError, match="Pinned file changed"):
        resolve_spec(path, smoke=False)


def test_actual_sft_graph_constructs_without_admitting_unprepared_rl(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-02a/marin/")
    resolved = resolve_spec(SPEC, smoke=False)
    terminal = build_spec_graph(resolved, "2026.11.09.12", through="sft")
    validate_graph(terminal, tmp_path, "s3://marin-us-east-02a/marin/")
    rows = json.loads((tmp_path / "graph.json").read_text())
    assert [row["stage"].rsplit("/", 1)[-1] for row in rows] == ["base", "inputs", "sft"]
    assert rows[0]["source"] == resolved["initializer"]["uri"]
    assert terminal.runtime_args["skyrl_execution"].priority == "production"
    resolved["blockers"] = ["Missing admitted agentic manifest"]
    with pytest.raises(ValueError, match="Run is not admitted"):
        build_spec_graph(resolved, "2026.11.09.12", through="archive")


@pytest.mark.parametrize("stage", ["coordinator", "rl", "evaluation", "endpoint"])
def test_new_submission_rejects_stale_interactive_priority(stage):
    resolved = resolve_spec(SPEC, smoke=False)
    if stage == "coordinator":
        runtime = resolved["execution"]["coordinator"]
    elif stage == "endpoint":
        runtime = resolved[stage]
    else:
        runtime = resolved[stage]["execution"]
    runtime["priority"] = "interactive"
    with pytest.raises(ValueError, match="production"):
        allocation_peaks(resolved)


def test_rl_smoke_stops_before_eval_and_never_replays_sft(monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-02a/marin/")
    resolved = resolve_spec(SPEC, smoke=True)
    assert not stage_blockers(resolved, "rl")
    with pytest.raises(ValueError, match="adopted_sft"):
        build_spec_graph(resolved, "2026.11.10.1", through="rl")
    resolved["adopted_sft"] = {"version": "2026.11.09.12", "uri": "s3://test/committed-sft"}
    terminal = build_spec_graph(resolved, "2026.11.10.1", through="rl")
    names = [step.name for step in graph_handles([terminal])]
    assert any(name.endswith("/adopt-sft") for name in names)
    assert not any(name.endswith("/sft") or "micro-" in name or "archive" in name for name in names)
    assert terminal.runtime_args["skyrl_execution"].priority == "production"


def test_adopted_sft_full_graph_report_has_no_fresh_base_dependency(monkeypatch, tmp_path):
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-02a/marin/")
    monkeypatch.setenv("WANDB_API_KEY", "offline-validation-fixture")
    resolved = resolve_spec(SPEC, smoke=True)
    # Isolate graph wiring; this does not waive the live artifact-admission checks.
    resolved["blockers"] = []
    resolved["adopted_sft"] = {"version": "2026.11.09.12", "uri": "s3://test/committed-sft", "step": 746}
    handoff_root = tmp_path / "verified-handoff"
    (handoff_root / "hf").mkdir(parents=True)
    (handoff_root / "hf/config.json").write_text(
        json.dumps(
            {
                "model_type": "grug_moe",
                "num_attention_heads": 20,
                "num_key_value_heads": 5,
                "num_hidden_layers": 26,
                "global_every": 4,
            }
        )
    )
    resolved["adopted_handoff"] = {
        "uri": str(handoff_root),
        "source_checkpoint": "s3://test/committed-sft/checkpoints/step-746",
        "files": [],
    }
    resolved["adopted_endpoint"] = {
        "uri": "s3://test/verified-endpoint",
        "model_uri": str(handoff_root / "hf"),
        "tokenizer_revision": resolved["tokenizer"]["revision"],
        "settings": resolved["endpoint"],
        "files": [],
    }
    terminal = build_spec_graph(resolved, "2026.11.10.1", through="archive")
    names = [step.name for step in graph_handles([terminal])]
    validate_graph(terminal, tmp_path / "full-graph", "s3://marin-us-east-02a/marin/")
    executions = [
        value
        for step in graph_handles([terminal])
        for value in [step.runtime_args.get("skyrl_execution")]
        if value is not None
    ]
    assert {execution.coordinator_name for execution in executions} == {"long-loop-full-2026.11.10.1"}
    assert "long-loop/report" in names
    assert any(name.endswith("/adopt-sft") for name in names)
    assert not any(name.endswith("/sft") or name.endswith("/base") for name in names)
    handoff = next(step for step in graph_handles([terminal]) if step.name.endswith("/sft-handoff"))
    assert handoff.adopt_source == str(handoff_root)
    endpoint = next(step for step in graph_handles([terminal]) if step.name.endswith("/endpoint"))
    assert endpoint.adopt_source == "s3://test/verified-endpoint"
    assert any(step.name.endswith("/verify-endpoint") for step in endpoint.deps)
    assert any(step.name.endswith("/verify-handoff") for step in handoff.deps)
    resolved["rl"]["config"]["trainer"]["use_sample_packing"] = False
    invalid = build_spec_graph(resolved, "2026.11.10.2", through="rl")
    with pytest.raises(ValueError, match="context parallel is only supported with sample packing"):
        validate_graph(invalid, tmp_path / "invalid", "s3://marin-us-east-02a/marin/")


def test_endpoint_reuse_rejects_different_export():
    resolved = resolve_spec(SPEC, smoke=True)
    resolved["adopted_sft"] = {"version": "v", "uri": "s3://test/sft", "step": 746}
    resolved["adopted_handoff"] = {
        "uri": "s3://test/hf",
        "source_checkpoint": "s3://test/sft/checkpoints/step-746",
        "files": [],
    }
    resolved["adopted_endpoint"] = {"model_uri": "s3://test/other/hf"}
    with pytest.raises(ValueError, match="identical export"):
        build_spec_graph(resolved, "2026.11.10.3", through="rl")


def test_rl_keeps_child_execution_profile_without_gpu_probe_jobs():
    resolved = resolve_spec(SPEC, smoke=True)
    resolved["adopted_sft"] = {"version": "2026.11.09.12", "uri": "s3://test/sft", "step": 746}
    terminal = build_spec_graph(resolved, "2026.11.10.4", through="rl")
    execution = terminal.runtime_args["skyrl_execution"]
    assert execution.container_profile == resolved["rl"]["execution"]["container_profile"] == 4
    assert execution.memory == resolved["rl"]["execution"]["memory"]
    assert execution.memory != resolved["execution"]["coordinator"]["memory"]
    assert not any("preflight" in step.name or "probe" in step.name for step in graph_handles([terminal]))
    resolved["execution"]["coordinator"]["container_profile"] = None
    with pytest.raises(ValueError, match="privileged coordinator"):
        build_spec_graph(resolved, "2026.11.10.4", through="rl")


@pytest.mark.parametrize("override", [None, 345600])
def test_emitted_stage_timeouts_default_to_three_days(tmp_path, monkeypatch, override):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    monkeypatch.setenv("WANDB_API_KEY", "offline-fixture")
    resolved = resolve_spec(SPEC, smoke=False)
    expected = DEFAULT_TASK_TIMEOUT_SECONDS if override is None else override
    if override is not None:
        resolved["execution"]["coordinator"].update(
            job_timeout_seconds=override, coordinator_timeout_hours=override // 3600
        )
        resolved["rl"]["execution"]["job_timeout_seconds"] = override
        resolved["evaluation"]["execution"]["job_timeout_seconds"] = override
        resolved["execution"]["task_timeouts"] = {k: override for k in ("sft", "export", "eval_inputs", "archive")}
    remote = spec_pipeline.remote
    dispatched = {}

    def collect(fn, **kwargs):
        result = remote(fn, **kwargs)
        dispatched[fn.__name__] = result.timeout
        return result

    monkeypatch.setattr(spec_pipeline, "remote", collect)
    terminal = build_spec_graph(resolved, "2026.11.12", through="archive")
    stages = graph_handles([terminal])
    executions = [s.runtime_args["skyrl_execution"] for s in stages if "skyrl_execution" in s.runtime_args]
    assert executions
    assert all(e.job_timeout_seconds == expected for e in executions)
    assert all(e.coordinator_timeout_hours * 3600 == expected for e in executions)
    assert dispatched == {
        k: Duration.from_seconds(expected) for k in ("export_sft", "prepare_micro", "archive_graph_run")
    }
    trained = next(s for s in stages if s.name.endswith("/sft"))
    context = StepContext.for_run(str(tmp_path / "sft"), str(tmp_path), deps=trained.deps)
    assert trained.build_config(context)["recipe"]["task_timeout_seconds"] == expected
    learned = next(s for s in stages if s.name.endswith("/" + resolved["rl"]["name"]))
    rl_context = StepContext.for_run(
        str(tmp_path / "rl"), str(tmp_path), deps=learned.deps, runtime_args=learned.runtime_args
    )
    launch = learned.build_config(replace(rl_context, is_fingerprint=True))
    emitted = yaml.safe_load(launch.launch_config_yaml)["skyrl"]["trainer"]["callbacks"]
    assert not any(c["type"] == "wall_time_budget" for c in emitted)
    resolved["rl"]["wall_clock_budget"] = {"seconds": 43200, "drain_margin": 1800}
    budgeted = build_spec_graph(resolved, "2026.11.13", through="rl")
    budget_context = StepContext.for_run(
        str(tmp_path / "budgeted"), str(tmp_path), deps=budgeted.deps, runtime_args=budgeted.runtime_args
    )
    budget_launch = budgeted.build_config(replace(budget_context, is_fingerprint=True))
    actual = [
        c
        for c in yaml.safe_load(budget_launch.launch_config_yaml)["skyrl"]["trainer"]["callbacks"]
        if c["type"] == "wall_time_budget"
    ]
    assert actual == [{"type": "wall_time_budget", "seconds": 43200, "drain_margin": 1800, "groups_per_step": 64}]
    resolved["rl"]["execution"]["job_timeout_seconds"] = 32400
    with pytest.raises(ValueError, match="exceed the explicit budget"):
        build_spec_graph(resolved, "2026.11.13", through="rl")
