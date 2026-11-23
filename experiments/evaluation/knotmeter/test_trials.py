# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise retained attempts, check scope and reports through public boundaries."""

import copy
import dataclasses
import hashlib
import json
import shutil
import signal
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.client.job_info import JobInfo
from iris.cluster.types import Entrypoint, JobName
from iris.resources.state import JobState, TaskState
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from marin.external_dependencies import VLLM_GPU_RELEASE
from marin.inference.vllm_cache import VllmCompileIdentity
from rigging.provenance import Provenance
from rigging.timing import Timestamp

from experiments.evaluation.knotmeter.artifacts import read_record, retain_artifacts, unpack, write_record
from experiments.evaluation.knotmeter.cli import main as run_cli
from experiments.evaluation.knotmeter.cli import resolve_plan
from experiments.evaluation.knotmeter.collect import collect_trial, refresh_costs
from experiments.evaluation.knotmeter.config import Check, MeasurementOverrides, Stage, load_profile, resolved_series
from experiments.evaluation.knotmeter.coordinator import run_coordinator
from experiments.evaluation.knotmeter.recipe import benchmark_command
from experiments.evaluation.knotmeter.report import recorded_checks, report_campaign, throughput_change
from experiments.evaluation.knotmeter.run import create_server, run_checks
from experiments.evaluation.knotmeter.run import run as run_trial
from experiments.evaluation.knotmeter.runtime import materialize_release, resolve_release


def test_multihost_entrypoint_retains_a_hardware_failure_and_closes_its_coordinator(tmp_path, monkeypatch):
    profile = load_profile(Path("hero-gb200"), [])
    plan = {
        "trial_uri": str(tmp_path / "retained"),
        "manifest": {"series": resolved_series(profile, Stage.PROBE)},
    }
    info = JobInfo(task_id=JobName.from_string("/fixture/hero/0"), num_tasks=4, advertise_host="127.0.0.1")
    registrations = []

    @contextmanager
    def registered(name, address):
        row = {"name": name, "address": address, "active": True}
        registrations.append(row)
        try:
            yield
        finally:
            row["active"] = False

    original_check_output = subprocess.check_output

    def hardware_failure(command, **kwargs):
        if command[0] == "nvidia-smi":
            raise subprocess.CalledProcessError(1, command, stderr="No GPU in this CPU test")
        return original_check_output(command, **kwargs)

    # Iris metadata, registration and GPU processes are external boundaries.
    # Keep the shared gang launcher and its real local actor lifecycle.
    monkeypatch.setattr("experiments.evaluation.knotmeter.run.get_job_info", lambda: info)
    monkeypatch.setattr("marin.inference.iris_vllm.get_job_info", lambda: info)
    monkeypatch.setattr(
        "marin.inference.iris_vllm.iris_ctx", lambda: SimpleNamespace(registry=SimpleNamespace(registered=registered))
    )
    monkeypatch.setattr(subprocess, "check_output", hardware_failure)
    for name, value in {
        "IRIS_OUTPUT_DIR": str(tmp_path / "outputs"),
        "IRIS_TASK_ID": "/fixture/hero/0",
        "IRIS_ATTEMPT_UID": "local-hardware-failure",
        "IRIS_BUNDLE_ID": "local-source",
        "VLLM_HOST_IP": "127.0.0.1",
        "GLOO_SOCKET_IFNAME": "lo",
    }.items():
        monkeypatch.setenv(name, value)
    handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        function, args, kwargs = Entrypoint.from_callable(run_trial, plan).resolve()
        with pytest.raises(subprocess.CalledProcessError):
            function(*args, **kwargs)
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    receipt = read_record(str(tmp_path / "retained/attempts/local-hardware-failure/hosts/0/receipt.json"))
    assert receipt["status"] == "failed" and receipt["task_index"] == 0
    assert registrations and not any(row["active"] for row in registrations)
    unpack(Path(receipt["archive_uri"]), tmp_path / "restored")
    assert read_record(str(tmp_path / "restored/attempt.json"))["error"].startswith("CalledProcessError:")


@pytest.mark.parametrize("operation", ["plan", "run"])
def test_missing_stage_exits_before_hub_or_iris_access(tmp_path, monkeypatch, operation):
    def unexpected_io(*args, **kwargs):
        raise AssertionError("Missing stage must fail before external I/O")

    monkeypatch.setattr("huggingface_hub.HfApi.model_info", unexpected_io)
    monkeypatch.setattr("experiments.evaluation.knotmeter.cli.IrisClient.remote", unexpected_io)
    output = tmp_path / "plan.json"
    monkeypatch.setattr(sys, "argv", ["knotmeter", operation, "--campaign", "fixture", "--output", str(output)])
    with pytest.raises(SystemExit) as stopped:
        run_cli()
    assert stopped.value.code == 2
    assert not output.exists()


def test_probe_coordinator_stops_after_one_gpu_job(tmp_path, monkeypatch):
    result = trial(tmp_path, "probe", 0, diagnostic=True)
    plan = {
        "trial_id": "probe",
        "trial_uri": result["trial_uri"],
        "campaign_uri": str(tmp_path),
        "job_id": "/fixture/probe",
        "gpu_job_id": "/fixture/probe/serve",
        "marin_prefix": str(tmp_path),
        "series": result["manifest"]["series"],
        "vllm_release": result["manifest"]["vllm_release"],
        "source": {},
        "placement": {},
    }

    class Client:
        def __init__(self):
            self.jobs = []

        def submit(self, **kwargs):
            self.jobs.append(kwargs)
            return SimpleNamespace(job_id=JobName.from_string(plan["gpu_job_id"]), wait=lambda **kwargs: None)

        def job_status(self, job_id):
            return SimpleNamespace(job_id=job_id, state=JobState.SUCCEEDED, execution_cluster_id="fixture")

        def list_tasks(self, _job_id):
            return []

    client = Client()
    monkeypatch.setattr("experiments.evaluation.knotmeter.coordinator.iris_ctx", lambda: SimpleNamespace(client=client))
    monkeypatch.setattr(
        "experiments.evaluation.knotmeter.coordinator.prepare_workload",
        lambda *args, **kwargs: {"key": "fixture", "recipe": {"fixture": True}},
    )
    run_coordinator(plan)
    assert len(client.jobs) == 1
    saved = read_record(f"{plan['trial_uri']}/result.json")
    assert saved["manifest"]["series"]["stage"] == "probe"
    assert saved["manifest"]["concurrency"] == 8
    assert (tmp_path / "summary.md").is_file()
    assert not (tmp_path / "recommended.yaml").exists()


@pytest.mark.parametrize("stage", list(Stage))
@pytest.mark.parametrize("coverage", ["complete", "missing", "failed"])
def test_only_confirm_with_complete_checks_writes_a_recommendation(tmp_path, stage, coverage):
    result = trial(tmp_path, "candidate", 120)
    profile = load_profile(Path("snowball-h100"), [])
    profile = profile.model_copy(update={"checks": [Check(name="synthetic-fixture-check")]})
    # Matching Confirm's full recipe cannot promote an explicitly selected early stage.
    result["manifest"]["series"] = resolved_series(
        profile, stage, MeasurementOverrides(duration=3600, warmup_requests_per_lane=10, unsafe_override=False)
    )
    result["manifest"]["diagnostic"] = False
    if coverage == "missing":
        result["measurement"]["checks"] = []
    elif coverage == "failed":
        result["measurement"]["checks"][0]["status"] = "failed"
    write_record(f"{result['trial_uri']}/result.json", result)
    report_campaign(str(tmp_path))
    if stage == Stage.CONFIRM and coverage == "complete":
        report_campaign(str(tmp_path), recommendation="candidate")
        assert (tmp_path / "recommended.yaml").is_file()
        assert read_record(str(tmp_path / "decision.json"))["recommended_trial"] == "candidate"
    else:
        with pytest.raises(ValueError, match="valid Confirm"):
            report_campaign(str(tmp_path), recommendation="candidate")
        assert not (tmp_path / "recommended.yaml").exists()


def test_short_confirm_cannot_write_a_recommendation(tmp_path):
    result = trial(tmp_path, "short-confirm", 120)
    result["manifest"]["series"]["duration"] = 900
    write_record(f"{result['trial_uri']}/result.json", result)
    with pytest.raises(ValueError, match="valid Confirm"):
        report_campaign(str(tmp_path), recommendation="short-confirm")
    assert not (tmp_path / "recommended.yaml").exists()


def test_chat_template_survives_native_compilation_cache_identity(tmp_path):
    profile = load_profile(Path("snowball-h100"), ["model.chat_template=marin"])
    manifest = {"series": resolved_series(profile, Stage.CONFIRM), "vllm_release": dataclasses.asdict(VLLM_GPU_RELEASE)}
    server = create_server(manifest, tmp_path, None)
    template = Path(server.extra_cli_args[server.extra_cli_args.index("--chat-template") + 1])
    assert template.read_text() == MARIN_CHAT_TEMPLATE
    # Passing the literal template used to raise ENAMETOOLONG at this boundary,
    # before any model weights or native serving process could be loaded.
    identity = VllmCompileIdentity.from_vllm_args(
        model_name_or_path=server.model_name_or_path,
        extra_cli_args=tuple(server.extra_cli_args),
    )
    assert hashlib.sha256(template.read_bytes()).hexdigest() in identity.extra_cli_args


def test_launch_plan_survives_json_delivery_without_invalidating_worker_identity(monkeypatch):
    monkeypatch.setattr("experiments.evaluation.knotmeter.cli.pin_profile", lambda profile: profile)
    monkeypatch.setattr(
        "experiments.evaluation.knotmeter.cli.launch_provenance",
        lambda: Provenance("fixture", "b" * 40, False, "fixture", None, command_line=("knotmeter", "run")),
    )
    plan = resolve_plan(load_profile(Path("snowball-h100"), []), "fixture", stage=Stage.PROBE)
    assert json.loads(json.dumps(plan)) == plan


@pytest.mark.parametrize("corrupt", [False, True])
def test_cloud_runtime_installs_only_verified_bytes_and_keeps_durable_identity(tmp_path, monkeypatch, corrupt):
    source = tmp_path / "candidate.whl"
    source.write_bytes(b"immutable candidate fixture")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = json.loads(json.dumps(dataclasses.asdict(VLLM_GPU_RELEASE)))
    metadata["wheels"][0] |= {"url": "s3://fixture/runtime/candidate.whl", "sha256": expected}
    monkeypatch.setattr("experiments.evaluation.knotmeter.runtime.tempfile.gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(
        "experiments.evaluation.knotmeter.runtime.platform.machine", lambda: metadata["wheels"][0]["architecture"]
    )
    monkeypatch.setattr(
        "experiments.evaluation.knotmeter.runtime.copy_file", lambda uri, path: shutil.copyfile(source, path)
    )
    if corrupt:
        source.write_bytes(b"corrupt candidate")
        with pytest.raises(ValueError, match="checksum"):
            materialize_release(metadata, tmp_path)
        assert not (tmp_path / "runtime-wheel.json").exists()
    else:
        release = materialize_release(metadata, tmp_path)
        assert materialize_release(metadata, tmp_path).wheels[0].url == release.wheels[0].url
        assert release.wheels[0].url.startswith("file://")
        assert release.wheels[0].sha256 == expected
        receipt = read_record(str(tmp_path / "runtime-wheel.json"))
        assert receipt["uri"] == metadata["wheels"][0]["url"]
        assert receipt["sha256"] == expected
        assert metadata["wheels"][0]["url"].startswith("s3://")


def test_https_runtime_uses_the_native_release_without_materializing_bytes(tmp_path, monkeypatch):
    def unexpected_copy(*args):
        raise AssertionError("Normal HTTPS releases must use the native installer")

    monkeypatch.setattr("experiments.evaluation.knotmeter.runtime.copy_file", unexpected_copy)
    assert materialize_release(dataclasses.asdict(VLLM_GPU_RELEASE), tmp_path) == VLLM_GPU_RELEASE
    assert not (tmp_path / "runtime-wheel.json").exists()


@pytest.mark.parametrize("selector", ["release", "manifest"])
def test_named_runtime_flag_replaces_a_preset_selection(tmp_path, monkeypatch, selector):
    metadata = json.loads(json.dumps(dataclasses.asdict(VLLM_GPU_RELEASE)))
    manifest = tmp_path / "runtime.json"
    manifest.write_text(json.dumps(metadata))
    published = {
        "release": {"repository": "marin-community/vllm", "status": "released", "tag": metadata["release_tag"]},
        "validation": {"status": "passed"},
        "source": {"fork_commit": metadata["source_commit"]},
        "distribution": {"version": metadata["version"]},
        "abi": {"cuda_variant": metadata["torch_backend"], "torch_version": metadata["torch_version"]},
        "platforms": [
            {
                "architecture": wheel["architecture"],
                "sm_targets": wheel["sm_targets"],
                "wheel": {"filename": unquote(wheel["url"].rsplit("/", 1)[-1]), "sha256": wheel["sha256"]},
            }
            for wheel in metadata["wheels"]
        ],
    }
    profile = load_profile(Path("snowball-h100"), []).model_dump()
    opposite = "manifest" if selector == "release" else "release"
    profile["vllm"] = {opposite: "stale-selection", "revision": "a" * 40}
    preset = tmp_path / "recommended.yaml"
    preset.write_text(json.dumps(profile))
    output = tmp_path / "plan.json"
    selection = metadata["release_tag"] if selector == "release" else str(manifest)
    monkeypatch.setattr(
        "huggingface_hub.HfApi.model_info", lambda self, repo, *, revision: SimpleNamespace(sha=revision)
    )
    monkeypatch.setattr(
        "experiments.evaluation.knotmeter.runtime.requests.get",
        lambda *args, **kwargs: SimpleNamespace(content=json.dumps(published).encode(), raise_for_status=lambda: None),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "knotmeter",
            "plan",
            "--stage",
            "probe",
            "--profile",
            str(preset),
            "--campaign",
            "fixture",
            f"--vllm-{selector}",
            selection,
            "--output",
            str(output),
        ],
    )
    run_cli()
    plan = read_record(str(output))
    assert plan["vllm_release"] == metadata
    assert plan["series"]["profile"]["vllm"][selector] == selection
    assert plan["series"]["profile"]["vllm"][opposite] is None
    assert plan["series"]["profile"]["vllm"]["revision"] is None


@pytest.mark.parametrize("ambient", ["local_store", "gs://wrong-cloud/outputs"])
def test_plan_uses_target_storage_unless_explicitly_overridden(monkeypatch, ambient):
    monkeypatch.setattr("experiments.evaluation.knotmeter.cli.pin_profile", lambda profile: profile)
    monkeypatch.setenv("MARIN_PREFIX", ambient)
    profile = load_profile(Path("snowball-h100"), [])
    plan = resolve_plan(profile, "fixture", stage=Stage.PROBE)
    assert plan["marin_prefix"].startswith("s3://") and plan["marin_prefix"] != ambient
    assert (
        resolve_plan(profile, "fixture", stage=Stage.PROBE, marin_prefix="s3://fixture/outputs")["marin_prefix"]
        == "s3://fixture/outputs"
    )
    with pytest.raises(ValueError, match="shared"):
        resolve_plan(profile, "fixture", stage=Stage.PROBE, marin_prefix="local_store")


def trial(
    root: Path,
    name: str,
    rate: float,
    *,
    diagnostic: bool = False,
    measurement: MeasurementOverrides = MeasurementOverrides(),
) -> dict:
    profile = load_profile(Path("snowball-h100"), [])
    profile = profile.model_copy(update={"checks": [Check(name="synthetic-fixture-check")]})
    series = resolved_series(profile, Stage.PROBE if diagnostic else Stage.CONFIRM, measurement)
    runtime = json.loads(json.dumps(dataclasses.asdict(VLLM_GPU_RELEASE))) | {
        "release_tag": "fixture",
        "source_commit": "b" * 40,
    }
    return {
        "trial_id": name,
        "trial_uri": str(root / "trials" / name),
        "manifest": {
            "series": series,
            "diagnostic": diagnostic,
            "vllm_release": runtime,
            "workload": {"recipe": {"fixture": True}},
        },
        "job_state": JobState.SUCCEEDED.value,
        "job_succeeded": True,
        "attempts": [],
        "gpu_seconds": 3600.0,
        "cost_complete": True,
        "measurement": {
            "validity": {"submission_valid": not diagnostic, "protocol_valid": not diagnostic, "gang_valid": True},
            "checks": [
                {
                    "name": "synthetic-fixture-check",
                    "kind": "numerical",
                    "checkpoint_revision": profile.model.revision,
                    "runtime_source": "b" * 40,
                    "status": "passed",
                }
            ],
            "aggregate": {
                "request_metrics": {
                    "throughput": {"per_gpu": {"output_tput_tps": rate}},
                    "latency": {"ttft": {"p90": 5.0}, "e2el": {"p90": 35.0}},
                }
            },
        },
    }


def test_report_keeps_diagnostics_separate_and_pins_selected_runtime_and_load(tmp_path, monkeypatch):
    for result in (
        trial(tmp_path, "baseline", 100, measurement=MeasurementOverrides(concurrency=64)),
        trial(tmp_path, "candidate", 120, measurement=MeasurementOverrides(concurrency=64)),
        trial(tmp_path, "diagnostic", 9999, diagnostic=True),
    ):
        write_record(f"{result['trial_uri']}/result.json", result)
    output = report_campaign(str(tmp_path), recommendation="candidate", baseline="baseline")
    assert output == (tmp_path / "summary.md").read_text()
    decision = read_record(str(tmp_path / "decision.json"))
    assert decision["throughput_change"] == pytest.approx(0.2, abs=1e-12)
    assert decision["recommended_trial"] == "candidate"
    # A future change to Marin's selection must not retarget the saved recommendation.
    monkeypatch.setattr(
        "experiments.evaluation.knotmeter.runtime.VLLM_GPU_RELEASE",
        dataclasses.replace(VLLM_GPU_RELEASE, source_commit="c" * 40),
    )
    recommended = load_profile(tmp_path / "recommended.yaml", [])
    assert resolve_release(recommended.vllm)["source_commit"] == "b" * 40
    for stage in (Stage.SCREEN, Stage.CONFIRM):
        assert resolved_series(recommended, stage)["concurrency"] == 64
    assert report_campaign(str(tmp_path), baseline="baseline") == output
    before = read_record(str(tmp_path / "decision.json"))
    diagnostic_view = report_campaign(str(tmp_path), baseline="diagnostic")
    assert str(tmp_path / "recommended.yaml") in diagnostic_view
    assert read_record(str(tmp_path / "decision.json")) == before
    with pytest.raises(ValueError):
        report_campaign(str(tmp_path), recommendation="diagnostic", baseline="baseline")


@pytest.mark.parametrize("recorded_gpu_seconds", [0.0, 1800.0])
@pytest.mark.parametrize("check_status", ["passed", "failed"])
def test_incomplete_cost_preserves_usage_and_correctness_gate(tmp_path, recorded_gpu_seconds, check_status):
    result = trial(tmp_path, "candidate", 120)
    result["gpu_seconds"] = recorded_gpu_seconds
    result["cost_complete"] = False
    if recorded_gpu_seconds:
        result["attempts"] = [{"allocation": {"end_basis": "iris-observed-terminal"}}]
    result["measurement"]["checks"][0]["status"] = check_status
    write_record(f"{result['trial_uri']}/result.json", result)

    if check_status == "passed":
        report = report_campaign(str(tmp_path), recommendation="candidate")
        assert ("0.500 + unknown (estimated)" if recorded_gpu_seconds else "unknown") in report
        assert (tmp_path / "recommended.yaml").is_file()
        assert read_record(str(tmp_path / "decision.json"))["recommended_trial"] == "candidate"
    else:
        with pytest.raises(ValueError, match="passed required checks"):
            report_campaign(str(tmp_path), recommendation="candidate")
        assert not (tmp_path / "recommended.yaml").exists()

    saved = read_record(f"{result['trial_uri']}/result.json")
    assert saved["gpu_seconds"] == recorded_gpu_seconds and not saved["cost_complete"]
    assert saved["measurement"]["checks"][0]["status"] == check_status


def test_failed_check_measurement_preserves_native_protocol_and_blocks_recommendation(tmp_path):
    profile = load_profile(Path("snowball-h100"), [])
    normal = resolved_series(profile, Stage.CONFIRM)
    measured = resolved_series(profile, Stage.CONFIRM, measure_failed_checks=True)
    assert measured["duration"] == 3600
    assert not measured.get("diagnostic")
    args = (64, "http://localhost:8000", tmp_path / "exports")
    assert benchmark_command(measured, *args) == benchmark_command(normal, *args)
    assert "--unsafe-override" not in benchmark_command(measured, *args)

    result = trial(tmp_path, "failed-numerics", 120)
    result["manifest"]["series"]["measure_failed_checks"] = True
    result["measurement"]["checks"][0]["status"] = "failed"
    write_record(f"{result['trial_uri']}/result.json", result)
    report_campaign(str(tmp_path))
    with pytest.raises(ValueError, match="passed required checks"):
        report_campaign(str(tmp_path), recommendation="failed-numerics")
    assert not (tmp_path / "recommended.yaml").exists()


@pytest.mark.parametrize("field", ["command", "evidence"])
def test_external_numerical_check_changes_prevent_a_baseline_comparison(tmp_path, field):
    baseline = trial(tmp_path, "baseline", 100)
    candidate = trial(tmp_path, "candidate", 120)
    for result in (baseline, candidate):
        check = result["measurement"]["checks"][0]
        check |= {"command": ["check-model", "--policy", "original"], "evidence": ["s3://fixture/original.json"]}
    candidate["measurement"]["checks"][0][field] = ["different-qualification"]
    for result in (baseline, candidate):
        result["manifest"]["series"]["profile"]["checks"][0] |= {
            key: result["measurement"]["checks"][0][key] for key in ("command", "evidence")
        }
        write_record(f"{result['trial_uri']}/result.json", result)
    assert throughput_change(candidate, baseline) is None
    with pytest.raises(ValueError, match="matched model and workload controls"):
        report_campaign(str(tmp_path), recommendation="candidate", baseline="baseline")
    assert not (tmp_path / "recommended.yaml").exists()


@pytest.mark.parametrize("status_code", [Code.NOT_FOUND, Code.UNAVAILABLE])
def test_rejected_gpu_submission_remains_reportable_and_recoverable(tmp_path, monkeypatch, status_code):
    result = trial(tmp_path, "submission-failed", 100)
    plan = {
        "trial_id": result["trial_id"],
        "trial_uri": result["trial_uri"],
        "campaign_uri": str(tmp_path),
        "job_id": "/fixture/trial",
        "gpu_job_id": "/fixture/trial/serve",
        "controller": "fixture-cluster",
        "marin_prefix": str(tmp_path),
        "series": result["manifest"]["series"],
        "vllm_release": result["manifest"]["vllm_release"],
        "source": {},
        "placement": {},
    }

    class Client:
        def submit(self, **kwargs):
            raise ConnectError(Code.INVALID_ARGUMENT, "Synthetic GPU submission rejection")

        def job_status(self, _job):
            if str(_job) == plan["job_id"]:
                return SimpleNamespace(state=JobState.FAILED)
            raise ConnectError(self.status_code, "Synthetic controller status")

    client = Client()
    client.status_code = status_code
    monkeypatch.setattr("experiments.evaluation.knotmeter.coordinator.iris_ctx", lambda: SimpleNamespace(client=client))
    # Workload preparation is the external corpus/cache I/O boundary. Keep the
    # real coordinator, manifest freeze, submission, failure and recovery paths.
    monkeypatch.setattr(
        "experiments.evaluation.knotmeter.coordinator.prepare_workload",
        lambda *args, **kwargs: {"key": "fixture", "recipe": {"fixture": True}},
    )
    with pytest.raises(ConnectError, match="submission rejection"):
        run_coordinator(plan)
    saved = read_record(f"{plan['trial_uri']}/result.json")
    assert saved["manifest"]["workload"]["key"] == "fixture"
    assert saved["job_succeeded"] is False and saved["measurement"] is None
    assert saved["gpu_seconds"] == 0
    assert saved["cost_complete"] is (status_code == Code.NOT_FOUND)
    assert refresh_costs(plan) == saved
    client.status_code = Code.NOT_FOUND
    assert collect_trial(client, plan, tmp_path / "recovery") == saved
    monkeypatch.setattr(
        "experiments.evaluation.knotmeter.cli.connect_controller",
        lambda **kwargs: nullcontext(SimpleNamespace(url="http://fixture", credentials=None)),
    )
    monkeypatch.setattr(
        "experiments.evaluation.knotmeter.cli.IrisClient.remote", lambda *args, **kwargs: nullcontext(client)
    )
    monkeypatch.setattr(sys, "argv", ["knotmeter", "recover", "--trial-uri", plan["trial_uri"]])
    with pytest.raises(SystemExit) as recovered:
        run_cli()
    assert recovered.value.code == 1
    assert read_record(f"{plan['trial_uri']}/result.json") == saved
    report_campaign(str(tmp_path))
    with pytest.raises(ValueError, match="passed required checks"):
        report_campaign(str(tmp_path), recommendation=plan["trial_id"])
    assert not (tmp_path / "recommended.yaml").exists()


@pytest.mark.parametrize("criteria", ["matching", "different", "unrecorded"])
def test_report_compares_precision_specific_policies_only_with_matching_criteria(tmp_path, criteria):
    baseline = trial(tmp_path, "baseline", 100)
    candidate = trial(tmp_path, "fp8", 120)
    candidate["manifest"]["series"]["profile"]["quantization"] = "fp8"
    for result in (baseline, candidate):
        check = result["measurement"]["checks"][0]
        check |= {
            "policy": result["trial_id"],
            "policy_sha256": hashlib.sha256(result["trial_id"].encode()).hexdigest(),
        }
        if criteria != "unrecorded":
            check["criteria_sha256"] = "a" * 64
    if criteria == "different":
        candidate["measurement"]["checks"][0]["criteria_sha256"] = "b" * 64
    for result in (baseline, candidate):
        write_record(f"{result['trial_uri']}/result.json", result)
    if criteria == "matching":
        report_campaign(str(tmp_path), recommendation="fp8", baseline="baseline")
        decision = read_record(str(tmp_path / "decision.json"))
        assert decision["throughput_change"] == pytest.approx(0.2, abs=1e-12)
        assert decision["numerical_checks"][0]["policy"] == "fp8"
        assert load_profile(tmp_path / "recommended.yaml", []).quantization == "fp8"
    else:
        with pytest.raises(ValueError, match="matched model and workload controls"):
            report_campaign(str(tmp_path), recommendation="fp8", baseline="baseline")


@pytest.mark.parametrize("scope", ["matching", "other-checkpoint", "other-runtime", "failed"])
def test_existing_check_pass_transfers_only_to_its_checkpoint_and_runtime(tmp_path, scope):
    base = load_profile(Path("snowball-h100"), [])
    check = Check(
        name="numerical-fixture",
        command=[sys.executable, "-c", f"raise SystemExit({7 if scope == 'failed' else 0})"],
        checkpoint_revision="a" * 40 if scope == "other-checkpoint" else base.model.revision,
        runtime_source="c" * 40 if scope == "other-runtime" else "b" * 40,
    )
    profile = base.model_copy(update={"checks": [check]})
    checks = run_checks(profile, "http://fixture/v1", tmp_path, "b" * 40)
    result = trial(tmp_path, "candidate", 120)
    result["manifest"]["series"] = resolved_series(profile, Stage.CONFIRM)
    result["measurement"]["checks"] = checks
    write_record(f"{result['trial_uri']}/result.json", result)
    if scope == "matching":
        report_campaign(str(tmp_path), recommendation="candidate")
        assert (tmp_path / "recommended.yaml").is_file()
    else:
        if scope != "failed":
            assert checks[0]["regression_status"] == "passed"
            assert checks[0]["status"] == "pending"
        with pytest.raises(ValueError):
            report_campaign(str(tmp_path), recommendation="candidate")
        assert not (tmp_path / "recommended.yaml").exists()
        assert read_record(f"{result['trial_uri']}/result.json")["measurement"]["checks"] == checks


@pytest.mark.parametrize("missing", ["required-check", "submission-validity", "qualification", "job-success"])
def test_numerical_pass_does_not_replace_other_acceptance_requirements(tmp_path, missing):
    result = trial(tmp_path, "candidate", 120)
    if missing == "required-check":
        result["manifest"]["series"]["profile"]["checks"].append(
            Check(name="serving-contract", kind="serving").model_dump()
        )
    elif missing == "submission-validity":
        result["measurement"]["validity"]["submission_valid"] = False
    elif missing == "job-success":
        result |= {"job_succeeded": False, "job_state": "failed"}
    else:
        result["manifest"]["diagnostic"] = True
    write_record(f"{result['trial_uri']}/result.json", result)
    report_campaign(str(tmp_path))
    assert (tmp_path / "summary.md").is_file()
    with pytest.raises(ValueError):
        report_campaign(str(tmp_path), recommendation="candidate")
    assert not (tmp_path / "recommended.yaml").exists()


@pytest.mark.parametrize("name", [None, "smoke", "model-numerics", "other-checkpoint", "other-runtime"])
def test_missing_or_relabelled_numerical_check_cannot_qualify_a_trial(tmp_path, name):
    result = trial(tmp_path, "candidate", 120)
    if name in ("other-checkpoint", "other-runtime"):
        field = "checkpoint_revision" if name == "other-checkpoint" else "runtime_source"
        result["measurement"]["checks"][0][field] = "wrong-scope"
    else:
        result["measurement"]["checks"] = [] if name is None else [{"name": name, "kind": "serving", "status": "passed"}]
    write_record(f"{result['trial_uri']}/result.json", result)
    with pytest.raises(ValueError, match="passed required checks"):
        report_campaign(str(tmp_path), recommendation="candidate")
    assert not (tmp_path / "recommended.yaml").exists()


def test_plan_without_result_remains_visible_in_report(tmp_path):
    result = trial(tmp_path, "interrupted", 0)
    write_record(
        f"{result['trial_uri']}/plan.json",
        {"trial_id": "interrupted", "trial_uri": result["trial_uri"], "manifest": result["manifest"]},
    )
    output = report_campaign(str(tmp_path))
    assert "interrupted" in output and "unrecorded" in output
    assert "interrupted/plan.json" in output
    assert output == (tmp_path / "summary.md").read_text()


@pytest.mark.parametrize("partial_exports", [False, True])
@pytest.mark.parametrize("pod_finish", [True, False])
def test_recovery_counts_setup_and_failed_native_attempts_without_resubmitting(tmp_path, partial_exports, pod_finish):
    result = trial(tmp_path, "failed", 0)
    plan = {
        "trial_id": "failed",
        "trial_uri": result["trial_uri"],
        "gpu_job_id": "/fixture/trial/serve",
        "manifest": result["manifest"],
    }
    task_id = JobName.from_string("/fixture/trial/serve/0")
    attempts = []
    for number, uid, start, finish, scheduled, stopped in ((0, "one", 20, 50, 10, 45), (1, "two", 80, 100, 70, 95)):
        attempts.append(
            SimpleNamespace(
                attempt_number=number,
                attempt_uid=uid,
                started_at=Timestamp.from_seconds(start),
                finished_at=Timestamp.from_seconds(finish),
                state=TaskState.FAILED,
                pod_uid=f"pod-{uid}",
                node_name="fixture-node",
                error_message="fixture failure",
                terminal_reason="",
                output_archive=None,
            )
        )
        write_record(
            f"{plan['trial_uri']}/attempts/{uid}/allocation.json",
            {
                "pod_uid": f"pod-{uid}",
                "scheduled_unix": scheduled,
                "finished_unix": stopped if pod_finish else None,
            },
        )
        if partial_exports:
            point = tmp_path / f"worker-{uid}"
            (point / "aiperf_artifacts").mkdir(parents=True)
            (point / "aiperf_artifacts/profile_export.jsonl").write_text('{"partial":true}\n')
            record = {
                "manifest": plan["manifest"],
                "iris_attempt_uid": uid,
                "status": "failed",
                "checks": [
                    {
                        "name": "synthetic-fixture-check",
                        "kind": "numerical",
                        "status": "passed" if number == 0 else "failed",
                    }
                ],
            }
            write_record(str(point / "attempt.json"), record)
            prefix = f"{plan['trial_uri']}/attempts/{uid}/hosts/0"
            uri, digest = retain_artifacts(point, tmp_path / f"{uid}.tar.zst", prefix, name="native")
            write_record(f"{prefix}/receipt.json", record | {"archive_uri": uri, "archive_sha256": digest})
    task = SimpleNamespace(task_id=task_id, attempts=attempts, current_attempt_number=1)

    class Client:
        def job_status(self, _job):
            return SimpleNamespace(
                job_id=JobName.from_string(plan["gpu_job_id"]),
                state=JobState.FAILED,
                execution_cluster_id="fixture-cluster",
            )

        def list_tasks(self, _job):
            return [task]

        def task_status(self, _task):
            return task

        def submit(self, **_kwargs):
            raise AssertionError("Recovery must not submit compute")

    recovered = collect_trial(Client(), plan, tmp_path / "recovery")
    assert recovered["gpu_seconds"] == (480 if pod_finish else 560)
    assert recovered["cost_complete"] is True
    assert all(
        row["allocation"]["status"] == ("measured" if pod_finish else "estimated") for row in recovered["attempts"]
    )
    assert [row["attempt_uid"] for row in recovered["attempts"]] == ["one", "two"]
    if partial_exports:
        assert recovered["measurement"] is None
        assert recorded_checks(recovered)[0]["status"] == "failed"
    before = copy.deepcopy(read_record(f"{plan['trial_uri']}/result.json"))
    collect_trial(Client(), plan, tmp_path / "repeated-recovery")
    assert read_record(f"{plan['trial_uri']}/result.json") == before
