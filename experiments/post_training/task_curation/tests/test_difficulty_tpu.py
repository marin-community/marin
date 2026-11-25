# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise regional staging, versioned grading and submission guards before hardware rollout."""

import asyncio
import hashlib
import importlib
import json
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from finestore.rl.difficulty import SampleTask, Solver
from iris.cli.job import run as iris_run_command
from iris.client.client import NamespacedEndpointRegistry
from iris.cluster.types import EndpointAccess, Namespace, TaskAttempt
from iris.rpc.proto_display import priority_band_value
from marin.inference.config import EffectiveServing, ObjectStoreLoadMode, VllmCompilationCacheMode, VllmLauncherType
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.grader import verifyit_package
from taskcompendium.harbor.runtime import read_source_runtime
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
    judge_spec,
)
from verifyit.grade import GradingInfraError
from verifyit.modes.grade_judge import JudgeConnection, grade_judge_candidate
from verifyit.spec import JudgeSpec

from experiments.post_training.task_curation import difficulty_preparation
from experiments.post_training.task_curation.difficulty import campaign, remediate, tpu, tpu_launch
from experiments.post_training.task_curation.difficulty.importer import catalog_sources, population_revision, receipt_row
from experiments.post_training.task_curation.difficulty.ledger import append_rows, digest, read_rows
from experiments.post_training.task_curation.difficulty.planner import QualifyJudgeRoute
from experiments.post_training.task_curation.difficulty.state import (
    CatalogSource,
    CellStatus,
    CoveragePolicy,
    GapReason,
    cell_state,
)
from experiments.post_training.task_curation.difficulty.tpu_router import pool_router
from experiments.post_training.task_curation.difficulty_protocol import PINS
from experiments.post_training.task_curation.tests.review_server import JudgeReply


def judged_task():
    return TaskSpec(
        id="judge-task",
        context=ConversationInput(events=(TextMessage(role="user", content="Original question"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.TEXT,
        answer_format=PlainText(),
        grader=verifyit_package(
            JudgeSpec(model="unavailable-original-judge", references=("Four",), exact_gate=False)
        ).grader,
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )


def test_versioned_judge_prepares_original_panel_without_rewriting_its_identity(
    tmp_path, monkeypatch, difficulty_source
):
    task = judged_task()
    source = difficulty_source(task)
    frozen_sha = source.sample.tasks[0].task_sha256
    monkeypatch.setattr(difficulty_preparation, "panel_task_runtimes", lambda panel: ([task], [{}]))
    prepared = difficulty_preparation.prepare_source(source, tmp_path / "prepared", judge_version=campaign.JUDGE_VERSION)
    (result,) = prepared.tasks
    assert result.reason is None
    rewritten = TaskSpec.model_validate_json((Path(result.path) / "task-spec.json").read_text())
    assert judge_spec(rewritten.grader).model == campaign.SUBSTITUTE_JUDGE_MODEL
    assert judge_spec(rewritten.grader).references == ("Four",)
    assert source.sample.tasks[0].task_sha256 == frozen_sha
    assert (Path(result.path) / "judge-redeclaration.json").exists()
    read_source_runtime(Path(result.path)).require_task(rewritten)
    # A source contract mismatch still fails before staging, even with the new judge.
    changed = task.model_copy(update={"id": "different-task"})
    monkeypatch.setattr(difficulty_preparation, "panel_task_runtimes", lambda panel: ([changed], [{}]))
    with pytest.raises(ValueError, match="outside the frozen"):
        difficulty_preparation.prepare_source(source, tmp_path / "bad", judge_version=campaign.JUDGE_VERSION)


def test_regional_stage_creates_separate_campaign_and_valid_frozen_queue(tmp_path, monkeypatch, difficulty_source):
    task = judged_task()
    source = difficulty_source(task).model_dump(mode="json")
    panel_file = tmp_path / "panel.parquet"
    panel_file.write_bytes(b"frozen-panel-bytes")
    population = tmp_path / "population"
    population.mkdir()
    (population / ".artifact.json").write_text('{"result": {}}')
    inventory_path = tmp_path / "inventory.json"
    source["panel"].update(
        panel_path=str(panel_file),
        panel_sha256=hashlib.sha256(panel_file.read_bytes()).hexdigest(),
        input_artifact=str(population),
        evidence_url=str(inventory_path),
    )
    inventory_path.write_text(json.dumps(source["panel"]))
    queue_path = tmp_path / "sources.json"
    queue_path.write_text(json.dumps([source]))
    queue_sha = hashlib.sha256(queue_path.read_bytes()).hexdigest()
    original = catalog_sources(str(queue_path), queue_sha)[0]
    sealed = tmp_path / "seal"
    sealed.mkdir()
    catalog_payload = json.dumps([asdict(original)]).encode()
    (sealed / "catalog.json").write_bytes(catalog_payload)
    manifest = {
        "files": {"catalog.json": {"sha256": hashlib.sha256(catalog_payload).hexdigest()}},
        "sealed_at": "2026-11-24T23:00:00+00:00",
    }
    (sealed / "manifest.json").write_text(json.dumps(manifest))
    root = "gs://marin-us-central2/fixture"
    qualification = tmp_path / "qualified.json"
    qualification.write_text(
        json.dumps(
            {
                "judge_identity": campaign.identity_bundle()["judge"],
                "fixtures": [
                    {"expected": verdict, "reward": float(verdict == "CORRECT"), "detail": {"verdict": verdict}}
                    for verdict in ("CORRECT", "INCORRECT", "NOT_ATTEMPTED", "INCORRECT")
                ],
                "tokenization": {
                    "tools_thinking_probe": {"completion": {"usage": {"prompt_tokens": 3}}, "pinned_prompt_tokens": 3}
                },
            }
        )
    )

    def mapped(uri):
        return (
            StoragePath(str(tmp_path / "regional" / str(uri).removeprefix(root).lstrip("/")))
            if str(uri).startswith(root)
            else StoragePath(uri)
        )

    monkeypatch.setattr(campaign, "StoragePath", mapped)
    monkeypatch.setattr(tpu, "regional_worker", lambda: None)
    monkeypatch.setattr(campaign, "regional_worker", lambda: None)
    campaign.stage_campaign(
        root,
        str(sealed / "manifest.json"),
        hashlib.sha256((sealed / "manifest.json").read_bytes()).hexdigest(),
        str(qualification),
    )
    record = campaign.campaign_record(root + "/ledger")
    assert record["cell_key"][0] == "campaign_id"
    assert record["identity"]["judge"]["version"] == campaign.JUDGE_VERSION
    assert record["carry_forward"].startswith("none:")
    assert not read_rows(str(tmp_path / "regional/ledger"))
    new_queue, sha = record["queues"][0]
    rows = json.loads(StoragePath(new_queue).read_text())
    assert rows[0]["sample"] == source["sample"]
    assert rows[0]["panel"]["panel_path"].startswith(str(tmp_path / "regional"))
    assert hashlib.sha256(StoragePath(new_queue).read_bytes()).hexdigest() == sha
    (mapped(root + "/ledger/v1/campaign.json")).write_text(json.dumps({**record, "campaign_id": "wrong"}))
    with pytest.raises(ValueError, match="identity differs"):
        campaign.campaign_record(root + "/ledger")


def test_all_solver_configs_and_root_commands_fit_reserved_regional_allocation():
    assert sum(shape.chips * shape.replicas for shape in tpu.SHAPES.values()) == 256
    for solver, shape in tpu.SHAPES.items():
        config = tpu.tpu_config(solver, "gs://marin-us-central2/pinned-weights")
        assert config.instances == 1 and config.broker is None
        assert config.engine.launcher == VllmLauncherType.TPU
        assert config.engine.compilation_cache == VllmCompilationCacheMode.MANAGED
        assert config.model.object_store_load_mode == ObjectStoreLoadMode.STREAM
        assert config.model.tokenizer_revision == PINS[solver].revision
        assert config.iris.priority == priority_band_value("interactive")
        assert config.iris.worker_resources.regions == ["us-central2"]
        assert config.iris.max_retries_failure > 0 and config.iris.max_retries_preemption > 0
        assert shape.tensor_parallel * shape.data_parallel * shape.attention_data_parallel == shape.chips
        model = tpu.tpu_model(solver, "gs://marin-us-central2/fixture")
        # Regression: attnDP16 previously expanded these into a 262k-token /
        # 2048-sequence compile, which exhausted v4 HBM at its 65k-token bucket.
        assert model.serve.max_num_batched_tokens * shape.attention_data_parallel <= 16384
        assert model.serve.max_num_seqs * shape.attention_data_parallel <= 128
        if solver == Solver.LARGE:
            # Regression: aligned prefix states required 159 GiB of cache;
            # the pilot slice had 66 GiB available for the frozen context.
            assert "--enable-prefix-caching" not in model.serve.vllm_extra_args
            assert "--no-enable-prefix-caching" in model.serve.vllm_extra_args
            assert model.serve.max_model_len == 65536
        assert not any("eager" in argument for argument in config.engine.extra_args)
    # Snowball's 20 Q / 5 KV heads cannot be split across TP8 or TP16.
    assert tpu.SHAPES[Solver.SNOWBALL].tensor_parallel == 1
    for phase in ("stage", "pilot", "full"):
        commands = tpu_launch.commands_for_phase(phase, tpu.REGIONAL_ROOT, "fixture", "seal", "a" * 64, "qualified")
        for command in commands:
            assert command[command.index("--priority") + 1] == "interactive"
            assert command[command.index("--region") + 1] == "us-central2"
            assert command[command.index("--zone") + 1] == "us-central2-b"
            assert int(command[command.index("--max-retries") + 1]) > 0
            assert "--target-cluster=cw-us-east-08a" not in command
            # Use the real CLI parser: env flags consume KEY and VALUE separately.
            with iris_run_command.make_context("run", command[7:]) as context:
                assert context.params["env_vars"] == (("MARIN_TEMP_PREFIX", "gs://marin-us-central2"),)


def test_submission_gate_rejects_stale_validation_and_mac_staging(tmp_path):
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"passed": True, "source_digest": "stale"}))
    with pytest.raises(ValueError, match="stale"):
        tpu_launch.validate_gate(gate)
    with pytest.raises(ValueError, match="Iris worker"):
        tpu.regional_worker()


def test_versioned_grading_to_ledger_and_coverage_keeps_failed_grader_unscored(
    tmp_path, monkeypatch, difficulty_source, review_server
):
    first = judged_task()
    second = first.model_copy(update={"id": "bad-judge-task"})
    source = difficulty_source(first)
    sample = source.sample.model_copy(
        update={
            "population_count": 2,
            "tasks": [
                SampleTask(task_id=task.id, task_sha256=digest(task.model_dump(mode="json"))) for task in (first, second)
            ],
        }
    )
    source = source.model_copy(update={"sample": sample})
    monkeypatch.setattr(difficulty_preparation, "panel_task_runtimes", lambda panel: ([first, second], [{}, {}]))
    prepared = difficulty_preparation.prepare_source(source, tmp_path / "tasks", judge_version=campaign.JUDGE_VERSION)
    review_server.judge_replies.extend([JudgeReply("SCORE: 1"), JudgeReply("malformed"), JudgeReply("malformed")])
    connection = JudgeConnection(f"http://127.0.0.1:{review_server.server_port}/v1", "fixture")
    ledger = tmp_path / "ledger"
    catalog = CatalogSource(
        source.sample.source_id,
        population_revision(sample.model_dump(mode="json")),
        tuple((task.task_id, task.task_sha256) for task in sample.tasks),
        "fixture:queue",
        "a" * 64,
    )
    for index, task in enumerate(prepared.tasks):
        spec = TaskSpec.model_validate_json((Path(task.path) / "task-spec.json").read_text())
        judge = judge_spec(spec.grader)
        if index == 0:
            reward = grade_judge_candidate(judge, "Four.", connection=connection, runtime=None)
            attempt = {"status": "scored", "reward": reward.reward}
        else:
            with pytest.raises(GradingInfraError):
                grade_judge_candidate(judge, "Four.", connection=connection, runtime=None)
            attempt = {"status": "not_measured", "reason": "grader_infrastructure_failure:fixture_malformed_judge"}
        attempt.update(task_id=task.task_id, result={"finished_at": f"2026-11-24T23:00:0{index}+00:00"})
        path = StoragePath(str(tmp_path / f"attempt-{index}.json"))
        path.write_text(json.dumps(attempt))
        row = receipt_row(
            {"uri": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
            sample.model_dump(mode="json"),
            Solver.SMALL,
            "fixture:versioned-run",
            "unused",
        )
        append_rows(str(ledger), [row])
    rows = read_rows(str(ledger))
    assert [row.reward for row in rows] == [1.0, None]
    result = cell_state(rows, catalog, Solver.SMALL, CoveragePolicy())
    assert result.state == CellStatus.NEEDS_ATTEMPTS
    assert result.n_missing == 1
    assert result.gaps[0].reason == GapReason.GRADER_INFRASTRUCTURE_FAILURE


def test_slice_router_balances_live_requests_preserves_stream_and_does_not_replay_errors():
    async def run():
        calls = []
        all_started = asyncio.Event()

        class Body(httpx.AsyncByteStream):
            def __init__(self, payload):
                self.payload = payload

            async def __aiter__(self):
                for chunk in self.payload.splitlines(keepends=True):
                    yield chunk

        async def backend(request):
            calls.append((request.url.host, json.loads(request.content)))
            if len(calls) == 4:
                all_started.set()
            await all_started.wait()
            if request.url.host == "slice-a":
                return httpx.Response(503, stream=Body(b'{"error":"failed slice"}'))
            return httpx.Response(
                200,
                stream=Body(b'data: {"content":"Four"}\n\ndata: [DONE]\n\n'),
                headers={"content-type": "text/event-stream"},
            )

        app = pool_router(["slice-a", "slice-b"], lambda name: "http://" + name, transport=httpx.MockTransport(backend))
        payload = {"model": "frozen-solver", "stream": True, "messages": [{"role": "user", "content": "unchanged"}]}
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://pool") as client,
        ):
            responses = await asyncio.gather(*(client.post("/v1/chat/completions", json=payload) for _ in range(4)))
        assert sorted(response.status_code for response in responses) == [200, 200, 503, 503]
        assert sorted(host for host, body in calls) == ["slice-a", "slice-a", "slice-b", "slice-b"]
        assert all(body == payload for host, body in calls)
        assert all("data: [DONE]" in response.text for response in responses if response.status_code == 200)

    asyncio.run(run())


def test_staged_solver_can_qualify_while_other_regional_mirrors_finish():
    commands = tpu_launch.commands_for_phase("pilot", tpu.REGIONAL_ROOT, "fixture", "seal", "sha", "judge", Solver.SMALL)
    assert len(commands) == 1
    assert commands[0][commands[0].index("--solver") + 1] == "small"
    assert commands[0][commands[0].index("--replicas") + 1] == "1"


def test_runtime_rejects_non_unit_reward_without_changing_raw_attempt(tmp_path, difficulty_source):
    pytest.importorskip("harbor")
    ledger_worker = importlib.import_module("experiments.post_training.task_curation.difficulty.ledger_worker")
    source = difficulty_source(judged_task())
    path = StoragePath(str(tmp_path / "attempt.json"))
    raw = {"status": "scored", "task_id": "judge-task", "reward": 1.25}
    path.write_text(json.dumps(raw))
    meter = SimpleNamespace(finish=lambda key: 0.0, start=lambda key: None)
    ledger_worker.record_attempt(
        path,
        source=source,
        config=SimpleNamespace(solver=Solver.SMALL),
        output=StoragePath(str(tmp_path / "run/trials")),
        ledger=str(tmp_path / "ledger"),
        meter=meter,
        task_key="fixture",
    )
    (row,) = read_rows(str(tmp_path / "ledger"))
    assert row.outcome == ledger_worker.Outcome.UNSCORED
    assert row.reward is None and row.error_identity == "reward_outside_unit_interval"
    assert row.reason == GapReason.GRADER_INFRASTRUCTURE_FAILURE
    assert json.loads(path.read_text()) == raw


def test_tpu_judge_remediation_never_attempts_a_gpu_fallback(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger"
    identity = campaign.identity_bundle()
    campaign.immutable_json(
        StoragePath(str(ledger / "v1/campaign.json")), {"identity": identity, "campaign_id": digest(identity)}
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("A TPU judge repair must never launch a self-hosted GPU judge")

    monkeypatch.setattr(remediate.subprocess, "run", forbidden)
    result = remediate.repair_action(
        QualifyJudgeRoute("MiniMaxAI/MiniMax-M3-MXFP8", (("source", Solver.SMALL),), ()),
        tmp_path / "repair",
        str(ledger),
    )
    assert result["status"] == "blocked"
    assert result["reason"] == GapReason.JUDGE_ROUTE_UNQUALIFIED


def test_full_allocation_waits_for_cancelled_pilot_tasks_to_release_chips(tmp_path, monkeypatch):
    (tmp_path / "pools").mkdir()
    for solver in tpu.SHAPES:
        (tmp_path / "pools" / f"{solver.value}.json").write_text(
            json.dumps({"replicas": [{"engine_jobs": [solver.value]}]})
        )
    checks = []

    def status(command, **kwargs):
        checks.append(command[-1])
        state = "running" if len(checks) <= 3 else "killed"
        return SimpleNamespace(stdout=json.dumps({"job": {"task_state_counts": {state: 2}}}))

    monkeypatch.setattr(tpu_launch.subprocess, "run", status)
    monkeypatch.setattr(tpu_launch.time, "sleep", lambda seconds: None)
    tpu_launch.await_pilot_teardown(str(tmp_path))
    assert len(checks) == 6
    with pytest.raises(TimeoutError, match="refusing full allocation"):
        checks.clear()
        tpu_launch.await_pilot_teardown(str(tmp_path), timeout_seconds=0)


@pytest.mark.parametrize("registry_failure", [False, True])
def test_tpu_owner_qualifies_registers_routes_retains_receipts_and_tears_down(
    tmp_path, monkeypatch, review_server, registry_failure
):
    """Run the whole owner, including its real HTTP router and publication boundary."""
    solver = Solver.SMALL
    model = tpu.SOLVER_MODELS[solver]
    review_server.model_ids = [model]
    endpoint = f"http://127.0.0.1:{review_server.server_port}/v1"
    (tmp_path / "staging").mkdir()
    (tmp_path / "staging/small.json").write_text(
        json.dumps(
            {
                "model": model,
                "revision": PINS[solver].revision,
                "weights": "gs://marin-us-central2/fixture",
            }
        )
    )
    lifecycle = []
    addresses = {}
    sweeps = []

    class EndOwner(Exception):
        pass

    @contextmanager
    def remote(config):
        lifecycle.append("slice_enter")
        try:
            yield SimpleNamespace(
                ready_direct_session=lambda: SimpleNamespace(
                    effective_serving=EffectiveServing(8, 1, 1, 2, 65536),
                    model=SimpleNamespace(endpoint=SimpleNamespace(base_url=endpoint)),
                ),
                endpoint_name="slice-fixture",
                jobs=[SimpleNamespace(job_id="/benfeuer/atlas-difficulty-engines-small-v4-fixture-pilot/inference")],
            )
        finally:
            lifecycle.append("slice_exit")

    class Cluster:
        def register_endpoint(self, *, name, address, metadata, access, task_attempt):
            assert access == EndpointAccess.ENDPOINT_ACCESS_LINK
            assert metadata == {"model": model}
            lifecycle.append("register")
            if registry_failure:
                raise EndOwner("registry unavailable")
            addresses[name] = address
            return name

        def unregister_endpoint(self, endpoint_id):
            del addresses[endpoint_id]
            lifecycle.append("unregister")

    # Use the actual Iris namespace boundary. The receipt is consumed by sibling
    # reconcile jobs, which cannot resolve an owner's unqualified local name.
    registry = NamespacedEndpointRegistry(
        Cluster(), Namespace("benfeuer/pool-owner"), TaskAttempt.from_wire("/benfeuer/pool-owner/0:0")
    )

    async def warm(endpoint, model, chips, *, concurrency):
        sweeps.append(concurrency)
        return {"output_tokens_per_second": concurrency * 2, "output_tokens_per_second_per_gpu": concurrency * 2 / chips}

    publish = tpu.publish_pool_receipt

    def publish_then_exit(root, solver, receipt):
        # The actual socket, ASGI lifespan, discovery and streaming proxy all run.
        address = addresses[receipt["endpoint_name"]]
        response = httpx.get(address + "/models")
        assert response.status_code == 200
        assert response.json() == {"data": [{"id": model}]}
        publish(root, solver, receipt)
        raise EndOwner("qualified owner stopped by test")

    monkeypatch.setattr(tpu, "regional_worker", lambda: None)
    monkeypatch.setattr(tpu, "configure_coreweave_s3", lambda: None)
    monkeypatch.setattr(tpu, "require_secret", lambda name: "fixture")
    monkeypatch.setattr(tpu, "remote_inference", remote)
    monkeypatch.setattr(tpu, "probe_endpoint", lambda *args: {"qualified": True})
    monkeypatch.setattr(tpu, "long_context_probe", lambda *args: {"prompt_tokens": 49009})
    monkeypatch.setattr(tpu, "warm_throughput", warm)
    monkeypatch.setattr(tpu, "get_job_info", lambda: SimpleNamespace(advertise_host="127.0.0.1"))
    monkeypatch.setattr(
        tpu,
        "iris_ctx",
        lambda: SimpleNamespace(
            client=SimpleNamespace(resolve_endpoint=lambda name: endpoint.removesuffix("/v1")),
            registry=registry,
        ),
    )
    monkeypatch.setattr(tpu, "publish_pool_receipt", publish_then_exit)
    with pytest.raises(EndOwner):
        tpu.run_tpu_pool(solver, str(tmp_path), 1)
    assert sweeps == [8, 32, 128]
    assert lifecycle == (
        ["slice_enter", "register", "slice_exit"]
        if registry_failure
        else ["slice_enter", "register", "unregister", "slice_exit"]
    )
    (qualification,) = (tmp_path / "qualification/small").glob("*.json")
    archived = json.loads(qualification.read_text())
    assert archived["replicas"][0]["warm_throughput"]["output_tokens_per_second_per_chip"] == 32
    assert (tmp_path / "pools/small.json").exists() is not registry_failure


def test_pilot_teardown_uses_receipt_owners_across_generations_and_rejects_foreign_jobs(tmp_path):
    (tmp_path / "pools").mkdir()
    owners = []
    for index, solver in enumerate(tpu.SHAPES):
        owner = f"/benfeuer/atlas-difficulty-engines-{solver.value}-v4-r{index}-pilot"
        owners.append(owner)
        (tmp_path / "pools" / f"{solver.value}.json").write_text(
            json.dumps({"replicas": [{"engine_jobs": [owner + "/inference-a", owner + "/inference-b"]}]})
        )
    assert tpu_launch.pilot_owners(str(tmp_path)) == sorted(owners)
    (tmp_path / "pools/small.json").write_text(
        json.dumps({"replicas": [{"engine_jobs": ["/someone/foreign/inference"]}]})
    )
    with pytest.raises(ValueError, match="owned pilot"):
        tpu_launch.pilot_owners(str(tmp_path))
