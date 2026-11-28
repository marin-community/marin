# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise replica replacement through the owner, live router and persisted receipts."""

import asyncio
import copy
import json
import threading
from contextlib import contextmanager
from types import SimpleNamespace

import httpx
import pytest
from finestore.rl.difficulty import LARGE_DYNAMIC_OUTPUT_PROTOCOL_ID, Solver
from iris.cluster.client.job_info import JobInfo, set_job_info
from iris.cluster.types import JobName
from marin.inference.config import EffectiveServing
from marin.inference.types import InferenceBackendState
from rigging.timing import ExponentialBackoff

from experiments.post_training.task_curation.difficulty import controller, runtime, tpu
from experiments.post_training.task_curation.difficulty.tpu_router import pool_router


@pytest.mark.parametrize("exhaust", [False, True])
def test_failed_replica_replacement_preserves_healthy_service_and_receipt_history(
    tmp_path, monkeypatch, review_server, exhaust
):
    solver = Solver.LARGE
    model = tpu.SOLVER_MODELS[solver]
    review_server.model_ids = [model]
    endpoint = f"http://127.0.0.1:{review_server.server_port}/v1"
    (tmp_path / "staging").mkdir()
    (tmp_path / "staging/large.json").write_text(
        json.dumps(dict(model=model, revision=tpu.PINS[solver].revision, weights="gs://marin-us-central2/fixture"))
    )
    initial = threading.Event()
    crash = threading.Event()
    degraded = threading.Event()
    release = threading.Event()
    cleanup = threading.Event()
    entered, closed, receipts, errors, resolved = [], [], [], [], []
    names_lock = threading.Lock()
    failed_name = None
    router_address = None

    class EndOwner(Exception):
        pass

    @contextmanager
    def remote(config):
        with names_lock:
            name = f"slice-{len(entered)}"
            entered.append(name)

        def ready():
            if cleanup.is_set():
                raise RuntimeError("test shutdown")
            if name not in ("slice-0", "slice-1", "slice-2", "slice-3"):
                assert release.wait(10)
                if exhaust:
                    raise RuntimeError("replacement did not qualify")
            return SimpleNamespace(
                effective_serving=EffectiveServing(16, 2, 1, 16, 65536),
                model=SimpleNamespace(endpoint=SimpleNamespace(base_url=endpoint)),
            )

        def alive():
            if cleanup.is_set() or (name == failed_name and crash.is_set()):
                raise RuntimeError("engine worker failed")

        session = SimpleNamespace(
            ready_direct_session=ready,
            direct_session=lambda: SimpleNamespace(model=SimpleNamespace(endpoint=SimpleNamespace(base_url=endpoint))),
            check_alive=alive,
            backend_state=lambda: InferenceBackendState.READY,
            endpoint_name=name,
            jobs=[SimpleNamespace(job_id="/benfeuer/fixture/" + name)],
        )
        try:
            yield session
        finally:
            closed.append(name)

    class Registry:
        @contextmanager
        def registered(self, name, address, metadata, access):
            nonlocal router_address
            router_address = address
            yield name

    def resolve(name):
        resolved.append(name)
        return router_address if name.startswith("/serve/") else endpoint.removesuffix("/v1")

    async def warm(*args, concurrency):
        return dict(output_tokens_per_second=concurrency, output_tokens_per_second_per_gpu=concurrency / 32)

    publish = tpu.publish_pool_receipt

    def published(root, solver, receipt):
        nonlocal failed_name
        publish(root, solver, receipt)
        receipts.append(copy.deepcopy(receipt))
        if not initial.is_set():
            failed_name = receipt["replicas"][0]["endpoint_name"]
            initial.set()
        elif len(receipt["replicas"]) == 3:
            degraded.set()
        elif any(item.get("generation", 1) > 1 for item in receipt["replicas"]):
            assert httpx.get(router_address + "/models").status_code == 200
            raise EndOwner("test complete after qualified replacement")

    monkeypatch.setattr(tpu, "regional_worker", lambda: None)
    monkeypatch.setattr(tpu, "configure_coreweave_s3", lambda: None)
    monkeypatch.setattr(tpu, "require_secret", lambda _: "fixture")
    monkeypatch.setattr(tpu, "remote_inference", remote)
    monkeypatch.setattr(tpu, "probe_endpoint", lambda *args: {"qualified": True})
    monkeypatch.setattr(tpu, "long_context_probe", lambda *args: {"prompt_tokens": 49009})
    monkeypatch.setattr(tpu, "warm_throughput", warm)
    monkeypatch.setattr(
        tpu, "iris_ctx", lambda: SimpleNamespace(client=SimpleNamespace(resolve_endpoint=resolve), registry=Registry())
    )
    monkeypatch.setattr(runtime, "iris_ctx", tpu.iris_ctx)
    monkeypatch.setattr(tpu, "publish_pool_receipt", published)
    monkeypatch.setattr(tpu, "REPLICA_HEALTH_INTERVAL", 0.01, raising=False)
    monkeypatch.setattr(
        tpu, "REPLICA_RETRY_BACKOFF", ExponentialBackoff(initial=0.001, maximum=0.001, jitter=0), raising=False
    )

    def owner():
        set_job_info(JobInfo(task_id=JobName.from_wire("/benfeuer/fixture/0"), advertise_host="127.0.0.1"))
        try:
            tpu.run_tpu_pool(solver, str(tmp_path), 4, solver_protocol_id=LARGE_DYNAMIC_OUTPUT_PROTOCOL_ID)
        except BaseException as error:
            errors.append(error)
        finally:
            set_job_info(None)

    thread = threading.Thread(target=owner)
    thread.start()
    try:
        assert initial.wait(10)
        crash.set()
        assert degraded.wait(10), errors
        healthy = {"slice-0", "slice-1", "slice-2", "slice-3"} - {failed_name}
        assert healthy.isdisjoint(closed)
        assert thread.is_alive()
        resolved.clear()
        for _ in range(3):
            assert httpx.get(router_address + "/models").status_code == 200
        assert set(resolved) and set(resolved) <= healthy
        assert Solver.LARGE in controller.qualified_tpu_solvers(str(tmp_path / "pools"))
        degraded_receipt = json.loads((tmp_path / "pools/large.json").read_text())
        assert degraded_receipt["chip_count"] == 128
        assert degraded_receipt["active_chip_count"] == 96
        assert len(degraded_receipt["recovering_replica_indices"]) == 1
        release.set()
        thread.join(10)
        assert not thread.is_alive()
        assert len(errors) == 1
        if exhaust:
            assert "exhausted its restart budget" in str(errors[0])
            assert len(entered) == 7  # Four initial replicas, then three replacements for the failed one.
        else:
            assert isinstance(errors[0], EndOwner)
            assert len(entered) == 5
            restored = json.loads((tmp_path / "pools/large.json").read_text())
            assert restored["endpoint_name"] == receipts[0]["endpoint_name"]
            assert restored["ready_at"] == receipts[0]["ready_at"]
            assert restored["active_chip_count"] == restored["chip_count"] == 128
            assert restored["recovering_replica_indices"] == []
            assert failed_name not in restored["endpoint_names"]
            history = [json.loads(path.read_text()) for path in (tmp_path / "pools/history/large").glob("*.json")]
            assert all(value in history for value in receipts)
            assert all(value["solver_protocol_id"] == LARGE_DYNAMIC_OUTPUT_PROTOCOL_ID for value in history)
            assert all(closed.index(failed_name) < closed.index(name) for name in healthy)
    finally:
        cleanup.set()
        release.set()
        thread.join(10)


def test_router_changes_qualified_endpoints_without_replaying_an_inflight_request():
    async def run():
        names = ["old"]
        entered, finish = asyncio.Event(), asyncio.Event()
        requests = []

        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                entered.set()
                await finish.wait()
                yield b"original response"

        class Replacement(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b"replacement response"

        async def backend(request):
            requests.append(request.url.host)
            if request.url.host == "old":
                return httpx.Response(200, stream=Stream())
            return httpx.Response(200, stream=Replacement())

        app = pool_router(lambda: tuple(names), lambda name: "http://" + name, transport=httpx.MockTransport(backend))
        async with (
            app.router.lifespan_context(app),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://router") as client,
        ):
            inflight = asyncio.create_task(client.post("/v1/chat/completions"))
            await entered.wait()
            names.clear()
            assert (await client.get("/v1/models")).status_code == 503
            names.append("new")
            assert (await client.post("/v1/chat/completions")).content == b"replacement response"
            finish.set()
            assert (await inflight).content == b"original response"
            assert requests == ["old", "new"]

    asyncio.run(run())
