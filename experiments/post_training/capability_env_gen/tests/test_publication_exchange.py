"""Construction -> queue -> publisher service -> returns -> import, end to end.

Everything runs locally: a directory queue, a directory object store holding the
captured layer, and a directory registry that validates exactly what the real
RegistryClient would push. The item-side path is the real
process_image_construction pending-publication branch.
"""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_generic_image_publication import _artifacts, _sha

from capability_pipeline import image_pipeline as pipeline
from capability_pipeline import publication_exchange as exchange
from capability_pipeline.image_publication_handoff import export_handoff_for_item
from capability_pipeline.oci_registry import RegistryError
from scripts import run_image_publisher_service as service_module
from scripts.run_image_publisher_service import LocalDirectoryRegistry, PublisherService

REGISTRY = "registry.example"
BUCKET = "marin-us-east-02a"
CAPTURE_PREFIX = f"s3://{BUCKET}/users/muchanem/envrootfs"
PASSWORD = "publisher-password-sentinel-7f3a"
T0 = 1_900_000_000.0


class World(SimpleNamespace):
    def construct(self, **kwargs):
        return pipeline.process_image_construction(
            item_root=self.item, capture_tools=self.tools, scripts_root=self.scripts,
            agent=SimpleNamespace(model="glm-5.3"), builder_session_ids={"builder-1"},
            command_runner=self.runner, review_base=self.review_base,
            publication_queue=self.queue, **kwargs)

    def service(self, *, registry_factory=None, prefixes=(CAPTURE_PREFIX,), clock=None, logs=None):
        return PublisherService(
            queue=self.queue, work_root=self.root / "publisher/work",
            credentials_file=self.credentials, private_credentials=self.root / "publisher/run/registry.json",
            expected_registry=REGISTRY,
            reader=exchange.ObjectStoreReader(exchange.LocalObjectClient(self.objects), list(prefixes)),
            registry_client_factory=registry_factory or LocalDirectoryRegistry.factory(self.registry),
            source_sha256="5" * 64, host="test-publisher", clock=clock or (lambda: T0),
            log=(logs.append if logs is not None else lambda record: None))

    def pending(self):
        roles = [self.role]
        return {"state": "pending_publication", "roles": roles, "builder_session_ids": ["builder-1"],
                "plan_path": str(self.plan), "workspace": str(self.frozen), "capture_tools": str(self.tools),
                "approval_path": str(self.approval),
                "capture_paths": {self.role: str(self.attempt / f"capture-{self.role}.json")},
                "publication_paths": {self.role: str(self.attempt / f"publication-{self.role}.json")}}


def _world(tmp_path, monkeypatch, *, role="candidate", payload=b"public", registry_host=None) -> World:
    source = tmp_path / "source"
    source.mkdir()
    workspace, tools, plan, approval, capture, layer = _artifacts(
        source, role=role, payload=payload, registry_host=registry_host)
    digest = _sha(layer)
    key = f"users/muchanem/envrootfs/{digest}.tar.gz"
    objects = tmp_path / "objects"
    (objects / BUCKET / key).parent.mkdir(parents=True)
    shutil.copy2(layer, objects / BUCKET / key)
    receipt = json.loads(capture.read_text())
    receipt["capture"].update(ok=True, object=f"s3://{BUCKET}/{key}", object_key=key)
    receipt["capture"]["capture"]["sha256"] = digest
    capture.write_text(json.dumps(receipt))

    run = tmp_path / "run"
    item = run / "items/task"
    attempt = item / "diagnostics/image-capture/attempt-source"
    frozen = attempt / "input/workspace"
    frozen.parent.mkdir(parents=True)
    shutil.copytree(workspace, frozen)
    plan_path = attempt / "input/plan.json"
    shutil.copy2(plan, plan_path)
    shutil.copy2(capture, attempt / f"capture-{role}.json")
    task = item / "workspace/task"
    task.mkdir(parents=True)
    pointer = json.loads(plan.read_text())["images"][0]["authored_image_pointer"]
    if role == "candidate":
        (task / "specification.json").write_text(json.dumps({"requirements": {"state": {"image": pointer}}, "steps": []}))
        (task / "binding.json").write_text(json.dumps({"environment": {"image": pointer}}))
    else:
        (task / "specification.json").write_text(json.dumps({"steps": [{"verifier": {"runtime": {"image": pointer}}}]}))
        (task / "binding.json").write_text(json.dumps({"environment": {}}))
    review_base = run / "image-reviews/task"
    shutil.copytree(approval.parent, review_base / attempt.name)
    credentials = tmp_path / "secret/credentials.json"
    credentials.parent.mkdir()
    credentials.write_text(json.dumps({"registry": REGISTRY, "user": "capability-publisher", "password": PASSWORD}))

    monkeypatch.setattr(pipeline, "prepare_construction_capture", lambda *_: {
        "attempt": str(attempt), "plan_path": str(plan_path), "workspace": str(frozen)})
    monkeypatch.setattr(pipeline, "_review_matches_task", lambda *a: None)
    monkeypatch.setattr(pipeline, "migrate_image_pointers", lambda **kw: {
        "roles": {r: {"publication": str(p)} for r, p in kw["publication_paths"].items()}})
    world = World(root=tmp_path, item=item, attempt=attempt, plan=plan_path, frozen=frozen, tools=tools,
                  approval=review_base / attempt.name / "approval.json", review_base=review_base,
                  scripts=tmp_path / "scripts", objects=objects, registry=tmp_path / "registry",
                  queue=exchange.LocalQueue(tmp_path / "queue"), credentials=credentials, role=role,
                  exchange_root=run / "image-publication/task" / attempt.name, commands=[])
    world.scripts.mkdir()

    def runner(command):
        world.commands.append(command)
        if Path(command[0]).name == "capture_generic_task_image.py":
            pytest.fail("capture already exists")
        assert Path(command[0]).name == "probe_generic_task_image.py"
        Path(command[command.index("--output") + 1]).write_text("{}")
        return SimpleNamespace(returncode=0)

    world.runner = runner
    return world


def _all_bytes(*roots: Path) -> bytes:
    return b"".join(path.read_bytes() for root in roots if root.exists()
                    for path in root.rglob("*") if path.is_file())


@pytest.mark.parametrize("role", ["candidate", "private_verifier"])
def test_end_to_end_construction_queue_publisher_import_and_cold_pull(tmp_path, monkeypatch, role):
    world = _world(tmp_path, monkeypatch, role=role)
    first = world.construct()
    assert first["state"] == "pending_publication"
    assert first["retryable"] is True
    assert first["reason"] == "awaiting_publisher_heartbeat_stale"  # no service has run yet
    assert first["backoff_seconds"] >= 60
    packet_sha = first["packet_sha"]
    assert first["submitted_at"]
    assert first["capture_paths"][role] == str(world.attempt / f"capture-{role}.json")  # legacy keys kept
    archive = world.queue.get(exchange.request_key(packet_sha))
    assert exchange.sha256(archive) == packet_sha
    record = json.loads((world.exchange_root / "request.json").read_text())
    assert record["packet_sha256"] == packet_sha and record["packet_bytes"] == len(archive)
    assert not (world.exchange_root / "handoff.tar.gz").exists()  # the queue holds the copy

    again = world.construct()
    assert again["packet_sha"] == packet_sha  # idempotent, content-addressed resubmission

    logs = []
    publisher = world.service(logs=logs)
    publisher.serve(poll_seconds=0, drain=True)
    assert publisher.counts == {"published": 1}
    result = exchange.parse_result(world.queue.get(exchange.result_key(packet_sha)), packet_sha)
    assert result["state"] == "published" and result["attempt"] == 1
    assert result["images"][role].startswith(REGISTRY + "/capability-env-gen/test-candidate@sha256:")
    heartbeat = json.loads(world.queue.get(exchange.HEARTBEAT_KEY))
    assert heartbeat["state"] == "running" and heartbeat["source_sha256"] == "5" * 64
    manifest_digest = result["images"][role].rpartition("@")[2]
    assert (world.registry / "capability-env-gen/test-candidate/manifests" / manifest_digest.replace(":", "-")).is_file()

    done = world.construct()
    assert done["state"] == "ready"
    assert done["reason"] == "reviewed_images_migrated"
    imported = world.attempt / f"publication-{role}.json"
    assert imported.read_bytes() == world.queue.get(exchange.receipt_key(packet_sha, role))
    receipt = json.loads(imported.read_text())
    assert receipt["state"] == "published_pending_cold_pull"
    assert receipt["layer_transport"]["object"].startswith(CAPTURE_PREFIX + "/")
    cold = [command for command in world.commands if Path(command[0]).name == "probe_generic_task_image.py"]
    assert len(cold) == 1 and cold[0][cold[0].index("--publication") + 1] == str(imported)

    # The registry credential never left the publisher: not in the queue, the
    # item, the retained exchange, or any log line; the private copy is gone.
    assert PASSWORD.encode() not in _all_bytes(world.queue.root, world.item, world.exchange_root.parent)
    assert PASSWORD not in json.dumps(logs)
    assert not (world.root / "publisher/run/registry.json").exists()


def test_object_storage_plan_host_is_corrected_and_accepted_on_import(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch, registry_host="marin-us-east-02a.cwobject.com")
    world.construct()
    world.service().serve(poll_seconds=0, drain=True)
    assert world.construct()["state"] == "ready"
    receipt = json.loads((world.attempt / "publication-candidate.json").read_text())
    assert receipt["registry_host_corrected"]["to"] == REGISTRY
    assert receipt["publication"]["image"].startswith(REGISTRY + "/")


def test_rootfs_review_rejection_is_terminal_for_the_item(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch, payload=b"private")  # private bytes at a public path
    world.construct()
    world.service().serve(poll_seconds=0, drain=True)
    held = world.construct()
    assert held["state"] == "failed_terminal"
    assert held["failure_stage"] == "image_publication"
    assert held["rejection_class"] == "rootfs_review_failed"
    assert held["retryable"] is False
    assert "candidate: RootfsReviewError" in held["reason"] and "private content" in held["reason"]
    assert held["builder_repairable"] is True
    assert "required_ready_hashes" in held["issues"][0]
    assert "image-capture-request.json" in held["issues"][0]
    assert not (world.attempt / "publication-candidate.json").exists()


def test_capture_object_outside_allowed_prefix_is_a_policy_hold_not_a_rejection(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    sha = world.construct()["packet_sha"]
    world.service(prefixes=("s3://other-bucket/elsewhere",)).serve(poll_seconds=0, drain=True)
    result = exchange.parse_result(world.queue.get(exchange.result_key(sha)), sha)
    assert result["state"] == "transient_failure"
    assert "outside the publisher's allowed prefixes" in result["reason"]


def test_missing_capture_object_is_rejected(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    world.construct()
    for path in world.objects.rglob("*.tar.gz"):
        path.unlink()
    world.service().serve(poll_seconds=0, drain=True)
    held = world.construct()
    assert held["state"] == "failed_terminal"
    assert held["rejection_class"] == "capture_object_missing"
    assert held["builder_repairable"] is False


class _FlakyRegistry:
    def __init__(self, root, failures):
        self.inner, self.failures = LocalDirectoryRegistry.factory(root), failures

    def __call__(self, host, repository, user, password):
        client = self.inner(host, repository, user, password)
        outer = self

        class Client:
            def __init__(self):
                self.host, self.repository = client.host, client.repository

            def publish_layer(self, *args, **kwargs):
                if outer.failures:
                    outer.failures -= 1
                    raise RegistryError("registry HTTP failure: 503")
                return client.publish_layer(*args, **kwargs)

        return Client()


def test_transient_failure_backs_off_resubmits_and_then_imports(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    now = [T0]
    first = exchange.exchange_publication(item_root=world.item, pending=world.pending(),
                                          exchange_root=world.exchange_root, queue=world.queue,
                                          clock=lambda: now[0])
    sha = first["packet_sha"]
    flaky = _FlakyRegistry(world.registry, failures=1)
    world.service(registry_factory=flaky, clock=lambda: now[0]).serve(poll_seconds=0, drain=True)
    result = exchange.parse_result(world.queue.get(exchange.result_key(sha)), sha)
    assert result["state"] == "transient_failure"
    assert result["reason"].startswith("candidate: registry_publish_failed: RegistryError")

    now[0] = T0 + 10
    backoff = exchange.exchange_publication(item_root=world.item, pending=world.pending(),
                                            exchange_root=world.exchange_root, queue=world.queue,
                                            clock=lambda: now[0])
    assert backoff["reason"] == "publisher_transient_failure_backoff" and backoff["retryable"] is True
    assert 280 <= backoff["backoff_seconds"] <= 300
    assert world.queue.get(exchange.resubmit_key(sha, 2)) is None

    now[0] = T0 + exchange.RESUBMIT_BASE_SECONDS + 1
    resubmitted = exchange.exchange_publication(item_root=world.item, pending=world.pending(),
                                                exchange_root=world.exchange_root, queue=world.queue,
                                                clock=lambda: now[0])
    assert resubmitted["reason"] == "publisher_transient_failure_resubmitted"
    assert resubmitted["publication_attempt"] == 2
    assert world.queue.get(exchange.resubmit_key(sha, 2)) is not None

    world.service(registry_factory=flaky, clock=lambda: now[0]).serve(poll_seconds=0, drain=True)
    assert exchange.parse_result(world.queue.get(exchange.result_key(sha)), sha)["attempt"] == 2
    assert exchange.exchange_publication(item_root=world.item, pending=world.pending(),
                                         exchange_root=world.exchange_root, queue=world.queue,
                                         clock=lambda: now[0]) is None
    assert (world.attempt / "publication-candidate.json").is_file()


def test_transient_retries_are_bounded(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    sha = world.construct()["packet_sha"]
    world.queue.put(exchange.result_key(sha), exchange.result_document(
        packet_sha=sha, attempt=exchange.MAX_PUBLICATION_ATTEMPTS, state="transient_failure",
        reason="registry_publish_failed: RegistryError: registry transport failed"))
    for attempt in range(2, exchange.MAX_PUBLICATION_ATTEMPTS + 1):
        world.queue.put(exchange.resubmit_key(sha, attempt), b"{}")
    held = world.construct()
    assert held["state"] == "pending_publication"
    assert held["retryable"] is False
    assert held["reason"] == "publisher_transient_retries_exhausted"
    # The publisher does not retry on its own either.
    publisher = world.service()
    publisher.serve(poll_seconds=0, drain=True)
    assert publisher.counts == {}


def test_queue_unreachable_is_a_retryable_hold(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)

    class Down:
        def __getattr__(self, name):
            def fail(*args, **kwargs):
                raise ConnectionError("object store down")
            return fail

    world.queue = Down()
    held = world.construct()
    assert held["state"] == "pending_publication"
    assert held["retryable"] is True
    assert held["reason"] == "queue_unreachable"
    assert held["error_type"] == "ConnectionError"
    assert len(held["packet_sha"]) == 64
    assert held["backoff_seconds"] == exchange.QUEUE_UNREACHABLE_BACKOFF_SECONDS


def test_default_queue_without_credentials_or_rigging_holds_retryably(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    world.queue = None
    monkeypatch.setenv(exchange.QUEUE_ENV, "s3://marin-us-east-02a/users/test/publication")
    monkeypatch.setattr(exchange, "FsspecQueue", lambda uri: (_ for _ in ()).throw(ImportError("rigging")))
    held = world.construct()
    assert held["reason"] == "queue_unreachable" and held["retryable"] is True


def test_disabled_queue_restores_the_manual_hold(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    world.queue = None
    monkeypatch.setenv(exchange.QUEUE_ENV, "off")
    held = world.construct()
    assert held["state"] == "pending_publication"
    assert held["reason"] == "publication_queue_disabled" and held["retryable"] is False
    assert not world.exchange_root.exists()


def test_submission_is_deterministic_and_uploaded_once(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    puts = []
    inner = world.queue

    class Counting:
        def __getattr__(self, name):
            return getattr(inner, name)

        def put(self, key, data):
            puts.append(key)
            inner.put(key, data)

    world.queue = Counting()
    first, second = world.construct(), world.construct()
    assert first["packet_sha"] == second["packet_sha"]
    assert puts == [exchange.request_key(first["packet_sha"])]
    one = export_handoff_for_item(world.item, world.pending(), tmp_path / "a.tar.gz")
    two = export_handoff_for_item(world.item, world.pending(), tmp_path / "b.tar.gz")
    assert one["archive_sha256"] == two["archive_sha256"] == first["packet_sha"]


def test_published_packets_are_never_republished_even_after_restart(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    sha = world.construct()["packet_sha"]
    world.service().serve(poll_seconds=0, drain=True)
    before = world.queue.get(exchange.result_key(sha))
    restarted = world.service()
    restarted.serve(poll_seconds=0, drain=True)
    assert restarted.counts == {} and sha in restarted.final
    assert world.queue.get(exchange.result_key(sha)) == before


def test_crash_after_partial_returns_is_redone_safely(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    sha = world.construct()["packet_sha"]
    # A previous publisher died after uploading a (torn) receipt but before result.json.
    world.queue.put(exchange.receipt_key(sha, "candidate"), b"torn")
    world.queue.put(exchange.started_key(sha, 1), json.dumps({"starts": 1}).encode())
    world.service().serve(poll_seconds=0, drain=True)
    assert json.loads(world.queue.get(exchange.started_key(sha, 1)))["starts"] == 2
    assert world.construct()["state"] == "ready"


def test_crash_loops_are_bounded_per_attempt(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    sha = world.construct()["packet_sha"]
    world.queue.put(exchange.started_key(sha, 1), json.dumps({"starts": service_module.MAX_STARTS_PER_ATTEMPT}).encode())
    world.service().serve(poll_seconds=0, drain=True)
    result = exchange.parse_result(world.queue.get(exchange.result_key(sha)), sha)
    assert result["state"] == "transient_failure"
    assert result["reason"].startswith("publisher_interrupted_repeatedly")


def test_forged_or_corrupt_packets_are_rejected_as_data(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    garbage = b"not a tarball"
    garbage_sha = exchange.sha256(garbage)
    world.queue.put(exchange.request_key(garbage_sha), garbage)
    wrong_sha = "a" * 64
    world.queue.put(exchange.request_key(wrong_sha), garbage)
    world.service().serve(poll_seconds=0, drain=True)
    garbage_result = exchange.parse_result(world.queue.get(exchange.result_key(garbage_sha)), garbage_sha)
    assert garbage_result["state"] == "rejected" and garbage_result["rejection_class"] == "packet_invalid"
    mismatch = exchange.parse_result(world.queue.get(exchange.result_key(wrong_sha)), wrong_sha)
    assert mismatch["rejection_class"] == "packet_digest_mismatch"


def test_broken_credential_degrades_without_consuming_attempts(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    sha = world.construct()["packet_sha"]
    world.credentials.write_text(json.dumps({"registry": "other.example", "user": "u", "password": PASSWORD}))
    logs = []
    publisher = world.service(logs=logs)
    publisher.serve(poll_seconds=0, drain=True)
    heartbeat = json.loads(world.queue.get(exchange.HEARTBEAT_KEY))
    assert heartbeat["state"] == "degraded"
    assert heartbeat["reason"] == "credential_registry_differs_from_expected"
    assert world.queue.get(exchange.result_key(sha)) is None
    assert world.queue.get(exchange.started_key(sha, 1)) is None
    assert PASSWORD not in json.dumps(logs)


def test_stale_heartbeat_is_reported_while_waiting(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    world.queue.put(exchange.HEARTBEAT_KEY, json.dumps({"epoch": T0 - 30}).encode())
    fresh = exchange.exchange_publication(item_root=world.item, pending=world.pending(),
                                          exchange_root=world.exchange_root, queue=world.queue,
                                          clock=lambda: T0)
    assert fresh["reason"] == "awaiting_publisher" and fresh["publisher_heartbeat_age_seconds"] == 30
    stale = exchange.exchange_publication(item_root=world.item, pending=world.pending(),
                                          exchange_root=world.exchange_root, queue=world.queue,
                                          clock=lambda: T0 + 3600)
    assert stale["reason"] == "awaiting_publisher_heartbeat_stale"
    assert stale["backoff_seconds"] == 600  # grows with time since submission, capped


def test_receipt_that_differs_from_its_result_record_is_not_imported(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    sha = world.construct()["packet_sha"]
    world.service().serve(poll_seconds=0, drain=True)
    world.queue.put(exchange.receipt_key(sha, "candidate"), b"{}")
    held = world.construct()
    assert held["reason"] == "publisher_result_invalid" and held["retryable"] is True
    assert not (world.attempt / "publication-candidate.json").exists()


def test_service_cli_dry_run_drains_a_local_queue(tmp_path, monkeypatch, capsys):
    world = _world(tmp_path, monkeypatch)
    sha = world.construct()["packet_sha"]
    code = service_module.main([
        "--queue", str(world.queue.root), "--work-root", str(tmp_path / "cli-work"),
        "--credentials-file", str(world.credentials),
        "--private-credentials", str(tmp_path / "cli-run/registry.json"),
        "--expected-registry", REGISTRY, "--capture-object-prefix", CAPTURE_PREFIX,
        "--dry-run-object-root", str(world.objects), "--dry-run-registry-root", str(world.registry),
        "--drain", "--poll-seconds", "0",
    ])
    assert code == 0
    assert exchange.parse_result(world.queue.get(exchange.result_key(sha)), sha)["state"] == "published"
    output = capsys.readouterr().out
    assert '"event": "packet_result"' in output and PASSWORD not in output
    assert world.construct()["state"] == "ready"


def test_result_documents_are_strict():
    sha = "b" * 64
    with pytest.raises(ValueError):
        exchange.parse_result(b'{"schema_version": "x"}', sha)
    with pytest.raises(ValueError):
        exchange.result_document(packet_sha=sha, attempt=1, state="published", reason="ok",
                                 roles=["candidate"], receipts={}, packet_manifest_sha256="c" * 64)
    with pytest.raises(ValueError):
        exchange.result_document(packet_sha=sha, attempt=1, state="rejected", reason="no class")
    assert exchange.parse_result(None, sha) is None
    with pytest.raises(ValueError):
        exchange.request_key("../../etc")


@pytest.mark.parametrize("bound", ["MAX_PACKET_BYTES", "MAX_PACKET_EXPANDED_BYTES"])
def test_packet_over_capacity_is_a_bounded_hold_not_a_rejection(tmp_path, monkeypatch, bound):
    world = _world(tmp_path, monkeypatch)
    sha = world.construct()["packet_sha"]
    monkeypatch.setattr(service_module, bound, 10)
    world.service().serve(poll_seconds=0, drain=True)
    result = exchange.parse_result(world.queue.get(exchange.result_key(sha)), sha)
    assert result["state"] == "transient_failure"
    assert result["reason"].startswith("publisher_capacity_exceeded")


def test_waiting_polls_reuse_the_retained_packet(tmp_path, monkeypatch):
    world = _world(tmp_path, monkeypatch)
    from capability_pipeline import image_publication_handoff as handoff

    calls = []
    real = handoff.handoff_packet
    monkeypatch.setattr(handoff, "handoff_packet", lambda *a, **kw: calls.append(1) or real(*a, **kw))
    shas = {world.construct()["packet_sha"] for _ in range(3)}
    assert len(shas) == 1 and len(calls) == 1
    request = world.queue.root / exchange.request_key(shas.pop())
    request.unlink()  # queue lost the request: rebuilt byte-identically under the same key
    again = world.construct()
    assert request.is_file() and exchange.sha256(request.read_bytes()) == again["packet_sha"]
    assert len(calls) == 2
    world.service().serve(poll_seconds=0, drain=True)
    request.unlink()  # import after the queue copy vanished: rebuilt and verified locally
    assert world.construct()["state"] == "ready" and len(calls) == 3
