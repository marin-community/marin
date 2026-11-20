# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Job-prefix-scoped tokens: minting, verification, and the owner-gated gates
(submit, cancel, kick, exec-adjacent endpoint mint) plus task-token inheritance."""

import json
from urllib.parse import quote

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.bundle import BundleStore
from iris.cluster.config import AuthConfig
from iris.cluster.controller import ops, reads
from iris.cluster.controller.auth import (
    DEFAULT_SCOPED_TOKEN_TTL_SECONDS,
    MAX_SCOPED_TOKEN_TTL_SECONDS,
    ControllerAuth,
    NativeProxyIdentityAuthenticator,
    create_controller_auth,
)
from iris.cluster.controller.controller import with_task_token
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.cluster.controller.ops.task import Assignment
from iris.cluster.controller.projections.run_templates import RunTemplatesProjection
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.service import ControllerServiceImpl
from iris.cluster.types import JobName, WorkerId
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.auth import authorize_job_prefix
from iris.testing.controller import make_job_request, make_worker_metadata, query_task, register_worker, submit_job
from iris.testing.transitions import WorkerTaskUpdates, apply_task_observations
from iris.time_proto import duration_to_proto
from rigging.server_auth import AuthDecision, AuthRequest, VerifiedIdentity, identity_scope
from rigging.timing import Duration, Timestamp
from rigging.token_authority import JwtSigner, generate_ed25519_keypair, signing_key_from_private_pem

_ALICE = "alice"
_PREFIX = "/alice/exp2-"
_INSIDE_JOB = "/alice/exp2-train"
_OUTSIDE_JOB = "/alice/other-train"


def _enforcing_auth() -> ControllerAuth:
    return create_controller_auth(
        AuthConfig(trusted_cidrs=["10.0.0.0/8"]),
        cluster_name="test-cluster",
        signing_key_pem=generate_ed25519_keypair().private_pem,
    )


def _scoped_identity(auth: ControllerAuth, user: str = _ALICE, prefix: str = _PREFIX) -> VerifiedIdentity:
    """A verified identity carrying the prefix claim, exactly as a minted token yields."""
    return auth.jwt_manager.verify(auth.jwt_manager.create_scoped_token(user, prefix, "iris_scoped_test"))


@pytest.fixture
def scoped_auth() -> ControllerAuth:
    return _enforcing_auth()


@pytest.fixture
def endpoint_service(state):
    return EndpointServiceImpl(db=state._db)


@pytest.fixture
def auth_service(state, mock_controller, tmp_path, log_client, scoped_auth, endpoint_service):
    return ControllerServiceImpl(
        controller=mock_controller,
        bundle_store=BundleStore(storage_dir=str(tmp_path / "bundles")),
        log_client=log_client,
        db=state._db,
        auth=scoped_auth,
        endpoint_service=endpoint_service,
    )


def _launch(service, name: str):
    return service.launch_job(make_job_request(name), None)


def _terminate(service, job_wire: str):
    return service.terminate_job(controller_pb2.Controller.TerminateJobRequest(job_id=job_wire), None)


def _kick(service, targets, desired_state=job_pb2.TASK_STATE_PREEMPTED):
    return service.kick_tasks(
        controller_pb2.Controller.KickTasksRequest(targets=targets, desired_state=desired_state, reason=""),
        None,
    )


def _job_state(state, job_wire: str) -> int:
    with state._db.read_snapshot() as tx:
        return reads.get_job_state(tx, JobName.from_wire(job_wire))


# --- Mint RPC -----------------------------------------------------------------


def test_mint_scoped_token_returns_a_token_for_the_callers_own_prefix(auth_service, scoped_auth):
    with identity_scope(VerifiedIdentity(user_id=_ALICE, role="user")):
        resp = auth_service.mint_scoped_token(controller_pb2.Controller.MintScopedTokenRequest(prefix=_PREFIX), None)

    assert resp.prefix == _PREFIX
    assert resp.HasField("expires_at")
    identity = scoped_auth.jwt_manager.verify(resp.token)
    assert (identity.user_id, identity.role, identity.prefix) == (_ALICE, "user", _PREFIX)


def test_mint_scoped_token_denies_a_prefix_owned_by_another_user(auth_service):
    with identity_scope(VerifiedIdentity(user_id=_ALICE, role="user")), pytest.raises(ConnectError) as exc:
        auth_service.mint_scoped_token(controller_pb2.Controller.MintScopedTokenRequest(prefix="/bob/exp2-"), None)
    assert exc.value.code is Code.PERMISSION_DENIED


def test_mint_scoped_token_denies_a_scoped_caller_a_wider_or_sibling_prefix(auth_service):
    """Delegation only narrows: a scoped token cannot mint outside its own prefix."""
    with (
        identity_scope(VerifiedIdentity(user_id=_ALICE, role="user", prefix=_PREFIX)),
        pytest.raises(ConnectError) as exc,
    ):
        auth_service.mint_scoped_token(controller_pb2.Controller.MintScopedTokenRequest(prefix="/alice/other-"), None)
    assert exc.value.code is Code.PERMISSION_DENIED


def test_mint_scoped_token_allows_a_scoped_caller_a_sub_prefix(auth_service, scoped_auth):
    with identity_scope(VerifiedIdentity(user_id=_ALICE, role="user", prefix=_PREFIX)):
        resp = auth_service.mint_scoped_token(
            controller_pb2.Controller.MintScopedTokenRequest(prefix=f"{_PREFIX}train-"), None
        )
    identity = scoped_auth.jwt_manager.verify(resp.token)
    assert identity.prefix == f"{_PREFIX}train-"


def test_mint_scoped_token_denies_a_scoped_task_token_a_sibling_prefix(auth_service):
    with (
        identity_scope(VerifiedIdentity(user_id=_ALICE, role="task", job_id="/alice/exp2-train", prefix=_PREFIX)),
        pytest.raises(ConnectError) as exc,
    ):
        auth_service.mint_scoped_token(controller_pb2.Controller.MintScopedTokenRequest(prefix="/alice/other-"), None)
    assert exc.value.code is Code.PERMISSION_DENIED


def test_mint_scoped_token_lets_an_admin_mint_for_another_user(auth_service, scoped_auth):
    with identity_scope(VerifiedIdentity(user_id="operator", role="admin")):
        resp = auth_service.mint_scoped_token(controller_pb2.Controller.MintScopedTokenRequest(prefix=_PREFIX), None)

    identity = scoped_auth.jwt_manager.verify(resp.token)
    # The token acts as the prefix's user, never as an admin identity.
    assert (identity.user_id, identity.role, identity.prefix) == (_ALICE, "user", _PREFIX)


@pytest.mark.parametrize(
    "prefix",
    ["benfeuer/exp2-", "", "/benfeuer", "/benfeuer/", "/benfeuer//x", "/benfeuer/exp 2-"],
    ids=["relative", "empty", "bare-user", "trailing-slash", "empty-segment", "whitespace"],
)
def test_mint_scoped_token_rejects_malformed_prefix(auth_service, prefix):
    with identity_scope(VerifiedIdentity(user_id=_ALICE, role="user")), pytest.raises(ConnectError) as exc:
        auth_service.mint_scoped_token(controller_pb2.Controller.MintScopedTokenRequest(prefix=prefix), None)
    assert exc.value.code is Code.INVALID_ARGUMENT


def test_mint_scoped_token_clamps_ttl_to_the_ceiling(auth_service, monkeypatch):
    now = Timestamp.from_ms(1_800_000_000_000)
    monkeypatch.setattr(Timestamp, "now", classmethod(lambda cls: now))
    request = controller_pb2.Controller.MintScopedTokenRequest(
        prefix=_PREFIX, ttl=duration_to_proto(Duration.from_seconds(400 * 86400))
    )
    with identity_scope(VerifiedIdentity(user_id=_ALICE, role="user")):
        resp = auth_service.mint_scoped_token(request, None)
    assert resp.expires_at.epoch_ms == now.epoch_ms() + MAX_SCOPED_TOKEN_TTL_SECONDS * 1000


def test_mint_scoped_token_uses_the_default_ttl(auth_service, monkeypatch):
    now = Timestamp.from_ms(1_800_000_000_000)
    monkeypatch.setattr(Timestamp, "now", classmethod(lambda cls: now))
    with identity_scope(VerifiedIdentity(user_id=_ALICE, role="user")):
        resp = auth_service.mint_scoped_token(controller_pb2.Controller.MintScopedTokenRequest(prefix=_PREFIX), None)
    assert resp.expires_at.epoch_ms == now.epoch_ms() + DEFAULT_SCOPED_TOKEN_TTL_SECONDS * 1000


# --- Verification: fail closed ------------------------------------------------


def test_scoped_token_with_malformed_prefix_claim_is_rejected(scoped_auth):
    """Fail closed: a malformed prefix claim denies the whole token."""
    signer = JwtSigner(signing_key_from_private_pem(generate_ed25519_keypair().private_pem), issuer="test-cluster")
    token = signer.mint(
        {"sub": _ALICE, "role": "user", "jti": "k", "prefix": "garbage"}, audience="iris", ttl_seconds=60
    )
    with pytest.raises(ValueError):
        scoped_auth.jwt_manager.verify(token)


# --- Native identity stamp ------------------------------------------------------

_STAMP_HEADER = "x-iris-verified-identity"


def _stamp_request(payload: dict):
    return AuthRequest(token=None, headers={_STAMP_HEADER: quote(json.dumps(payload))})


def test_native_stamp_carries_the_prefix_claim():
    """The native listener's verified-identity stamp keeps a scoped token scoped."""
    outcome = NativeProxyIdentityAuthenticator().authenticate(
        _stamp_request({"user_id": _ALICE, "role": "user", "prefix": _PREFIX})
    )
    assert outcome.decision == AuthDecision.AUTHENTICATED
    assert outcome.identity.prefix == _PREFIX


def test_native_stamp_with_malformed_prefix_is_rejected():
    outcome = NativeProxyIdentityAuthenticator().authenticate(
        _stamp_request({"user_id": _ALICE, "role": "user", "prefix": "garbage"})
    )
    assert outcome.decision == AuthDecision.REJECTED


def test_native_stamp_without_a_prefix_stays_unscoped():
    outcome = NativeProxyIdentityAuthenticator().authenticate(_stamp_request({"user_id": _ALICE, "role": "user"}))
    assert outcome.decision == AuthDecision.AUTHENTICATED
    assert outcome.identity.prefix is None


# --- Prefix gate ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("job_name", "allowed"),
    [
        pytest.param("/alice/exp2-train", True, id="inside"),
        pytest.param("/alice/exp2-train/child", True, id="inside-child"),
        pytest.param("/alice/exp2-train/0", True, id="inside-task"),
        pytest.param("/alice/other-train", False, id="outside"),
        pytest.param("/alice/exp2x-train", False, id="sibling-extension"),
        pytest.param("/bobby/exp2-train", False, id="other-user"),
    ],
)
def test_authorize_job_prefix_matches_the_token_scope(job_name, allowed):
    with identity_scope(VerifiedIdentity(user_id=_ALICE, role="user", prefix=_PREFIX)):
        if allowed:
            authorize_job_prefix(job_name)  # must not raise
            return
        with pytest.raises(ConnectError) as exc:
            authorize_job_prefix(job_name)
    assert exc.value.code is Code.PERMISSION_DENIED


def test_authorize_job_prefix_passes_without_a_prefix_claim():
    with identity_scope(VerifiedIdentity(user_id=_ALICE, role="user")):
        authorize_job_prefix("/bob/anything")  # unscoped: unchanged behavior


# --- Submit / cancel gates -----------------------------------------------------


def test_scoped_token_submits_inside_its_prefix(auth_service, scoped_auth):
    with identity_scope(_scoped_identity(scoped_auth)):
        resp = _launch(auth_service, _INSIDE_JOB)
    assert resp.job_id == _INSIDE_JOB


def test_scoped_token_submit_outside_its_prefix_is_refused(auth_service, scoped_auth):
    with identity_scope(_scoped_identity(scoped_auth)), pytest.raises(ConnectError) as exc:
        _launch(auth_service, _OUTSIDE_JOB)
    assert exc.value.code is Code.PERMISSION_DENIED


def test_scoped_token_cancels_inside_its_prefix(auth_service, scoped_auth, state):
    with identity_scope(VerifiedIdentity(user_id=_ALICE, role="user")):
        _launch(auth_service, _INSIDE_JOB)

    with identity_scope(_scoped_identity(scoped_auth)):
        _terminate(auth_service, _INSIDE_JOB)
    assert _job_state(state, _INSIDE_JOB) == job_pb2.JOB_STATE_KILLED


def test_scoped_token_cancel_outside_its_prefix_is_refused(auth_service, scoped_auth):
    with identity_scope(VerifiedIdentity(user_id=_ALICE, role="user")):
        _launch(auth_service, _OUTSIDE_JOB)

    with identity_scope(_scoped_identity(scoped_auth)), pytest.raises(ConnectError) as exc:
        _terminate(auth_service, _OUTSIDE_JOB)
    assert exc.value.code is Code.PERMISSION_DENIED


# --- Task-token inheritance ----------------------------------------------------


def test_task_token_inherits_the_submitters_prefix(auth_service, scoped_auth, state):
    """A job submitted with a scoped token mints task tokens with the same prefix."""
    with identity_scope(_scoped_identity(scoped_auth)):
        _launch(auth_service, _INSIDE_JOB)

    job_id = JobName.from_wire(_INSIDE_JOB)
    with state._db.read_snapshot() as tx:
        template = state._db.caches[RunTemplatesProjection].get(tx, job_id)
    assert template.submitting_prefix == _PREFIX

    stamped = with_task_token(template, job_id, scoped_auth)
    identity = scoped_auth.jwt_manager.verify(stamped.task_token)
    assert (identity.role, identity.user_id, identity.prefix) == ("task", _ALICE, _PREFIX)


def test_task_token_of_an_unscoped_submission_has_no_prefix(auth_service, state):
    auth = _enforcing_auth()
    with identity_scope(VerifiedIdentity(user_id=_ALICE, role="user")):
        _launch(auth_service, _OUTSIDE_JOB)

    job_id = JobName.from_wire(_OUTSIDE_JOB)
    with state._db.read_snapshot() as tx:
        template = state._db.caches[RunTemplatesProjection].get(tx, job_id)
    assert template.submitting_prefix == ""

    stamped = with_task_token(template, job_id, auth)
    assert auth.jwt_manager.verify(stamped.task_token).prefix is None


def test_task_token_refuses_a_child_outside_the_prefix(auth_service, scoped_auth):
    """A scoped job's tasks cannot act outside the subtree the token named."""
    with identity_scope(VerifiedIdentity(user_id=_ALICE, role="user")):
        _launch(auth_service, _INSIDE_JOB)

    task_identity = VerifiedIdentity(user_id=_ALICE, role="task", job_id=_INSIDE_JOB, prefix=_PREFIX)
    with identity_scope(task_identity):
        child = _launch(auth_service, f"{_INSIDE_JOB}/worker")
        assert child.job_id == f"{_INSIDE_JOB}/worker"
        with pytest.raises(ConnectError) as exc:
            _launch(auth_service, "/alice/elsewhere/child")
    assert exc.value.code is Code.PERMISSION_DENIED


def test_task_token_without_prefix_keeps_whole_user_scope(auth_service):
    with identity_scope(VerifiedIdentity(user_id=_ALICE, role="user")):
        _launch(auth_service, _INSIDE_JOB)

    task_identity = VerifiedIdentity(user_id=_ALICE, role="task", job_id=_INSIDE_JOB)
    with identity_scope(task_identity):
        resp = _launch(auth_service, "/alice/unrelated-job")
    assert resp.job_id == "/alice/unrelated-job"


# --- Unscoped and admin behavior unchanged --------------------------------------


def test_unscoped_identity_keeps_whole_user_scope(auth_service):
    with identity_scope(VerifiedIdentity(user_id=_ALICE, role="user")):
        first = _launch(auth_service, "/alice/one")
        second = _launch(auth_service, "/alice/two")
    assert (first.job_id, second.job_id) == ("/alice/one", "/alice/two")

    with identity_scope(VerifiedIdentity(user_id=_ALICE, role="user")):
        _terminate(auth_service, "/alice/two")


def test_admin_identity_is_not_prefix_scoped(auth_service, state):
    with identity_scope(VerifiedIdentity(user_id="operator", role="admin")):
        resp = _launch(auth_service, "/bob/job")
    assert resp.job_id == "/bob/job"

    with identity_scope(VerifiedIdentity(user_id="operator", role="admin")):
        _terminate(auth_service, "/bob/job")
    assert _job_state(state, "/bob/job") == job_pb2.JOB_STATE_KILLED


# --- Endpoint mint gate ----------------------------------------------------------


def _register_endpoint(endpoint_service, name: str, task_id: JobName, state):
    endpoint_service.register_endpoint(
        controller_pb2.Controller.RegisterEndpointRequest(
            name=name,
            address="up:8000",
            task_id=task_id.to_wire(),
            attempt_id=query_task(state, task_id).current_attempt_id,
        ),
        None,
    )


def test_scoped_token_mints_endpoint_token_inside_its_prefix(auth_service, scoped_auth, endpoint_service, state):
    task = submit_job(state, _INSIDE_JOB, make_job_request(_INSIDE_JOB))[0].task_id
    _register_endpoint(endpoint_service, "/serve/foo", task, state)

    with identity_scope(_scoped_identity(scoped_auth)):
        resp = auth_service.mint_endpoint_token(
            controller_pb2.Controller.MintEndpointTokenRequest(endpoint_name="/serve/foo"), None
        )
    assert scoped_auth.jwt_manager.verify(resp.token).audience == "/serve/foo"


def test_scoped_token_endpoint_mint_outside_its_prefix_is_refused(auth_service, scoped_auth, endpoint_service, state):
    task = submit_job(state, _OUTSIDE_JOB, make_job_request(_OUTSIDE_JOB))[0].task_id
    _register_endpoint(endpoint_service, "/serve/bar", task, state)

    with identity_scope(_scoped_identity(scoped_auth)), pytest.raises(ConnectError) as exc:
        auth_service.mint_endpoint_token(
            controller_pb2.Controller.MintEndpointTokenRequest(endpoint_name="/serve/bar"), None
        )
    assert exc.value.code is Code.PERMISSION_DENIED


# --- Kick gate -------------------------------------------------------------------


def _assign_and_run(state, task_id: JobName, worker_id: WorkerId):
    """Drive a PENDING task to RUNNING on ``worker_id`` (attempt 0)."""
    with state._db.transaction() as cur:
        ops.task.assign(cur, [Assignment(task_id=task_id, worker_id=worker_id)], health=state._health)
    with state._db.transaction() as cur:
        apply_task_observations(
            cur,
            [
                WorkerTaskUpdates(
                    worker_id=worker_id,
                    updates=[
                        TaskUpdate(task_id=task_id, attempt_id=0, new_state=job_pb2.TASK_STATE_RUNNING),
                    ],
                )
            ],
            health=state._health,
            now=Timestamp.now(),
        )


def test_scoped_token_kicks_inside_its_prefix_and_is_refused_outside(auth_service, scoped_auth, state):
    inside = submit_job(state, _INSIDE_JOB, make_job_request(_INSIDE_JOB))[0].task_id
    outside = submit_job(state, _OUTSIDE_JOB, make_job_request(_OUTSIDE_JOB))[0].task_id
    worker = register_worker(state, "w0", "10.0.0.1", make_worker_metadata())
    _assign_and_run(state, inside, worker)
    _assign_and_run(state, outside, worker)

    with identity_scope(_scoped_identity(scoped_auth)):
        resp = _kick(auth_service, [inside.to_wire()])
        assert resp.results[0].queued

        with pytest.raises(ConnectError) as exc:
            _kick(auth_service, [outside.to_wire()])
    assert exc.value.code is Code.PERMISSION_DENIED
