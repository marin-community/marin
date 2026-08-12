# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from datetime import UTC, datetime, timedelta

import pytest
from iac.gcp.cloud_run_rollback import (
    CloudRunHealthVerifier,
    CloudRunRestApi,
    CloudRunRevisionBackend,
    CloudRunRevisionSnapshot,
    CloudRunServiceSnapshot,
)
from iac.rollback import Release, RollbackError


class FakeCloudRunApi:
    def __init__(
        self,
        service: CloudRunServiceSnapshot,
        revisions: tuple[CloudRunRevisionSnapshot, ...],
    ):
        self.current_service = service
        self._revisions = revisions
        self.traffic_updates: list[tuple[str, str]] = []

    def service(self) -> CloudRunServiceSnapshot:
        return self.current_service

    def revisions(self) -> tuple[CloudRunRevisionSnapshot, ...]:
        return self._revisions

    def set_traffic(self, revision: str, *, etag: str) -> None:
        self.traffic_updates.append((revision, etag))
        self.current_service = CloudRunServiceSnapshot(
            name=self.current_service.name,
            uri=self.current_service.uri,
            etag=f"{etag}-next",
            reconciling=False,
            active_revision=revision,
        )


def _revision(name: str, age: int, *, ready: bool = True) -> CloudRunRevisionSnapshot:
    return CloudRunRevisionSnapshot(
        name=name,
        created_at=datetime(2026, 8, 12, tzinfo=UTC) - timedelta(minutes=age),
        ready=ready,
        image=f"us-central1-docker.pkg.dev/project/service/image@sha256:{age:064x}",
    )


def test_cloud_run_backend_uses_actual_traffic_and_etag() -> None:
    service = CloudRunServiceSnapshot(
        name="projects/project/locations/us-central1/services/service",
        uri="https://service.run.app",
        etag="etag-1",
        reconciling=False,
        active_revision="service-00002",
    )
    revisions = (_revision("service-00003", 0), _revision("service-00002", 1), _revision("service-00001", 2))
    api = FakeCloudRunApi(service, revisions)
    backend = CloudRunRevisionBackend(api, poll_interval=0, activation_timeout=1)
    history = backend.history()
    target = history.releases[-1]

    backend.begin_activation(target, expected_current=history.current.name, expected_version=history.version)
    backend.wait_active(target)

    assert history.current.name == "service-00002"
    assert api.traffic_updates == [("service-00001", "etag-1")]


def test_cloud_run_recovery_overrides_pending_traffic_change() -> None:
    service = CloudRunServiceSnapshot(
        name="projects/project/locations/us-central1/services/service",
        uri="https://service.run.app",
        etag="etag-pending",
        reconciling=True,
        active_revision="service-00002",
    )
    api = FakeCloudRunApi(service, (_revision("service-00002", 1), _revision("service-00001", 2)))
    backend = CloudRunRevisionBackend(api, poll_interval=0, activation_timeout=1)

    backend.recover(Release("service-00002", datetime(2026, 8, 12, tzinfo=UTC), True))

    assert api.traffic_updates == [("service-00002", "etag-pending")]


def test_cloud_run_backend_rejects_stale_service_version() -> None:
    service = CloudRunServiceSnapshot(
        name="projects/project/locations/us-central1/services/service",
        uri="https://service.run.app",
        etag="etag-new",
        reconciling=False,
        active_revision="service-00002",
    )
    api = FakeCloudRunApi(service, (_revision("service-00002", 1), _revision("service-00001", 2)))
    backend = CloudRunRevisionBackend(api, poll_interval=0, activation_timeout=1)

    with pytest.raises(RollbackError, match="configuration changed"):
        backend.begin_activation(
            Release("service-00001", datetime(2026, 8, 12, tzinfo=UTC), True),
            expected_current="service-00002",
            expected_version="etag-old",
        )

    assert api.traffic_updates == []


class FakeTokenProvider:
    def get_token(self) -> str:
        return "iap-token"


class FakeResponse:
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text

    def json(self):
        raise AssertionError("health responses are not decoded as JSON")

    def raise_for_status(self) -> None:
        raise AssertionError("health responses do not use raise_for_status")


class FakeHealthSession:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = responses
        self.requests: list[tuple[str, dict[str, str]]] = []

    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> FakeResponse:
        self.requests.append((url, headers))
        return self.responses.pop(0)


def test_cloud_run_health_verifier_retries_through_iap() -> None:
    session = FakeHealthSession([FakeResponse(503, "starting"), FakeResponse(200, "ok")])
    verifier = CloudRunHealthVerifier(
        "https://service.run.app/healthz",
        token_provider=FakeTokenProvider(),
        session=session,
        timeout=1,
        poll_interval=0,
    )

    verifier.verify(Release("service-00001", datetime(2026, 8, 12, tzinfo=UTC), True))

    assert session.requests == [
        ("https://service.run.app/healthz", {"Proxy-Authorization": "Bearer iap-token"}),
        ("https://service.run.app/healthz", {"Proxy-Authorization": "Bearer iap-token"}),
    ]


class FakeJsonResponse:
    status_code = 200
    text = ""

    def __init__(self, body: dict):
        self.body = body

    def json(self) -> dict:
        return self.body

    def raise_for_status(self) -> None:
        return


class FakeJsonSession:
    def __init__(self):
        self.patch_request: tuple[str, dict] | None = None

    def get(self, url: str, **kwargs) -> FakeJsonResponse:
        if url.endswith("/revisions"):
            return FakeJsonResponse(
                {
                    "revisions": [
                        {
                            "name": "projects/project/locations/us-central1/services/service/revisions/service-00002",
                            "createTime": "2026-08-12T20:00:00Z",
                            "conditions": [{"type": "Ready", "state": "CONDITION_SUCCEEDED"}],
                            "containers": [{"image": "image@sha256:abc"}],
                            "labels": {"commit-sha": "0123456"},
                        }
                    ]
                }
            )
        return FakeJsonResponse(
            {
                "name": "projects/project/locations/us-central1/services/service",
                "uri": "https://service.run.app",
                "etag": "etag-1",
                "reconciling": False,
                "trafficStatuses": [
                    {
                        "type": "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST",
                        "percent": 100,
                    }
                ],
                "latestReadyRevision": (
                    "projects/project/locations/us-central1/services/service/revisions/service-00002"
                ),
                "terminalCondition": {"state": "CONDITION_SUCCEEDED"},
            }
        )

    def patch(self, url: str, **kwargs) -> FakeJsonResponse:
        self.patch_request = (url, kwargs)
        return FakeJsonResponse({"name": "operations/traffic-update"})


def test_cloud_run_rest_api_maps_revision_history_and_traffic_patch() -> None:
    session = FakeJsonSession()
    api = CloudRunRestApi(session, project="project", region="us-central1", service="service")

    service = api.service()
    revisions = api.revisions()
    api.set_traffic("service-00002", etag=service.etag)

    assert service.active_revision == "service-00002"
    assert revisions == (
        CloudRunRevisionSnapshot(
            name="service-00002",
            created_at=datetime(2026, 8, 12, 20, tzinfo=UTC),
            ready=True,
            image="image@sha256:abc",
            source_revision="0123456",
        ),
    )
    assert session.patch_request is not None
    _, request = session.patch_request
    assert request["params"] == {"updateMask": "traffic"}
    assert request["json"] == {
        "name": "projects/project/locations/us-central1/services/service",
        "etag": "etag-1",
        "traffic": [
            {
                "type": "TRAFFIC_TARGET_ALLOCATION_TYPE_REVISION",
                "revision": "service-00002",
                "percent": 100,
            }
        ],
    }
