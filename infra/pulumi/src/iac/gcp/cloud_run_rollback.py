# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Roll a Cloud Run service back using its retained immutable revisions."""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, cast
from urllib.parse import quote

import click
import google.auth
import requests
from google.auth import exceptions as google_auth_exceptions
from google.auth.transport.requests import AuthorizedSession
from rigging.auth import (
    MARIN_DESKTOP_OAUTH_CLIENT,
    IapCredentialsUnavailable,
    IapLoginRequired,
    IapServiceAccountTokenProvider,
    TokenProvider,
)
from rigging.credentials import iap_edge_provider

from iac.rollback import (
    Release,
    ReleaseHistory,
    RollbackError,
    RollbackPlan,
    execute_rollback,
    rollback_plan,
)

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
CLOUD_RUN_API = "https://run.googleapis.com/v2"
DEFAULT_ACTIVATION_TIMEOUT = 300.0
DEFAULT_HEALTH_TIMEOUT = 60.0
DEFAULT_POLL_INTERVAL = 2.0
HTTP_REQUEST_TIMEOUT = 30.0
REVISION_TRAFFIC_TYPE = "TRAFFIC_TARGET_ALLOCATION_TYPE_REVISION"
READY_CONDITION = "Ready"
SUCCEEDED_CONDITION_STATE = "CONDITION_SUCCEEDED"


class HttpResponse(Protocol):
    status_code: int
    text: str

    def json(self) -> Any:
        """Decode the response body."""

    def raise_for_status(self) -> None:
        """Raise for an unsuccessful response."""


class JsonHttpSession(Protocol):
    def get(self, url: str, **kwargs) -> HttpResponse:
        """Issue an HTTP GET."""

    def patch(self, url: str, **kwargs) -> HttpResponse:
        """Issue an HTTP PATCH."""


class HealthHttpSession(Protocol):
    def get(self, url: str, *, headers: dict[str, str], timeout: float) -> HttpResponse:
        """Issue an application health request."""


@dataclass(frozen=True)
class CloudRunServiceSnapshot:
    """Cloud Run fields required to coordinate a traffic update."""

    name: str
    uri: str
    etag: str
    reconciling: bool
    active_revision: str | None
    terminal_error: str | None = None


@dataclass(frozen=True)
class CloudRunRevisionSnapshot:
    """Immutable Cloud Run revision metadata used for release selection."""

    name: str
    created_at: datetime
    ready: bool
    image: str | None = None
    source_revision: str | None = None

    def release(self) -> Release:
        return Release(
            name=self.name,
            created_at=self.created_at,
            platform_ready=self.ready,
            artifact=self.image,
            source_revision=self.source_revision,
        )


class CloudRunApi(Protocol):
    """The Cloud Run API operations required by the rollback backend."""

    def service(self) -> CloudRunServiceSnapshot:
        """Return current service traffic and reconciliation state."""

    def revisions(self) -> tuple[CloudRunRevisionSnapshot, ...]:
        """Return retained revisions."""

    def set_traffic(self, revision: str, *, etag: str) -> None:
        """Conditionally move all traffic to one revision."""


def _short_name(resource_name: str) -> str:
    return resource_name.rsplit("/", maxsplit=1)[-1]


def _active_revision(traffic_statuses: object, latest_ready_revision: object) -> str | None:
    if not isinstance(traffic_statuses, list):
        return None
    active = [target for target in traffic_statuses if isinstance(target, dict) and target.get("percent", 0) > 0]
    if len(active) != 1 or active[0].get("percent") != 100:
        return None
    revision = active[0].get("revision")
    if isinstance(revision, str) and revision:
        return _short_name(revision)
    if active[0].get("type") != "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST":
        return None
    if not isinstance(latest_ready_revision, str) or not latest_ready_revision:
        return None
    return _short_name(latest_ready_revision)


def _terminal_error(condition: object) -> str | None:
    if not isinstance(condition, dict) or condition.get("state") != "CONDITION_FAILED":
        return None
    reason = condition.get("reason")
    message = condition.get("message")
    return ": ".join(str(value) for value in (reason, message) if value) or "Cloud Run reconciliation failed"


def _ready(conditions: object) -> bool:
    if not isinstance(conditions, list):
        return False
    return any(
        isinstance(condition, dict)
        and condition.get("type") == READY_CONDITION
        and condition.get("state") == SUCCEEDED_CONDITION_STATE
        for condition in conditions
    )


def _parse_time(value: object) -> datetime:
    if not isinstance(value, str):
        raise RollbackError("Cloud Run revision has no createTime")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _source_revision(labels: object) -> str | None:
    if not isinstance(labels, dict):
        return None
    for key in ("marin-source-revision", "commit-sha"):
        value = labels.get(key)
        if isinstance(value, str) and value:
            return value
    return None


class CloudRunRestApi:
    """Minimal authenticated Cloud Run v2 REST client."""

    def __init__(
        self,
        session: JsonHttpSession,
        *,
        project: str,
        region: str,
        service: str,
    ):
        self._session = session
        self._service_name = (
            f"projects/{quote(project, safe='')}/locations/{quote(region, safe='')}/"
            f"services/{quote(service, safe='')}"
        )
        self._service_url = f"{CLOUD_RUN_API}/{self._service_name}"

    @classmethod
    def from_adc(cls, *, project: str, region: str, service: str) -> "CloudRunRestApi":
        try:
            credentials, _ = google.auth.default(scopes=[CLOUD_PLATFORM_SCOPE])
        except google_auth_exceptions.DefaultCredentialsError as exc:
            raise RollbackError(
                "Cloud Run API credentials are unavailable; run `gcloud auth application-default login`"
            ) from exc
        return cls(AuthorizedSession(credentials), project=project, region=region, service=service)

    def _json(self, response: HttpResponse) -> Mapping[str, Any]:
        try:
            response.raise_for_status()
            body = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise RollbackError(f"Cloud Run API request failed: {exc}") from exc
        if not isinstance(body, dict):
            raise RollbackError("Cloud Run API returned a non-object response")
        return cast(Mapping[str, Any], body)

    def _get(self, url: str, **kwargs) -> Mapping[str, Any]:
        try:
            response = self._session.get(url, **kwargs)
        except requests.RequestException as exc:
            raise RollbackError(f"Cloud Run API request failed: {exc}") from exc
        return self._json(response)

    def _patch(self, url: str, **kwargs) -> Mapping[str, Any]:
        try:
            response = self._session.patch(url, **kwargs)
        except requests.RequestException as exc:
            raise RollbackError(f"Cloud Run API request failed: {exc}") from exc
        return self._json(response)

    def service(self) -> CloudRunServiceSnapshot:
        body = self._get(self._service_url, timeout=HTTP_REQUEST_TIMEOUT)
        name = body.get("name")
        uri = body.get("uri")
        etag = body.get("etag")
        if not all(isinstance(value, str) and value for value in (name, uri, etag)):
            raise RollbackError("Cloud Run service response is missing name, uri, or etag")
        return CloudRunServiceSnapshot(
            name=cast(str, name),
            uri=cast(str, uri),
            etag=cast(str, etag),
            reconciling=bool(body.get("reconciling", False)),
            active_revision=_active_revision(body.get("trafficStatuses"), body.get("latestReadyRevision")),
            terminal_error=_terminal_error(body.get("terminalCondition")),
        )

    def revisions(self) -> tuple[CloudRunRevisionSnapshot, ...]:
        revisions: list[CloudRunRevisionSnapshot] = []
        page_token: str | None = None
        url = f"{self._service_url}/revisions"
        while True:
            params = {"pageSize": 100}
            if page_token is not None:
                params["pageToken"] = page_token
            body = self._get(url, params=params, timeout=HTTP_REQUEST_TIMEOUT)
            for raw in body.get("revisions", []):
                if not isinstance(raw, dict):
                    raise RollbackError("Cloud Run revisions response contains a non-object revision")
                name = raw.get("name")
                if not isinstance(name, str) or not name:
                    raise RollbackError("Cloud Run revision has no name")
                containers = raw.get("containers")
                image = None
                if isinstance(containers, list) and containers and isinstance(containers[0], dict):
                    raw_image = containers[0].get("image")
                    image = raw_image if isinstance(raw_image, str) else None
                revisions.append(
                    CloudRunRevisionSnapshot(
                        name=_short_name(name),
                        created_at=_parse_time(raw.get("createTime")),
                        ready=_ready(raw.get("conditions")),
                        image=image,
                        source_revision=_source_revision(raw.get("labels")),
                    )
                )
            raw_page_token = body.get("nextPageToken")
            page_token = raw_page_token if isinstance(raw_page_token, str) and raw_page_token else None
            if page_token is None:
                return tuple(revisions)

    def set_traffic(self, revision: str, *, etag: str) -> None:
        body = {
            "name": self._service_name,
            "etag": etag,
            "traffic": [
                {
                    "type": REVISION_TRAFFIC_TYPE,
                    "revision": revision,
                    "percent": 100,
                }
            ],
        }
        self._patch(
            self._service_url,
            params={"updateMask": "traffic"},
            json=body,
            timeout=HTTP_REQUEST_TIMEOUT,
        )


class CloudRunRevisionBackend:
    """Derive and activate releases from Cloud Run's service and revision APIs."""

    def __init__(
        self,
        api: CloudRunApi,
        *,
        activation_timeout: float = DEFAULT_ACTIVATION_TIMEOUT,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ):
        self._api = api
        self._activation_timeout = activation_timeout
        self._poll_interval = poll_interval

    def history(self) -> ReleaseHistory:
        service = self._api.service()
        if service.reconciling:
            raise RollbackError(f"Cloud Run service {_short_name(service.name)} is reconciling")
        if service.active_revision is None:
            raise RollbackError("Cloud Run service must have exactly one revision receiving 100% of traffic")
        releases = tuple(
            revision.release()
            for revision in sorted(self._api.revisions(), key=lambda item: item.created_at, reverse=True)
        )
        current = next((release for release in releases if release.name == service.active_revision), None)
        if current is None:
            raise RollbackError(f"serving revision {service.active_revision} is absent from retained history")
        return ReleaseHistory(current=current, releases=releases, version=service.etag)

    def service_uri(self) -> str:
        return self._api.service().uri

    def begin_activation(
        self,
        release: Release,
        *,
        expected_current: str,
        expected_version: str,
    ) -> None:
        service = self._api.service()
        if service.reconciling:
            raise RollbackError(f"Cloud Run service {_short_name(service.name)} began reconciling")
        if service.active_revision != expected_current:
            raise RollbackError(
                f"Cloud Run traffic changed from {expected_current} to {service.active_revision}; plan the rollback again"
            )
        if service.etag != expected_version:
            raise RollbackError("Cloud Run service configuration changed; plan the rollback again")
        self._api.set_traffic(release.name, etag=expected_version)

    def wait_active(self, release: Release) -> None:
        deadline = time.monotonic() + self._activation_timeout
        while True:
            service = self._api.service()
            if not service.reconciling:
                if service.active_revision == release.name:
                    return
                if service.terminal_error is not None:
                    raise RollbackError(service.terminal_error)
                raise RollbackError(
                    f"Cloud Run reconciliation completed on {service.active_revision}, expected {release.name}"
                )
            if time.monotonic() >= deadline:
                raise RollbackError(f"timed out waiting for Cloud Run revision {release.name}")
            time.sleep(self._poll_interval)

    def recover(self, release: Release) -> None:
        service = self._api.service()
        if not service.reconciling and service.active_revision == release.name:
            return
        self._api.set_traffic(release.name, etag=service.etag)
        self.wait_active(release)


class CloudRunHealthVerifier:
    """Poll an IAP-protected application endpoint after traffic activation."""

    def __init__(
        self,
        url: str,
        *,
        token_provider: TokenProvider,
        session: HealthHttpSession | None = None,
        timeout: float = DEFAULT_HEALTH_TIMEOUT,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
    ):
        self._url = url
        self._token_provider = token_provider
        self._session = session or requests.Session()
        self._timeout = timeout
        self._poll_interval = poll_interval

    def verify(self, release: Release) -> None:
        deadline = time.monotonic() + self._timeout
        last_result = "no response"
        while True:
            try:
                token = self._token_provider.get_token()
            except (IapCredentialsUnavailable, IapLoginRequired, google_auth_exceptions.GoogleAuthError) as exc:
                raise RollbackError(f"IAP credentials are unavailable: {exc}") from exc
            if not token:
                raise RollbackError("IAP token provider returned no token")
            try:
                response = self._session.get(
                    self._url,
                    headers={"Proxy-Authorization": f"Bearer {token}"},
                    timeout=min(HTTP_REQUEST_TIMEOUT, self._timeout),
                )
                if 200 <= response.status_code < 300:
                    return
                last_result = f"HTTP {response.status_code}: {response.text[:200]}"
            except requests.RequestException as exc:
                last_result = str(exc)
            if time.monotonic() >= deadline:
                raise RollbackError(f"revision {release.name} failed application health verification: {last_result}")
            time.sleep(self._poll_interval)


def _release_description(release: Release) -> str:
    details = [release.name, release.created_at.isoformat()]
    if release.artifact:
        details.append(release.artifact)
    if release.source_revision:
        details.append(f"source={release.source_revision}")
    return " | ".join(details)


def _iap_token_provider(login: str) -> TokenProvider:
    human = iap_edge_provider(login)
    if human is not None:
        return human
    return IapServiceAccountTokenProvider(MARIN_DESKTOP_OAUTH_CLIENT.client_id)


@click.command()
@click.option("--project", required=True, help="GCP project containing the Cloud Run service.")
@click.option("--region", required=True, help="Cloud Run service region.")
@click.option("--service", required=True, help="Cloud Run service name.")
@click.option("--health-path", required=True, help="Application health path, beginning with '/'.")
@click.option("--to", "target", help="Exact retained revision; defaults to the previous ready revision.")
@click.option("--iap-login", default="marin", show_default=True, help="Rigging login used for IAP verification.")
@click.option("--activation-timeout", default=DEFAULT_ACTIVATION_TIMEOUT, show_default=True, type=float)
@click.option("--health-timeout", default=DEFAULT_HEALTH_TIMEOUT, show_default=True, type=float)
@click.option("-y", "--yes", is_flag=True, help="Skip the image-only rollback confirmation.")
def main(
    project: str,
    region: str,
    service: str,
    health_path: str,
    target: str | None,
    iap_login: str,
    activation_timeout: float,
    health_timeout: float,
    yes: bool,
) -> None:
    """Move all traffic to a previous Cloud Run revision and verify it."""
    if not health_path.startswith("/") or health_path.startswith("//"):
        raise click.UsageError("--health-path must begin with one '/' character")
    try:
        api = CloudRunRestApi.from_adc(project=project, region=region, service=service)
        backend = CloudRunRevisionBackend(api, activation_timeout=activation_timeout)
        plan: RollbackPlan = rollback_plan(backend.history(), target=target)
        health_url = f"{backend.service_uri().rstrip('/')}{health_path}"
        verifier = CloudRunHealthVerifier(
            health_url,
            token_provider=_iap_token_provider(iap_login),
            timeout=health_timeout,
        )
        click.echo(f"Current: {_release_description(plan.current)}")
        click.echo(f"Target:  {_release_description(plan.target)}")
        click.echo(f"Health:  {health_url}")
        if not yes:
            click.confirm(
                "This changes only Cloud Run traffic; database and secret versions are unchanged. Continue?",
                abort=True,
            )
        execute_rollback(backend, verifier, plan)
    except RollbackError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Cloud Run service {service} now serves {plan.target.name}.")


if __name__ == "__main__":
    main()
