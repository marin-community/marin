# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read live GCP IAM policies for the drift scan (`iam_scan`).

One authenticated session hits each API's `getIamPolicy` for exactly the resources iam_data.py
declares — the project, the KMS key, every secret, bucket, Artifact Registry repo, and service
account — plus Service Usage for the enabled-API set the orphaned-agent check needs. Every
permission required here is already in the `marinGcpResourcePreviewer` custom role the CI preview
account holds, except `serviceusage.services.list`, whose absence degrades to skipping the
orphaned-agent check rather than failing the scan.
"""

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

import google.auth
import requests
from google.auth.transport.requests import AuthorizedSession

from iac.gcp.iam import GcpRoleGrant
from iac.gcp.iam_scan import Binding, Container

logger = logging.getLogger(__name__)

_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


@dataclass(frozen=True)
class ResourceTarget:
    """A resource to read a live IAM policy for, identified the same way iam_data.py declares it
    so the resulting `Binding.resource` diffs directly against the declared set."""

    container: Container
    resource: str


def iter_resources(
    resources: Iterable[tuple[Container, str, tuple[GcpRoleGrant, ...]]],
) -> Iterator[ResourceTarget]:
    """The live-read target for each `iam_scan.declared_resources` entry. Sharing that one
    derivation is what keeps a live `Binding.resource` identical to its declared counterpart."""
    return (ResourceTarget(container, resource) for container, resource, _ in resources)


class IamReader(Protocol):
    """The live-read surface `iam_scan` consumes, so tests can inject canned policies."""

    def bindings(self, target: ResourceTarget, project: str) -> list[Binding]: ...

    def enabled_services(self, project: str) -> frozenset[str] | None: ...


def _policy_url(target: ResourceTarget, project: str) -> tuple[str, str]:
    """Return (http_method, url) for a resource's getIamPolicy. Methods differ per API: Resource
    Manager and IAM take POST, the rest GET."""
    container, resource = target.container, target.resource
    if container is Container.PROJECT:
        return "POST", f"https://cloudresourcemanager.googleapis.com/v1/projects/{resource}:getIamPolicy"
    if container is Container.KMS:
        return "GET", f"https://cloudkms.googleapis.com/v1/{resource}:getIamPolicy"
    if container is Container.SECRET:
        return "GET", f"https://secretmanager.googleapis.com/v1/projects/{project}/secrets/{resource}:getIamPolicy"
    if container is Container.BUCKET:
        return "GET", f"https://storage.googleapis.com/storage/v1/b/{resource}/iam"
    if container is Container.ARTIFACT_REPOSITORY:
        location, repository = resource.split("/", 1)
        return (
            "GET",
            f"https://artifactregistry.googleapis.com/v1/projects/{project}/locations/{location}"
            f"/repositories/{repository}:getIamPolicy",
        )
    assert container is Container.SERVICE_ACCOUNT, container
    return "POST", f"https://iam.googleapis.com/v1/projects/{project}/serviceAccounts/{resource}:getIamPolicy"


def parse_policy_bindings(target: ResourceTarget, policy: dict) -> list[Binding]:
    """Flatten a getIamPolicy response's `bindings` into one `Binding` per member. Conditional
    bindings are included unconditionally: the condition is dropped, matching how `iam_scan`
    diffs (see its `Binding` docstring)."""
    bindings: list[Binding] = []
    for entry in policy.get("bindings", ()):
        role = entry["role"]
        for member in entry.get("members", ()):
            bindings.append(Binding(target.container, target.resource, role, member))
    return bindings


class GcpIamReader:
    """Reads live IAM through Application Default Credentials. In CI those resolve to the
    `pulumi-ci` service account via Workload Identity Federation, exactly as `pulumi preview`."""

    def __init__(self, session: AuthorizedSession | None = None) -> None:
        if session is None:
            credentials, _ = google.auth.default(scopes=[_CLOUD_PLATFORM_SCOPE])
            session = AuthorizedSession(credentials)
        self._session = session

    def _get_json(self, method: str, url: str, *, params: dict | None = None) -> dict:
        response = self._session.request(method, url, params=params)
        response.raise_for_status()
        return response.json()

    def bindings(self, target: ResourceTarget, project: str) -> list[Binding]:
        """The live bindings on one resource. An HTTP failure (a deleted resource, a missing
        read grant) is logged and treated as no bindings so one bad resource can't abort the
        weekly scan; the trade-off is that drift on an unreadable resource goes unreported until
        the read succeeds. A parse error is a bug and propagates."""
        method, url = _policy_url(target, project)
        try:
            policy = self._get_json(method, url)
        except requests.RequestException as error:
            logger.warning("skipping %s %s: %s", target.container, target.resource, error)
            return []
        return parse_policy_bindings(target, policy)

    def enabled_services(self, project: str) -> frozenset[str] | None:
        url = f"https://serviceusage.googleapis.com/v1/projects/{project}/services"
        services: set[str] = set()
        page_token: str | None = None
        while True:
            params = {"filter": "state:ENABLED", "pageSize": "200"}
            if page_token:
                params["pageToken"] = page_token
            try:
                data = self._get_json("GET", url, params=params)
            except requests.RequestException as error:
                logger.warning("serviceusage unavailable, skipping orphaned-agent check: %s", error)
                return None
            for service in data.get("services", ()):
                name = service.get("config", {}).get("name")
                if name:
                    services.add(name)
            page_token = data.get("nextPageToken")
            if not page_token:
                return frozenset(services)
