# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Issue pickup with policy-bound markers, safe retry and bounded pagination."""

import re
from enum import StrEnum
from urllib.parse import urlsplit, urlunsplit

import requests
from pydantic import BaseModel, ConfigDict
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.task_curation.backend_census_manifest import seal_manifest
from experiments.post_training.task_curation.ingest_stages.graph import StageJob
from experiments.post_training.task_curation.ingest_stages.queue import QueueClaim, verify_claim
from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef, read_ref

MAX_COMMENT_PAGES = 100
COMMENT_PAGE_SIZE = 100
HTTP_TIMEOUT = 30


class PickupAction(StrEnum):
    COMMENT = "comment"
    CLOSED = "closed"
    NO_ISSUE = "no-issue"


class Pickup(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    action: PickupAction
    issue: str | None
    comment: str | None


def issue_api(issue: str) -> str:
    parts = urlsplit(issue)
    if (
        parts.scheme != "https"
        or parts.netloc != "github.com"
        or re.fullmatch(r"/[^/]+/[^/]+/issues/[0-9]+", parts.path) is None
        or parts.query
        or parts.fragment
    ):
        raise ValueError("requester must be a canonical HTTPS GitHub issue")
    return urlunsplit((parts.scheme, "api.github.com", "/repos" + parts.path, "", ""))


class Requester:
    """Use an explicitly authenticated session; never serialize its credentials."""

    def __init__(self, session: requests.Session):
        self.session = session

    def pickup(self, claim: QueueClaim) -> Pickup:
        """Reuse a marker on serial retries; callers must serialize same-policy runs."""
        selection = claim.selection
        issue = selection.requester_issue
        if issue is None:
            return Pickup(action=PickupAction.NO_ISSUE, issue=None, comment=None)
        endpoint = issue_api(issue)
        marker = f"<!-- atlas-ingest:pickup:{selection.policy_sha256} -->"
        matches = []
        for page in range(1, MAX_COMMENT_PAGES + 1):
            response = self.session.get(
                endpoint + "/comments",
                params={"per_page": COMMENT_PAGE_SIZE, "page": page},
                timeout=HTTP_TIMEOUT,
                allow_redirects=False,
            )
            response.raise_for_status()
            if response.status_code != 200:
                raise ValueError("GitHub comment listing returned an unexpected status")
            comments = response.json()
            if not isinstance(comments, list):
                raise ValueError("GitHub comment listing did not return a page")
            matches.extend(comment for comment in comments if marker in (comment["body"] or ""))
            if len(comments) < COMMENT_PAGE_SIZE:
                break
        else:
            raise ValueError("requester comment history exceeds the bounded pickup scan")
        if len(matches) > 1:
            raise ValueError("requester issue has duplicate pickup markers; reconcile before retry")
        if matches:
            return Pickup(action=PickupAction.COMMENT, issue=issue, comment=matches[0]["html_url"])
        response = self.session.get(endpoint, timeout=HTTP_TIMEOUT, allow_redirects=False)
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError("GitHub issue lookup returned an unexpected status")
        document = response.json()
        if "pull_request" in document:
            raise ValueError("the queue requester is a pull request rather than an issue")
        if document["state"] == "closed":
            return Pickup(action=PickupAction.CLOSED, issue=issue, comment=None)
        if document["state"] != "open":
            raise ValueError("GitHub returned an unknown issue state")
        response = self.session.post(
            endpoint + "/comments",
            json={
                "body": (
                    f"🤖 atlas-ingest picked up `{selection.source_id}` for the reviewed policy "
                    f"`{selection.policy_sha256}`. Admission will follow the sealed ingest receipts.\n\n{marker}"
                )
            },
            timeout=HTTP_TIMEOUT,
            allow_redirects=False,
        )
        response.raise_for_status()
        if response.status_code != 201:
            raise ValueError("GitHub pickup creation returned an unexpected status")
        comment = response.json()
        if marker not in comment["body"]:
            raise ValueError("GitHub pickup response did not preserve the policy marker")
        return Pickup(action=PickupAction.COMMENT, issue=issue, comment=comment["html_url"])


def claim_stage(job: StageJob, requester: Requester) -> dict[str, ArtifactRef]:
    """Consume the host's sealed queue claim before creating or reusing its pickup."""
    claim = QueueClaim.model_validate(read_ref(job.inputs.artifacts["queue-claim"]))
    verify_claim(claim, job.execution.policy)
    pickup = requester.pickup(claim)
    return {"pickup": seal_manifest(pickup.model_dump(mode="json"), prefix_join(job.output_root, "pickup.json"))}
