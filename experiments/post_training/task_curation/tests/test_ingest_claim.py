# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Queue CAS and issue pickup survive retries without losing unrelated work."""

import fcntl
import json
from functools import partial
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
import requests
from requests.adapters import BaseAdapter
from taskcompendium.identity import canonical_sha256

from experiments.post_training.task_curation.backend_census_manifest import seal_manifest
from experiments.post_training.task_curation.ingest_stages.graph import IngestExecution, StageJob, execute_stage
from experiments.post_training.task_curation.ingest_stages.ingest_config import load_ingest_config
from experiments.post_training.task_curation.ingest_stages.queue import claim_queue, select_queue
from experiments.post_training.task_curation.ingest_stages.receipts import IngestStage, StageInputs, load_receipt
from experiments.post_training.task_curation.ingest_stages.requester import Pickup, PickupAction, Requester, claim_stage
from experiments.post_training.task_curation.release_stages.manifest import read_ref

CONFIGS = Path(__file__).parents[1] / "ingest_stages/configs"
ISSUE = "https://github.com/example/project/issues/17"


class IssueTransport(BaseAdapter):
    """A persisted HTTP boundary that can lose the response after creating a comment."""

    def __init__(self):
        super().__init__()
        self.comments: list[dict] = []
        self.state = "open"
        self.lose_creation_response = False

    def send(self, request, **kwargs):
        url = urlsplit(request.url)
        assert url.hostname == "api.github.com"
        assert not kwargs["stream"]
        response = requests.Response()
        response.request = request
        response.status_code = 200
        if request.method == "POST":
            body = json.loads(request.body)["body"]
            comment = {"body": body, "html_url": f"{ISSUE}#issuecomment-{len(self.comments) + 1}"}
            self.comments.append(comment)
            if self.lose_creation_response:
                self.lose_creation_response = False
                raise requests.ConnectionError("response lost after durable creation")
            response.status_code = 201
            payload = comment
        elif url.path.endswith("/comments"):
            query = parse_qs(url.query)
            page, size = int(query["page"][0]), int(query["per_page"][0])
            payload = self.comments[(page - 1) * size : page * size]
        else:
            payload = {"state": self.state}
        response._content = json.dumps(payload).encode()
        return response

    def close(self):
        pass


@pytest.fixture
def issue_client():
    transport = IssueTransport()
    with requests.Session() as session:
        session.mount("https://api.github.com/", transport)
        yield Requester(session), transport


def queue_row(policy, label=None, issue=None):
    return (
        f"| {label or policy.source_id} | https://huggingface.co/datasets/{policy.upstream.repository} "
        f"| Pending{f'; Request: {issue}.' if issue else ''} | P1 |\n"
    )


def claimed_job(tmp_path, issue=ISSUE):
    policy = load_ingest_config(CONFIGS / "litmus-train.json")
    path = tmp_path / "CANDIDATES.md"
    path.write_text(queue_row(policy, issue=issue))
    claim = claim_queue(path, policy, select_queue(path.read_bytes(), policy))
    reference = seal_manifest(claim.model_dump(mode="json"), str(tmp_path / "queue-claim.json"))
    execution = IngestExecution(
        policy=policy,
        source_revision="a" * 40,
        artifact_version="2026.11.28",
        artifact_root=str(tmp_path / "artifacts"),
        worker_argv=("python", "-m", "worker"),
        initial_artifacts={"queue-claim": reference},
    )
    inputs = StageInputs(
        stage=IngestStage.CLAIM,
        source_id=policy.source_id,
        config_sha256=policy.digest(),
        source_revision=execution.source_revision,
        worker_argv_sha256=canonical_sha256({"argv": list(execution.worker_argv)}),
        dependencies={},
        artifacts=execution.initial_artifacts,
    )
    return StageJob(execution=execution, inputs=inputs, output_root=str(tmp_path / "claim"))


def test_legacy_repackager_claim_selects_one_folder_and_preserves_concurrent_append(tmp_path):
    policy = load_ingest_config(CONFIGS / "mimo-repair.json")
    path = tmp_path / "CANDIDATES.md"
    first = queue_row(policy, label="FWeindel FineEnvs v2+v3")
    selected = queue_row(policy, label="FWeindel MiMo code v1")
    path.write_text(first + selected + queue_row(policy, label="FWeindel Endless v2+v3"))
    selection = select_queue(path.read_bytes(), policy)
    concurrent = "| Other:new | elsewhere | Pending | P2 |\n"
    with path.open("a") as handle:
        handle.write(concurrent)

    claim = claim_queue(path, policy, selection)
    retry = claim_queue(path, policy, selection)

    assert retry == claim
    rows = path.read_text().splitlines()
    assert rows[0] == first.rstrip()
    assert rows[-1] == concurrent.rstrip()
    assert rows[1].count("atlas-ingest:claim:") == 1
    assert "mimo" in claim.selection.source_id
    assert claim.selection.row == selected.rstrip()


def test_claim_refuses_a_changed_selected_row_without_overwriting_it(tmp_path):
    policy = load_ingest_config(CONFIGS / "litmus-train.json")
    path = tmp_path / "CANDIDATES.md"
    path.write_text(queue_row(policy))
    selection = select_queue(path.read_bytes(), policy)
    changed = path.read_text().replace("Pending", "Excluded: license doubt")
    path.write_text(changed)

    with pytest.raises(ValueError, match="selected queue row changed"):
        claim_queue(path, policy, selection)

    assert path.read_text() == changed


def test_claim_refuses_the_shared_queue_lock_without_writing(tmp_path):
    policy = load_ingest_config(CONFIGS / "litmus-train.json")
    path = tmp_path / "CANDIDATES.md"
    original = queue_row(policy)
    path.write_text(original)
    selection = select_queue(path.read_bytes(), policy)
    with (tmp_path / ".candidates.lock").open("a") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            claim_queue(path, policy, selection)

    assert path.read_text() == original


def test_ambiguous_repackager_rows_refuse_a_claim(tmp_path):
    policy = load_ingest_config(CONFIGS / "mimo-repair.json")
    path = tmp_path / "CANDIDATES.md"
    original = queue_row(policy, label="FWeindel MiMo v1 code") + queue_row(policy, label="FWeindel MiMo v1 other")
    path.write_text(original)

    with pytest.raises(ValueError, match="exactly one queue row"):
        select_queue(path.read_bytes(), policy)

    assert path.read_text() == original


def test_claim_stage_recovers_a_lost_pickup_response_and_seals_one_comment(tmp_path, issue_client):
    requester, transport = issue_client
    job = claimed_job(tmp_path)
    transport.comments = [{"body": "earlier request discussion", "html_url": ISSUE} for _ in range(100)]
    transport.lose_creation_response = True

    with pytest.raises(requests.ConnectionError):
        execute_stage(partial(claim_stage, requester=requester), job)
    assert not (Path(job.output_root) / "pickup.json").exists()
    # Closure after the successful creation must still reuse that pickup.
    transport.state = "closed"

    artifact = execute_stage(partial(claim_stage, requester=requester), job)
    receipt = load_receipt(artifact.receipt, job.inputs)
    pickup = Pickup.model_validate(read_ref(receipt.outputs["pickup"]))

    assert len(transport.comments) == 101
    assert transport.comments[-1]["body"].startswith("🤖")
    assert pickup.action is PickupAction.COMMENT
    assert pickup.comment == transport.comments[-1]["html_url"]


@pytest.mark.parametrize("issue", [ISSUE, None])
def test_closed_or_absent_requester_seals_without_posting(tmp_path, issue_client, issue):
    requester, transport = issue_client
    transport.state = "closed"
    job = claimed_job(tmp_path, issue)

    artifact = execute_stage(partial(claim_stage, requester=requester), job)
    pickup = Pickup.model_validate(read_ref(load_receipt(artifact.receipt, job.inputs).outputs["pickup"]))

    assert transport.comments == []
    assert pickup.action is (PickupAction.CLOSED if issue else PickupAction.NO_ISSUE)


def test_duplicate_pickup_markers_refuse_to_seal_or_create_another_comment(tmp_path, issue_client):
    requester, transport = issue_client
    job = claimed_job(tmp_path)
    marker = f"<!-- atlas-ingest:pickup:{job.execution.policy.digest()} -->"
    transport.comments = [{"body": marker, "html_url": ISSUE}] * 2

    with pytest.raises(ValueError, match="duplicate pickup markers"):
        execute_stage(partial(claim_stage, requester=requester), job)

    assert len(transport.comments) == 2
    assert not (Path(job.output_root) / "pickup.json").exists()
