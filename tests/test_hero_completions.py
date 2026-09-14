# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import UTC, date, datetime, timedelta

import httpx
import numpy as np
import pytest
from marin.publish import sites

from experiments.grug.moe_hero_ep.ops.vibe_check.completions import (
    Checkpoint,
    Completion,
    Entry,
    JobStatus,
    Phase,
    Prompt,
    Queue,
    SampleRequest,
    SampleResult,
    SampleStore,
    SamplingSpec,
    StopReason,
    reconcile,
)
from experiments.grug.moe_hero_ep.ops.vibe_check.generation import generate
from experiments.grug.moe_hero_ep.ops.vibe_check.publishing import (
    COMMENT_MARKER,
    publish_daily,
    render_report,
    report_manifest,
    update_issue_comment,
)

NOW = datetime(2026, 9, 12, 10, tzinfo=UTC)


@pytest.fixture
def sample_request():
    return SampleRequest(
        checkpoint=Checkpoint(
            uri="s3://checkpoints/step-6000", run_id="hero", step=6000, timestamp=NOW.isoformat(), metadata_digest="abc"
        ),
        spec=SamplingSpec(
            release="test-v1",
            batch_size=1,
            prompts=(Prompt(id="add", text="def add(a, b):", seed=0, source_url="https://example.org"),),
            tokenizer="test",
            tokenizer_revision="a" * 40,
            model={},
            temperature=0,
            max_new_tokens=2,
            context_length=4,
        ),
        source_revision="b" * 40,
        target_cluster="test",
    )


def completed(request):
    return SampleResult(
        request=request,
        completions=(
            Completion(
                prompt_id="add",
                prompt_token_ids=(1,),
                token_ids=(2, 3),
                text="a + b",
                stop_reason=StopReason.MAX_NEW_TOKENS,
            ),
        ),
        completed_at=NOW.isoformat(),
        eos_token_id=0,
    )


class JobService:
    """External job service with unique names and retained terminal states."""

    def __init__(self):
        self.jobs: dict[str, JobStatus] = {}
        self.requests: dict[str, SampleRequest] = {}
        self.lose_submit_response = False
        self.unavailable = False

    def status(self, name):
        if self.unavailable:
            raise ConnectionError("job service unavailable")
        return self.jobs.get(name, JobStatus.MISSING)

    def submit(self, entry):
        if entry.job_name in self.jobs:
            return
        self.jobs[entry.job_name] = JobStatus.RUNNING
        self.requests[entry.job_name] = entry.request
        if self.lose_submit_response:
            raise ConnectionError("response lost after submission")


def test_queue_recovers_lost_submission_and_waits_for_teardown(tmp_path, sample_request):
    request = sample_request
    store = SampleStore(str(tmp_path))
    jobs = JobService()
    jobs.lose_submit_response = True
    with pytest.raises(ConnectionError):
        reconcile(store, jobs, [request], NOW)
    # A fresh invocation has a different main revision, but must keep the submitted request.
    changed_main = request.model_copy(update={"source_revision": "c" * 40})
    older = request.model_copy(
        update={"checkpoint": request.checkpoint.model_copy(update={"step": 3000, "uri": "s3://checkpoints/step-3000"})}
    )
    jobs.lose_submit_response = False
    store.save_result(completed(request))
    queue = reconcile(store, jobs, [changed_main, older], NOW + timedelta(hours=1))
    assert queue.entries[request.sample_id].phase == Phase.ACTIVE
    assert len(jobs.jobs) == 1
    name = next(iter(jobs.jobs))
    assert jobs.requests[name].source_revision == request.source_revision
    jobs.jobs[name] = JobStatus.SUCCEEDED
    queue = reconcile(store, jobs, [changed_main, older], NOW + timedelta(hours=2))
    assert queue.entries[request.sample_id].phase == Phase.COMPLETE
    assert queue.entries[older.sample_id].phase == Phase.ACTIVE
    assert len(jobs.jobs) == 2


def test_all_permanent_requests_survive_missed_ticks_and_retry_budget(tmp_path, sample_request):
    request = sample_request
    store, jobs = SampleStore(str(tmp_path)), JobService()
    requests = [
        request.model_copy(
            update={
                "checkpoint": request.checkpoint.model_copy(update={"step": step, "uri": f"s3://checkpoints/{step}"})
            }
        )
        for step in [6000, 12000, 18000]
    ]
    queue = reconcile(store, jobs, requests, NOW)
    assert len(queue.entries) == 3
    assert queue.entries[requests[-1].sample_id].phase == Phase.ACTIVE
    for attempt in range(3):
        active = next(entry for entry in queue.entries.values() if entry.phase == Phase.ACTIVE)
        jobs.jobs[active.job_name] = JobStatus.FAILED
        queue = reconcile(store, jobs, [], NOW + timedelta(days=attempt + 1))
    assert queue.entries[requests[-1].sample_id].phase == Phase.FAILED
    assert queue.entries[requests[-1].sample_id].attempt == 3
    assert queue.entries[requests[1].sample_id].phase == Phase.ACTIVE
    assert queue.entries[requests[0].sample_id].phase == Phase.QUEUED


def test_service_error_does_not_start_another_allocation(tmp_path, sample_request):
    request = sample_request
    store, jobs = SampleStore(str(tmp_path)), JobService()
    reconcile(store, jobs, [request], NOW)
    jobs.unavailable = True
    with pytest.raises(ConnectionError):
        reconcile(store, jobs, [request], NOW + timedelta(days=3))
    assert len(jobs.jobs) == 1
    queue, _ = store.read_queue()
    assert queue.entries[request.sample_id].attempt == 1


def test_capacity_waits_do_not_use_the_sampling_failure_budget(tmp_path, sample_request):
    store, jobs = SampleStore(str(tmp_path)), JobService()
    queue = reconcile(store, jobs, [sample_request], NOW)
    for hour in [0, 6, 12, 18]:
        entry = queue.entries[sample_request.sample_id]
        jobs.jobs[entry.job_name] = JobStatus.DEFERRED
        queue = reconcile(store, jobs, [], NOW + timedelta(hours=hour))
        assert queue.entries[sample_request.sample_id].phase == Phase.QUEUED
        queue = reconcile(store, jobs, [], NOW + timedelta(hours=hour + 6))
        assert queue.entries[sample_request.sample_id].phase == Phase.ACTIVE
    entry = queue.entries[sample_request.sample_id]
    assert entry.failures == 0
    jobs.jobs[entry.job_name] = JobStatus.FAILED
    queue = reconcile(store, jobs, [], NOW + timedelta(hours=25))
    assert queue.entries[sample_request.sample_id].failures == 1
    assert queue.entries[sample_request.sample_id].phase == Phase.ACTIVE
    assert queue.entries[sample_request.sample_id].error == ""


def test_absent_attempt_recovers_before_deadline_and_fails_after_it(tmp_path, sample_request):
    store, jobs = SampleStore(str(tmp_path)), JobService()
    queue = reconcile(store, jobs, [sample_request], NOW)
    jobs.jobs.clear()  # The service lost the logical job after the queue committed the request.
    recovered = reconcile(store, jobs, [], NOW + timedelta(hours=2))
    assert recovered.entries[sample_request.sample_id].job_name == queue.entries[sample_request.sample_id].job_name
    assert len(jobs.jobs) == 1
    jobs.jobs.clear()
    expired = reconcile(store, jobs, [], NOW + timedelta(hours=49))
    assert expired.entries[sample_request.sample_id].phase == Phase.FAILED
    assert jobs.jobs == {}


def test_changed_prompts_create_distinct_history_without_replacing_results(tmp_path, sample_request):
    request = sample_request
    store, jobs = SampleStore(str(tmp_path)), JobService()
    store.save_result(completed(request))
    changed = request.model_copy(update={"spec": request.spec.model_copy(update={"temperature": 0.5})})
    queue = reconcile(store, jobs, [request, changed], NOW)
    assert len(queue.entries) == 2
    assert len(jobs.jobs) == 1
    assert store.result(request) == completed(request)
    assert store.result(changed) is None


def test_existing_result_recovers_a_missing_queue_without_allocating_gpus(tmp_path, sample_request):
    store, jobs = SampleStore(str(tmp_path)), JobService()
    store.save_result(completed(sample_request))
    new_main = sample_request.model_copy(update={"source_revision": "d" * 40})
    queue = reconcile(store, jobs, [new_main], NOW)
    assert jobs.jobs == {}
    assert queue.entries[new_main.sample_id].phase == Phase.COMPLETE
    assert queue.entries[new_main.sample_id].request.source_revision == sample_request.source_revision


def test_issue_update_recovers_lost_response_and_preserves_human_comments(monkeypatch):
    comments = [{"id": 1, "user": {"login": "researcher"}, "body": COMMENT_MARKER}]
    lose_response = True
    client_type = httpx.Client

    def serve(request):
        nonlocal lose_response
        if request.method == "GET":
            return httpx.Response(200, json=comments)
        body = json.loads(request.content)["body"]
        if request.method == "POST":
            comments.append({"id": 2, "user": {"login": "github-actions[bot]"}, "body": body})
            if lose_response:
                lose_response = False
                raise httpx.ReadError("Response lost", request=request)
            return httpx.Response(201, json=comments[-1])
        assert request.method == "PATCH"
        comments[1]["body"] = body
        return httpx.Response(200, json=comments[1])

    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client_type(transport=httpx.MockTransport(serve), **kwargs))
    with pytest.raises(httpx.ReadError):
        update_issue_comment(f"first report {COMMENT_MARKER}", "test-token")
    update_issue_comment(f"second report {COMMENT_MARKER}", "test-token")
    assert len(comments) == 2
    assert comments[0]["body"] == COMMENT_MARKER
    assert comments[1]["body"] == f"second report {COMMENT_MARKER}"


@pytest.mark.parametrize(
    ("prompt_ids", "predicted", "expected_ids", "reason"),
    [
        ([1], 0, (0,), StopReason.EOS),
        ([1], 2, (2, 2), StopReason.MAX_NEW_TOKENS),
        ([1, 2, 3], 2, (2,), StopReason.CONTEXT_LIMIT),
        ([1, 2, 3, 4], 2, (), StopReason.CONTEXT_LIMIT),
    ],
)
def test_generation_stops_at_the_correct_boundary(sample_request, prompt_ids, predicted, expected_ids, reason):
    request = sample_request

    def logits(tokens, positions):
        return np.eye(5)[[predicted]]

    result = generate(request.spec, [prompt_ids], eos_token_id=0, logits=logits, decode=str)[0]
    assert result.token_ids == expected_ids
    assert result.stop_reason == reason


def test_sampling_streams_do_not_depend_on_prompt_order_or_early_eos(sample_request):
    request = sample_request
    other = Prompt(id="other", text="other", seed=27, source_url="https://example.org")
    spec = request.spec.model_copy(
        update={
            "prompts": (request.spec.prompts[0], other),
            "batch_size": 2,
            "temperature": 1.0,
            "context_length": 20,
            "max_new_tokens": 12,
        }
    )

    def logits(tokens, positions):
        return np.tile(np.array([-1000, 1, 1, 1, 1]), (tokens.shape[0], 1))

    pair = generate(spec, [[1], [2]], eos_token_id=0, logits=logits, decode=str)
    reversed_spec = spec.model_copy(update={"prompts": tuple(reversed(spec.prompts))})
    reverse = generate(reversed_spec, [[2], [1]], eos_token_id=0, logits=logits, decode=str)
    alone = generate(
        spec.model_copy(update={"prompts": (other,), "batch_size": 1}), [[2]], eos_token_id=0, logits=logits, decode=str
    )
    assert pair[0].token_ids == reverse[1].token_ids
    assert pair[1].token_ids == reverse[0].token_ids == alone[0].token_ids

    def early_eos(tokens, positions):
        scores = logits(tokens, positions)
        scores[tokens[:, 0] == 1] = [1000, -1000, -1000, -1000, -1000]
        return scores

    early = generate(spec, [[1], [2]], eos_token_id=0, logits=early_eos, decode=str)
    assert early[0].stop_reason == StopReason.EOS
    assert early[1].token_ids == alone[0].token_ids


def test_daily_publication_retry_preserves_snapshot_and_does_not_resample(tmp_path, monkeypatch, sample_request):
    request = sample_request
    public = tmp_path / "public"
    monkeypatch.setattr(sites, "PUBLIC_ROOT", str(public))
    store = SampleStore(str(tmp_path / "private"))
    store.save_result(completed(request))
    store.save_queue(
        Queue(entries={request.sample_id: Entry(request=request, phase=Phase.COMPLETE)}, inventory_at=NOW), None
    )
    comments = []

    def fail_comment(body):
        raise ConnectionError("GitHub unavailable")

    with pytest.raises(ConnectionError):
        publish_daily(store, date(2026, 9, 12), fail_comment)
    page = public / "hero/completions/2026.09.12/index.html"
    first_page = page.read_text()
    queue, version = store.read_queue()
    store.save_queue(queue.model_copy(update={"inventory_at": NOW + timedelta(hours=3)}), version)
    url = publish_daily(store, date(2026, 9, 12), comments.append)
    assert page.read_text() == first_page
    assert SampleResult.model_validate_json(
        (public / f"hero/completions/results/{request.sample_id}.json").read_bytes()
    ) == completed(request)
    assert store.read_queue()[0].published_date == "2026-09-12"
    assert url in comments[0]
    assert len(comments) == 1
    publish_daily(store, date(2026, 9, 12), comments.append)
    assert len(comments) == 1


def test_report_data_cannot_close_its_script_element(sample_request):
    request = sample_request
    poisoned = request.model_copy(
        update={"checkpoint": request.checkpoint.model_copy(update={"run_id": "</script><script>alert(1)</script>"})}
    )
    manifest = report_manifest(Queue(entries={poisoned.sample_id: Entry(request=poisoned)}), "2026-09-12")
    html = render_report(manifest)
    embedded = html.split('<script id="report-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert "</script>" not in embedded
    assert json.loads(embedded)["entries"][0]["run_id"] == poisoned.checkpoint.run_id
