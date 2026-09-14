# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import UTC, date, datetime

import httpx
import numpy as np
import pytest
from iris.resources.state import JobState
from marin.publish import sites

from experiments.grug.moe_hero_ep.ops.vibe_check.completions import (
    Checkpoint,
    Completion,
    Prompt,
    SampleRequest,
    SampleResult,
    SampleStore,
    SamplingSpec,
    StopReason,
)
from experiments.grug.moe_hero_ep.ops.vibe_check.generation import generate
from experiments.grug.moe_hero_ep.ops.vibe_check.jobs import submit_pending
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
        self.jobs: dict[str, JobState] = {}
        self.requests: dict[str, SampleRequest] = {}
        self.lose_submit_response = False
        self.unavailable = False

    def states(self):
        if self.unavailable:
            raise ConnectionError("job service unavailable")
        return dict(self.jobs)

    def submit(self, request, name):
        if name in self.jobs:
            return
        self.jobs[name] = JobState.RUNNING
        self.requests[name] = request
        if self.lose_submit_response:
            raise ConnectionError("response lost after submission")


def test_retries_recover_service_errors_without_overlapping_jobs(tmp_path, sample_request):
    request = sample_request
    store = SampleStore(str(tmp_path))
    jobs = JobService()
    jobs.lose_submit_response = True
    with pytest.raises(ConnectionError):
        submit_pending(store, jobs, [request])
    # A fresh invocation has a different main revision, but must keep the submitted request.
    changed_main = request.model_copy(update={"source_revision": "c" * 40})
    older = request.model_copy(
        update={"checkpoint": request.checkpoint.model_copy(update={"step": 3000, "uri": "s3://checkpoints/step-3000"})}
    )
    jobs.lose_submit_response = False
    submit_pending(store, jobs, [changed_main, older])
    assert len(jobs.jobs) == 1
    name = next(iter(jobs.jobs))
    assert jobs.requests[name].source_revision == request.source_revision
    jobs.unavailable = True
    with pytest.raises(ConnectionError):
        submit_pending(store, jobs, [changed_main, older])
    assert len(jobs.jobs) == 1
    jobs.unavailable = False
    jobs.jobs[name] = JobState.FAILED
    submit_pending(store, jobs, [changed_main, older])
    assert len(jobs.jobs) == 2
    retry = list(jobs.jobs)[-1]
    assert jobs.requests[retry] == request
    store.save_result(completed(request))
    submit_pending(store, jobs, [])
    assert len(jobs.jobs) == 2  # A saved result does not mean GPU teardown finished.
    jobs.jobs[retry] = JobState.SUCCEEDED
    submit_pending(store, jobs, [])
    assert list(jobs.requests.values())[-1] == older


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
    newest = request.model_copy(update={"checkpoint": request.checkpoint.model_copy(update={"step": 24000})})
    submit_pending(store, jobs, requests)
    assert {row.sample_id for row in store.requests()} == {row.sample_id for row in requests}
    for state in [JobState.FAILED, JobState.UNSCHEDULABLE, JobState.SUCCEEDED]:
        active = next(name for name, state in jobs.jobs.items() if state == JobState.RUNNING)
        assert jobs.requests[active] == requests[-1]
        jobs.jobs[active] = state
        submit_pending(store, jobs, [newest] if state == JobState.SUCCEEDED else [])
    assert store.failed(requests[-1])
    assert list(jobs.requests.values())[-1] == newest
    store.save_result(completed(newest))
    store.save_result(completed(requests[1]))
    jobs.jobs.clear()  # Iris prunes terminal job history.
    submit_pending(store, jobs, [])
    assert len(jobs.jobs) == 1
    assert jobs.requests[next(iter(jobs.jobs))] == requests[0]


def test_result_written_during_status_read_completes_last_attempt(tmp_path, monkeypatch, sample_request):
    store, jobs = SampleStore(str(tmp_path)), JobService()
    submit_pending(store, jobs, [sample_request])
    for _ in range(2):
        jobs.jobs[list(jobs.jobs)[-1]] = JobState.FAILED
        submit_pending(store, jobs, [])

    def finish_job():
        store.save_result(completed(sample_request))
        return dict.fromkeys(jobs.jobs, JobState.SUCCEEDED)

    monkeypatch.setattr(jobs, "states", finish_job)
    submit_pending(store, jobs, [])
    assert store.result(sample_request) == completed(sample_request)
    assert not store.failed(sample_request)
    assert len(jobs.jobs) == 3


def test_prompt_changes_add_samples_and_source_changes_reuse_results(tmp_path, sample_request):
    request = sample_request
    store, jobs = SampleStore(str(tmp_path)), JobService()
    store.save_result(completed(request))
    new_main = request.model_copy(update={"source_revision": "d" * 40})
    submit_pending(store, jobs, [new_main])
    assert jobs.jobs == {}
    assert store.result(request) == completed(request)

    changed_prompt = request.spec.prompts[0].model_copy(update={"text": "def subtract(a, b):"})
    changed = new_main.model_copy(update={"spec": request.spec.model_copy(update={"prompts": (changed_prompt,)})})
    submit_pending(store, jobs, [new_main, changed])
    assert len(store.requests()) == 2
    assert len(jobs.jobs) == 1
    assert store.result(request) == completed(request)
    assert store.result(changed) is None


def test_results_require_the_full_bank_and_keep_the_first_success(tmp_path, sample_request):
    prompt = sample_request.spec.prompts[0].model_copy(update={"id": "other"})
    request = sample_request.model_copy(
        update={"spec": sample_request.spec.model_copy(update={"prompts": (*sample_request.spec.prompts, prompt)})}
    )
    partial = completed(sample_request).model_dump()
    partial["request"] = request.model_dump()
    with pytest.raises(ValueError):
        SampleResult.model_validate(partial)
    partial["completions"] += ({**partial["completions"][0], "prompt_id": "other"},)
    result = SampleResult.model_validate(partial)
    store = SampleStore(str(tmp_path))
    store.save_result(result)
    store.save_result(result.model_copy(update={"completed_at": "2026-09-13T10:00:00+00:00"}))
    assert store.result(request) == result


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
    batches = generate(spec.model_copy(update={"batch_size": 1}), [[1], [2]], eos_token_id=0, logits=logits, decode=str)
    assert batches == pair
    assert pair[0].token_ids == reverse[1].token_ids
    assert pair[1].token_ids == reverse[0].token_ids == alone[0].token_ids

    def early_eos(tokens, positions):
        scores = logits(tokens, positions)
        scores[tokens[:, 0] == 1] = [1000, -1000, -1000, -1000, -1000]
        return scores

    early = generate(spec, [[1], [2]], eos_token_id=0, logits=early_eos, decode=str)
    assert early[0].stop_reason == StopReason.EOS
    assert early[1].token_ids == alone[0].token_ids


def test_daily_publication_preserves_history_across_retries_and_new_days(tmp_path, monkeypatch, sample_request):
    request = sample_request
    public = tmp_path / "public"
    monkeypatch.setattr(sites, "PUBLIC_ROOT", str(public))
    store = SampleStore(str(tmp_path / "private"))
    store.save_result(completed(request))
    comments = []

    def fail_comment(body):
        raise ConnectionError("GitHub unavailable")

    with pytest.raises(ConnectionError):
        publish_daily(store, date(2026, 9, 12), fail_comment)
    page = public / "rav/hero-completions/2026.09.12/index.html"
    first_page = page.read_text()
    newer = request.model_copy(update={"checkpoint": request.checkpoint.model_copy(update={"step": 24000})})
    store.save_result(completed(newer))
    url = publish_daily(store, date(2026, 9, 12), comments.append)
    assert page.read_text() == first_page
    assert newer.sample_id not in first_page
    assert SampleResult.model_validate_json(
        (public / f"rav/hero-completions/results/{request.sample_id}.json").read_bytes()
    ) == completed(request)
    assert url in comments[0]
    publish_daily(store, date(2026, 9, 12), comments.append)
    assert len(comments) == 1
    latest = public / "rav/hero-completions/latest/index.html"
    assert f'href="{url}"' in latest.read_text()
    next_url = publish_daily(store, date(2026, 9, 13), comments.append)
    assert next_url != url
    assert f'href="{next_url}"' in latest.read_text()
    assert page.read_text() == first_page
    assert len(comments) == 2
    assert next_url in comments[1]
    next_page = (public / "rav/hero-completions/2026.09.13/index.html").read_text()
    assert request.sample_id in next_page and newer.sample_id in next_page
    assert url in next_page


def test_report_data_cannot_close_its_script_element(sample_request):
    request = sample_request
    poisoned = request.model_copy(
        update={"checkpoint": request.checkpoint.model_copy(update={"run_id": "</script><script>alert(1)</script>"})}
    )
    manifest = report_manifest([completed(poisoned)], "2026-09-12", "")
    html = render_report(manifest)
    embedded = html.split('<script id="report-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert json.loads(embedded)["entries"][0]["run_id"] == poisoned.checkpoint.run_id
