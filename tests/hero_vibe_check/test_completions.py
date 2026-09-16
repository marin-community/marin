# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from datetime import UTC, date, datetime

import httpx
import numpy as np
import pytest
from iris.client.workload_codec import job_status_from_proto
from iris.resources.state import JobState
from iris.rpc import job_pb2
from marin.publish import sites

from experiments.grug.moe_hero_ep.ops.vibe_check.completions import (
    Checkpoint,
    Completion,
    Prompt,
    PromptCompletions,
    SampleRequest,
    SampleResult,
    SampleStore,
    SamplingSpec,
    StopReason,
    digest,
)
from experiments.grug.moe_hero_ep.ops.vibe_check.generation import generate
from experiments.grug.moe_hero_ep.ops.vibe_check.jobs import sample_job_names, submit_pending
from experiments.grug.moe_hero_ep.ops.vibe_check.publishing import (
    COMMENT_MARKER,
    publish_reports,
    render_report,
    report_entry,
    report_manifest,
    update_issue_comment,
)
from experiments.grug.moe_hero_ep.ops.vibe_check.status import render_sampling_summary

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
            completions_per_prompt=3,
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
            PromptCompletions(
                prompt_id="add",
                prompt_token_ids=(1,),
                samples=tuple(
                    Completion(
                        sample_index=index,
                        seed=request.spec.prompts[0].seed + index,
                        token_ids=(2, 3),
                        text="a + b",
                        stop_reason=StopReason.MAX_NEW_TOKENS,
                    )
                    for index in range(request.spec.completions_per_prompt)
                ),
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

    def submit(self, request, name, _priority_band):
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
        submit_pending(store, jobs, [request], spec=sample_request.spec)
    # A fresh invocation has a different main revision, but must keep the submitted request.
    changed_main = request.model_copy(update={"source_revision": "c" * 40})
    older = request.model_copy(
        update={"checkpoint": request.checkpoint.model_copy(update={"step": 3000, "uri": "s3://checkpoints/step-3000"})}
    )
    jobs.lose_submit_response = False
    submit_pending(store, jobs, [changed_main, older], spec=sample_request.spec)
    assert len(jobs.jobs) == 1
    name = next(iter(jobs.jobs))
    assert jobs.requests[name].source_revision == request.source_revision
    jobs.unavailable = True
    with pytest.raises(ConnectionError):
        submit_pending(store, jobs, [changed_main, older], spec=sample_request.spec)
    assert len(jobs.jobs) == 1
    jobs.unavailable = False
    jobs.jobs[name] = JobState.FAILED
    submit_pending(store, jobs, [changed_main, older], spec=sample_request.spec)
    assert len(jobs.jobs) == 2
    retry = list(jobs.jobs)[-1]
    assert jobs.requests[retry] == request
    store.save_result(completed(request))
    submit_pending(store, jobs, [], spec=sample_request.spec)
    assert len(jobs.jobs) == 2  # A saved result does not mean GPU teardown finished.
    jobs.jobs[retry] = JobState.SUCCEEDED
    submit_pending(store, jobs, [], spec=sample_request.spec)
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
    submit_pending(store, jobs, requests, spec=sample_request.spec)
    assert {row.sample_id for row in store.requests(sample_request.spec)} == {row.sample_id for row in requests}
    for state in [JobState.FAILED, JobState.UNSCHEDULABLE, JobState.SUCCEEDED]:
        active = next(name for name, job_state in jobs.jobs.items() if job_state == JobState.RUNNING)
        assert jobs.requests[active] == requests[-1]
        jobs.jobs[active] = state
        if state == JobState.UNSCHEDULABLE:
            jobs.jobs.clear()  # History deletion between attempts must not reset the budget.
        submit_pending(store, jobs, [newest] if state == JobState.SUCCEEDED else [], spec=sample_request.spec)
    assert store.retries_exhausted(requests[-1])
    assert sum(row == requests[-1] for row in jobs.requests.values()) == 3
    assert list(jobs.requests.values())[-1] == newest
    store.save_result(completed(newest))
    store.save_result(completed(requests[1]))
    jobs.jobs.clear()  # Iris prunes terminal job history.
    submit_pending(store, jobs, [], spec=sample_request.spec)
    assert len(jobs.jobs) == 1
    assert jobs.requests[next(iter(jobs.jobs))] == requests[0]


def test_result_written_during_status_read_completes_last_attempt(tmp_path, monkeypatch, sample_request):
    store, jobs = SampleStore(str(tmp_path)), JobService()
    submit_pending(store, jobs, [sample_request], spec=sample_request.spec)
    for _ in range(2):
        jobs.jobs[list(jobs.jobs)[-1]] = JobState.FAILED
        submit_pending(store, jobs, [], spec=sample_request.spec)

    def finish_job():
        store.save_result(completed(sample_request))
        return dict.fromkeys(jobs.jobs, JobState.SUCCEEDED)

    monkeypatch.setattr(jobs, "states", finish_job)
    submit_pending(store, jobs, [], spec=sample_request.spec)
    assert store.result(sample_request.sample_id) == completed(sample_request)
    assert not store.retries_exhausted(sample_request)
    assert len(jobs.jobs) == 3


def test_prompt_changes_add_samples_and_source_changes_reuse_results(tmp_path, sample_request):
    request = sample_request
    store, jobs = SampleStore(str(tmp_path)), JobService()
    store.save_result(completed(request))
    new_main = request.model_copy(update={"source_revision": "d" * 40})
    submit_pending(store, jobs, [new_main], spec=sample_request.spec)
    assert jobs.jobs == {}
    assert store.result(request.sample_id) == completed(request)

    changed_prompt = request.spec.prompts[0].model_copy(update={"text": "def subtract(a, b):"})
    changed = new_main.model_copy(update={"spec": request.spec.model_copy(update={"prompts": (changed_prompt,)})})
    submit_pending(store, jobs, [changed], spec=changed.spec)
    assert store.requests(changed.spec) == [changed]
    assert store.requests(sample_request.spec) == [new_main]
    assert len(jobs.jobs) == 1
    assert store.result(request.sample_id) == completed(request)
    assert changed.sample_id not in store.completed_ids()


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
    missing_sample = result.model_dump()
    missing_sample["completions"][0]["samples"] = missing_sample["completions"][0]["samples"][:-1]
    with pytest.raises(ValueError, match="every sample index"):
        SampleResult.model_validate(missing_sample)
    duplicate_sample = result.model_dump(mode="json")
    duplicate_sample["completions"][0]["samples"][1] = duplicate_sample["completions"][0]["samples"][0]
    with pytest.raises(ValueError, match="every sample index"):
        SampleResult.model_validate(duplicate_sample)
    store = SampleStore(str(tmp_path))
    store.save_result(result)
    store.save_result(result.model_copy(update={"completed_at": "2026-09-13T10:00:00+00:00"}))
    assert store.result(request.sample_id) == result


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

    result = generate(
        request.spec, [prompt_ids], eos_token_id=0, logits=logits, decode=lambda ids: "".join(map(str, ids))
    )[0]
    assert len(result.samples) == 3
    assert all(sample.token_ids == expected_ids and sample.stop_reason == reason for sample in result.samples)


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

    pair = generate(spec, [[1], [2]], eos_token_id=0, logits=logits, decode=lambda ids: "".join(map(str, ids)))
    reversed_spec = spec.model_copy(update={"prompts": tuple(reversed(spec.prompts))})
    reverse = generate(
        reversed_spec, [[2], [1]], eos_token_id=0, logits=logits, decode=lambda ids: "".join(map(str, ids))
    )
    alone = generate(
        spec.model_copy(update={"prompts": (other,), "batch_size": 1}),
        [[2]],
        eos_token_id=0,
        logits=logits,
        decode=lambda ids: "".join(map(str, ids)),
    )
    batches = generate(
        spec.model_copy(update={"batch_size": 1}),
        [[1], [2]],
        eos_token_id=0,
        logits=logits,
        decode=lambda ids: "".join(map(str, ids)),
    )
    assert batches == pair
    assert pair[0] == reverse[1]
    assert pair[1] == reverse[0] == alone[0]
    assert len({sample.token_ids for sample in pair[1].samples}) == 3
    assert [sample.seed for sample in pair[1].samples] == [27, 28, 29]

    def early_eos(tokens, positions):
        scores = logits(tokens, positions)
        scores[tokens[:, 0] == 1] = [1000, -1000, -1000, -1000, -1000]
        return scores

    early = generate(spec, [[1], [2]], eos_token_id=0, logits=early_eos, decode=lambda ids: "".join(map(str, ids)))
    assert all(sample.stop_reason == StopReason.EOS for sample in early[0].samples)
    assert early[1] == alone[0]


def report_data(path):
    embedded = path.read_text().split('<script id="report-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    return json.loads(embedded)


def test_report_retains_history_and_only_advances_with_usable_results(tmp_path, monkeypatch, sample_request):
    public = tmp_path / "public"
    monkeypatch.setattr(sites, "PUBLIC_ROOT", str(public))
    store = SampleStore(str(tmp_path / "private"))
    comments = []
    # One corrupt historical result must not block publication of usable results.
    legacy_result = tmp_path / "private/results/legacy.json"
    legacy_result.parent.mkdir(parents=True)
    legacy_result.write_text("invalid historical result")
    url = publish_reports(store, date(2026, 9, 12), comments.append, spec=sample_request.spec)
    latest = public / "rav/hero-completions/latest/index.html"
    assert not latest.exists()
    assert not comments
    assert not list((tmp_path / "private/reports").glob("*"))
    assert not (public / "rav/hero-completions/2026.09.12/index.html").exists()

    store.save_request(sample_request)
    store.save_result(completed(sample_request))
    assert publish_reports(store, date(2026, 9, 12), comments.append, spec=sample_request.spec) == url
    assert [entry["id"] for entry in report_data(latest)["entries"]] == [sample_request.sample_id]
    daily = public / "rav/hero-completions/2026.09.12/index.html"
    assert report_data(daily)["entries"] == report_data(latest)["entries"]
    original = latest.read_bytes()
    newer_spec = sample_request.spec.model_copy(update={"release": "test-v2"})
    publish_reports(store, date(2026, 9, 13), comments.append, spec=newer_spec)
    assert latest.read_bytes() == original
    assert len(comments) == 1

    legacy = completed(sample_request).model_dump(mode="json")
    del legacy["request"]["spec"]["completions_per_prompt"]
    del legacy["request"]["spec"]["release"]
    del legacy["completed_at"]
    del legacy["eos_token_id"]
    row = legacy["completions"][0]
    sample = row.pop("samples")[0]
    del sample["sample_index"]
    del sample["token_scores"]
    row.update(sample)
    legacy_id = digest({key: legacy["request"][key] for key in ("checkpoint", "spec")})
    (tmp_path / f"private/results/{legacy_id}.json").write_text(json.dumps(legacy))
    newer = sample_request.model_copy(update={"spec": newer_spec})
    store.save_result(completed(newer))
    corrupt = completed(sample_request).model_dump(mode="json")
    corrupt["request"]["checkpoint"]["metadata_digest"] = "changed"
    (tmp_path / "private/results/wrong-provenance.json").write_text(json.dumps(corrupt))
    publish_reports(store, date(2026, 9, 13), comments.append, spec=newer_spec)
    entries = report_data(latest)["entries"]
    assert [entry["id"] for entry in entries] == [newer.sample_id, sample_request.sample_id, legacy_id]
    assert json.loads((public / f"rav/hero-completions/results/{legacy_id}.json").read_text()) == legacy
    assert [entry["id"] for entry in report_data(daily)["entries"]] == [sample_request.sample_id]


@pytest.mark.parametrize("failure", ["upload", "comment"])
def test_current_report_advances_while_daily_history_survives_retries(tmp_path, monkeypatch, sample_request, failure):
    public = tmp_path / "public"
    monkeypatch.setattr(sites, "PUBLIC_ROOT", str(public))
    store = SampleStore(str(tmp_path / "private"))
    store.save_request(sample_request)
    store.save_result(completed(sample_request))
    comments = []
    publish_site = sites.publish_site

    def lose_upload_response(*args, **kwargs):
        publish_site(*args, **kwargs)
        raise ConnectionError("Upload response lost")

    def fail_comment(body):
        raise ConnectionError("GitHub unavailable")

    with monkeypatch.context() as patch:
        if failure == "upload":
            patch.setattr(sites, "publish_site", lose_upload_response)
        with pytest.raises(ConnectionError):
            publish_reports(
                store,
                date(2026, 9, 12),
                fail_comment if failure == "comment" else comments.append,
                spec=sample_request.spec,
            )
    page = public / "rav/hero-completions/2026.09.12/index.html"
    first_page = page.read_bytes()
    newer = sample_request.model_copy(
        update={"checkpoint": sample_request.checkpoint.model_copy(update={"step": 24000})}
    )
    store.save_request(newer)
    store.save_result(completed(newer))
    url = publish_reports(store, date(2026, 9, 12), comments.append, spec=sample_request.spec)
    latest = public / "rav/hero-completions/latest/index.html"
    assert page.read_bytes() == first_page
    assert [entry["step"] for entry in report_data(latest)["entries"]] == [24000, 6000]
    assert [entry["step"] for entry in report_data(page)["entries"]] == [6000]
    for request in [sample_request, newer]:
        assert SampleResult.model_validate_json(
            (public / f"rav/hero-completions/results/{request.sample_id}.json").read_bytes()
        ) == completed(request)
    assert url in comments[-1]
    publish_reports(store, date(2026, 9, 12), comments.append, spec=sample_request.spec)
    assert page.read_bytes() == first_page
    assert [entry["step"] for entry in report_data(latest)["entries"]] == [24000, 6000]
    assert len(comments) == 1
    unchanged = latest.read_bytes()
    publish_reports(store, date(2026, 9, 13), comments.append, spec=sample_request.spec)
    assert latest.read_bytes() == unchanged
    newest = sample_request.model_copy(
        update={"checkpoint": sample_request.checkpoint.model_copy(update={"step": 30000})}
    )
    store.save_result(completed(newest))
    with monkeypatch.context() as patch:
        patch.setattr(sites, "publish_site", lose_upload_response)
        with pytest.raises(ConnectionError):
            publish_reports(store, date(2026, 9, 13), comments.append, spec=sample_request.spec)
    assert latest.read_bytes() == unchanged
    publish_reports(store, date(2026, 9, 13), comments.append, spec=sample_request.spec)
    assert page.read_bytes() == first_page
    next_page = public / "rav/hero-completions/2026.09.13/index.html"
    assert [entry["step"] for entry in report_data(next_page)["entries"]] == [30000, 24000, 6000]
    assert report_data(next_page)["previous_url"].endswith("/2026.09.12/index.html")


@pytest.mark.parametrize(
    ("state", "pending_reason", "error"),
    [
        (job_pb2.JOB_STATE_PENDING, "Queued for peer cw-us-east-08a to report free capacity", ""),
        (job_pb2.JOB_STATE_RUNNING, "", ""),
        (job_pb2.JOB_STATE_FAILED, "", "Restore failed: <tensor>"),
    ],
)
def test_summary_shows_checkpoint_job_state_and_diagnostics(sample_request, state, pending_reason, error):
    name = sample_job_names(sample_request)[0]
    job = job_status_from_proto(
        job_pb2.JobStatus(job_id=f"/hero-completions/{name}", state=state, pending_reason=pending_reason, error=error)
    )
    summary = render_sampling_summary([sample_request], {"previous-result"}, [job], "https://iris.oa.dev")
    assert "<td>6000</td>" in summary
    assert f'<a href="https://iris.oa.dev/#/job/%2Fhero-completions%2F{name}">' in summary
    assert f"<td>{job.state.value}</td>" in summary
    if pending_reason:
        assert f"<td>{pending_reason}</td>" in summary
    if error:
        assert "Restore failed: &lt;tensor&gt;" in summary
        assert "<tensor>" not in summary


def test_report_data_cannot_close_its_script_element(sample_request):
    request = sample_request
    poisoned = request.model_copy(
        update={"checkpoint": request.checkpoint.model_copy(update={"run_id": "</script><script>alert(1)</script>"})}
    )
    entry = report_entry(poisoned.sample_id, completed(poisoned).model_dump(mode="json"))
    manifest = report_manifest([entry], "2026-09-12", "")
    html = render_report(manifest)
    embedded = html.split('<script id="report-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert json.loads(embedded)["entries"][0]["run_id"] == poisoned.checkpoint.run_id
