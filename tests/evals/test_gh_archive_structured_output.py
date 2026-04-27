# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
from pathlib import Path

from experiments.evals.gh_archive_structured_output import (
    GH_ARCHIVE_OPTIONAL_EVENT_TYPES,
    GH_ARCHIVE_REQUIRED_EVENT_TYPES,
    gh_archive_structured_output_raw_validation_sets,
)
from marin.datakit.download.gh_archive import GhArchiveDownloadConfig, download_gh_archive_events, gh_archive_step


def _event(
    *,
    event_type: str,
    event_id: str,
    before_sha: str = "a" * 40,
    after_sha: str = "b" * 40,
    html_url: str = "https://api.github.com/repos/marin-community/marin/issues/comments/112233445566",
) -> dict:
    return {
        "id": event_id,
        "type": event_type,
        "created_at": "2024-02-01T12:34:56Z",
        "actor": {"id": 9132456789, "login": "octocat", "node_id": "MDQ6VXNlcjE="},
        "repo": {"id": 9988776655, "name": "marin-community/marin"},
        "payload": {
            "before": before_sha,
            "after": after_sha,
            "head": "c" * 40,
            "comment": {
                "id": 112233445566,
                "html_url": html_url,
            },
        },
    }


def _read_jsonl_gz(path: Path) -> list[dict]:
    rows: list[dict] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def test_download_gh_archive_events_filters_masks_and_serializes(tmp_path: Path):
    base_url = "https://example-gh-archive"
    cfg = GhArchiveDownloadConfig(
        output_path=str(tmp_path / "gh_archive_eval"),
        start_date="2024-02-01",
        end_date="2024-02-01",
        start_hour=0,
        end_hour=1,
        base_url=base_url,
        event_types=(*GH_ARCHIVE_REQUIRED_EVENT_TYPES, *GH_ARCHIVE_OPTIONAL_EVENT_TYPES),
        max_events_per_event_type=None,
        request_timeout=30,
    )

    url_to_events = {
        f"{base_url}/2024-02-01-0.json.gz": [
            _event(event_type="PushEvent", event_id="1234567890123"),
            _event(event_type="WatchEvent", event_id="1234567890124"),
            _event(event_type="PullRequestEvent", event_id="1234567890125"),
        ],
        f"{base_url}/2024-02-01-1.json.gz": [
            _event(event_type="IssuesEvent", event_id="1234567890126"),
            _event(
                event_type="IssueCommentEvent",
                event_id="1234567890127",
                html_url="https://example.com/[maintainer=@hsjobeki]",
            ),
            _event(event_type="WorkflowRunEvent", event_id="1234567890128"),
        ],
    }

    def read_hour_events(url: str, timeout: int):
        assert timeout == 30
        return url_to_events.get(url, ())

    result = download_gh_archive_events(cfg, read_hour_events=read_hour_events)

    assert result["counts"] == {
        "IssueCommentEvent": 1,
        "IssuesEvent": 1,
        "PullRequestEvent": 1,
        "PushEvent": 1,
        "WorkflowRunEvent": 1,
    }

    push_rows = _read_jsonl_gz(tmp_path / "gh_archive_eval" / "PushEvent" / "part-00000.jsonl.gz")
    assert len(push_rows) == 1
    push_text = push_rows[0]["text"]
    push_payload = json.loads(push_text)
    assert push_payload["id"] == "<INT_13>"
    assert push_payload["payload"]["after"] == "<SHA_40>"
    assert push_payload["payload"]["before"] == "<SHA_40>"
    assert push_payload["payload"]["comment"]["html_url"].endswith("/issues/comments/<INT_12>")
    assert push_payload["repo"]["id"] == "<INT_10>"
    assert push_payload["created_at"] == "<DATE:2024-02-01>"
    assert push_text == json.dumps(push_payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    comment_rows = _read_jsonl_gz(tmp_path / "gh_archive_eval" / "IssueCommentEvent" / "part-00000.jsonl.gz")
    assert len(comment_rows) == 1
    comment_payload = json.loads(comment_rows[0]["text"])
    assert comment_payload["payload"]["comment"]["html_url"] == "https://example.com/[maintainer=@hsjobeki]"

    assert not (tmp_path / "gh_archive_eval" / "WatchEvent").exists()


def test_gh_archive_structured_output_raw_validation_sets_paths_and_tags():
    datasets = gh_archive_structured_output_raw_validation_sets(raw_root="gs://example-bucket/raw/gha")

    push = datasets["gh_archive_structured_output/PushEvent"]
    workflow = datasets["gh_archive_structured_output/WorkflowRunEvent"]

    assert push.input_path == "gs://example-bucket/raw/gha/PushEvent/*.jsonl.gz"
    assert push.tags == (
        "gh_archive_structured_output",
        "epic:5005",
        "issue:5098",
        "event_type:PushEvent",
    )
    assert workflow.input_path == "gs://example-bucket/raw/gha/WorkflowRunEvent/*.jsonl.gz"


def test_gh_archive_raw_validation_sets_can_drop_optional_event_types():
    datasets = gh_archive_structured_output_raw_validation_sets(
        raw_root="gs://example-bucket/raw/gha",
        include_optional_event_types=False,
    )
    assert "gh_archive_structured_output/WorkflowRunEvent" not in datasets


def test_gh_archive_step_hash_attrs_include_window_and_caps():
    step = gh_archive_step(
        start_date="2024-02-01",
        end_date="2024-02-02",
        start_hour=6,
        end_hour=8,
        event_types=("PushEvent", "IssuesEvent"),
        max_events_per_event_type=64,
        base_url="https://example-gh-archive",
    )
    assert step.hash_attrs["start_date"] == "2024-02-01"
    assert step.hash_attrs["end_date"] == "2024-02-02"
    assert step.hash_attrs["start_hour"] == 6
    assert step.hash_attrs["end_hour"] == 8
    assert step.hash_attrs["event_types"] == ("PushEvent", "IssuesEvent")
    assert step.hash_attrs["max_events_per_event_type"] == 64
    assert step.hash_attrs["base_url"] == "https://example-gh-archive"
