# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Current completion reports, fixed daily snapshots, and one GitHub issue comment."""

import json
import logging
from collections.abc import Callable
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from marin.publish import sites
from rigging.filesystem.conditional_object import conditional_object
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.grug.moe_hero_ep.ops.vibe_check.completions import SampleResult, SampleStore

logger = logging.getLogger(__name__)

REPORT_USER = "rav"
REPORT_SLUG = "hero-completions"
LATEST_REPORT_KEY = f"{REPORT_USER}/{REPORT_SLUG}/latest/index.html"
COMMENT_MARKER = "<!-- hero-checkpoint-completions-v1 -->"
ISSUES_API = "https://api.github.com/repos/marin-community/marin/issues"
ISSUE_API = f"{ISSUES_API}/8827"
COMMENT_PAGE_SIZE = 100


def public_result_key(sample_id: str) -> str:
    return f"{REPORT_USER}/{REPORT_SLUG}/results/{sample_id}.json"


def report_manifest(results: list[SampleResult], report_date: str, previous_url: str) -> dict:
    requests = sorted(
        (result.request for result in results), key=lambda row: (row.checkpoint.step, row.sample_id), reverse=True
    )
    return {
        "date": report_date,
        "previous_url": previous_url,
        "entries": [
            {
                "id": request.sample_id,
                "run_id": request.checkpoint.run_id,
                "step": request.checkpoint.step,
                "url": prefix_join(sites.PUBLIC_URL_BASE, public_result_key(request.sample_id)),
            }
            for request in requests
        ],
    }


def render_report(manifest: dict) -> str:
    # A script element ends at </script> even when it contains JSON. Escape every '<'.
    data = json.dumps(manifest, ensure_ascii=True).replace("<", "\\u003c")
    return Path(__file__).with_name("report.html").read_text().replace("__REPORT_DATA__", data)


def publish_current(manifest: dict) -> str:
    """Replace the current report with the available completed results."""
    target_path = StoragePath(sites.PUBLIC_ROOT) / LATEST_REPORT_KEY
    target_path.parent.mkdirs()
    fs, path = url_to_fs(str(target_path))
    # fs.open passes content-type and cache metadata to GCS; open_url does not.
    with fs.open(
        path, "wb", content_type="text/html; charset=utf-8", fixed_key_metadata={"cache_control": "no-store"}
    ) as handle:
        handle.write(render_report(manifest).encode())
    return prefix_join(sites.PUBLIC_URL_BASE, LATEST_REPORT_KEY)


def update_issue_comment(body: str, token: str) -> None:
    """Upsert only the marked GitHub Actions comment, including recovery after a lost POST response."""
    with httpx.Client(
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}, timeout=30
    ) as client:
        comment_ids = []
        page = 1
        while True:
            response = client.get(f"{ISSUE_API}/comments", params={"per_page": COMMENT_PAGE_SIZE, "page": page})
            response.raise_for_status()
            comments = response.json()
            comment_ids.extend(
                comment["id"]
                for comment in comments
                if comment["user"]["login"] == "github-actions[bot]" and COMMENT_MARKER in comment["body"]
            )
            if len(comments) < COMMENT_PAGE_SIZE:
                break
            page += 1
        if len(comment_ids) > 1:
            logger.warning("Multiple managed comments %s; updating the newest", comment_ids)
        if comment_ids:
            response = client.patch(
                f"{ISSUES_API}/comments/{max(comment_ids)}",
                json={"body": body},
            )
        else:
            response = client.post(f"{ISSUE_API}/comments", json={"body": body})
        response.raise_for_status()


def publish_daily(store: SampleStore, day: date, results: list[SampleResult]) -> str:
    """Publish a fixed daily snapshot after the first result arrives.

    The workflow serializes callers. A retry uses the saved snapshot, even if
    more results arrive before publication succeeds.
    """
    report_date = day.isoformat()
    published = sorted((store.root / "reports/*.url").glob(), key=lambda path: path.name)
    previous_url = published[-1].read_text() if published else ""
    if published and published[-1].name >= f"{report_date}.url":
        return previous_url
    if not results:
        return previous_url
    snapshot = conditional_object(str(store.root / f"reports/{report_date}.json"))
    saved = snapshot.read()
    if saved is None:
        manifest = report_manifest(results, report_date, previous_url)
        snapshot.write(json.dumps(manifest).encode(), expected_version=None)
    else:
        manifest = json.loads(saved.data)
    with TemporaryDirectory(prefix="hero-completion-report-") as directory:
        source = Path(directory) / "index.html"
        source.write_text(render_report(manifest))
        site = sites.publish_site(
            source,
            user=REPORT_USER,
            slug=REPORT_SLUG,
            version=day.strftime("%Y.%m.%d"),
            title="Hero checkpoint completions",
            summary="Completed hero samples, with a daily history report.",
        )
    conditional_object(str(store.root / f"reports/{report_date}.url")).write(site.url.encode(), expected_version=None)
    return site.url


def publish_reports(store: SampleStore, day: date, comment: Callable[[str], None]) -> str:
    """Update the current report and retain one nonempty snapshot per report day."""
    results = store.results()
    for result in results:
        key = result.request.sample_id
        target = conditional_object(prefix_join(sites.PUBLIC_ROOT, public_result_key(key)))
        if target.version() is None:
            target.write(result.model_dump_json().encode(), expected_version=None)
    published = sorted((store.root / "reports/*.url").glob(), key=lambda path: path.name)
    previous_url = published[-1].read_text() if published else ""
    manifest = report_manifest(results, day.isoformat(), previous_url)
    latest = publish_current(manifest)
    daily_url = publish_daily(store, day, results)
    links = f"[Current completions]({latest})"
    if daily_url:
        links += f" · [Daily snapshot]({daily_url})"
    comment(
        f"🤖 Hero checkpoint completions · {day.isoformat()} UTC\n\n"
        f"{links}\n\n"
        f"{len(results)} completed sample sets. "
        "The current report updates hourly. Dated snapshots stay fixed.\n\n"
        f"{COMMENT_MARKER}"
    )
    return latest
