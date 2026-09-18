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
from pydantic import BaseModel, Field
from rigging.filesystem.conditional_object import conditional_object
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.grug.moe_hero_ep.ops.vibe_check.completions import SampleStore, SamplingSpec, digest

logger = logging.getLogger(__name__)

REPORT_USER = "rav"
REPORT_SLUG = "hero-completions"
REPORT_URL_PATTERN = "reports/*.url"
LATEST_REPORT_KEY = f"{REPORT_USER}/{REPORT_SLUG}/latest/index.html"
COMMENT_MARKER = "<!-- hero-checkpoint-completions-v1 -->"
ISSUES_API = "https://api.github.com/repos/marin-community/marin/issues"
ISSUE_API = f"{ISSUES_API}/8827"
COMMENT_PAGE_SIZE = 100
CATALOG_KEY = "reports/catalog.json"
UNKNOWN_RUN = "Unknown run"
UNKNOWN_RELEASE = "Unknown sampling version"


class ReportEntry(BaseModel):
    id: str
    url: str
    step: int | None = Field(default=None, ge=0)
    run_id: str = UNKNOWN_RUN
    release: str = UNKNOWN_RELEASE
    spec_id: str = ""
    completed_at: str = ""


def report_entry(sample_id: str, data: dict) -> ReportEntry:
    """Read report metadata without imposing the current sampler schema on historical results."""
    rows = data.get("completions", [])
    rows = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    samples = [sample for row in rows for sample in (row["samples"] if isinstance(row.get("samples"), list) else [row])]
    if not any(
        isinstance(sample, dict)
        and (
            isinstance(sample.get("text"), str)
            or (
                isinstance(sample.get("token_scores"), list)
                and any(
                    isinstance(token, dict) and isinstance(token.get("text"), str) for token in sample["token_scores"]
                )
            )
        )
        for sample in samples
    ):
        raise ValueError("Result has no completed samples")
    request = data.get("request") or {}
    if not isinstance(request, dict):
        raise ValueError("Result request must be an object")
    checkpoint = request.get("checkpoint") or {}
    spec = request.get("spec") or {}
    if not isinstance(checkpoint, dict) or not isinstance(spec, dict):
        raise ValueError("Checkpoint and sampling specification must be objects")
    # Hash the original fields. New schema defaults would change historical request IDs.
    if checkpoint and spec and digest({"checkpoint": checkpoint, "spec": spec}) != sample_id:
        raise ValueError("Result provenance does not match its filename")
    return ReportEntry(
        id=sample_id,
        url=prefix_join(sites.PUBLIC_URL_BASE, public_result_key(sample_id)),
        step=checkpoint.get("step"),
        run_id=checkpoint.get("run_id") or UNKNOWN_RUN,
        release=spec.get("release") or UNKNOWN_RELEASE,
        spec_id=digest(spec) if spec else "",
        completed_at=data.get("completed_at") or "",
    )


def public_result_key(sample_id: str) -> str:
    return f"{REPORT_USER}/{REPORT_SLUG}/results/{sample_id}.json"


def report_manifest(
    entries: list[ReportEntry], report_date: str, previous_url: str, *, current_spec_id: str = ""
) -> dict:
    entries = sorted(
        entries,
        key=lambda row: (
            row.step if row.step is not None else -1,
            row.spec_id == current_spec_id,
            row.completed_at,
            row.id,
        ),
        reverse=True,
    )
    return {
        "date": report_date,
        "previous_url": previous_url,
        "entries": [entry.model_dump() for entry in entries],
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


def publish_daily(store: SampleStore, day: date, manifest: dict) -> str:
    """Publish a fixed daily snapshot after the first result arrives.

    The workflow serializes callers. A retry uses the saved snapshot, even if
    more results arrive before publication succeeds.
    """
    report_date = day.isoformat()
    published = sorted((store.root / REPORT_URL_PATTERN).glob(), key=lambda path: path.name)
    previous_url = published[-1].read_text() if published else ""
    if published and published[-1].name >= f"{report_date}.url":
        return previous_url
    if not manifest["entries"]:
        return previous_url
    snapshot = conditional_object(str(store.root / f"reports/{report_date}.json"))
    saved = snapshot.read()
    if saved is None:
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


def publish_reports(store: SampleStore, day: date, comment: Callable[[str], None], *, spec: SamplingSpec) -> str:
    """Add completed results to the catalog and publish only when new usable results exist."""
    catalog = conditional_object(str(store.root / CATALOG_KEY))
    saved = catalog.read()
    entries = [ReportEntry.model_validate(row) for row in json.loads(saved.data)["entries"]] if saved else []
    known = {entry.id for entry in entries}
    added = []
    for sample_id in sorted(store.completed_ids() - known):
        raw = StoragePath(store.result_uri(sample_id)).read_bytes()
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("Result must be a JSON object")
            entry = report_entry(sample_id, data)
        except ValueError as error:
            logger.warning("Skip unusable result %s: %s", sample_id, error)
            continue
        target = conditional_object(prefix_join(sites.PUBLIC_ROOT, public_result_key(sample_id)))
        if target.version() is None:
            target.write(raw, expected_version=None)
        added.append(entry)
    latest = prefix_join(sites.PUBLIC_URL_BASE, LATEST_REPORT_KEY)
    if not added:
        logger.info("No new usable completions; retain the current report (%d catalog entries)", len(entries))
        return latest
    entries.extend(added)
    published = sorted((store.root / REPORT_URL_PATTERN).glob(), key=lambda path: path.name)
    previous_url = published[-1].read_text() if published else ""
    manifest = report_manifest(entries, day.isoformat(), previous_url, current_spec_id=digest(spec.model_dump()))
    daily_url = publish_daily(store, day, manifest)
    latest = publish_current(manifest)
    links = f"[Current completions]({latest})"
    if daily_url:
        links += f" · [Daily snapshot]({daily_url})"
    comment(
        f"🤖 Hero checkpoint completions · {day.isoformat()} UTC\n\n"
        f"{links}\n\n"
        f"{len(entries)} completed sample sets across sampling versions. "
        "The current report updates when new results arrive. Dated snapshots stay fixed.\n\n"
        f"{COMMENT_MARKER}"
    )
    catalog.write(json.dumps(manifest).encode(), expected_version=saved.version if saved else None)
    return latest
