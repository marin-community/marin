# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Daily public history reports and one updated GitHub issue comment."""

import json
import logging
from collections.abc import Callable
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from rigging.filesystem.conditional_object import ConditionalWriteError, conditional_object

from marin.evaluation.completions import Phase, Queue, SampleStore
from marin.publish import sites

REPORT_USER = "hero"
REPORT_SLUG = "completions"
COMMENT_MARKER = "<!-- hero-checkpoint-completions-v1 -->"
ISSUE_API = "https://api.github.com/repos/marin-community/marin/issues/8827"
logger = logging.getLogger(__name__)


def report_manifest(queue: Queue, report_date: str) -> dict:
    entries = sorted(queue.entries.items(), key=lambda pair: (pair[1].request.checkpoint.step, pair[0]), reverse=True)
    return {
        "date": report_date,
        "inventory_at": queue.inventory_at.isoformat() if queue.inventory_at else None,
        "previous_url": queue.report_url,
        "counts": {phase.value: sum(entry.phase == phase for _, entry in entries) for phase in Phase},
        "entries": [
            {
                "id": key,
                "run_id": entry.request.checkpoint.run_id,
                "step": entry.request.checkpoint.step,
                "phase": entry.phase.value,
                "error": entry.error,
                "url": (
                    f"{sites.PUBLIC_URL_BASE}/{REPORT_USER}/{REPORT_SLUG}/results/{key}.json"
                    if entry.phase == Phase.COMPLETE
                    else None
                ),
            }
            for key, entry in entries
        ],
    }


def render_report(manifest: dict) -> str:
    # A script element ends at </script> even when it contains JSON. Escape every '<'.
    data = json.dumps(manifest, ensure_ascii=True).replace("<", "\\u003c")
    return Path(__file__).with_name("completion_report.html").read_text().replace("__REPORT_DATA__", data)


def update_issue_comment(body: str, token: str) -> None:
    """Upsert only the marked GitHub Actions comment, including recovery after a lost POST response."""
    with httpx.Client(
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}, timeout=30
    ) as client:
        comment_ids = []
        page = 1
        while True:
            response = client.get(f"{ISSUE_API}/comments", params={"per_page": 100, "page": page})
            response.raise_for_status()
            comments = response.json()
            comment_ids.extend(
                comment["id"]
                for comment in comments
                if comment["user"]["login"] == "github-actions[bot]" and COMMENT_MARKER in comment["body"]
            )
            if len(comments) < 100:
                break
            page += 1
        if len(comment_ids) > 1:
            logger.warning("Multiple managed comments %s; updating the newest", comment_ids)
        if comment_ids:
            response = client.patch(
                f"https://api.github.com/repos/marin-community/marin/issues/comments/{max(comment_ids)}",
                json={"body": body},
            )
        else:
            response = client.post(f"{ISSUE_API}/comments", json={"body": body})
        response.raise_for_status()


def publish_daily(store: SampleStore, day: date, comment: Callable[[str], None]) -> str:
    """Publish once per UTC day, with retries independent of GPU sampling.

    The scheduled workflow serializes invocations. The first attempt freezes the daily
    snapshot, so a retry cannot change an already linked daily report.
    """
    report_date = day.isoformat()
    queue, version = store.read_queue()
    if queue.published_date >= report_date:
        return queue.report_url
    snapshot = conditional_object(f"{store.root}/reports/{report_date}.json")
    saved = snapshot.read()
    if saved is None:
        snapshot.write(queue.model_dump_json().encode(), expected_version=None)
        frozen = queue
    else:
        frozen = Queue.model_validate_json(saved.data)
    for key, entry in frozen.entries.items():
        if entry.phase != Phase.COMPLETE:
            continue
        target = conditional_object(f"{sites.PUBLIC_ROOT}/{REPORT_USER}/{REPORT_SLUG}/results/{key}.json")
        if target.version() is not None:
            continue
        result = store.result(entry.request)
        if result is None:
            raise FileNotFoundError(f"Completed sample has no result: {key}")
        try:
            target.write(result.model_dump_json().encode(), expected_version=None)
        except ConditionalWriteError:
            raise RuntimeError(f"Another publisher wrote sample {key}") from None
    manifest = report_manifest(frozen, report_date)
    with TemporaryDirectory(prefix="hero-completion-report-") as directory:
        source = Path(directory) / "index.html"
        source.write_text(render_report(manifest))
        site = sites.publish_site(
            source,
            user=REPORT_USER,
            slug=REPORT_SLUG,
            version=day.strftime("%Y.%m.%d"),
            title="Hero checkpoint completions",
            summary="Every permanent checkpoint, with a daily history report.",
        )
    latest = sites.publish_site_alias(site, user=REPORT_USER, slug=REPORT_SLUG)
    counts = manifest["counts"]
    comment(
        f"🤖 Hero checkpoint completions · {report_date} UTC\n\n"
        f"[Full checkpoint history]({latest}) · [This daily report]({site.url})\n\n"
        f"{counts['complete']} complete, {counts['active']} active, "
        f"{counts['queued']} queued, {counts['failed']} failed. "
        "Select two checkpoints and a prompt to compare the samples. "
        "The report includes raw results and generation settings.\n\n"
        f"Inventory checked: {manifest['inventory_at'] or 'not yet available'}.\n\n{COMMENT_MARKER}"
    )
    store.save_queue(queue.model_copy(update={"published_date": report_date, "report_url": site.url}), version)
    return site.url
