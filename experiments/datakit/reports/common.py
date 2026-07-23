# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared plumbing for the per-stage datakit reports.

Every stage report is one StepSpec whose fn aggregates that stage's per-source
artifacts (counters + a bounded read of site/sample parquet), renders a single
self-contained HTML page from a template in ``templates/``, and returns a
:class:`StageReport`. Templates embed their data as JSON via the ``__DATA__``
placeholder, so a report works offline and can be hosted anywhere.
"""

import html
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from pydantic import BaseModel
from rigging.filesystem import StoragePath

TEMPLATES = Path(__file__).parent / "templates"
REPORT_FILE = "report.html"


class StageReport(BaseModel):
    """Outcome of a stage-report step: the rendered page plus its headline stats.

    Persisted as the step's ``.artifact``; ``stats`` mirrors the numbers shown
    on the page so downstream tooling can read them without parsing HTML.
    """

    version: str = "v1"
    html_path: str
    stats: dict[str, Any]


def render_template(template_name: str, *, title: str, data: dict) -> str:
    """Render ``templates/<template_name>`` with ``__TITLE__`` and ``__DATA__`` filled in."""
    blob = json.dumps(data).replace("</", "<\\/")
    return (TEMPLATES / template_name).read_text().replace("__TITLE__", html.escape(title)).replace("__DATA__", blob)


def write_report(output_path: str, page: str) -> str:
    """Write the rendered page to ``<output_path>/report.html`` and return its path."""
    path = f"{output_path.rstrip('/')}/{REPORT_FILE}"
    with StoragePath(path).open("w") as fh:
        fh.write(page)
    return path


def sample_rows(directory: str, columns: list[str], limit: int) -> list[dict]:
    """Read up to ``limit`` rows of ``columns`` from the parquet files under ``directory``.

    Files are visited in sorted order via a single-level ``*.parquet`` glob (a
    recursive glob makes s3fs ``HeadObject`` the prefix, which the CW object
    store rejects), streaming batches until the limit is reached.
    """
    out: list[dict] = []
    for f in sorted(str(m) for m in StoragePath(f"{directory.rstrip('/')}/*.parquet").glob()):
        with StoragePath(f).open("rb") as fh:
            for batch in pq.ParquetFile(fh).iter_batches(batch_size=min(limit, 4096), columns=columns):
                rows = batch.to_pylist()
                out.extend(rows[: limit - len(out)])
                if len(out) >= limit:
                    return out
    return out
