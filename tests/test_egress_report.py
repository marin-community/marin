# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from scripts.ops.egress_report import _compose_message


def test_message_reports_region_coverage_and_scopes_no_offenders_to_it() -> None:
    message = _compose_message(
        hours=24.0,
        date="2026-11-05",
        total=25,
        gs_lines=37506,
        region_known_lines=6158,
        offenders=[],
        report_url="https://gist.github.com/owner/id",
    )

    assert message.split("\n") == [
        "**Daily cross-region egress report** (24h window, UTC 2026-11-05)",
        "- total cross-region mentions: **25**",
        "- worker region known for **16%** of 37,506 `gs://` log lines",
        "- report: <https://gist.github.com/owner/id>",
        "",
        "_No offenders among the 16% of lines with a known region._",
    ]
