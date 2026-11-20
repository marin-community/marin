# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Emit campaign queues from the Atlas Difficulty coverage document."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from infra.marina.apps.rl_data_catalog.server.difficulty_coverage import DifficultyCoverage, campaign_queue

logger = logging.getLogger(__name__)

MARINA_URL = "https://marina.oa.dev"
"""The CLI's default query target; mirrors ``marina.client.DEFAULT_MARINA_URL`` without
importing the marina client, which the marin-core test environment does not install."""


def published_queue(marina_url: str, applet_id: str) -> dict[str, Any]:
    """Fetch the published coverage document through the app's authenticated query channel.

    The Atlas is IAP-protected, so the fetch uses the marina query client the
    loaders use, not a bare HTTP request: the caller names the applet whose
    rebuild published the coverage.
    """
    # marina is not a dependency of the marin-core test environment this module's
    # tests run under; import it where the query channel is actually used.
    from marina.client import applet_query

    query = applet_query(marina_url, applet_id)
    rows = query(
        """
        SELECT c.coverage FROM atlas_difficulty_coverage c
        JOIN atlas_current a ON a.load_id = c.load_id WHERE a.name = 'latest' LIMIT 1
        """,
        {},
    )["rows"]
    if not rows:
        raise ValueError("The current load has no published Difficulty coverage; run load_difficulty first")
    return rows[0]["coverage"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    queue = commands.add_parser("queue", help="Emit the next campaign source queue from NEEDS_MEASUREMENT.")
    source = queue.add_mutually_exclusive_group(required=True)
    source.add_argument("--applet-id", help="The Atlas applet whose rebuild published the coverage.")
    source.add_argument("--coverage", type=Path, help="A published coverage artifact file.")
    queue.add_argument("--marina-url", default=MARINA_URL)
    queue.add_argument("--declarations", type=Path, help="The served catalog's declaration map (file mode).")
    queue.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command != "queue":
        raise ValueError(f"Unknown command: {args.command}")
    if args.applet_id:
        document = published_queue(args.marina_url, args.applet_id)
        args.output.write_text(json.dumps(document.get("queue", []), indent=2) + "\n")
        logger.info(
            "Queued %d sources needing measurement from applet %s", len(document.get("queue", [])), args.applet_id
        )
        return
    if not args.declarations:
        raise ValueError("File mode needs --declarations alongside --coverage")
    document = DifficultyCoverage.model_validate(json.loads(args.coverage.read_text()))
    rows = campaign_queue(document, json.loads(args.declarations.read_text()))
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    logger.info("Queued %d sources needing measurement -> %s", len(rows), args.output)


if __name__ == "__main__":
    main()
