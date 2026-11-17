# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Emit campaign queues from the Atlas Difficulty coverage document."""

import argparse
import json
import logging
import urllib.request
from pathlib import Path
from typing import Any

from infra.marina.apps.rl_data_catalog.server.difficulty_coverage import DifficultyCoverage, campaign_queue

logger = logging.getLogger(__name__)


def served_queue(url: str) -> dict[str, Any]:
    """Fetch the published coverage document from the served Atlas app."""
    endpoint = url.rstrip("/") + "/api/difficulty-coverage"
    with urllib.request.urlopen(endpoint) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    queue = commands.add_parser("queue", help="Emit the next campaign source queue from NEEDS_MEASUREMENT.")
    source = queue.add_mutually_exclusive_group(required=True)
    source.add_argument("--from-url", help="The served Atlas app URL publishing /api/difficulty-coverage.")
    source.add_argument("--coverage", type=Path, help="A published coverage artifact.")
    queue.add_argument("--declarations", type=Path, help="The served catalog's declaration map (file mode).")
    queue.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command != "queue":
        raise ValueError(f"Unknown command: {args.command}")
    if args.from_url:
        document = served_queue(args.from_url)
        args.output.write_text(json.dumps(document.get("queue", []), indent=2) + "\n")
        logger.info("Queued %d sources needing measurement from %s", len(document.get("queue", [])), args.from_url)
        return
    if not args.declarations:
        raise ValueError("File mode needs --declarations alongside --coverage")
    document = DifficultyCoverage.model_validate(json.loads(args.coverage.read_text()))
    rows = campaign_queue(document, json.loads(args.declarations.read_text()))
    args.output.write_text(json.dumps(rows, indent=2) + "\n")
    logger.info("Queued %d sources needing measurement -> %s", len(rows), args.output)


if __name__ == "__main__":
    main()
