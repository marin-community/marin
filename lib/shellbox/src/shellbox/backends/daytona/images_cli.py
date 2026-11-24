# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Operate Daytona images without ad-hoc scripts.

``shellbox-daytona-images`` compares a project's image inventory with
supplied image specs, creates the missing images before a campaign and
prunes the Shellbox images that no current spec needs.
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from daytona import AsyncDaytonaV2Client

from shellbox.backends.daytona.connection import DaytonaConnection
from shellbox.backends.daytona.images import IMAGE_PREFIX, ImageRequest, ensure_image, list_images, prune_images


def _load_specs(source: str) -> list[ImageRequest]:
    """Read the image-spec interchange emitted by ``trial_driver image-specs``."""
    text = sys.stdin.read() if source == "-" else Path(source).read_text()
    payload = json.loads(text)
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not all(isinstance(entry, dict) for entry in payload):
        raise SystemExit("Image specs must be one JSON object or a list of objects")
    return [ImageRequest.from_dict(entry) for entry in payload]


def _connection(arguments: argparse.Namespace) -> DaytonaConnection:
    api_key = os.environ.get(arguments.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing Daytona credential environment variable: {arguments.api_key_env}")
    return DaytonaConnection(api_key=api_key)


def _print_rows(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> None:
    widths = [len(column) for column in header]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row, strict=True)]
    print("  ".join(column.ljust(width) for column, width in zip(header, widths, strict=True)))
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True)))


async def _inventory(client: AsyncDaytonaV2Client, *, as_json: bool) -> None:
    entries = [
        {
            "id": image.id,
            "name": image.name,
            "status": image.status,
            "region": image.region,
            "origin": image.origin,
            "bytes": image.bytes,
            "shellbox": bool(image.name and image.name.startswith(IMAGE_PREFIX)),
            "created_at": image.created_at.isoformat(),
        }
        for image in await list_images(client)
    ]
    if as_json:
        print(json.dumps(entries, indent=2, sort_keys=True))
        return
    rows = [
        (entry["name"] or "-", entry["status"], entry["origin"], entry["region"], entry["created_at"])
        for entry in entries
    ]
    _print_rows(("NAME", "STATUS", "ORIGIN", "REGION", "CREATED"), rows)


async def _plan(client: AsyncDaytonaV2Client, requests: list[ImageRequest]) -> None:
    present = {image.name: image for image in await list_images(client)}
    rows = []
    for request in requests:
        image = present.get(request.name)
        rows.append((request.name, "missing" if image is None else image.status))
    _print_rows(("NAME", "STATUS"), rows)


async def _ensure(client: AsyncDaytonaV2Client, requests: list[ImageRequest]) -> None:
    rows = [(await ensure_image(client, request), "ready") for request in requests]
    _print_rows(("NAME", "STATUS"), rows)


async def _prune(client: AsyncDaytonaV2Client, arguments: argparse.Namespace) -> None:
    if arguments.keep_specs is None and arguments.older_than_days is None:
        raise SystemExit("prune needs --keep-specs, --older-than-days or both")
    keep = (
        frozenset(request.name for request in _load_specs(arguments.keep_specs)) if arguments.keep_specs else frozenset()
    )
    created_before = (
        datetime.now(UTC) - timedelta(days=arguments.older_than_days) if arguments.older_than_days is not None else None
    )
    pruned = await prune_images(client, keep=keep, created_before=created_before, delete=arguments.delete)
    _print_rows(
        ("NAME", "CREATED", "ACTION"), [(image.name, image.created_at.isoformat(), image.action) for image in pruned]
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shellbox-daytona-images",
        description="Inventory, plan and require Daytona images identified by Shellbox image sources.",
    )
    parser.add_argument(
        "--api-key-env",
        required=True,
        help="Environment variable holding the selected account's API key.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inventory", help="List every named image in the project.")
    inventory.add_argument("--json", action="store_true", help="Emit JSON.")
    for name, help_text in (
        ("plan", "Compare required images with the project inventory."),
        ("ensure", "Create required images that are missing and wait until they are ready."),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--specs", required=True, help="Image-spec JSON file; use - to read stdin.")
    prune = commands.add_parser("prune", help="List, or with --delete delete, Shellbox images that no kept spec names.")
    prune.add_argument("--keep-specs", help="Image-spec JSON whose images are kept; use - to read stdin.")
    prune.add_argument("--older-than-days", type=float, help="Select only images created more than this long ago.")
    prune.add_argument("--delete", action="store_true", help="Delete the selected images; without it, only list them.")
    return parser


async def _run(arguments: argparse.Namespace) -> None:
    async with _connection(arguments).client_factory()() as client:
        if arguments.command == "inventory":
            await _inventory(client, as_json=arguments.json)
        elif arguments.command == "plan":
            await _plan(client, _load_specs(arguments.specs))
        elif arguments.command == "ensure":
            await _ensure(client, _load_specs(arguments.specs))
        else:
            await _prune(client, arguments)


def main() -> None:
    asyncio.run(_run(_parser().parse_args()))


if __name__ == "__main__":
    main()
