#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Post one-way notifications to Marin Discord channels via webhooks.

Webhook URLs are resolved per channel from, in order:
1. ``DISCORD_WEBHOOK_<CHANNEL>`` environment variable (e.g.
   ``DISCORD_WEBHOOK_INTERNAL_DISCUSS``). GH Actions injects these from repo
   secrets.
2. ``gcloud secrets versions access`` against
   ``marin-discord-webhook-<channel>`` in the active gcloud project. This is
   the local-CLI fallback so users don't have to manage env vars.

CLI:
    uv run python scripts/ops/discord.py -c internal-discuss -m "hello"
    echo "release notes..." | uv run python scripts/ops/discord.py -c code-review

Library:
    from scripts.ops.discord import post
    post("code-review", "PR #123 ready for review")
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import urllib.request

CHANNELS = ("internal-discuss", "code-review")
GCLOUD_SECRET_PREFIX = "marin-discord-webhook-"

log = logging.getLogger(__name__)


def _env_var_for(channel: str) -> str:
    return f"DISCORD_WEBHOOK_{channel.upper().replace('-', '_')}"


def _webhook_url(channel: str) -> str:
    assert channel in CHANNELS, f"unknown channel {channel!r}; known: {CHANNELS}"
    if url := os.environ.get(_env_var_for(channel)):
        return url.strip()
    secret_name = f"{GCLOUD_SECRET_PREFIX}{channel}"
    log.debug("env var unset; pulling %s from gcloud secret manager", secret_name)
    try:
        return subprocess.check_output(
            ["gcloud", "secrets", "versions", "access", "latest", f"--secret={secret_name}"],
            text=True,
            stderr=subprocess.PIPE,
        ).strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"could not resolve webhook for channel {channel!r}: env var "
            f"{_env_var_for(channel)} unset and gcloud secret {secret_name!r} "
            f"unreadable ({e.stderr.strip()})"
        ) from e


def post(channel: str, message: str, *, username: str | None = None) -> None:
    """Post ``message`` to ``channel``. Raises on non-2xx response."""
    if not message.strip():
        raise ValueError("refusing to post empty message")
    payload: dict[str, str] = {"content": message}
    if username:
        payload["username"] = username
    req = urllib.request.Request(
        _webhook_url(channel),
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            # Discord's edge rejects requests with the default Python urllib UA.
            "User-Agent": "marin-discord-notifier (+https://github.com/marin-community/marin)",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        if resp.status >= 300:
            raise RuntimeError(f"discord webhook {channel!r} returned {resp.status}: {resp.read()!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("-c", "--channel", required=True, choices=CHANNELS)
    p.add_argument("-m", "--message", help="message text; reads stdin if omitted")
    p.add_argument("--username", help="override the webhook's display name")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")
    msg = args.message if args.message is not None else sys.stdin.read()
    post(args.channel, msg, username=args.username)


if __name__ == "__main__":
    main()
