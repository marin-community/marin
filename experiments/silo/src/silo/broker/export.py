# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Copy a running broker's snapshots into a broker state document.

    python -m silo.broker.export --out s3://.../broker-state.json

Run it (as an Iris job on the broker's cluster: the broker has no off-cluster
address) BEFORE replacing a broker that predates persistence. The replacement,
started with ``--state-url`` pointing at the same document, then restores every
snapshot -- same names, ids, recipes and states -- instead of starting empty.

Reads only the public ``GET /snapshots`` API, so it works against old brokers --
including a jammed one, whose request pool queues every call: pages are large,
each request may wait --timeout seconds, and a failed page is retried.
Environment: SILO_API_TOKEN, and SILO_BROKER_RESOLVE_URL or SILO_BROKER_URL.
Refuses to overwrite an existing document unless --force.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from silo.auth import HEADER_API_TOKEN, bearer
from silo.broker.core import STATE_VERSION, snapshot_state_from_api
from silo.broker.state import open_state_store
from silo.http import HttpClient
from silo.model import utcnow

logger = logging.getLogger("silo.broker.export")


def _page(broker: HttpClient, page: int, page_size: int, attempts: int, timeout: float) -> dict:
    query = {"page": page, "limit": page_size}
    for attempt in range(attempts - 1):
        try:
            return broker.get("/snapshots", query=query, timeout=timeout)
        except Exception:
            logger.warning("page %d failed; retrying", page, exc_info=True)
            time.sleep(5 * (attempt + 1))
    return broker.get("/snapshots", query=query, timeout=timeout)


def fetch_snapshots(broker: HttpClient, *, page_size: int = 500, attempts: int = 5, timeout: float = 300) -> list[dict]:
    items: list[dict] = []
    page = 1
    while True:
        body = _page(broker, page, page_size, attempts, timeout)
        items.extend(body["items"])
        if page >= int(body["total_pages"]):
            return items
        page += 1


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="state document URL (local path or fsspec URL)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing state document")
    parser.add_argument("--page-size", type=int, default=500)
    parser.add_argument("--timeout", type=float, default=300, help="seconds each page request may wait")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stdout)

    url = os.environ.get("SILO_BROKER_URL")
    if not url:
        resolve = os.environ["SILO_BROKER_RESOLVE_URL"]
        url = HttpClient(resolve, timeout=30).get("/whoami")["url"]
    broker = HttpClient(url, headers={HEADER_API_TOKEN: bearer(os.environ["SILO_API_TOKEN"])}, timeout=120)

    store = open_state_store(args.out)
    assert store is not None
    if store.load() is not None and not args.force:
        sys.exit(f"{store.url} already holds broker state; pass --force to overwrite it")

    items = fetch_snapshots(broker, page_size=args.page_size, timeout=args.timeout)
    snapshots, skipped = [], 0
    for item in items:
        try:
            snapshots.append(snapshot_state_from_api(item))
        except (KeyError, TypeError) as error:
            skipped += 1
            logger.error("skipping snapshot %r: %s", item.get("name"), error)
    if not snapshots and items:
        sys.exit("no snapshot could be converted; refusing to write an empty state document")
    store.save(
        {"version": STATE_VERSION, "saved_at": utcnow(), "exported_from": url, "snapshots": snapshots, "hosts": []}
    )
    by_state: dict[str, int] = {}
    for snapshot in snapshots:
        by_state[snapshot["state"]] = by_state.get(snapshot["state"], 0) + 1
    print(f"exported {len(snapshots)} snapshot(s) from {url} to {store.url}: {by_state}; skipped {skipped}")


if __name__ == "__main__":
    main()
