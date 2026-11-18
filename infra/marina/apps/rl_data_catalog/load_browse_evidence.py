# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Import sealed regional review/rollout shards into the private preview only."""

import argparse
import gzip
import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import requests
from marina.client import applet_query
from requests.adapters import HTTPAdapter
from rigging.auth import StaticTokenProvider
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.secrets import require_secret
from urllib3.util.retry import Retry

from infra.marina.apps.rl_data_catalog.load_reviews import load_content_reviews
from infra.marina.apps.rl_data_catalog.load_rollouts import load_rollouts
from infra.marina.apps.rl_data_catalog.prepare_preview import PREVIEW_APP_ID
from infra.marina.apps.rl_data_catalog.role_projection import ManifestReference, manifest_bytes


@contextmanager
def catalog_session():
    """Retry lost HTTP replies only for this loader's idempotent SQL/readback operations."""
    with requests.Session() as session:
        retry = Retry(total=3, connect=3, read=3, status=0, backoff_factor=0.5, allowed_methods={"GET", "POST"})
        session.mount("https://", HTTPAdapter(max_retries=retry))
        session.mount("http://", HTTPAdapter(max_retries=retry))
        yield session


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", nargs=2, action="append", required=True, metavar=("URI", "SHA256"))
    parser.add_argument("--load-id", required=True)
    parser.add_argument("--service-url", required=True)
    parser.add_argument("--auth-token-secret", required=True)
    parser.add_argument("--workers", type=int, choices=range(1, 9), required=True)
    parser.add_argument("--receipt", required=True)
    args = parser.parse_args()
    configure_coreweave_s3()
    if StoragePath(args.receipt).exists():
        raise FileExistsError(args.receipt)
    provider = StaticTokenProvider(require_secret(args.auth_token_secret))
    query = applet_query(args.service_url, PREVIEW_APP_ID, token_provider=provider)
    current = query("SELECT load_id FROM atlas_current WHERE name='latest'", {})["rows"][0]["load_id"]
    if current != args.load_id:
        raise ValueError("Selected catalog changed; do not load evidence into another data cut")
    tasks = []
    for uri, digest in args.index:
        index = json.loads(manifest_bytes(ManifestReference(uri=uri, sha256=digest)))
        if index["load_id"] != args.load_id:
            raise ValueError("Regional evidence was staged against another catalog load")
        tasks.extend((index["kind"], record) for record in index["receipts"] if record["status"] == "complete")
    if len({(kind, row["identity"]) for kind, row in tasks}) != len(tasks):
        raise ValueError("Duplicate evidence sets in the selected shard indexes")

    def load(task):
        kind, record = task
        staged = json.loads(gzip.decompress(manifest_bytes(ManifestReference.model_validate(record["reference"]))))
        with catalog_session() as session:
            task_query = applet_query(args.service_url, PREVIEW_APP_ID, session=session, token_provider=provider)
            if kind == "qc":
                result = load_content_reviews(staged, args.load_id, record["source_id"], task_query)
            else:
                result = load_rollouts(staged["measurement"], staged["metadata"], staged["rows"], task_query)
        receipt = {
            "kind": kind,
            "identity": record["identity"],
            "source_id": record["source_id"],
            "stage_reference": record["reference"],
            "result": result,
        }
        print(json.dumps(receipt), flush=True)
        return receipt

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(load, tasks))
    if query("SELECT load_id FROM atlas_current WHERE name='latest'", {})["rows"][0]["load_id"] != args.load_id:
        raise ValueError("Selected catalog changed while loading browser evidence")
    result = {"load_id": args.load_id, "app_id": PREVIEW_APP_ID, "indexes": args.index, "results": results}
    payload = (json.dumps(result, indent=2) + "\n").encode()
    output = StoragePath(args.receipt)
    output.write_bytes(payload)
    if output.read_bytes() != payload:
        raise ValueError("Regional load receipt readback differs")
    print(json.dumps({"complete": True, "receipt": args.receipt, "sets": len(results)}), flush=True)


if __name__ == "__main__":
    main()
