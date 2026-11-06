# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Operator-only OpenFGA model publication and authoritative standing relationships."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import requests


class OpenFGAOperator:
    def __init__(self, api_url: str, token_file: Path, *, ca_bundle: str | bool = True):
        self.api_url = api_url.rstrip("/")
        self.token_file = token_file
        self.ca_bundle = ca_bundle

    def request(self, path: str, payload: dict) -> dict:
        response = requests.post(
            f"{self.api_url}/{path}",
            headers={"Authorization": f"Bearer {self.token_file.read_text().strip()}"},
            json=payload,
            verify=self.ca_bundle,
            timeout=15,
            allow_redirects=False,
        )
        if response.status_code not in {200, 201, 204}:
            raise RuntimeError(f"OpenFGA operator request failed: {path}: HTTP {response.status_code}")
        return response.json() if response.content else {}

    def publish(self, name: str, model: dict) -> tuple[str, str]:
        store_id = self.request("stores", {"name": name})["id"]
        model_id = self.request(f"stores/{store_id}/authorization-models", model)["authorization_model_id"]
        return store_id, model_id

    def write(self, store_id: str, model_id: str, *, writes: list[dict], deletes: list[dict]) -> None:
        payload = {"authorization_model_id": model_id}
        if writes:
            payload["writes"] = {"tuple_keys": writes}
        if deletes:
            payload["deletes"] = {"tuple_keys": deletes}
        if writes or deletes:
            self.request(f"stores/{quote(store_id, safe='')}/write", payload)

    def reconcile(self, store_id: str, model_id: str, relationships: list[dict]) -> None:
        """Own every standing tuple in this dedicated store; runtime grants live elsewhere."""
        existing = []
        continuation = ""
        while True:
            result = self.request(
                f"stores/{quote(store_id, safe='')}/read", {"page_size": 100, "continuation_token": continuation}
            )
            existing.extend(row["key"] for row in result.get("tuples", []))
            continuation = result.get("continuation_token", "")
            if not continuation:
                break

        def key(row: dict) -> str:
            return json.dumps(row, sort_keys=True, separators=(",", ":"))

        current = {key(row): row for row in existing}
        desired = {key(row): row for row in relationships}
        writes = [row for item, row in desired.items() if item not in current]
        deletes = [row for item, row in current.items() if item not in desired]
        # OpenFGA limits one Write transaction to 100 tuple mutations.
        for offset in range(0, len(deletes), 100):
            self.write(store_id, model_id, writes=[], deletes=deletes[offset : offset + 100])
        for offset in range(0, len(writes), 100):
            self.write(store_id, model_id, writes=writes[offset : offset + 100], deletes=[])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--ca-file")
    parser.add_argument("--name")
    parser.add_argument("--store-id")
    parser.add_argument("--model-id")
    parser.add_argument("--model", type=Path, default=Path(__file__).parent / "policy/model.json")
    parser.add_argument("--relationships", type=Path, required=True)
    args = parser.parse_args()
    operator = OpenFGAOperator(args.url, args.token_file, ca_bundle=args.ca_file or True)
    if args.name:
        if args.store_id or args.model_id:
            parser.error("--name creates a new store; do not combine it with existing IDs")
        store_id, model_id = operator.publish(args.name, json.loads(args.model.read_text()))
    else:
        if not args.store_id or not args.model_id:
            parser.error("supply --name or both --store-id and --model-id")
        store_id, model_id = args.store_id, args.model_id
    operator.reconcile(store_id, model_id, json.loads(args.relationships.read_text()))
    print(
        json.dumps(
            {
                "store_id": store_id,
                "model_id": model_id,
                "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
