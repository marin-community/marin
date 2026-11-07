# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Seed one synthetic KV-v2 canary and a separate, bounded read-only identity."""

import argparse
import hashlib
import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

import requests
from rigging.openbao import OpenBaoClient

from infra.auth.bootstrap import OperatorClient
from infra.auth.secret_bindings import SecretConfig

KV_MOUNT = "auth-proof"
CANARY_NAME = "task-canary"
CANARY_POLICY = "marin-task-canary"


@dataclass(frozen=True)
class CanarySecret:
    binding: SecretConfig
    expected_sha256: str


def seed_canary(client: OperatorClient, *, token_file: Path, ttl: int) -> CanarySecret:
    """Create only synthetic proof data; keep its native token in a private file."""
    if not 1 <= ttl <= 900:
        raise ValueError("Canary identity lifetime must be 1-900 seconds")
    mounts = client.request("GET", "sys/mounts")["data"]
    mount = mounts.get(KV_MOUNT + "/")
    if mount is None:
        client.request("POST", f"sys/mounts/{KV_MOUNT}", {"type": "kv", "options": {"version": "2"}})
    elif mount["type"] != "kv" or mount["options"].get("version") != "2":
        raise ValueError("The proof KV mount is occupied by a different secrets engine")
    data = {"canary": secrets.token_hex(32)}
    client.request("POST", f"{KV_MOUNT}/data/{CANARY_NAME}", {"data": data})
    client.request(
        "PUT",
        f"sys/policies/acl/{CANARY_POLICY}",
        {"policy": f'path "{KV_MOUNT}/data/{CANARY_NAME}" {{ capabilities = ["read"] }}'},
    )
    result = client.request(
        "POST",
        "auth/token/create",
        {
            "policies": [CANARY_POLICY],
            "no_default_policy": True,
            "ttl": f"{ttl}s",
            "explicit_max_ttl": f"{ttl}s",
            "renewable": False,
        },
    )
    token_file.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with token_file.open("x", opener=lambda path, flags: os.open(path, flags, 0o600)) as output:
        output.write(result["auth"]["client_token"])
    reader = OpenBaoClient(client.address, token_file, ca_bundle=client.ca_file or True)
    if reader.request("GET", f"/{KV_MOUNT}/data/{CANARY_NAME}")["data"]["data"] != data:
        raise ValueError("The read-only canary identity returned unexpected data")
    for path in (f"/{KV_MOUNT}/data/other", "/transit/keys/marin-grants"):
        try:
            reader.request("GET", path)
        except requests.HTTPError as error:
            if error.response.status_code != 403:
                raise ValueError("The canary identity did not receive an explicit access denial") from None
        else:
            raise ValueError("The canary identity can access an unapproved provider path")
    return CanarySecret(
        SecretConfig(name=CANARY_NAME, mount=KV_MOUNT, path=CANARY_NAME, token_file=token_file),
        hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True)
    parser.add_argument("--ca-file", required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--ttl", type=int, required=True)
    args = parser.parse_args()
    result = seed_canary(
        OperatorClient(args.address, os.environ["BAO_TOKEN"], args.ca_file), token_file=args.token_file, ttl=args.ttl
    )
    print(
        json.dumps({"binding": result.binding.model_dump(mode="json"), "expected_canary_sha256": result.expected_sha256})
    )


if __name__ == "__main__":
    main()
