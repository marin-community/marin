# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone AWS credential_process helper, mounted into task Pods.

This file must run with only Python's standard library: the task's environment
need not contain Iris, Rigging, Connect or protobuf.
"""

import argparse
import fcntl
import json
import os
import shlex
import ssl
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import Request, urlopen

TASK_CREDENTIAL_FILE_ENV = "IRIS_TASK_CREDENTIAL_FILE"
TASK_CREDENTIAL_DIR = "/var/run/iris/credentials"
TASK_CREDENTIAL_FILE = f"{TASK_CREDENTIAL_DIR}/session.json"
TASK_CREDENTIAL_HELPER = f"{TASK_CREDENTIAL_DIR}/credential_process.py"
TASK_AWS_CONFIG_FILE = f"{TASK_CREDENTIAL_DIR}/aws-config"
TASK_CA_FILE = "/var/run/iris/ca/ca.crt"
TASK_CREDENTIAL_CACHE = "/var/run/iris/credential-cache"
REFRESH_MARGIN = 900


@dataclass(frozen=True)
class TaskCredentialFile:
    controller: str
    attempt_uid: str
    refresh_capability: str = field(repr=False)
    profiles: dict[str, dict]
    ca_file: str = TASK_CA_FILE


def read_task_credential_file(path: str) -> TaskCredentialFile:
    return TaskCredentialFile(**json.loads(Path(path).read_text()))


def _rpc(config: TaskCredentialFile, method: str, body: dict, token: str = "") -> dict:
    headers = {"Content-Type": "application/json", "Connect-Protocol-Version": "1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        config.controller.rstrip("/") + "/iris.cluster.ControllerService/" + method,
        data=json.dumps(body).encode(),
        headers=headers,
    )
    context = ssl.create_default_context(cafile=config.ca_file)
    with urlopen(request, context=context, timeout=45) as response:
        return json.load(response)


def storage_credentials(path: str, profile: str, cache_dir: str = TASK_CREDENTIAL_CACHE) -> dict:
    """Return standard process credentials, sharing refreshes across local SDKs."""
    config = read_task_credential_file(path)
    if profile not in config.profiles:
        raise ValueError("Unknown task storage profile")
    root = Path(cache_dir) / str(os.getuid())
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    cache_key = f"{config.attempt_uid}-{profile}"
    if "/" in cache_key or ".." in cache_key:
        raise ValueError("Invalid credential cache key")
    with open(root / f"{cache_key}.lock", "a+") as lock:
        os.fchmod(lock.fileno(), 0o600)
        fcntl.flock(lock, fcntl.LOCK_EX)
        cache_path = root / f"{cache_key}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            if cached["expires_epoch"] > time.time() + REFRESH_MARGIN:
                return cached["credentials"]
        token = _rpc(
            config,
            "RenewTaskSession",
            {
                "attemptUid": config.attempt_uid,
                "refreshCapability": config.refresh_capability,
            },
        )["token"]
        response = _rpc(config, "GetTaskCredentials", {"profile": profile}, token)
        # iris.time.Timestamp wraps epoch milliseconds, encoded as an int64 string.
        expiry = int(response["expiresAt"]["epochMs"]) / 1000
        result = {
            "Version": 1,
            "AccessKeyId": response["accessKeyId"],
            "SecretAccessKey": response["secretAccessKey"],
            "SessionToken": response.get("sessionToken", ""),
            "Expiration": datetime.fromtimestamp(expiry, UTC).isoformat(),
        }
        temporary = root / f"{cache_key}.new"
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as output:
            json.dump({"expires_epoch": expiry, "credentials": result}, output)
        temporary.replace(cache_path)
        return result


def aws_config(config: TaskCredentialFile, path: str) -> str:
    """Return AWS config sections that refresh each granted profile through the task helper."""
    sections = []
    for name, profile in config.profiles.items():
        command = shlex.join(["python3", TASK_CREDENTIAL_HELPER, "--file", path, "--profile", name])
        sections.append(
            f"[profile {name}]\ncredential_process = {command}\nregion = {profile['region']}\n"
            f"endpoint_url = {profile['endpoint']}\ns3 =\n    addressing_style = {profile['addressing_style']}\n"
        )
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=os.environ.get(TASK_CREDENTIAL_FILE_ENV, TASK_CREDENTIAL_FILE))
    parser.add_argument("--profile")
    parser.add_argument("--cache-dir", default=TASK_CREDENTIAL_CACHE)
    args = parser.parse_args()
    if not args.profile:
        parser.error("--profile is required")
    # The SDK reads its credential_process response from the private stdout pipe.
    print(json.dumps(storage_credentials(args.file, args.profile, args.cache_dir)))


if __name__ == "__main__":
    main()
