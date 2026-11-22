# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared file-batch transport for the Numina classifier and golden judge."""

import fcntl
import hashlib
import json
import os
import re
import shutil
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from experiments.post_training.task_curation.datasets.lean.numina_autoinformalization_quality import read_records

API = "https://generativelanguage.googleapis.com/v1beta/"
MAX_FILE_BYTES = 2_000_000_000


def save_json(path: Path, value: dict) -> None:
    """Replace local state atomically after flushing it to disk."""
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def api_request(path: str, body: dict | None = None) -> dict:
    request = Request(
        API + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"], "Content-Type": "application/json"},
    )
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def upload_file(path: Path, display_name: str) -> str:
    """Upload JSONL through Google's resumable Files API without generating model output."""
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise ValueError("Input file exceeds Google's 2 GB limit")
    request = Request(
        "https://generativelanguage.googleapis.com/upload/v1beta/files",
        data=json.dumps({"file": {"displayName": display_name}}).encode(),
        headers={
            "x-goog-api-key": os.environ["GEMINI_API_KEY"],
            "Content-Type": "application/json",
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(size),
            "X-Goog-Upload-Header-Content-Type": "application/jsonl",
        },
    )
    with urlopen(request, timeout=120) as response:
        upload_url = response.headers["X-Goog-Upload-URL"]
    parsed = urlparse(upload_url)
    if parsed.scheme != "https" or parsed.hostname != "generativelanguage.googleapis.com":
        raise ValueError("Unexpected Google upload URL")
    request = Request(
        upload_url,
        data=path.read_bytes(),
        headers={
            "Content-Length": str(size),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        },
    )
    with urlopen(request, timeout=300) as response:
        return json.load(response)["file"]["name"]


def run_batch(requests: dict[str, dict], model: str, directory: Path) -> dict[str, dict]:
    """Submit once, wait for completion, and return results keyed by source row.

    Saved requests and job state prevent resubmission after interruptions. An ambiguous
    submission stops with its display name recorded for reconciliation in Google AI Studio.
    """
    if not requests or not re.fullmatch(r"[A-Za-z0-9._-]+", model):
        raise ValueError("Batch requires source rows and a model ID")
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "state.json"
    serialized = "".join(
        json.dumps({"key": key, "request": request}, sort_keys=True) + "\n" for key, request in sorted(requests.items())
    )
    fingerprint = hashlib.sha256((model + "\n" + serialized).encode()).hexdigest()
    with (directory / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state: dict = json.loads(state_path.read_text()) if state_path.exists() else {"fingerprint": fingerprint}
        if state["fingerprint"] != fingerprint:
            raise ValueError("Existing batch has different inputs or settings; use another output path")
        result_path = directory / "responses.jsonl"
        if state.get("complete"):
            items = read_records(result_path)
        else:
            if state.get("submitting") and not state.get("job"):
                raise ValueError(
                    f"Submission uncertain for {state['display_name']}. Check Google AI Studio and "
                    f"record its job name as 'job' in {state_path} before resuming."
                )
            os.environ["GEMINI_API_KEY"]
            if not state.get("file"):
                path = directory / "requests.jsonl"
                path.write_text(serialized)
                state["display_name"] = f"numina-{fingerprint[:20]}"
                state["file"] = upload_file(path, state["display_name"])
                save_json(state_path, state)
            if not state.get("job"):
                info = api_request(state["file"])
                while info["state"] == "PROCESSING":
                    time.sleep(60)
                    info = api_request(state["file"])
                if info["state"] != "ACTIVE":
                    raise ValueError(f"Input file unavailable: {info['state']}")
                state["submitting"] = True
                save_json(state_path, state)
                job = api_request(
                    f"models/{model}:batchGenerateContent",
                    {"batch": {"displayName": state["display_name"], "inputConfig": {"fileName": state["file"]}}},
                )
                state["job"] = job["name"]
                save_json(state_path, state)
                print(f"Submitted {len(requests)} rows. Waiting for {state['job']}.", flush=True)
            operation = api_request(state["job"])
            while not operation.get("done"):
                time.sleep(60)
                operation = api_request(state["job"])
            save_json(directory / "completed.json", operation)
            if "error" in operation:
                raise ValueError(f"Batch failed: {operation['error']}")
            name = operation["response"]["responsesFile"]
            if not re.fullmatch(r"files/[A-Za-z0-9_-]+", name):
                raise ValueError("Unexpected output file name")
            request = Request(
                f"https://generativelanguage.googleapis.com/download/v1beta/{name}:download?alt=media",
                headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
            )
            temporary = result_path.with_suffix(".tmp")
            with urlopen(request, timeout=300) as response, temporary.open("wb") as stream:
                shutil.copyfileobj(response, stream)
            temporary.replace(result_path)
            items = read_records(result_path)
        results = {item["key"]: item for item in items}
        if len(results) != len(items) or results.keys() != requests.keys():
            raise ValueError("Missing, duplicate, or unexpected result keys")
        save_json(state_path, {**state, "complete": True})
        return results
