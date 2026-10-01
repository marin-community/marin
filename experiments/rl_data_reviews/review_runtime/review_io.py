# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local artifacts and stateless model calls shared by review stages."""

import dataclasses
import datetime
import enum
import hashlib
import importlib.metadata
import json
import shutil
import sys
import traceback
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def json_text(value: Any) -> str:
    return json.dumps(value, default=json_value, ensure_ascii=False, allow_nan=False)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json_text(value) + "\n")
    temporary.replace(path)


def append_event(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json_text({"at": utc_now(), **value}) + "\n")
        stream.flush()


def execution_error(error: Exception, root: Path, verifier_executed: bool) -> dict:
    """Persist a failed native execution with its traceback and verifier state."""
    append_event(
        root / "verifier-trace.jsonl",
        {
            "event": "execution_error",
            "type": type(error).__name__,
            "detail": str(error),
            "traceback": traceback.format_exc(),
        },
    )
    return {
        "verification": {
            "status": "error",
            "score": None,
            "passed": None,
            "reason": str(error),
            "diagnostics": {"exception_type": type(error).__name__},
        },
        "verifier_executed": verifier_executed,
        "done": False,
    }


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_completion(
    model: dict, messages: list[dict], directory: Path, parameters: dict, *, api_key: str | None
) -> dict:
    """Persist a fresh request and its full response without authentication headers."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = {**parameters, "model": model["name"], "messages": messages, "stream": False, "n": 1}
    write_json(directory / "request.json", payload)
    headers = {"Content-Type": "application/json"}
    if api_key is not None:
        headers["Authorization"] = "Bearer " + api_key
    request = urllib.request.Request(
        model["base_url"].rstrip("/") + "/chat/completions",
        data=json_text(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=model["timeout"]) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        (directory / "http-error.txt").write_bytes(error.read())
        raise
    (directory / "response.json").write_bytes(raw)
    result = json.loads(raw)
    if len(result["choices"]) != 1:
        raise ValueError("Expected exactly one model completion")
    return result


NATIVE_MODULE_PREFIXES = (
    "skyrl_gym.",
    "skyrl_train.trajectory_runners.",
    "harbor.verifier.",
    "tasktrove_verify.",
    "verifyit.",
)


def native_module(name: str) -> bool:
    return name == "verifyit" or name.startswith(NATIVE_MODULE_PREFIXES)


def verifyit_distribution() -> dict[str, str]:
    """Identify the installed verifier package and its immutable VCS source."""
    distribution = importlib.metadata.distribution("verifyit")
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is None:
        raise ValueError("Installed verifyit package has no source URL provenance")
    source = json.loads(direct_url)
    commit = source.get("vcs_info", {}).get("commit_id")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("Installed verifyit package has no immutable source commit")
    return {"version": distribution.version, "source_url": source["url"], "source_commit": commit}


@contextmanager
def native_calls():
    """Record native modules whose Python functions ran during this attempt."""
    called = set()
    previous = sys.getprofile()

    def profile(frame, event, _argument):
        if event == "call":
            name = frame.f_globals.get("__name__", "")
            if native_module(name):
                called.add(name)

    sys.setprofile(profile)
    try:
        yield called
    finally:
        sys.setprofile(previous)


def capture_native_sources(root: Path, called: set[str]) -> None:
    """Copy imported verifier modules so reviewers see the code actually loaded."""
    index = []
    verifyit = (
        verifyit_distribution()
        if any(name == "verifyit" or name.startswith("verifyit.") for name in sys.modules)
        else None
    )
    for name, module in sorted(sys.modules.copy().items()):
        if not native_module(name):
            continue
        origin = getattr(module, "__file__", None)
        if origin is None or not origin.endswith(".py"):
            continue
        source = Path(origin)
        destination = root / "native-code" / (name.replace(".", "/") + ".py")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        entry: dict[str, Any] = {
            "module": name,
            "origin": str(source),
            "path": str(destination.relative_to(root)),
            "sha256": digest(destination),
            "called_in_attempt": name in called,
        }
        if name == "verifyit" or name.startswith("verifyit."):
            entry["package"] = verifyit
        index.append(entry)
    write_json(root / "native-code-index.json", index)
