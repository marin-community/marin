# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gzip
import io
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import draccus
import fsspec
import requests
import zstandard as zstd
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_limiter import Limiter
from pyarrow.parquet import ParquetFile

app = Flask(__name__, static_folder="build")
SERVER_ROOT = Path(__file__).resolve().parent

CLOUD_STORAGE_PREFIXES = ("gs://", "s3://")

limiter = Limiter(app)


@dataclass(frozen=True)
class ServerConfig:
    """
    Specifies how to launch a data browser (e.g., what files to expose and how).
    Any sort of permissions and throttling policies should be specified here.
    """

    root_paths: list[str]
    """Paths (and their descendents) to allow access to."""

    blocked_paths: list[str] | None = None
    """Paths to explicitly block access to, even if under root_paths."""

    max_size: int | None = None
    """Maximum size of a file to read in bytes (for text files), when reading a
    JSON or downloading a file."""

    max_lines: int | None = None
    """Maximum number of lines to read from a jsonl or parquet file.
    This is the maximum number of lines that will be read in a single request,
    which means that even if we're only requesting a single line, if the portion
    of the file requested is past the first `max_lines` lines, we would exceed
    this limit."""

    port: int | None = None
    """Port to bind the Flask development server to when running `main` directly. Defaults to 5000 in debug."""


class Server:
    """
    The only state that the server has right now is the filesystems.
    Note that utils.py has fsspec utilities that we could call directly which
    would simplify the code, but the filesystem objects wouldn't be cached.
    """

    def __init__(self, config: ServerConfig):
        self.config = config

        # Lazily instantiate remote filesystems to avoid triggering cloud
        # auth during local-only development.
        self.fs_cache = {
            None: fsspec.filesystem("local"),
        }

    def fs(self, path: str):
        """Automatically figure out the filesystem to use based on the `path`."""
        protocol, _ = fsspec.core.split_protocol(path)
        if protocol not in self.fs_cache:
            if protocol == "gs":
                self.fs_cache[protocol] = fsspec.filesystem("gcs")
            elif protocol == "s3":
                self.fs_cache[protocol] = fsspec.filesystem("s3")
            else:
                self.fs_cache[protocol] = fsspec.filesystem("local")
        return self.fs_cache[protocol]


server: Server | None = None


def resolve_path(path: str) -> str:
    """Resolve a path to an absolute path, except for cloud storage paths."""
    return path if path.startswith(CLOUD_STORAGE_PREFIXES) else os.path.realpath(path)


def canonicalize_request_path(path: str) -> str:
    """Convert user-provided local paths into absolute paths under allowed roots."""

    if path.startswith(CLOUD_STORAGE_PREFIXES):
        return path

    if os.path.isabs(path):
        return os.path.realpath(path)

    normalized = path
    while normalized.startswith("./"):
        normalized = normalized[2:]

    roots = server.config.root_paths if server and server.config else []
    for root_path in roots:
        base = os.path.basename(root_path.rstrip("/"))
        # Accept either `basename` or `basename/...` as shorthand for the root
        if normalized == base:
            return root_path
        if normalized.startswith(f"{base}/") or normalized.startswith(f"{base}\\"):
            suffix = normalized[len(base) + 1 :]
            return os.path.realpath(os.path.join(root_path, suffix))

    # Fall back to resolving relative to the data_browser directory
    return os.path.realpath(str((SERVER_ROOT / normalized).resolve()))


def list_files(path: str) -> dict:
    """List all files in the given path."""
    protocol, _ = fsspec.core.split_protocol(path)  # e.g., "gs"
    files = server.fs(path).ls(path, detail=True, refresh=True)

    # If path = "gs://", the file names listed don't have the "gs://" prefix
    # (but still represents an absolute path), so we add it in.
    def name_to_path(name: str) -> str:
        if protocol is not None:
            return f"{protocol}://{name}"
        return name

    # Replace file with path and filter out blocked paths
    files = []
    for file in server.fs(path).ls(path, detail=True, refresh=True):
        file_path = name_to_path(file["name"])
        # Only include files that are not blocked
        if has_permissions(file_path, server.config.root_paths, server.config.blocked_paths):
            files.append({"path": file_path, **file})

    return {
        "type": "directory",
        "files": files,
    }


def read_json_file(path: str) -> dict:
    """Reads a JSON file."""
    with server.fs(path).open(path) as f:
        if server.config.max_size is None:
            raw = f.read()
        else:
            raw = f.read(server.config.max_size)  # Don't OOM on huge files
            if len(raw) == server.config.max_size:
                return {
                    "type": "json",
                    "error": f"File too large (exceeded {server.config.max_size} bytes)",
                }
        data = json.loads(raw)

    return {
        "type": "json",
        "data": data,
    }


def is_too_many_lines(offset: int, count: int) -> bool:
    return server.config.max_lines is not None and offset + count > server.config.max_lines


def is_too_large(size: int) -> bool:
    return server.config.max_size is not None and size > server.config.max_size


def read_executor_status_file(path: str, offset: int, count: int) -> dict:
    # Ensure we only read a max of server.config.max_lines lines
    if is_too_many_lines(offset, count):
        return {"error": f"Only {server.config.max_lines} lines are allowed to be read at a time"}

    # Read the lines from the file, enforcing max_size if configured.
    with server.fs(path).open(path, "r") as f:
        lines: list[str] = []
        if server.config.max_size is None:
            for raw_line in f:
                line = raw_line.strip()
                if line:
                    lines.append(line)
        else:
            bytes_read = 0
            for raw_line in f:
                # Approximate byte size using UTF-8 encoding
                bytes_read += len(raw_line.encode("utf-8"))
                if is_too_large(bytes_read):
                    return {
                        "type": "jsonl",
                        "error": f"File too large (exceeded {server.config.max_size} bytes)",
                    }
                line = raw_line.strip()
                if line:
                    lines.append(line)

    if not lines:
        return {"type": "jsonl", "items": []}

    # new format: a single status token like SUCCESS/RUNNING/etc.
    if len(lines) == 1 and not lines[0].startswith("{"):
        if offset > 0:
            return {"type": "jsonl", "items": []}
        return {"type": "jsonl", "items": [{"status": lines[0]}]}

    # legacy format: JSON-lines event log.
    events: list[dict] = []
    for line in lines[offset : offset + count]:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return {"type": "jsonl", "items": events}


def read_text_file(
    path: str, get_json: bool, offset: int, count: int, gzipped: bool = False, zstded: bool = False
) -> dict:
    """
    Reads a range of lines (offset to offset + count) from a text file (possibly compressed using gzip or zstd).
    Interpret each line as a JSON if `get_json` is set.
    """
    # Ensure we only read a max of server.config.max_lines lines
    if is_too_many_lines(offset, count):
        return {"error": f"Only {server.config.max_lines} lines are allowed to be read at a time"}
    with server.fs(path).open(path, "rb") as f:
        # Unzip
        if gzipped:
            f = gzip.GzipFile(fileobj=f)
        if zstded:
            f = zstd.ZstdDecompressor().stream_reader(f)

        # Assume text (Unicode)
        f = io.TextIOWrapper(f, encoding="utf-8", errors="ignore")

        # Seek to line `offset` (note: O(offset) time!)
        for _ in range(offset):
            if not f.readline():
                break

        # Read `count` lines
        items = []
        for _ in range(count):
            line = f.readline()
            if not line:
                break

            items.append(json.loads(line) if get_json else line)

    return {
        "type": "jsonl" if get_json else "text",
        "items": items,
    }


def read_parquet_file(path: str, offset: int, count: int) -> dict:
    """Reads a range of records (offset to offset + count) from a parquet file."""
    # Ensure we only read a max of server.config.max_lines lines
    if is_too_many_lines(offset, count):
        return {"error": f"Only {server.config.max_lines} lines are allowed to be read at a time"}

    pf = ParquetFile(path)
    # Note: can make this more efficient by skipping the first offset without reading into memory
    rows = next(pf.iter_batches(batch_size=offset + count))[offset:]
    return {
        "type": "parquet",
        "items": json.loads(rows.to_pandas().to_json(orient="records")),
    }


def has_permissions(path: str, root_paths: list[str], blocked_paths: list[str] | None = None) -> bool:
    """Returns whether the user can access `path` according to the permissions."""

    # Check if path is blocked
    if blocked_paths:
        for blocked_path_pattern in blocked_paths:
            # For cloud storage paths, check if the path starts with the blocked pattern
            if path.startswith(CLOUD_STORAGE_PREFIXES):
                if path.startswith(blocked_path_pattern):
                    return False
            else:
                # For local paths, use os.path.commonpath for consistent logic
                if not os.path.isabs(path):
                    # Don't allow relative paths
                    return False
                try:
                    if os.path.commonpath([path, blocked_path_pattern]) == blocked_path_pattern:
                        # The path is blocked if it's a subpath of the blocked pattern
                        return False
                except ValueError:
                    # Paths don't have a common base, so not blocked by this pattern
                    continue

    # Check if path is allowed under root_paths
    for allowed_path in root_paths:
        # For cloud storage paths, check if the resolved path starts with the allowed path
        if path.startswith(CLOUD_STORAGE_PREFIXES):
            if path.startswith(allowed_path):
                return True
        else:
            if not os.path.isabs(path):
                # Don't allow relative paths
                return False
            if os.path.commonpath([path, allowed_path]) == allowed_path:
                # The path is allowed if it's a subpath of the allowed path
                return True
    return False


# Sanity checks to ensure has_permissions works as expected
assert has_permissions("gs://marin-us-central2/test/test.txt", ["gs://marin-us-central2"])
assert not has_permissions("gs://marin-us-central2/test/test.txt", [])
assert not has_permissions("/etc/hosts", [])
assert has_permissions("/app/var/test", ["/app/var"])
assert not has_permissions("/app/various", ["/app/var"])
assert not has_permissions("../app/var", ["/app/var"])
assert not has_permissions("../etc/hosts", ["/etc"])

# Test blocked paths functionality
assert has_permissions(
    "gs://marin-us-central2/allowed/file.txt", ["gs://marin-us-central2"], ["gs://marin-us-central2/blocked"]
)
assert not has_permissions(
    "gs://marin-us-central2/blocked/file.txt", ["gs://marin-us-central2"], ["gs://marin-us-central2/blocked"]
)
assert not has_permissions(
    "gs://marin-us-central2/blocked/subdir/file.txt", ["gs://marin-us-central2"], ["gs://marin-us-central2/blocked"]
)
# Test that both with and without trailing slash work
assert not has_permissions(
    "gs://marin-us-central2/blocked", ["gs://marin-us-central2"], ["gs://marin-us-central2/blocked"]
)
assert not has_permissions(
    "gs://marin-us-central2/blocked/", ["gs://marin-us-central2"], ["gs://marin-us-central2/blocked"]
)


@app.route("/api/config", methods=["GET"])
def config():
    # Later: only send the necessary parts of config
    return jsonify(asdict(server.config))


@app.route("/api/download", methods=["GET"])
@limiter.limit("10 per minute")
def download():
    path = request.args.get("path")
    if not path:
        return jsonify({"error": "No path specified"})
    sanitized_path = canonicalize_request_path(path)
    if not has_permissions(sanitized_path, server.config.root_paths, server.config.blocked_paths):
        return jsonify({"error": f"No permission to access: {path}"})
    if not server.fs(sanitized_path).exists(sanitized_path):
        return jsonify({"error": f"Path does not exist: {path}"})

    if is_too_large(server.fs(sanitized_path).size(sanitized_path)):
        return jsonify({"error": f"File too large (exceeded {server.config.max_size} bytes)"})

    try:
        file_handle = server.fs(sanitized_path).open(sanitized_path, "rb")
        return Response(file_handle, content_type="application/octet-stream")
    except ValueError as e:
        return jsonify({"error": str(e)})


@app.route("/api/view", methods=["GET"])
@limiter.limit("60 per minute")
def view():
    path = request.args.get("path")
    offset = int(request.args.get("offset", 0))
    count = int(request.args.get("count", 1))

    try:
        if not path:
            return jsonify({"error": "No path specified"})
        sanitized_path = canonicalize_request_path(path)
        if not has_permissions(sanitized_path, server.config.root_paths, server.config.blocked_paths):
            return jsonify({"error": f"No permission to access: {path}"})
        if not server.fs(sanitized_path).exists(sanitized_path):
            return jsonify({"error": f"Path does not exist: {path}"})

        # Directory - check permissions before listing
        if server.fs(sanitized_path).isdir(sanitized_path):
            return jsonify(list_files(sanitized_path))

        # jsonl files
        if sanitized_path.endswith(".jsonl"):
            return jsonify(read_text_file(path=sanitized_path, get_json=True, offset=offset, count=count))
        if any(sanitized_path.endswith(ext) for ext in [".json.gz", ".ndjson.gz", ".jsonl.gz"]):
            # json.gz is because Dolma files are named like this (should be jsonl.gz)
            return jsonify(read_text_file(path=sanitized_path, get_json=True, offset=offset, count=count, gzipped=True))
        if sanitized_path.endswith(".jsonl.zstd") or sanitized_path.endswith(".jsonl.zst"):
            return jsonify(read_text_file(path=sanitized_path, get_json=True, offset=offset, count=count, zstded=True))

        # parquet files (what Hugging Face datasets use)
        if sanitized_path.endswith(".parquet"):
            return jsonify(read_parquet_file(path=sanitized_path, offset=offset, count=count))

        # json files (.SUCCESS is used in Marin to keep track of progress)
        if sanitized_path.endswith(".json") or sanitized_path.endswith(".SUCCESS"):
            return jsonify(read_json_file(sanitized_path))

        # .executor_info files are also rendered as json
        if sanitized_path.endswith(".executor_info"):
            return jsonify(read_json_file(sanitized_path))

        # render .executor_status files as jsonl
        # Note: there are two formats of .executor_status files; we convert both to jsonl.
        if sanitized_path.endswith(".executor_status"):
            return jsonify(read_executor_status_file(path=sanitized_path, offset=offset, count=count))

        # Assume text file (treated as a list of lines)
        return jsonify(read_text_file(path=sanitized_path, gzipped=False, get_json=False, offset=offset, count=count))

    except Exception as e:
        print(f"EXCEPTION: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)})


@app.route("/")
@app.route("/view")
@app.route("/view/")
@app.route("/experiment")
@app.route("/experiment/")
def serve():
    if os.environ.get("DEV") == "true":
        return proxy_to_dev_server("")
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def static_proxy(path):
    if os.environ.get("DEV") == "true":
        return proxy_to_dev_server(path)
    return send_from_directory(app.static_folder, path)


def proxy_to_dev_server(path):
    """Proxy requests to the development server running on port 3000

    This implements a basic HTTP reverse proxy pattern where we forward the original request
    to the target server (dev server) and then strip out hop-by-hop headers that should not
    be forwarded (these headers are meant only for a single transport-level connection).
    See https://www.rfc-editor.org/rfc/rfc2616#section-13.5.1
    """
    try:
        resp = requests.get(f"http://localhost:3000/{path}")
    except requests.exceptions.RequestException:
        return Response(
            "React dev server not running on http://localhost:3000.\n"
            "Run: cd data_browser && uv run python run-dev.py --config conf/local.conf\n",
            status=502,
            content_type="text/plain; charset=utf-8",
        )
    excluded_headers = ["content-encoding", "content-length", "transfer-encoding", "connection"]
    headers = [(name, value) for (name, value) in resp.raw.headers.items() if name.lower() not in excluded_headers]
    return Response(resp.content, resp.status_code, headers)


def standardize_config(config: ServerConfig) -> ServerConfig:
    """Replace relative paths with absolute paths, except for cloud storage paths."""
    absolute_root_paths = [resolve_path(path) for path in config.root_paths]
    return replace(config, root_paths=absolute_root_paths)


@draccus.wrap()
def main(config: ServerConfig):
    print("ServerConfig:", config)

    config = standardize_config(config)

    global server
    server = Server(config)

    debug = os.environ.get("DEV") == "true"
    assert debug, "This function must be run in debug mode"
    port = config.port if config.port is not None else (5000 if debug else 80)
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    main()
