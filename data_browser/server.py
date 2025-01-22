import gzip
import io
import json
import os
from dataclasses import asdict, dataclass, replace

import draccus
import fsspec
import requests
import zstandard as zstd
from flask import Flask, Response, jsonify, request, send_from_directory
from pyarrow.parquet import ParquetFile

app = Flask(__name__, static_folder="build")

CLOUD_STORAGE_PREFIXES = ("gs://", "s3://")


@dataclass(frozen=True)
class ServerConfig:
    """
    Specifies how to launch a data browser (e.g., what files to expose and how).
    Any sort of permissions and throttling policies should be specified here.
    """

    root_paths: list[str]
    """Paths (and their descendents) to allow access to."""


class Server:
    """
    The only state that the server has right now is the filesystems.
    Note that utils.py has fsspec utilities that we could call directly which
    would simplify the code, but the filesystem objects wouldn't be cached.
    """

    def __init__(self, config: ServerConfig):
        self.config = config

        self.fs_cache = {
            None: fsspec.filesystem("local"),
            "gs": fsspec.filesystem("gcs"),
            "s3": fsspec.filesystem("s3"),
        }

    def fs(self, path: str):
        """Automatically figure out the filesystem to use based on the `path`."""
        protocol, _ = fsspec.core.split_protocol(path)
        return self.fs_cache[protocol]


server: Server | None = None


def resolve_path(path: str) -> str:
    """Resolve a path to an absolute path, except for cloud storage paths."""
    return path if path.startswith(CLOUD_STORAGE_PREFIXES) else os.path.realpath(path)


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

    # Replace file with path
    files = [{"path": name_to_path(file["name"]), **file} for file in files]

    return {
        "type": "directory",
        "files": files,
    }


def read_json_file(path: str) -> dict:
    """Reads a JSON file."""
    MAX_BYTES = 100 * 1024 * 1024
    with server.fs(path).open(path) as f:
        raw = f.read(MAX_BYTES)  # Don't OOM on huge files
        if len(raw) == MAX_BYTES:
            return {
                "type": "json",
                "error": f"File too large (exceeded {MAX_BYTES} bytes)",
            }
        data = json.loads(raw)

    return {
        "type": "json",
        "data": data,
    }


def read_text_file(
    path: str, get_json: bool, offset: int, count: int, gzipped: bool = False, zstded: bool = False
) -> dict:
    """
    Reads a range of lines (offset to offset + count) from a text file (possibly compressed using gzip or zstd).
    Interpret each line as a JSON if `get_json` is set.
    """
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
    pf = ParquetFile(path)
    # Note: can make this more efficient by skipping the first offset without reading into memory
    rows = next(pf.iter_batches(batch_size=offset + count))[offset:]
    return {
        "type": "parquet",
        "items": json.loads(rows.to_pandas().to_json(orient="records")),
    }


def has_permissions(path: str, root_paths: list[str]) -> bool:
    """Returns whether the user can access `path` according to the permissions."""
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


@app.route("/api/config", methods=["GET"])
def config():
    # Later: only send the necessary parts of config
    return jsonify(asdict(server.config))


@app.route("/api/download", methods=["GET"])
def download():
    path = request.args.get("path")
    if not path:
        return jsonify({"error": "No path specified"})
    if not has_permissions(path, server.config.root_paths):
        return jsonify({"error": f"No permission to access: {path}"})
    if not server.fs(path).exists(path):
        return jsonify({"error": f"Path does not exist: {path}"})
    
    # Stream the file directly from cloud storage or the filesystem
    return Response(server.fs(path).open(path, "rb"), content_type='application/octet-stream')


@app.route("/api/view", methods=["GET"])
def view():
    path = request.args.get("path")
    offset = int(request.args.get("offset", 0))
    count = int(request.args.get("count", 1))

    try:
        if not path:
            return jsonify({"error": "No path specified"})
        if not has_permissions(path, server.config.root_paths):
            return jsonify({"error": f"No permission to access: {path}"})
        if not server.fs(path).exists(path):
            return jsonify({"error": f"Path does not exist: {path}"})

        # Directory
        if server.fs(path).isdir(path):
            return jsonify(list_files(path))

        # jsonl files
        if path.endswith(".jsonl"):
            return jsonify(read_text_file(path=path, get_json=True, offset=offset, count=count))
        if path.endswith(".json.gz") or path.endswith(".ndjson.gz") or path.endswith(".jsonl.gz"):
            # json.gz is because Dolma files are named like this (should be jsonl.gz)
            return jsonify(read_text_file(path=path, get_json=True, offset=offset, count=count, gzipped=True))
        if path.endswith(".jsonl.zstd") or path.endswith(".jsonl.zst"):
            return jsonify(read_text_file(path=path, get_json=True, offset=offset, count=count, zstded=True))

        # parquet files (what Hugging Face datasets use)
        if path.endswith(".parquet"):
            return jsonify(read_parquet_file(path=path, offset=offset, count=count))

        # json files (.SUCCESS is used in Marin to keep track of progress)
        if path.endswith(".json") or path.endswith(".SUCCESS"):
            return jsonify(read_json_file(path))

        # .executor_info files are also rendered as json
        if path.endswith(".executor_info"):
            return jsonify(read_json_file(path))

        # render .executor_status files as jsonl
        if path.endswith(".executor_status"):
            return jsonify(read_text_file(path=path, get_json=True, offset=offset, count=count))

        # Assume text file (treated as a list of lines)
        return jsonify(read_text_file(path=path, gzipped=False, get_json=False, offset=offset, count=count))

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
    resp = requests.get(f"http://localhost:3000/{path}")
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
    app.run(host="0.0.0.0", port=5000 if debug else 80, debug=debug)


if __name__ == "__main__":
    main()
