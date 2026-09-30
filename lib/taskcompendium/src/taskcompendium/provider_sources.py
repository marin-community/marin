# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage pinned Python tool providers without fetching source during a trial.

Installed providers use ``python:module:Class``. Git providers use
``python+git+https://host/repo@<40-hex-sha>:module:Class``. The trusted export
caller supplies a checkout of that exact commit; trials use only staged files.
"""

import ast
import base64
import hashlib
import importlib
import importlib.util
import inspect
import json
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from taskcompendium.path_validation import validate_relative_file_path, validate_relative_file_paths

PROVIDER_SOURCES_DIR = "provider_sources"
SOURCE_MANIFEST = ".taskcompendium-provider-manifest.json"
MAX_PROVIDER_FILE_BYTES = 64 * 1024 * 1024
MAX_PROVIDER_SOURCE_BYTES = 256 * 1024 * 1024
MAX_PROVIDER_FILES = 4096
MAX_PROVIDER_MANIFEST_BYTES = 2 * 1024 * 1024
_source_cache: dict[str, Path] = {}
_source_cache_dir: tempfile.TemporaryDirectory[str] | None = None
_source_cache_lock = threading.RLock()


@dataclass(frozen=True)
class GitProviderLocator:
    """A repository commit and Python class named by a provider binding."""

    url: str
    commit: str
    module: str
    class_name: str


@dataclass(frozen=True)
class GitSourceFile:
    path: str
    mode: str
    blob: str


def parse_git_provider(provider: str) -> GitProviderLocator | None:
    """Parse a pinned HTTPS Git provider reference, or return None for Python imports."""
    if not provider.startswith("python+git+"):
        return None
    parts = provider.removeprefix("python+git+").rsplit(":", maxsplit=2)
    if len(parts) != 3:
        raise ValueError("Git provider requires a repository, module, and class")
    source, module, class_name = parts
    url, at, commit = source.rpartition("@")
    parsed = urlsplit(url)
    if (
        not at
        or parsed.scheme != "https"
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or not parsed.path.strip("/")
        or any(character.isspace() for character in url)
        or "@" in parsed.path
        or parsed.query
        or parsed.fragment
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
        or not class_name.isidentifier()
        or not all(part.isidentifier() for part in module.split("."))
    ):
        raise ValueError("Git provider must be python+git+https://host/repo@<40-hex-sha>:module:Class")
    return GitProviderLocator(url=url, commit=commit, module=module, class_name=class_name)


def _git(checkout: Path, *args: str) -> bytes:
    result = subprocess.run(("git", "-C", str(checkout), *args), capture_output=True, check=False, timeout=60)
    if result.returncode:
        raise ValueError(
            f"Git provider source is invalid: git {' '.join(args)}: {result.stderr.decode(errors='replace')}"
        )
    return result.stdout


def _canonical_url(url: str) -> str:
    if url.startswith("git@") and ":" in url:
        host, path = url.removeprefix("git@").split(":", maxsplit=1)
        url = f"https://{host}/{path}"
    elif url.startswith("ssh://git@"):
        url = "https://" + url.removeprefix("ssh://git@")
    return url.removesuffix(".git").rstrip("/")


def _git_object_digest(kind: str, content: bytes) -> bytes:
    header = f"{kind} {len(content)}\0".encode()
    return hashlib.sha1(header + content, usedforsecurity=False).digest()


def _git_tree_digest(files: list[tuple[str, str, bytes]]) -> bytes:
    root: dict[str, object] = {}
    for path, mode, blob in files:
        node = root
        parts = path.split("/")
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            assert isinstance(child, dict)
            node = child
        node[parts[-1]] = (mode, blob)

    def hash_node(node: dict[str, object]) -> bytes:
        content = bytearray()
        for name, value in sorted(node.items(), key=lambda item: item[0] + ("/" if isinstance(item[1], dict) else "")):
            if isinstance(value, dict):
                mode, digest = "40000", hash_node(value)
            else:
                assert isinstance(value, tuple)
                mode, digest = value
            content.extend(mode.encode() + b" " + name.encode() + b"\0" + digest)
        return _git_object_digest("tree", bytes(content))

    return hash_node(root)


def validate_git_provider_checkout(provider: str, checkout: Path) -> GitProviderLocator:
    """Verify a trusted caller's local checkout against a Git provider locator."""
    locator = parse_git_provider(provider)
    if locator is None:
        raise ValueError("Provider is not Git-pinned")
    checkout = checkout.resolve(strict=True)
    if (
        not checkout.is_dir()
        or Path(_git(checkout, "rev-parse", "--show-toplevel").decode().strip()).resolve() != checkout
    ):
        raise ValueError("Git provider source must be the repository root")
    if _git(checkout, "rev-parse", "HEAD").decode().strip() != locator.commit:
        raise ValueError("Git provider checkout differs from pinned commit")
    if _canonical_url(_git(checkout, "remote", "get-url", "origin").decode().strip()) != _canonical_url(locator.url):
        raise ValueError("Git provider checkout origin differs from pinned URL")
    if _git(checkout, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Git provider checkout must be clean")
    return locator


def _source_files(checkout: Path, commit: str) -> list[GitSourceFile]:
    entries = _git(checkout, "ls-tree", "-r", "-z", commit).split(b"\0")
    files: list[GitSourceFile] = []
    for entry in entries:
        if not entry:
            continue
        metadata, path_bytes = entry.split(b"\t", maxsplit=1)
        mode, kind, blob = metadata.decode().split()
        path = path_bytes.decode("utf-8")
        if "__pycache__" in Path(path).parts or path.endswith((".pyc", ".pyo")):
            raise ValueError(f"Git provider contains generated Python cache: {path}")
        if mode not in {"100644", "100755"} or kind != "blob":
            raise ValueError(f"Git provider contains a symlink, submodule, or special file: {path}")
        files.append(GitSourceFile(path=path, mode=mode, blob=blob))
    if len(files) > MAX_PROVIDER_FILES:
        raise ValueError("Git provider source exceeds file limit")
    validate_relative_file_paths(file.path for file in files)
    return files


def stage_git_provider(provider: str, checkout: Path, destination: Path) -> None:
    """Copy pinned Git blobs into a private task package with a digest manifest."""
    locator = validate_git_provider_checkout(provider, checkout)
    files = _source_files(checkout, locator.commit)
    validate_relative_file_paths([*(file.path for file in files), SOURCE_MANIFEST])
    destination.mkdir(parents=True, exist_ok=False)
    total_bytes = 0
    manifest_files: list[dict[str, object]] = []
    package_name = locator.module.split(".")[0]
    for file in files:
        path, mode, blob = file.path, file.mode, file.blob
        size = int(_git(checkout, "cat-file", "-s", blob).decode())
        total_bytes += size
        if size > MAX_PROVIDER_FILE_BYTES or total_bytes > MAX_PROVIDER_SOURCE_BYTES:
            raise ValueError("Git provider source exceeds size limits")
        payload = _git(checkout, "cat-file", "blob", blob)
        if len(payload) != size:
            raise ValueError(f"Git provider blob length differs: {path}")
        if path.endswith(".py") and path.startswith((f"src/{package_name}/", f"{package_name}/")):
            for node in ast.walk(ast.parse(payload, filename=path)):
                if isinstance(node, ast.Import) and any(
                    imported.name == package_name or imported.name.startswith(f"{package_name}.")
                    for imported in node.names
                ):
                    raise ValueError(f"Git provider must use relative imports within its package: {path}")
                if (
                    isinstance(node, ast.ImportFrom)
                    and node.level == 0
                    and node.module is not None
                    and (node.module == package_name or node.module.startswith(f"{package_name}."))
                ):
                    raise ValueError(f"Git provider must use relative imports within its package: {path}")
        target = destination.joinpath(*path.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        executable = mode == "100755"
        target.chmod(0o755 if executable else 0o644)
        manifest_files.append(
            {"path": path, "sha256": hashlib.sha256(payload).hexdigest(), "git_blob": blob, "executable": executable}
        )
    manifest = {
        "url": locator.url,
        "commit": locator.commit,
        "commit_object": base64.b64encode(_git(checkout, "cat-file", "commit", locator.commit)).decode(),
        "files": manifest_files,
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    if len(encoded.encode()) > MAX_PROVIDER_MANIFEST_BYTES:
        raise ValueError("Git provider source manifest exceeds size limit")
    (destination / SOURCE_MANIFEST).write_text(encoded)


def validate_staged_git_provider(provider: str, source: Path) -> None:
    """Reject altered, extra, or escaping files before Harbor launches."""
    locator = parse_git_provider(provider)
    if locator is None:
        raise ValueError("Provider is not Git-pinned")
    if any(path.is_symlink() for path in (source, source.parent, source.parent.parent)) or not source.is_dir():
        raise ValueError("Missing or unsafe Git provider source")
    manifest_path = source / SOURCE_MANIFEST
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Missing Git provider source manifest")
    if manifest_path.stat().st_size > MAX_PROVIDER_MANIFEST_BYTES:
        raise ValueError("Git provider source manifest exceeds size limit")
    manifest = json.loads(manifest_path.read_text())
    if manifest["url"] != locator.url or manifest["commit"] != locator.commit:
        raise ValueError("Git provider source manifest differs from binding")
    commit_object = base64.b64decode(manifest["commit_object"], validate=True)
    if _git_object_digest("commit", commit_object).hex() != locator.commit:
        raise ValueError("Git provider commit object differs from binding")
    tree_line = commit_object.split(b"\n", maxsplit=1)[0]
    if not tree_line.startswith(b"tree ") or len(tree_line) != 45:
        raise ValueError("Git provider commit has no valid tree")
    pinned_tree = bytes.fromhex(tree_line.removeprefix(b"tree ").decode())
    if not isinstance(manifest["files"], list) or len(manifest["files"]) > MAX_PROVIDER_FILES:
        raise ValueError("Git provider source manifest exceeds file limit")
    expected: set[str] = {SOURCE_MANIFEST}
    total_bytes = 0
    tree_files: list[tuple[str, str, bytes]] = []
    for item in manifest["files"]:
        path = str(validate_relative_file_path(item["path"]))
        if path in expected:
            raise ValueError(f"Duplicate Git provider source path: {path}")
        expected.add(path)
        target = source.joinpath(*path.split("/"))
        if not target.is_file() or target.is_symlink() or not target.resolve().is_relative_to(source.resolve()):
            raise ValueError(f"Missing or unsafe Git provider source file: {path}")
        if target.stat().st_size > MAX_PROVIDER_FILE_BYTES:
            raise ValueError("Git provider source exceeds size limits")
        payload = target.read_bytes()
        total_bytes += len(payload)
        if len(payload) > MAX_PROVIDER_FILE_BYTES or total_bytes > MAX_PROVIDER_SOURCE_BYTES:
            raise ValueError("Git provider source exceeds size limits")
        if hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise ValueError(f"Git provider source digest mismatch: {path}")
        blob = _git_object_digest("blob", payload)
        if blob.hex() != item["git_blob"]:
            raise ValueError(f"Git provider source blob differs from commit: {path}")
        if bool(target.stat().st_mode & 0o100) != item["executable"]:
            raise ValueError(f"Git provider source executable bit differs: {path}")
        tree_files.append((path, "100755" if item["executable"] else "100644", blob))
    if any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("Git provider source contains unexpected files or symlinks")
    observed = {str(path.relative_to(source)) for path in source.rglob("*") if not path.is_dir()}
    if observed != expected:
        raise ValueError("Git provider source contains unexpected files or symlinks")
    if _git_tree_digest(tree_files) != pinned_tree:
        raise ValueError("Git provider source tree differs from pinned commit")


def import_staged_provider(provider: str, source: Path) -> type:
    """Import an isolated copy of a verified package without modifying its snapshot."""
    locator = parse_git_provider(provider)
    if locator is None:
        raise ValueError("Provider is not Git-pinned")
    validate_staged_git_provider(provider, source)
    cache_key = hashlib.sha256((provider + "\n").encode() + (source / SOURCE_MANIFEST).read_bytes()).hexdigest()
    global _source_cache_dir
    with _source_cache_lock:
        cached_source = _source_cache.get(cache_key)
        if cached_source is None:
            if _source_cache_dir is None:
                _source_cache_dir = tempfile.TemporaryDirectory(prefix="taskcompendium-providers-")
            cached_source = Path(_source_cache_dir.name) / cache_key
            shutil.copytree(source, cached_source)
            validate_staged_git_provider(provider, cached_source)
            _source_cache[cache_key] = cached_source
        package_root = cached_source / "src" if (cached_source / "src").is_dir() else cached_source
        top_level = locator.module.split(".")[0]
        package_dir = package_root / top_level
        init_file = package_dir / "__init__.py"
        if not init_file.is_file():
            raise ValueError("Git provider must be an importable Python package")
        alias = f"_taskcompendium_git_{cache_key[:24]}"
        # Relative imports stay in this alias, even if another copy of the package is installed.
        if alias not in sys.modules:
            package_spec = importlib.util.spec_from_file_location(
                alias, init_file, submodule_search_locations=[str(package_dir)]
            )
            if package_spec is None or package_spec.loader is None:
                raise ValueError("Git provider package cannot be imported")
            package = importlib.util.module_from_spec(package_spec)
            sys.modules[alias] = package
            try:
                package_spec.loader.exec_module(package)
            except Exception:
                del sys.modules[alias]
                raise
        module_name = alias + locator.module.removeprefix(top_level)
        module = importlib.import_module(module_name)
        module_file = Path(inspect.getfile(module)).resolve()
        if not module_file.is_relative_to(package_root.resolve()):
            raise ValueError(f"Git provider module was already imported from another source: {locator.module}")
        provider_class = getattr(module, locator.class_name)
        if not inspect.isclass(provider_class) or not Path(inspect.getfile(provider_class)).resolve().is_relative_to(
            package_root.resolve()
        ):
            raise ValueError("Git provider path must name a class in the pinned source")
        return provider_class
