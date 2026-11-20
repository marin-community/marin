"""Build Harbor task sidechannels for Nemotron Ultra's released SWE rows.

The released RLVR blends contain pivot-state verifier records rather than full
repository environments. The campaign path keeps only exact TaskTrove proxy
states so Daytona can reuse TaskTrove's content-addressed snapshots. The older
SWE-Gym and R2E conversion helpers remain available for Iris/gVisor workflows.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import re
import shlex
import shutil
import tarfile
import tempfile
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from infra.rl_data.sources import (
    NEMOTRON_ULTRA_REVISION,
    NEMOTRON_ULTRA_RL_DATASET,
    NEMOTRON_ULTRA_SWE_AGENT,
    TASKTROVE_PROXY_PATH_KEY,
)

TASKTROVE_DATASET = "open-thoughts/TaskTrove"
TASKTROVE_REVISION = "131d8a8470c7a81113baac898c0c232db3f5ae31"
TASKTROVE_SWE_PROXY_PARQUET = "laion__nemotron-gym-agentic-swe-pivot-v4/tasks.parquet"
SWEGYM_DATASET = "SWE-Gym/SWE-Gym"
RLVR_BLEND_FILES = ("rlvr1.jsonl", "rlvr2.jsonl")
ENVIRONMENT_DIR = "environment"
R2E_TEST_INFO_PATH = "tests/test_info.json"
LEGACY_R2E_TEST_INFO_PATH = "/workspace/metadata.json"
R2E_TEST_INFO_ABSOLUTE_PATH = f"/{R2E_TEST_INFO_PATH}"


@dataclass(frozen=True)
class SWEGymSelection:
    tasks: list[dict[str, Any]]
    matched_ids: set[str]
    reconstructed_count: int


@dataclass(frozen=True)
class TaskTroveScan:
    tasks: list[dict[str, Any]]
    matched_ids: set[str]
    templates_by_repo: dict[str, bytes]


@dataclass
class SWEGymBuildContext:
    files: dict[str, tuple[bytes, int]]
    task_paths: list[str]


_TASK_TOML = """\
version = "1.0"

[agent]
timeout_sec = 900.0

[metadata]
author_name = "R2E-Gym"
author_email = "r2egym@example.com"
difficulty = "hard"
category = "software-engineering"
tags = ["r2egym", "code-repair", "bug-fixing"]

[verifier]
restart_environment = false
timeout_sec = 720.0
"""

# Ported from OpenThoughts-Agent's R2E-Gym Harbor adapter. The upstream image
# owns /root/run_tests.sh and /r2e_tests; this wrapper only parses its result.
_TEST_SH = r"""#!/bin/bash
set -e
source ~/.bashrc

ln -sf /testbed/.venv /root/.venv
ln -sf /testbed/.venv/bin/python /root/.local/bin/python
ln -sf /testbed/.venv/bin/python /root/.local/bin/python3
find /testbed/.venv/bin -type f -executable -exec ln -sf {} /root/.local/bin/ \;
export PATH=/testbed/.venv/bin:$PATH
uv pip install chardet

find . -name '*.pyc' -delete
find . -name '__pycache__' -exec rm -rf {} +
find /r2e_tests -name '*.pyc' -delete 2>/dev/null || true
find /r2e_tests -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true

for path in run_tests.sh r2e_tests; do
    if [ -e "/testbed/$path" ]; then mv "/testbed/$path" /root/; fi
done
if [ -d /r2e_tests ]; then mv /r2e_tests /root/r2e_tests; fi
if [ -d /root/r2e_tests ]; then ln -sf /root/r2e_tests /testbed/r2e_tests; fi

mkdir -p /logs/verifier /tests
cat > /tests/calculate_reward.py <<'PY'
import json
import re
import sys
from pathlib import Path


def parse_log_pytest(log):
    if log is None or "short test summary info" not in log:
        return {}
    statuses = {}
    for line in log.split("short test summary info", 1)[1].strip().splitlines():
        if "PASSED" in line:
            statuses[".".join(line.split("::")[1:])] = "PASSED"
        elif "FAILED" in line:
            statuses[".".join(line.split("::")[1:]).split(" - ")[0]] = "FAILED"
        elif "ERROR" in line:
            statuses[".".join(line.split("::")[1:]).split(" - ")[0]] = "ERROR"
    return statuses


def decolor(values):
    return {re.sub(r"\u001b\[\d+m", "", key): value for key, value in values.items()}


def reward(parsed, expected_json):
    parsed = {key.split(" - ")[0]: value for key, value in sorted(decolor(parsed).items())}
    expected = {key.split(" - ")[0]: value for key, value in sorted(decolor(json.loads(expected_json)).items())}
    if len(parsed) != len(expected):
        return 0.0
    return float(all(not key or expected.get(key) == value for key, value in parsed.items()))


metadata = json.loads(Path("__R2E_TEST_INFO_PATH__").read_text())
value = reward(parse_log_pytest(Path(sys.argv[1]).read_text()), metadata.get("expected_output_json", "{}"))
Path("/logs/verifier/reward.txt").write_text(str(value))
print(f"Reward: {value}")
PY

if [ ! -f /root/run_tests.sh ]; then
    echo 0 > /logs/verifier/reward.txt
    exit 1
fi
bash /root/run_tests.sh 2>&1 | tee /tmp/test_output.txt
python3 /tests/calculate_reward.py /tmp/test_output.txt
test -f /logs/verifier/reward.txt
""".replace("__R2E_TEST_INFO_PATH__", R2E_TEST_INFO_ABSOLUTE_PATH)


def collect_swe_instance_ids(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    """Extract unique SWE instance IDs from raw RLVR rows."""
    result: set[str] = set()
    for row in rows:
        agent_ref = row.get("agent_ref")
        if not isinstance(agent_ref, Mapping) or agent_ref.get("name") != NEMOTRON_ULTRA_SWE_AGENT:
            continue
        metadata = row.get("metadata")
        instance_id = metadata.get("instance_id") if isinstance(metadata, Mapping) else None
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError("Nemotron Ultra SWE row is missing metadata.instance_id")
        result.add(instance_id)
    return result


def _archive_json(archive: bytes, suffix: str) -> Mapping[str, Any] | None:
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(suffix):
                source = tar.extractfile(member)
                if source is None:
                    return None
                value = json.load(source)
                return value if isinstance(value, Mapping) else None
    return None


@dataclass(frozen=True)
class SWEProxyKey:
    trajectory_id: str
    step: int
    turn: int
    depth: int
    instance_id: str
    agent_cls: str


def _proxy_key(values: Mapping[str, Any], *, label: str) -> SWEProxyKey:
    trajectory_id = values.get("trajectory_id")
    instance_id = values.get("instance_id")
    agent_cls = values.get("agent_cls")
    dimensions = {name: values.get(name) for name in ("step", "turn", "depth")}
    if not isinstance(trajectory_id, (str, int)) or isinstance(trajectory_id, bool):
        raise ValueError(f"{label} is missing trajectory_id")
    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError(f"{label} is missing instance_id")
    if not isinstance(agent_cls, str) or not agent_cls:
        raise ValueError(f"{label} is missing agent_cls")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in dimensions.values()):
        raise ValueError(f"{label} is missing integer step, turn, or depth")
    return SWEProxyKey(
        trajectory_id=str(trajectory_id),
        step=dimensions["step"],
        turn=dimensions["turn"],
        depth=dimensions["depth"],
        instance_id=instance_id,
        agent_cls=agent_cls,
    )


def _raw_swe_proxy_key(row: Mapping[str, Any]) -> SWEProxyKey:
    info = row.get("info")
    metadata = row.get("metadata")
    if not isinstance(info, Mapping) or not isinstance(metadata, Mapping):
        raise ValueError("Nemotron Ultra SWE row is missing info or metadata")
    return _proxy_key({**info, **metadata, "trajectory_id": row.get("trajectory_id")}, label="Nemotron Ultra SWE row")


def tasktrove_swe_proxy_index(rows: Iterable[Mapping[str, Any]]) -> dict[SWEProxyKey, dict[str, Any]]:
    """Index exact TaskTrove pivot-state proxies without altering their archives."""
    result: dict[SWEProxyKey, dict[str, Any]] = {}
    paths: set[str] = set()
    for row in rows:
        path = row.get("path")
        archive_value = row.get("task_binary")
        if not isinstance(path, str):
            raise TypeError("TaskTrove proxy path must be a string")
        if not isinstance(archive_value, (bytes, bytearray, memoryview)):
            raise TypeError("TaskTrove proxy task_binary must be bytes")
        safe_path = _safe_task_path(path)
        archive = bytes(archive_value)
        metadata = _archive_json(archive, "metadata.json")
        if metadata is None:
            raise ValueError(f"TaskTrove proxy {path!r} has no metadata.json")
        key = _proxy_key(metadata, label=f"TaskTrove proxy {path!r}")
        if key in result:
            raise ValueError(f"duplicate TaskTrove SWE proxy key {key!r}")
        if safe_path in paths:
            raise ValueError(f"duplicate TaskTrove SWE proxy path {safe_path!r}")
        result[key] = {"path": safe_path, "task_binary": archive}
        paths.add(safe_path)
    return result


def bind_tasktrove_swe_proxies(
    rows: Iterable[Mapping[str, Any]],
    proxies: Mapping[SWEProxyKey, Mapping[str, Any]],
) -> Iterable[Mapping[str, Any]]:
    """Bind exact proxy paths to SWE rows and omit SWE states TaskTrove lacks."""
    for row in rows:
        agent_ref = row.get("agent_ref")
        if not isinstance(agent_ref, Mapping) or agent_ref.get("name") != NEMOTRON_ULTRA_SWE_AGENT:
            yield row
            continue
        proxy = proxies.get(_raw_swe_proxy_key(row))
        if proxy is None:
            continue
        path = proxy.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("TaskTrove proxy index contains an invalid path")
        bound = copy.deepcopy(row)
        metadata = bound["metadata"]
        metadata[TASKTROVE_PROXY_PATH_KEY] = path
        yield bound


def select_tasktrove_swe_proxy_tasks(
    desired_paths: set[str], tasktrove_rows: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Select byte-identical proxy archives for every requested task path."""
    selected: list[dict[str, Any]] = []
    matched: set[str] = set()
    for row in tasktrove_rows:
        path = row.get("path")
        archive = row.get("task_binary")
        if path not in desired_paths:
            continue
        if not isinstance(path, str) or not isinstance(archive, (bytes, bytearray, memoryview)):
            raise TypeError("TaskTrove proxy rows require string path and binary task_binary")
        if path in matched:
            raise ValueError(f"duplicate TaskTrove SWE proxy path {path!r}")
        selected.append({"path": _safe_task_path(path), "task_binary": bytes(archive)})
        matched.add(path)
    missing = desired_paths - matched
    if missing:
        raise ValueError(f"TaskTrove has no SWE proxy for {len(missing)} paths: {sorted(missing)[:5]}")
    return selected


def _safe_task_path(value: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise ValueError(f"unsafe task path {value!r}")
    return value


def _index_swegym_source_rows(
    desired_ids: set[str],
    source_rows: Iterable[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    source_by_id: dict[str, Mapping[str, Any]] = {}
    for row in source_rows:
        instance_id = row.get("instance_id")
        if instance_id not in desired_ids:
            continue
        if not isinstance(instance_id, str):
            raise ValueError("SWE-Gym source row has an invalid instance_id")
        if instance_id in source_by_id:
            raise ValueError(f"duplicate SWE-Gym source instance {instance_id!r}")
        source_by_id[instance_id] = row
    return source_by_id


def _select_tasktrove_rows_and_templates(
    desired_ids: set[str],
    tasktrove_rows: Iterable[Mapping[str, Any]],
    needed_repos: set[str],
) -> TaskTroveScan:
    selected: list[dict[str, Any]] = []
    matched: set[str] = set()
    template_by_repo: dict[str, bytes] = {}
    for row in tasktrove_rows:
        archive_value = row.get("task_binary")
        if not isinstance(archive_value, (bytes, bytearray, memoryview)):
            raise TypeError("TaskTrove task_binary must be bytes")
        archive = bytes(archive_value)
        config = _archive_json(archive, "tests/config.json")
        instance_id = config.get("instance_id") if config else None
        repo = config.get("repo") if config else None
        if isinstance(repo, str) and repo in needed_repos and repo not in template_by_repo:
            template_by_repo[repo] = archive
        if instance_id not in desired_ids:
            continue
        if instance_id in matched:
            raise ValueError(f"duplicate TaskTrove SWE-Gym instance {instance_id!r}")
        path = row.get("path")
        if not isinstance(path, str):
            raise TypeError("TaskTrove task path must be a string")
        selected.append({"path": _safe_task_path(path), "task_binary": archive})
        matched.add(str(instance_id))
    return TaskTroveScan(selected, matched, template_by_repo)


def _select_and_reconstruct_swegym_tasks(
    desired_ids: set[str],
    tasktrove_rows: Iterable[Mapping[str, Any]],
    source_rows: Iterable[Mapping[str, Any]],
) -> SWEGymSelection:
    """Select direct TaskTrove matches and reconstruct evidenced missing rows."""
    source_by_id = _index_swegym_source_rows(desired_ids, source_rows)

    needed_repos = {
        str(row["repo"])
        for row in source_by_id.values()
        if isinstance(row.get("repo"), str) and row["instance_id"] in desired_ids
    }
    scan = _select_tasktrove_rows_and_templates(desired_ids, tasktrove_rows, needed_repos)
    selected = scan.tasks
    matched = scan.matched_ids

    reconstructed = 0
    for instance_id in sorted(desired_ids - matched):
        source = source_by_id.get(instance_id)
        repo = source.get("repo") if source else None
        template = scan.templates_by_repo.get(str(repo))
        if source is None or template is None:
            continue
        selected.append(reconstruct_swegym_task(template, source))
        matched.add(instance_id)
        reconstructed += 1
    return SWEGymSelection(selected, matched, reconstructed)


def _instruction(problem_statement: str, base_commit: str) -> str:
    return f"""\
<uploaded_files>
/testbed
</uploaded_files>

I've uploaded a python code repository in the directory /testbed. Consider the following issue description:

<issue_description>
{problem_statement}
</issue_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?
I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way!
Also the development Python environment is already set up for you (i.e., all dependencies already installed), so you don't need to install other packages.
Your task is to make the minimal changes to non-test files in the /testbed directory to ensure the <issue_description> is satisfied.

Follow these steps to resolve the issue:

1. EXPLORATION: First, thoroughly explore the repository structure using tools like `find` and `grep`.
- Identify all files mentioned in the problem statement
- Locate where the issue occurs in the codebase
- Understand the surrounding context and dependencies
- Use `grep` to search for relevant functions, classes, or error messages

2. Assess whether you can reproduce the issue:
    - Create a script at /testbed/reproduce_issue.py that demonstrates the error.
    - Execute this script to confirm the error behavior.
    - You should reproduce the issue before fixing it.
    - Your reproduction script should also assert the expected behavior for the fixed code.

3. IMPLEMENTATION: Edit the source code to implement your chosen solution.
- Make minimal, focused changes to fix the issue

4. VERIFICATION: Test your implementation thoroughly.
- Run your reproduction script to verify the fix works
- Add edge cases to your test script to ensure comprehensive coverage
- Run existing tests related to the modified code to ensure you haven't broken anything

5. FINAL REVIEW: Carefully re-read the problem description and compare your changes with the base commit {base_commit}.
- Ensure you've fully addressed all requirements

Be thorough in your exploration, testing, and reasoning. It's fine if your thinking process is lengthy - quality and completeness are more important than brevity.
"""


def _tar_bytes(files: Mapping[str, tuple[bytes, int]], *, directories: Iterable[str] = ()) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz", format=tarfile.PAX_FORMAT) as archive:
        for name in sorted(directories):
            info = tarfile.TarInfo(name.rstrip("/") + "/")
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info)
        for name, (content, mode) in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = mode
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            archive.addfile(info, io.BytesIO(content))
    return output.getvalue()


def _archive_files(archive: bytes) -> dict[str, tuple[bytes, int]]:
    files: dict[str, tuple[bytes, int]] = {}
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as source:
        for member in source.getmembers():
            if not member.isfile():
                continue
            stream = source.extractfile(member)
            if stream is None:
                raise ValueError(f"could not read TaskTrove archive member {member.name!r}")
            files[member.name] = (stream.read(), member.mode)
    return files


_IMAGE_REFERENCE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]*$")


def _task_toml_with_image(content: bytes, image: str) -> bytes:
    if not _IMAGE_REFERENCE.fullmatch(image):
        raise ValueError(f"invalid prebuilt image reference {image!r}")
    config = tomllib.loads(content.decode())
    environment = config.get("environment")
    if environment is not None:
        if not isinstance(environment, Mapping):
            raise ValueError("task.toml [environment] must be a table")
        existing_image = environment.get("docker_image")
        if existing_image == image:
            return content
        if existing_image is not None:
            raise ValueError("task.toml already defines a different environment.docker_image")
        rendered = content.decode()
        rendered, count = re.subn(
            r"(?m)^(\[environment\][ \t]*(?:#.*)?)$",
            lambda match: f"{match.group(1)}\ndocker_image = {json.dumps(image)}",
            rendered,
            count=1,
        )
        if count != 1:
            raise ValueError("task.toml defines [environment] in an unsupported form")
        return rendered.encode()
    return content.rstrip() + b"\n\n" + f"[environment]\ndocker_image = {json.dumps(image)}\n".encode()


def _r2e_metadata(files: Mapping[str, tuple[bytes, int]]) -> Mapping[str, Any] | None:
    value = files.get(R2E_TEST_INFO_PATH)
    if value is None:
        return None
    metadata = json.loads(value[0])
    if not isinstance(metadata, Mapping) or metadata.get("source") != "r2egym":
        return None
    return metadata


def _environment_context(files: Mapping[str, tuple[bytes, int]]) -> dict[str, tuple[bytes, int]]:
    context: dict[str, tuple[bytes, int]] = {}
    for name, value in files.items():
        path = PurePosixPath(name)
        if not path.parts or path.parts[0] != ENVIRONMENT_DIR:
            continue
        relative = PurePosixPath(*path.parts[1:])
        if not relative.parts or relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe environment archive path {name!r}")
        context[relative.as_posix()] = value
    if "Dockerfile" not in context:
        raise ValueError("SWE-Gym task does not contain environment/Dockerfile")
    return context


def _environment_digest(context: Mapping[str, tuple[bytes, int]]) -> str:
    digest = hashlib.sha256()
    for name, (content, mode) in sorted(context.items()):
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(f"{mode:o}".encode())
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def _swegym_contexts(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, SWEGymBuildContext], int]:
    """Group SWE-Gym environments by digest and count skipped R2E-Gym tasks."""
    contexts: dict[str, SWEGymBuildContext] = {}
    r2e_count = 0
    for row in rows:
        archive = row.get("task_binary")
        path = row.get("path")
        if not isinstance(archive, (bytes, bytearray, memoryview)) or not isinstance(path, str):
            raise TypeError("task rows require string path and binary task_binary")
        files = _archive_files(bytes(archive))
        if _r2e_metadata(files) is not None:
            r2e_count += 1
            continue
        context = _environment_context(files)
        digest = _environment_digest(context)
        entry = contexts.setdefault(digest, SWEGymBuildContext(files=context, task_paths=[]))
        if entry.files != context:
            raise ValueError(f"environment digest collision for {digest}")
        entry.task_paths.append(path)
    return contexts, r2e_count


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_swegym_build_contexts(source_tasks: Path, output_dir: Path) -> dict[str, Any]:
    """Write each unique SWE-Gym environment as a content-addressed build context."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output_dir}")
    rows = pq.read_table(source_tasks).to_pylist()
    contexts, r2e_count = _swegym_contexts(rows)
    manifest = {
        "source": {"tasks": str(source_tasks), "sha256": _file_sha256(source_tasks)},
        "counts": {
            "tasks": len(rows),
            "swegym_tasks": len(rows) - r2e_count,
            "unique_contexts": len(contexts),
            "r2egym_tasks": r2e_count,
        },
        "contexts": [
            {
                "digest": digest,
                "task_count": len(entry.task_paths),
                "task_paths": sorted(entry.task_paths),
            }
            for digest, entry in sorted(contexts.items())
        ],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        for digest, entry in contexts.items():
            context_root = staging / digest
            for name, (content, mode) in entry.files.items():
                target = context_root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                target.chmod(mode)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


_DIGEST_PINNED_IMAGE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*@sha256:[0-9a-f]{64}$")


def _convert_task_to_prebuilt(row: Mapping[str, Any], swegym_images: Mapping[str, str]) -> dict[str, Any]:
    files = _archive_files(bytes(row["task_binary"]))
    r2e = _r2e_metadata(files)
    if r2e is None:
        image = swegym_images[_environment_digest(_environment_context(files))]
    else:
        image = r2e.get("docker_image")
        if not isinstance(image, str) or not image:
            raise ValueError(f"R2E task {row['path']!r} has no docker_image")
        test_script = files.get("tests/test.sh")
        if test_script is not None:
            files["tests/test.sh"] = (
                test_script[0].replace(LEGACY_R2E_TEST_INFO_PATH.encode(), R2E_TEST_INFO_ABSOLUTE_PATH.encode()),
                test_script[1],
            )
    task_toml = files.get("task.toml")
    if task_toml is None:
        raise ValueError(f"task {row['path']!r} has no task.toml")
    files["task.toml"] = (_task_toml_with_image(task_toml[0], image), task_toml[1])
    environment_prefix = f"{ENVIRONMENT_DIR}/"
    files = {name: value for name, value in files.items() if not name.startswith(environment_prefix)}
    return {"path": row["path"], "task_binary": _tar_bytes(files, directories=(ENVIRONMENT_DIR,))}


def _write_task_artifact(output_dir: Path, rows: list[dict[str, Any]], provenance: Mapping[str, Any]) -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        table = pa.Table.from_pylist(rows, schema=pa.schema([("path", pa.string()), ("task_binary", pa.binary())]))
        pq.write_table(table, staging / "tasks.parquet", compression="zstd")
        (staging / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
        staging.rename(output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def convert_swe_tasks_to_prebuilt(
    source_tasks: Path,
    output_dir: Path,
    swegym_images: Mapping[str, str],
) -> dict[str, Any]:
    """Replace task Dockerfiles with prebuilt image references, failing closed on gaps."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output_dir}")
    rows = pq.read_table(source_tasks).to_pylist()
    contexts, r2e_count = _swegym_contexts(rows)
    missing = sorted(contexts.keys() - swegym_images.keys())
    unexpected = sorted(swegym_images.keys() - contexts.keys())
    if missing or unexpected:
        raise ValueError(f"SWE-Gym image map mismatch: missing published images={missing}; unexpected={unexpected}")
    mutable = sorted(image for image in swegym_images.values() if not _DIGEST_PINNED_IMAGE.fullmatch(image))
    if mutable:
        raise ValueError(f"SWE-Gym images must be digest-pinned: {mutable}")

    converted = [_convert_task_to_prebuilt(row, swegym_images) for row in rows]

    provenance = {
        "source": {"tasks": str(source_tasks), "sha256": _file_sha256(source_tasks)},
        "counts": {"total": len(rows), "swegym": len(rows) - r2e_count, "r2egym": r2e_count},
        "swegym_images": dict(sorted(swegym_images.items())),
    }
    _write_task_artifact(output_dir, converted, provenance)
    return provenance


def _required_string(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"SWE-Gym reconstruction row is missing {field!r}")
    return value


def _diff_paths(patch: str) -> list[str]:
    paths = []
    for path in re.findall(r"^\+\+\+ b/(.+)$", patch, flags=re.MULTILINE):
        if path not in paths:
            paths.append(path)
    return paths


def _replace_test_array(script: str, name: str, values: list[str]) -> str:
    pattern = re.compile(rf"(?ms)^(?P<indent>[ \t]*){name}=\(\n.*?^(?P=indent)\)")

    def replacement(match: re.Match[str]) -> str:
        indent = match.group("indent")
        entries = "\n".join(f"{indent}    {shlex.quote(value)}" for value in values)
        return f"{indent}{name}=(\n{entries}\n{indent})"

    result, count = pattern.subn(replacement, script, count=1)
    if count != 1:
        raise ValueError(f"TaskTrove template does not define {name}")
    return result


def _swegym_solution_script(repo: str, commit: str, test_patch: str, solution_patch: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
        raise ValueError(f"unsafe SWE-Gym repository {repo!r}")
    if not re.fullmatch(r"[0-9a-f]{40}|[A-Za-z0-9_.-]+", commit):
        raise ValueError(f"unsafe SWE-Gym commit {commit!r}")
    if "TEST_PATCH_EOF" in test_patch or "SOLUTION_PATCH_EOF" in solution_patch:
        raise ValueError("SWE-Gym patch contains a reserved heredoc delimiter")
    return f"""\
#!/usr/bin/env bash
set -Eeuo pipefail

source /opt/miniconda3/bin/activate
conda activate testbed
cd /testbed
if [ ! -d repo ]; then
    git clone https://github.com/{repo}.git repo
fi
cd repo
git checkout {commit}

cat <<'TEST_PATCH_EOF' | git apply --whitespace=nowarn --apply
{test_patch.rstrip()}
TEST_PATCH_EOF
cat <<'SOLUTION_PATCH_EOF' | git apply --whitespace=nowarn --apply
{solution_patch.rstrip()}
SOLUTION_PATCH_EOF
"""


def reconstruct_swegym_task(template_archive: bytes, row: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild a missing SWE-Gym task from a same-repository TaskTrove template."""
    instance_id = _required_string(row, "instance_id")
    repo = _required_string(row, "repo")
    commit = _required_string(row, "base_commit")
    problem = _required_string(row, "problem_statement")
    solution_patch = _required_string(row, "patch")
    test_patch = _required_string(row, "test_patch")
    pass_to_pass = row.get("PASS_TO_PASS")
    fail_to_pass = row.get("FAIL_TO_PASS")
    if not isinstance(pass_to_pass, list) or not all(isinstance(value, str) for value in pass_to_pass):
        raise ValueError(f"SWE-Gym reconstruction row {instance_id!r} has invalid PASS_TO_PASS")
    if not isinstance(fail_to_pass, list) or not all(isinstance(value, str) for value in fail_to_pass):
        raise ValueError(f"SWE-Gym reconstruction row {instance_id!r} has invalid FAIL_TO_PASS")

    files = _archive_files(template_archive)
    required_template_files = {
        "environment/Dockerfile",
        "task.toml",
        "tests/install_trusted_test_patch.sh",
        "tests/install_trusted_test_paths.sh",
        "tests/test.sh",
        "tests/test_state.py",
    }
    missing = required_template_files - files.keys()
    if missing:
        raise ValueError(f"TaskTrove reconstruction template is missing files: {sorted(missing)}")
    template_config = _archive_json(template_archive, "tests/config.json")
    if template_config is None or template_config.get("repo") != repo:
        raise ValueError(f"TaskTrove reconstruction template is not for repository {repo!r}")

    test_script = files["tests/test.sh"][0].decode()
    old_commit = _required_string(template_config, "base_commit")
    if old_commit not in test_script:
        raise ValueError("TaskTrove reconstruction template test script does not contain its base commit")
    test_script = test_script.replace(old_commit, commit)
    test_script = _replace_test_array(test_script, "PASS_TESTS", pass_to_pass)
    test_script = _replace_test_array(test_script, "FAIL_TESTS", fail_to_pass)

    metadata = {
        "source": SWEGYM_DATASET,
        "split": "train",
        "instance_id": instance_id,
        "repo": repo,
        "base_commit": commit,
        "version": row.get("version"),
        "pass_to_pass": pass_to_pass,
        "fail_to_pass": fail_to_pass,
        "created_at": row.get("created_at"),
        "reconstructed_from_tasktrove_template": template_config.get("instance_id"),
    }
    config = {**metadata, "patch": solution_patch, "test_patch_length": len(test_patch)}
    test_paths = sorted({value.split("::", 1)[0] for value in [*pass_to_pass, *fail_to_pass] if value})
    patch_paths = _diff_paths(test_patch)
    if not patch_paths:
        raise ValueError(f"SWE-Gym reconstruction row {instance_id!r} test patch has no target paths")

    files.update(
        {
            "instruction.md": (_instruction(problem, commit).encode(), 0o644),
            "metadata.json": ((json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode(), 0o644),
            "solution/solve.sh": (
                _swegym_solution_script(repo, commit, test_patch, solution_patch).encode(),
                0o755,
            ),
            "tests/config.json": ((json.dumps(config, indent=2, sort_keys=True) + "\n").encode(), 0o644),
            "tests/test.sh": (test_script.encode(), files["tests/test.sh"][1]),
            "tests/test_patch.diff": (test_patch.encode(), 0o644),
            "tests/trusted_patch_paths.txt": (("\n".join(patch_paths) + "\n").encode(), 0o644),
            "tests/trusted_test_paths.txt": (("\n".join(test_paths or patch_paths) + "\n").encode(), 0o644),
        }
    )
    return {"path": _safe_task_path(instance_id.casefold()), "task_binary": _tar_bytes(files)}


def make_r2e_task(instance_id: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Package one upstream R2E-Gym image as a Harbor task archive."""
    commit = row.get("commit_hash")
    image = row.get("docker_image")
    problem = row.get("problem_statement")
    expected = row.get("expected_output_json")
    if not all(isinstance(value, str) and value for value in (commit, image, problem, expected)):
        raise ValueError(f"R2E-Gym row for {instance_id!r} is missing required string fields")
    metadata = {
        "instance_id": instance_id,
        "docker_image": image,
        "base_commit": commit,
        "problem_statement": problem,
        "repo_name": row.get("repo_name"),
        "expected_output_json": expected,
        "source": "r2egym",
    }
    metadata_bytes = (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode()
    files = {
        "instruction.md": (_instruction(problem, commit).encode(), 0o644),
        "task.toml": (_task_toml_with_image(_TASK_TOML.encode(), image), 0o644),
        R2E_TEST_INFO_PATH: (metadata_bytes, 0o644),
        "tests/test.sh": (_TEST_SH.encode(), 0o755),
    }
    return {"path": _safe_task_path(instance_id.casefold()), "task_binary": _tar_bytes(files)}


def select_r2e_tasks(desired_ids: set[str], rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], set[str]]:
    """Match remaining blend IDs to R2E-Gym's globally unique commit hashes."""
    by_commit: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for row in rows:
        commit = row.get("commit_hash")
        if not isinstance(commit, str) or not commit:
            raise ValueError("R2E-Gym row is missing commit_hash")
        if commit in by_commit:
            raise ValueError(f"duplicate R2E-Gym commit hash {commit!r}")
        by_commit[commit] = (str(row.get("repo_name", "")), row)

    selected: list[dict[str, Any]] = []
    matched: set[str] = set()
    for instance_id in sorted(desired_ids):
        commit = instance_id.rsplit("-", 1)[-1]
        match = by_commit.get(commit)
        if match is None:
            continue
        selected.append(make_r2e_task(instance_id, match[1]))
        matched.add(instance_id)
    return selected, matched


def compose_swe_tasks(
    desired_ids: set[str],
    swegym_rows: Iterable[Mapping[str, Any]],
    r2e_rows: Iterable[Mapping[str, Any]],
    *,
    swegym_source_rows: Iterable[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Build exactly one task for every unique SWE ID in the released blend."""
    swegym_selection = _select_and_reconstruct_swegym_tasks(desired_ids, swegym_rows, swegym_source_rows)
    swegym = swegym_selection.tasks
    swegym_ids = swegym_selection.matched_ids
    r2e, r2e_ids = select_r2e_tasks(desired_ids - swegym_ids, r2e_rows)
    overlap = swegym_ids & r2e_ids
    if overlap:
        raise ValueError(f"SWE instances matched both sources: {sorted(overlap)[:5]}")
    missing = desired_ids - swegym_ids - r2e_ids
    if missing:
        raise ValueError(f"No Harbor source for {len(missing)} SWE instances: {sorted(missing)[:5]}")
    rows = sorted([*swegym, *r2e], key=lambda row: str(row["path"]))
    if len({row["path"] for row in rows}) != len(rows):
        raise ValueError("composed Harbor task paths are not unique")
    counts = {"total": len(rows), "swegym": len(swegym)}
    if swegym_selection.reconstructed_count:
        counts["swegym_reconstructed"] = swegym_selection.reconstructed_count
    counts["r2egym"] = len(r2e)
    return rows, counts


def _load_blend_swe_proxy_paths(revision: str, proxies: Mapping[SWEProxyKey, Mapping[str, Any]]) -> set[str]:
    paths: set[str] = set()
    for filename in RLVR_BLEND_FILES:
        path = hf_hub_download(
            repo_id=NEMOTRON_ULTRA_RL_DATASET,
            repo_type="dataset",
            filename=filename,
            revision=revision,
        )
        with open(path) as source:
            bound = bind_tasktrove_swe_proxies((json.loads(line) for line in source), proxies)
            for row in bound:
                agent_ref = row.get("agent_ref")
                if not isinstance(agent_ref, Mapping) or agent_ref.get("name") != NEMOTRON_ULTRA_SWE_AGENT:
                    continue
                metadata = row["metadata"]
                paths.add(str(metadata[TASKTROVE_PROXY_PATH_KEY]))
    return paths


def tasktrove_swe_proxy_rows(revision: str = TASKTROVE_REVISION):
    """Stream the pinned TaskTrove semantic next-action proxy archives."""
    path = hf_hub_download(
        repo_id=TASKTROVE_DATASET,
        repo_type="dataset",
        filename=TASKTROVE_SWE_PROXY_PARQUET,
        revision=revision,
    )
    for batch in pq.ParquetFile(path).iter_batches(columns=["path", "task_binary"], batch_size=64):
        yield from batch.to_pylist()


def load_tasktrove_swe_proxy_index(revision: str = TASKTROVE_REVISION) -> dict[SWEProxyKey, dict[str, Any]]:
    """Load the pinned TaskTrove proxy archive index."""
    return tasktrove_swe_proxy_index(tasktrove_swe_proxy_rows(revision))


def prepare_swe_task_artifact(
    output_dir: Path,
    *,
    desired_paths: set[str] | None = None,
    blend_files: tuple[str, ...] = RLVR_BLEND_FILES,
    blend_revision: str = NEMOTRON_ULTRA_REVISION,
    tasktrove_revision: str = TASKTROVE_REVISION,
) -> dict[str, Any]:
    """Write the exact TaskTrove SWE proxy archives used by the blends.

    Without ``desired_paths``, every proxy the RLVR blends reference is written. Callers
    that pass their own paths also name the blend files the paths came from.
    """
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output_dir}")
    proxy_rows = list(tasktrove_swe_proxy_rows(tasktrove_revision))
    proxies = tasktrove_swe_proxy_index(proxy_rows)
    desired_paths = _load_blend_swe_proxy_paths(blend_revision, proxies) if desired_paths is None else desired_paths
    if not desired_paths:
        raise ValueError("Nemotron Ultra SWE task preparation requires at least one TaskTrove proxy path")
    rows = select_tasktrove_swe_proxy_tasks(desired_paths, proxy_rows)
    counts = {"total": len(rows), "tasktrove_proxy": len(rows)}
    provenance = {
        "blend": {
            "dataset": NEMOTRON_ULTRA_RL_DATASET,
            "revision": blend_revision,
            "files": list(blend_files),
        },
        "tasktrove_proxy": {
            "dataset": TASKTROVE_DATASET,
            "revision": tasktrove_revision,
            "file": TASKTROVE_SWE_PROXY_PARQUET,
        },
        "counts": counts,
    }
    _write_task_artifact(output_dir, rows, provenance)
    return provenance
