# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Linux root regressions for the protected SHARED source-scorer adapter.

The fixtures embed the original Prime family scorer logic (extracted without
behavioral edits by the ingest lane) with one documented adaptation: the fixture
``ENV`` PATH names a shim directory so the scorer finds this test environment's
pytest-capable interpreter instead of the source conda layout. Runtime
protection is exercised exactly as published: bootstrap through the task's own
setup command, agent-phase edits as the agent UID, grading through the shipped
``tests/test.sh`` launcher.
"""

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import sysconfig
import textwrap
import zlib
from dataclasses import dataclass
from pathlib import Path

import pytest
import tomlkit

from taskcompendium.harbor.shared_source import (
    SCALE_SWE_PROFILE,
    SOURCE_ARCHIVE,
    SOURCE_PAYLOAD,
    SWE_LEGO_PROFILE,
    SWE_REBENCH_PROFILE,
    harden_shared_source,
)
from taskcompendium.integrity import AGENT_HOME, PRIVATE_CONTEXT, PRIVATE_DIRECTORY, SharedDependencyState

pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(sys.platform != "linux" or os.geteuid() != 0, reason="UID isolation requires Linux root"),
]

WORKSPACE = Path("/testbed")
PROTECTED = Path("/opt/taskcompendium-shared")
BOOTSTRAP = Path("/opt/taskcompendium-shared-bootstrap")
LOGS = Path("/logs")
REPORTS = LOGS / "verifier"
SETUP_FILES = Path("/setup_files")
AGENT_UID, GRADER_UID = 10001, 10002

SHIM = "/tmp/taskcompendium-source-shim"
SOURCE_PREFIX = Path("/opt/miniconda3/envs/testbed")
SOURCE_ENV_PATH = (
    "{shim}:/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:"
    "/opt/conda/envs/testbed/bin:/opt/conda/bin:"
    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)

NATIVE_SCRIPTS = textwrap.dedent(
    '''\
    ENV = {"PATH": "@SHIM@:/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:"
                 "/opt/conda/envs/testbed/bin:/opt/conda/bin:"
                 "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
           "PAGER": "cat", "CI": "1"}

    RESTORE = r"""
    git checkout "$base" -- tests/ test/ Test/ Tests/ 2>/dev/null || true
    git ls-tree -r --name-only "$base" 2>/dev/null | while IFS= read -r path; do
      case "$path" in
        test_*.py|*/test_*.py|*_test.py|*/*_test.py|conftest.py|*/conftest.py)
          git checkout "$base" -- "$path" 2>/dev/null || true ;;
      esac
    done
    git ls-files --cached --others --exclude-standard 2>/dev/null | while IFS= read -r path; do
      case "$path" in
        tests/*|test/*|Test/*|Tests/*|test_*.py|*/test_*.py|*_test.py|*/*_test.py|conftest.py|*/conftest.py)
          if ! git cat-file -e "$base:$path" 2>/dev/null; then
            rm -f -- "$path"; git rm -q --cached -- "$path" 2>/dev/null || true
          fi ;;
      esac
    done
    """

    REVERT_TEST_FILES = r"""
    git apply --numstat "$patch" | cut -f3- | while IFS= read -r path; do
      path=${path#\\"}; path=${path%\\"}
      if git cat-file -e "$base:$path" 2>/dev/null; then
        git checkout "$base" -- "$path" 2>/dev/null || true
      else
        rm -f -- "$path"
      fi
    done
    """

    REPO_PATH = "/testbed"

    def build_eval_script(test_cmd):
        return f"""#!/bin/bash
    set -uo pipefail
    cd {REPO_PATH}
    set +u
    if [ -f /opt/miniconda3/bin/activate ]; then source /opt/miniconda3/bin/activate; fi
    conda activate testbed 2>/dev/null || true
    set -u
    set +e
    {test_cmd}
    PYTEST_EXIT=$?
    set -e
    echo ""
    echo "SWELEGO_PYTEST_EXIT=$PYTEST_EXIT"
    exit 0
    """

    ANSI_RE = __import__("re").compile(r"\\x1b\\[[0-?]*[ -/]*[@-~]")

    PYTEST_OUTCOMES = ("PASSED", "FAILED", "ERROR", "SKIPPED", "XFAIL", "XPASS")

    def parse_pytest_outcomes(output):
        outcomes = {}
        for raw_line in ANSI_RE.sub("", output or "").splitlines():
            parts = raw_line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            outcome, test_id = parts
            if outcome not in PYTEST_OUTCOMES or test_id.startswith("["):
                continue
            if outcome in ("FAILED", "ERROR", "XFAIL", "XPASS") and " - " in test_id:
                pieces = test_id.split(" - ")
                for index in range(1, len(pieces)):
                    node_id = " - ".join(pieces[:index])
                    if node_id.count("[") == node_id.count("]"):
                        test_id = node_id
                        break
            outcomes[test_id.rstrip()] = outcome
        return outcomes
    '''
)

SCALE_GRADE = textwrap.dedent(
    """\
    import json, os, re, subprocess
    from pathlib import Path
    from native_scripts import ENV, RESTORE

    CONFIG = Path("/tests/config.json")

    def main():
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        os.chdir(config["workdir"])
        env = {**os.environ, **ENV, "base": config["base_commit"]}
        subprocess.run(["sh", "-c", RESTORE], env=env, check=False)
        if config["f2p_patch"].strip():
            patch = Path("/tmp/scaleswe_f2p.patch")
            patch.write_text(config["f2p_patch"], encoding="utf-8")
            for command in (
                ["git", "apply", "--verbose", str(patch)],
                ["git", "apply", "--verbose", "--ignore-space-change", "--ignore-whitespace", str(patch)],
                ["patch", "--batch", "--fuzz=5", "-p1", "-i", str(patch)],
                ["git", "apply", "--verbose", "--reject", "--ignore-whitespace", str(patch)],
            ):
                if subprocess.run(command, env=env, check=False).returncode == 0:
                    break
        if config["f2p_script"]:
            Path("test_fail_to_pass.py").write_text(config["f2p_script"], encoding="utf-8")
        ids = Path("/tmp/scaleswe_test_ids.json")
        ids.write_text(json.dumps(config["test_ids"]))
        result = subprocess.run(["python", "/tests/scale_score.py", str(ids)], env=env, capture_output=True, text=True)
        print(result.stdout)
        matches = re.findall(r"<score>([0-9.]+)</score>", result.stdout)
        score = float(matches[-1]) if matches else 0.0
        Path("/logs/verifier/reward.txt").write_text(str(score))

    main()
    """
)

SCALE_SCORE = textwrap.dedent(
    """\
    import json, sys, xml.etree.ElementTree as ET
    import pytest

    XML = "/tmp/scaleswe_results.xml"

    def emit(score):
        print(f"<score>{score}</score>")
        sys.stdout.flush()

    def normalize(value):
        parts = value.strip().split("::")
        if parts and parts[0].endswith(".py"):
            parts[0] = parts[0][:-3]
        return ".".join(parts).replace("/", ".").strip(".")

    def all_passed(xml_content, expected):
        try:
            root = ET.fromstring(xml_content)
        except ET.ParseError:
            return False
        exact = set(expected)
        norm = {normalize(t): t for t in expected}
        matched = {}
        found = set()
        for tc in root.iter("testcase"):
            if tc.find("skipped") is not None:
                continue
            name, classname = tc.get("name", ""), tc.get("classname", "")
            file_attr = tc.get("file", "")
            status = "failed" if tc.find("failure") is not None or tc.find("error") is not None else "passed"
            for candidate in (f"{file_attr}::{name}" if file_attr else "", normalize(f"{classname}.{name}")):
                original = candidate if candidate in exact else norm.get(candidate)
                if original:
                    matched[original] = status
                    found.add(original)
                    break
        missing = [t for t in expected if t not in found]
        return bool(found) and all(s == "passed" for s in matched.values()) and not missing

    def main():
        with open(sys.argv[1], encoding="utf-8") as stream:
            expected = json.load(stream)
        if not expected:
            emit(0.0)
            return
        pytest.main(["-vv", f"--junitxml={XML}", "-o", "addopts=", "--rootdir=.", *expected])
        try:
            with open(XML, encoding="utf-8") as stream:
                xml_content = stream.read()
        except OSError:
            emit(0.0)
            return
        emit(1.0 if all_passed(xml_content, expected) else 0.0)

    main()
    """
)

LEGO_GRADE = textwrap.dedent(
    """\
    import json, os, subprocess
    from pathlib import Path
    from native_scripts import ENV, REVERT_TEST_FILES, build_eval_script, parse_pytest_outcomes

    def main():
        config = json.loads(Path("/tests/config.json").read_text(encoding="utf-8"))
        os.chdir("/testbed")
        env = {**os.environ, **ENV}
        reward = Path("/logs/verifier/reward.txt")
        if config["test_patch"].strip():
            patch = Path("/tmp/test_patch.patch")
            patch.write_text(config["test_patch"], encoding="utf-8")
            if config["base_commit"]:
                subprocess.run(
                    ["sh", "-c", REVERT_TEST_FILES],
                    env={**env, "patch": str(patch), "base": config["base_commit"]},
                    check=False,
                )
            for command in (
                ["git", "apply", "--whitespace=fix", str(patch)],
                ["patch", "--batch", "--fuzz=5", "-p1", "-i", str(patch)],
            ):
                if subprocess.run(command, env=env, check=False).returncode == 0:
                    break
            else:
                reward.write_text("0.0")
                return
        result = subprocess.run(
            ["bash", "-c", build_eval_script(config["test_cmd"])], env=env, capture_output=True, text=True
        )
        print(result.stdout)
        outcomes = parse_pytest_outcomes(result.stdout)
        required = config["test_ids"]
        reward.write_text(str(float(bool(required) and all(outcomes.get(test) == "PASSED" for test in required))))

    main()
    """
)

APP_BUGGY = "def add(a, b):\n    return a - b\n"
APP_FIXED = "def add(a, b):\n    return a + b\n"
GOLD_PATCH = "--- a/app.py\n+++ b/app.py\n@@ -1,2 +1,2 @@\n def add(a, b):\n-    return a - b\n+    return a + b\n"
SCALE_TEST = (
    "import sys, pathlib\n"
    "sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))\n"
    "from app import add\n\n\n"
    "def test_add():\n    assert add(1, 2) == 3\n\n\n"
    "def test_add_negative():\n    assert add(-1, -1) == -2\n"
)
# A later upstream commit changes the test expectations; the declared base still
# holds the original bytes, so a silent no-op restore must fail the NoP grade.
HEAD_TEST = SCALE_TEST.replace("assert add(1, 2) == 3", "assert add(1, 2) == -1").replace(
    "assert add(-1, -1) == -2", "assert add(-1, -1) == 0"
)
FAKE_TEST = SCALE_TEST.replace("assert add(1, 2) == 3", "assert True").replace("assert add(-1, -1) == -2", "assert True")
LEGO_TEST_PATCH = (
    "--- /dev/null\n"
    "+++ b/tests/nested/test_fixed.py\n"
    "@@ -0,0 +1,6 @@\n"
    "+import sys, pathlib\n"
    "+sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))\n"
    "+from app import add\n"
    "+\n"
    "+def test_add_fixed():\n"
    "+    assert add(1, 2) == 3\n"
)
LEGO_TEST_CMD = "python -m pytest -rA tests/nested/test_fixed.py"

SCALE_IDS = ["tests/test_app.py::test_add", "tests/test_app.py::test_add_negative"]
LEGO_IDS = ["tests/nested/test_fixed.py::test_add_fixed"]

SWE_REBENCH_TEST_PATCH = (
    "--- /dev/null\n"
    "+++ b/tests/test_new.py\n"
    "@@ -0,0 +1,6 @@\n"
    "+import sys, pathlib\n"
    "+sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))\n"
    "+from app import add\n"
    "+\n"
    "+def test_resolved():\n"
    "+    assert add(1, 2) == 3\n"
)
SWE_REBENCH_TEST_CMD = "python3 -m pytest -q tests/test_new.py"
SWE_REBENCH_IDS = ["tests/test_new.py::test_resolved"]
SWE_REBENCH_GRADE = (
    "import pathlib, sys\n"
    "log = pathlib.Path(sys.argv[1]).read_text()\n"
    "ok = '1 passed' in log and ' failed' not in log and 'error' not in log.lower()\n"
    "raise SystemExit(0 if ok else 1)\n"
)
SWE_REBENCH_TEST_SH = (
    "#!/bin/bash\n"
    "set -uo pipefail\n"
    "V=/logs/verifier\n"
    'mkdir -p "$V"\n'
    'echo 0 > "$V/reward.txt"\n'
    "git restore -- tests/test_app.py 2>/dev/null || rm -f -- tests/test_app.py\n"
    "git apply /tests/test.patch\n"
    f'timeout 300 {SWE_REBENCH_TEST_CMD} > "$V/test_output.log" 2>&1 || true\n'
    'if python3 /tests/grade.py "$V/test_output.log"; then\n'
    '  echo 1 > "$V/reward.txt"\n'
    "fi\n"
)

TEST_SH = b"#!/bin/sh\nset -eu\nmkdir -p /logs/verifier\npython /tests/grade.py\n"


@dataclass(frozen=True)
class SourceTask:
    family: str
    files: dict[str, bytes]
    base_commit: str


def git(*arguments: str) -> None:
    subprocess.run(
        ["git", *arguments],
        cwd=WORKSPACE,
        env={**os.environ, "GIT_CONFIG_GLOBAL": "/dev/null", "HOME": "/root"},
        check=True,
        capture_output=True,
    )


def build_repo() -> str:
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    WORKSPACE.mkdir(parents=True)
    (WORKSPACE / "app.py").write_text(APP_BUGGY)
    tests = WORKSPACE / "tests"
    tests.mkdir()
    (tests / "test_app.py").write_text(SCALE_TEST)
    # Pre-existing oddities the bootstrap must preserve for the agent.
    scratch = tests / "scratch_util.py"
    scratch.write_text("VALUE = 'preserve'\n")
    scratch.chmod(0o600)
    link = tests / "link_to_app.py"
    link.symlink_to("../app.py")
    (WORKSPACE / "untracked_note.txt").write_text("untracked\n")
    # A nested repository conftest inside the original restore scope.
    package = WORKSPACE / "pkg"
    package.mkdir()
    (package / "conftest.py").write_text("NESTED_CONFTEST = 'base'\n")
    git("init", "-q")
    git("config", "user.email", "fixture@example.test")
    git("config", "user.name", "Fixture")
    git("add", "app.py", "tests/test_app.py", "pkg/conftest.py")
    git("commit", "-qm", "base")
    base = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=WORKSPACE, capture_output=True, text=True, check=True
    ).stdout.strip()
    # A later upstream commit moves HEAD's test bytes away from the declared base.
    (tests / "test_app.py").write_text(HEAD_TEST)
    git("add", "tests/test_app.py")
    git("commit", "-qm", "upstream test change")
    return base


def package_files(family: str, base: str) -> tuple[dict[str, bytes], object]:
    if family == "swe-rebench":
        config = {
            "workdir": str(WORKSPACE),
            "base_commit": base,
            "test_patch": SWE_REBENCH_TEST_PATCH,
            "test_cmd": SWE_REBENCH_TEST_CMD,
            "test_files": ["tests/test_app.py"],
            "log_parser": "parse_log_pytest",
            "language": "python",
            "test_ids": SWE_REBENCH_IDS,
            "FAIL_TO_PASS": SWE_REBENCH_IDS,
            "PASS_TO_PASS": [],
        }
        files = {
            "instruction.md": b"Fix add so the tests pass.\n",
            # The declared per-task setup reproduces the clone+checkout state:
            # tracked files land exactly on the pinned base before bootstrap.
            "task.toml": (
                b"[environment]\ncpus = 4\nmemory_mb = 4096\nstorage_mb = 10240\n"
                b'[verifier]\nenvironment_mode = "shared"\n'
                b"\n[environment.setup]\n"
                b"command = 'git -C " + str(WORKSPACE).encode() + b" checkout " + base.encode() + b" -f -- .'\n"
            ),
            "environment/Dockerfile": b"FROM fixture.test/source@sha256:" + b"a" * 64 + b"\n",
            "tests/test.sh": SWE_REBENCH_TEST_SH.encode(),
            "tests/test.patch": SWE_REBENCH_TEST_PATCH.encode(),
            "tests/config.json": json.dumps(config).encode(),
            "tests/grade.py": SWE_REBENCH_GRADE.encode(),
            "solution/solve.sh": b"cd /testbed\n" + GOLD_PATCH.encode(),
            "solution/gold.patch": GOLD_PATCH.encode(),
        }
        return files, SWE_REBENCH_PROFILE
    if family == "scale":
        config = {
            "workdir": str(WORKSPACE),
            "base_commit": base,
            "f2p_patch": "",
            "f2p_script": "",
            "test_ids": SCALE_IDS,
        }
        profile = SCALE_SWE_PROFILE
        private = {"grade.py": SCALE_GRADE.encode(), "scale_score.py": SCALE_SCORE.encode()}
        test_sh = TEST_SH
    else:
        config = {
            "base_commit": base,
            "test_patch": LEGO_TEST_PATCH,
            "test_cmd": LEGO_TEST_CMD,
            "test_ids": LEGO_IDS,
        }
        profile = SWE_LEGO_PROFILE
        private = {"grade.py": LEGO_GRADE.encode()}
        test_sh = TEST_SH
    files = {
        "instruction.md": b"Fix add so the tests pass.\n",
        "task.toml": (
            b"[environment]\ncpus = 4\nmemory_mb = 4096\nstorage_mb = 10240\n"
            b'[verifier]\nenvironment_mode = "shared"\n'
        ),
        "environment/Dockerfile": b"FROM fixture.test/source@sha256:" + b"a" * 64 + b"\n",
        "tests/test.sh": test_sh,
        "tests/native_scripts.py": NATIVE_SCRIPTS.replace("@SHIM@", SHIM).encode(),
        "tests/config.json": json.dumps(config).encode(),
        **{f"tests/{name}": content for name, content in private.items()},
        "solution/solve.sh": b"cd /testbed\n" + GOLD_PATCH.encode(),
        "solution/gold.patch": GOLD_PATCH.encode(),
    }
    return files, profile


def stage_package(files: dict[str, bytes]) -> None:
    if BOOTSTRAP.exists():
        shutil.rmtree(BOOTSTRAP)
    BOOTSTRAP.mkdir(parents=True)
    (BOOTSTRAP / "control.py").write_bytes(files["environment/taskcompendium-shared/control.py"])
    (BOOTSTRAP / "shared_pytest.py").write_bytes(files["environment/taskcompendium-shared/shared_pytest.py"])
    private = Path(PRIVATE_DIRECTORY)
    private.mkdir(mode=0o700, parents=True)
    for name, content in files.items():
        if name.startswith(PRIVATE_CONTEXT + "/"):
            target = private / Path(name).name
            target.write_bytes(content)
            target.chmod(0o600)
    home = Path(AGENT_HOME)
    home.mkdir(parents=True, exist_ok=True)
    os.chown(home, AGENT_UID, AGENT_UID)
    for line in files["environment/Dockerfile"].decode().splitlines():
        if line.startswith("RUN "):
            subprocess.run(["/bin/sh", "-c", line.removeprefix("RUN ")], check=True)


def stage_and_bootstrap(files: dict[str, bytes], *, path: str | None = None) -> None:
    stage_package(files)
    environment = tomlkit.parse(files["task.toml"].decode())["environment"]
    command = environment["setup"]["command"]
    subprocess.run(
        ["bash", "-c", command],
        cwd="/",
        env={**os.environ, **environment["env"], "PATH": path or SHIM + ":" + os.environ["PATH"]},
        check=True,
        timeout=600,
    )
    assert not Path(SOURCE_PAYLOAD).exists()


def run_as_agent(program: str, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-I", "-c", program],
        user=AGENT_UID,
        group=AGENT_UID,
        extra_groups=(),
        env={"HOME": "/tmp", "PATH": "/usr/local/bin:/usr/bin:/bin", "GIT_CONFIG_GLOBAL": "/dev/null"},
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def run_grade(files: dict[str, bytes], timeout: float = 600) -> dict:
    result = subprocess.run(
        ["bash", "-c", files["tests/test.sh"].decode()],
        cwd="/",
        env={**os.environ, "PATH": SHIM + ":" + os.environ["PATH"]},
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    verdict = json.loads((REPORTS / "verdict.json").read_text())
    assert verdict["status"] == "scored", verdict
    assert (REPORTS / "verdict.json").stat().st_uid == 0
    assert float((REPORTS / "reward.txt").read_text()) == verdict["reward"]
    verdict["test_output"] = (REPORTS / "test_output.log").read_text()
    return verdict


def payload_of(files: dict[str, bytes]) -> dict:
    return json.loads(zlib.decompress(files[SOURCE_ARCHIVE]))


def protected_tree_hash() -> str:
    """Hash the protected root, excluding the controller's grade-time test script copy."""
    digest = hashlib.sha256()
    for path in sorted(PROTECTED.rglob("*")):
        if path.relative_to(PROTECTED).as_posix() == "original-test.sh":
            continue
        digest.update(path.relative_to(PROTECTED).as_posix().encode())
        if path.is_file() and not path.is_symlink():
            digest.update(b"\0")
            digest.update(path.read_bytes())
    return digest.hexdigest()


def apply_solution() -> None:
    result = run_as_agent(f"import pathlib\npathlib.Path('/testbed/app.py').write_text({APP_FIXED!r})\n")
    assert result.returncode == 0, result.stderr


def cleanup() -> None:
    for path in (WORKSPACE, PROTECTED, BOOTSTRAP, Path(PRIVATE_DIRECTORY), LOGS, SETUP_FILES):
        if path.exists() or path.is_symlink():
            shutil.rmtree(path) if path.is_dir() and not path.is_symlink() else path.unlink()
    for leftover in (
        "/tmp/taskcompendium-source-input.json",
        "/tmp/taskcompendium-source-test.patch",
        "/tmp/scaleswe_test_ids.json",
        "/tmp/scaleswe_results.xml",
        "/tmp/scaleswe_f2p.patch",
        "/tmp/test_patch.patch",
    ):
        Path(leftover).unlink(missing_ok=True)


@pytest.fixture
def shim() -> None:
    # The source scorer alias can change; the resolved control Python stays protected.
    directory = Path(SHIM)
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o755)
    python = directory / "python"
    python.write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    python.chmod(0o755)
    os.chown(python, AGENT_UID, AGENT_UID)
    python3 = directory / "python3"
    python3.write_text(f'#!/bin/sh\nexec {sys.executable} "$@"\n')
    python3.chmod(0o755)
    yield
    shutil.rmtree(directory, ignore_errors=True)


@pytest.fixture
def source_prefix(shim, request):
    """Use a full task Python prefix ahead of the scorer shim, as in Prime images."""
    cleanup()
    base = build_repo()
    files, profile = package_files(request.param, base)
    harden_shared_source(files, profile=profile, dependency_state=SharedDependencyState.AGENT_MUTABLE)
    (SOURCE_PREFIX / "bin").mkdir(parents=True)
    shutil.copy2(Path(sys.executable).resolve(), SOURCE_PREFIX / "bin/python3")
    (SOURCE_PREFIX / "bin/python").symlink_to("python3")
    stdlib = SOURCE_PREFIX / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    shutil.copytree(sysconfig.get_path("stdlib"), stdlib, ignore=shutil.ignore_patterns("site-packages", "__pycache__"))
    sites = stdlib / "site-packages"
    sites.mkdir()
    (sites / "fixture-dependencies.pth").write_text(f"import site;site.addsitedir({sysconfig.get_path('purelib')!r})\n")
    yield SourceTask(family=request.param, files=files, base_commit=base)
    cleanup()
    shutil.rmtree(SOURCE_PREFIX)


@pytest.mark.parametrize("source_prefix", ["scale", "lego"], indirect=True)
def test_task_python_stdlib_cannot_run_actor_code_as_root(source_prefix):
    stage_and_bootstrap(source_prefix.files, path=f"{SOURCE_PREFIX}/bin:{SHIM}:{os.environ['PATH']}")
    assert (BOOTSTRAP / "python").resolve() == SOURCE_PREFIX / "bin/python3"
    probe = f"""import pathlib
prefix=pathlib.Path({str(SOURCE_PREFIX)!r})
stdlib=prefix/'lib/python{sys.version_info.major}.{sys.version_info.minor}'
for target in (prefix/'bin/python3',stdlib/'json/__init__.py'):
    try:
        with target.open('ab') as stream:
            stream.write(b'\\nraise RuntimeError("actor ran as root")\\n')
    except PermissionError:
        continue
    raise AssertionError('actor modified root control runtime: '+str(target))
(stdlib/'site-packages/actor_dependency.py').write_text('VALUE = 42\\n')
"""
    result = run_as_agent(probe)
    assert result.returncode == 0, result.stderr
    assert run_grade(source_prefix.files)["reward"] == 0


@pytest.mark.parametrize("source_prefix", ["scale"], indirect=True)
def test_bootstrap_rejects_an_actor_writable_control_prefix(source_prefix):
    SOURCE_PREFIX.chmod(0o777)
    with pytest.raises(subprocess.CalledProcessError):
        stage_and_bootstrap(source_prefix.files, path=f"{SOURCE_PREFIX}/bin:{SHIM}:{os.environ['PATH']}")


@pytest.fixture()
def source_image(shim, request, monkeypatch):
    no_setup = request.param.endswith("-no-setup")
    preowned = request.param == "scale-preowned"
    family = "scale" if preowned else request.param.removesuffix("-no-setup")
    cleanup()
    base = build_repo()
    files, profile = package_files(family, base)
    if preowned:
        config = tomlkit.parse(files["task.toml"].decode())
        config["environment"]["setup"] = {
            "command": f"git -C /testbed reset --hard {base} && printf prepared > /testbed/public-ready.txt",
            "timeout_sec": 120.0,
        }
        files["task.toml"] = tomlkit.dumps(config).encode()
        subprocess.run(["chown", "-R", f"{AGENT_UID}:{AGENT_UID}", str(WORKSPACE)], check=True)
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(BOOTSTRAP / "source.gitconfig"))
    harden_shared_source(files, profile=profile, dependency_state=SharedDependencyState.AGENT_MUTABLE)
    if no_setup:
        stage_package(files)
    else:
        stage_and_bootstrap(files)
    yield SourceTask(family=family, files=files, base_commit=base)
    cleanup()


@pytest.mark.parametrize("source_image", ["scale-no-setup", "lego-no-setup", "swe-rebench-no-setup"], indirect=True)
def test_actor_cannot_read_bootstrap_payload_when_runner_skips_setup(source_image):
    payload = Path(SOURCE_PAYLOAD)
    assert payload.read_bytes()
    assert WORKSPACE.stat().st_uid == AGENT_UID
    probe = (
        "from pathlib import Path\n"
        "try:\n"
        f"    Path({str(payload)!r}).read_bytes()\n"
        "except PermissionError:\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('actor read private bootstrap payload')\n"
    )
    result = run_as_agent(probe)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("source_image", ["scale", "lego", "swe-rebench"], indirect=True)
@pytest.mark.parametrize("phase", ["nop", "oracle"])
def test_original_control_reward(source_image, phase):
    if phase == "oracle":
        apply_solution()
    verdict = run_grade(source_image.files)
    assert verdict["reward"] == (1.0 if phase == "oracle" else 0.0), verdict


@pytest.mark.parametrize("source_image", ["scale"], indirect=True)
def test_bootstrap_preserves_the_exact_pre_bootstrap_state(source_image):
    tests = WORKSPACE / "tests"
    scratch = tests / "scratch_util.py"
    link = tests / "link_to_app.py"
    assert scratch.read_text() == "VALUE = 'preserve'\n"
    assert stat.S_IMODE(scratch.stat().st_mode) == 0o600
    assert link.is_symlink() and os.readlink(link) == "../app.py"
    assert (WORKSPACE / "untracked_note.txt").read_text() == "untracked\n"
    # The agent-facing tree keeps HEAD's test bytes until grading.
    assert (tests / "test_app.py").read_text() == HEAD_TEST
    assert (WORKSPACE / "pkg/conftest.py").read_text() == "NESTED_CONFTEST = 'base'\n"


@pytest.mark.parametrize("source_image", ["scale"], indirect=True)
def test_restore_uses_the_declared_base_commit_not_head(source_image):
    # HEAD's test bytes pass with the buggy app; only a real $base restoration
    # keeps the NoP grade at zero.
    verdict = run_grade(source_image.files)
    assert verdict["reward"] == 0.0
    assert (WORKSPACE / "tests/test_app.py").read_text() == SCALE_TEST


@pytest.mark.parametrize("source_image", ["scale"], indirect=True)
def test_agent_edited_tests_and_forged_git_objects_cannot_fake_a_pass(source_image):
    program = (
        "import pathlib, subprocess\n"
        f"pathlib.Path('/testbed/tests/test_app.py').write_text({FAKE_TEST!r})\n"
        "def git(*args): subprocess.run(['git', *args], cwd='/testbed', capture_output=True)\n"
        "git('add', '-A')\n"
        "git('-c', 'user.email=a@b', '-c', 'user.name=a', 'commit', '-qm', 'fake')\n"
        "head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd='/testbed', capture_output=True, text=True)"
        ".stdout.strip()\n"
        f"git('replace', '{source_image.base_commit}', head)\n"
    )
    result = run_as_agent(program)
    assert result.returncode == 0, result.stderr
    verdict = run_grade(source_image.files)
    assert verdict["reward"] == 0.0
    # The frozen reference and the pinned .git both reverted the agent state.
    assert (WORKSPACE / "tests/test_app.py").read_text() == SCALE_TEST
    assert not list((WORKSPACE / ".git/refs/replace").glob("*"))


@pytest.mark.parametrize("source_image", ["lego"], indirect=True)
def test_committed_solution_survives_pristine_git_substitution(source_image):
    program = (
        "import pathlib, subprocess\n"
        f"pathlib.Path('/testbed/app.py').write_text({APP_FIXED!r})\n"
        "subprocess.run(['git', 'add', '-A'], cwd='/testbed', capture_output=True)\n"
        "subprocess.run(['git', '-c', 'user.email=a@b', '-c', 'user.name=a', 'commit', '-qm', 'fix'],"
        " cwd='/testbed', capture_output=True)\n"
    )
    result = run_as_agent(program)
    assert result.returncode == 0, result.stderr
    verdict = run_grade(source_image.files)
    assert verdict["reward"] == 1.0, verdict
    # The agent's commit history never reaches grading; the working tree survives.
    assert (WORKSPACE / "app.py").read_text() == APP_FIXED


@pytest.mark.parametrize("source_image", ["lego"], indirect=True)
def test_agent_created_patch_path_file_is_discarded(source_image):
    fake = (
        "import sys, pathlib\n"
        "sys.path.insert(0, str(pathlib.Path('/testbed')))\n"
        "from app import add\n"
        "\n"
        "def test_add_fixed():\n    assert True\n"
    )
    program = (
        "import pathlib\n"
        "pathlib.Path('/testbed/tests/nested').mkdir(parents=True)\n"
        f"pathlib.Path('/testbed/tests/nested/test_fixed.py').write_text({fake!r})\n"
    )
    result = run_as_agent(program)
    assert result.returncode == 0, result.stderr
    assert run_grade(source_image.files)["reward"] == 0.0
    # The frozen reference discarded the agent file and the patch recreated the real one.
    assert "assert add(1, 2) == 3" in (WORKSPACE / "tests/nested/test_fixed.py").read_text()


@pytest.mark.parametrize("source_image", ["lego"], indirect=True)
def test_hostile_nested_parent_symlink_cannot_redirect_patch_paths(source_image):
    before = protected_tree_hash()
    program = (
        "import pathlib\n"
        "nested = pathlib.Path('/testbed/tests/nested')\n"
        "nested.mkdir(parents=True)\n"
        "nested_rmdir = nested\n"
        "import shutil\n"
        "shutil.rmtree(nested_rmdir)\n"
        "nested.symlink_to('/opt/taskcompendium-shared/tests')\n"
    )
    result = run_as_agent(program)
    assert result.returncode == 0, result.stderr
    assert run_grade(source_image.files)["reward"] == 0.0
    # The patch path lands inside a repaired real directory, and the protected
    # root the symlink pointed at is byte-identical afterwards.
    assert not (WORKSPACE / "tests/nested").is_symlink()
    assert "assert add(1, 2) == 3" in (WORKSPACE / "tests/nested/test_fixed.py").read_text()
    assert protected_tree_hash() == before


@pytest.mark.parametrize("source_image", ["scale"], indirect=True)
def test_nested_parent_symlink_cannot_redirect_restore(source_image):
    before = protected_tree_hash()
    program = (
        "import shutil, pathlib\n"
        "pkg = pathlib.Path('/testbed/pkg')\n"
        "shutil.rmtree(pkg)\n"
        "pkg.symlink_to('/opt/taskcompendium-shared/reference')\n"
    )
    result = run_as_agent(program)
    assert result.returncode == 0, result.stderr
    assert run_grade(source_image.files)["reward"] == 0.0
    assert not (WORKSPACE / "pkg").is_symlink()
    assert (WORKSPACE / "pkg/conftest.py").read_text() == "NESTED_CONFTEST = 'base'\n"
    assert protected_tree_hash() == before


@pytest.mark.parametrize("source_image", ["scale", "lego", "swe-rebench"], indirect=True)
def test_actor_cannot_read_hidden_grading_material(source_image):
    private = [PROTECTED / "tests/config.json", PROTECTED / "tests/grade.py"]
    for path in private:
        assert path.is_file()
        assert path.read_bytes()
    probe = (
        "import os, pathlib\n"
        f"assert os.getuid() == {AGENT_UID}\n"
        f"for name in {[str(path) for path in private]!r}:\n"
        "    try:\n"
        "        pathlib.Path(name).read_bytes()\n"
        "    except PermissionError:\n"
        "        continue\n"
        "    raise AssertionError('Actor read hidden grader: ' + name)\n"
    )
    result = run_as_agent(probe)
    assert result.returncode == 0, result.stderr
    assert run_grade(source_image.files)["reward"] == 0.0


@pytest.mark.parametrize("source_image", ["scale", "lego"], indirect=True)
def test_agent_cannot_read_protected_material_or_forge_verdicts(source_image):
    # Model a source image that pre-creates a permissive /logs; a privileged
    # pre-created report directory still cannot smuggle a forged reward in.
    REPORTS.mkdir(parents=True, exist_ok=True)
    REPORTS.chmod(0o777)
    baseline = PROTECTED / "dependency-baseline.json"
    before = baseline.read_bytes()
    forge = (
        "import os, pathlib\n"
        "pathlib.Path('/logs/verifier/reward.txt').write_text('1')\n"
        "pathlib.Path('/logs/verifier/site-packages-diff.json').symlink_to('/opt/taskcompendium-shared/dependency-baseline.json')\n"
        "pathlib.Path('/logs/verifier/startup-hooks-diff.json').symlink_to('/opt/taskcompendium-shared/dependency-baseline.json')\n"
        "for target in ('/opt/taskcompendium-shared/contract.json',"
        " '/opt/taskcompendium-shared/tests/config.json',"
        " '/opt/taskcompendium-shared/materialize.patch'):\n"
        "    try:\n"
        "        pathlib.Path(target).read_bytes()\n"
        "    except OSError:\n"
        "        pass\n"
        "    else:\n"
        "        raise SystemExit('agent read protected material: ' + target)\n"
        "if os.path.exists('/setup_files/taskcompendium-source-input.zlib'):\n"
        "    raise SystemExit('private payload archive survived bootstrap')\n"
    )
    result = run_as_agent(forge)
    assert result.returncode == 0, result.stderr
    assert run_grade(source_image.files)["reward"] == 0.0
    assert baseline.read_bytes() == before


@pytest.mark.parametrize("source_image", ["scale"], indirect=True)
def test_background_writer_cannot_fake_a_pass(source_image):
    REPORTS.mkdir(parents=True, exist_ok=True)
    REPORTS.chmod(0o777)
    child_code = (
        "from pathlib import Path\n"
        "reward = Path('/logs/verifier/reward.txt')\n"
        "reward.write_text('1\\n')\n"
        "print('ready', flush=True)\n"
        "while True:\n"
        "    reward.write_text('1\\n')\n"
    )
    program = (
        "import subprocess, sys\n"
        "child = subprocess.Popen("
        f"[sys.executable, '-c', {child_code!r}], "
        "stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)\n"
        "assert child.stdout.readline().strip() == 'ready'\n"
    )
    result = run_as_agent(program)
    assert result.returncode == 0, result.stderr
    assert run_grade(source_image.files)["reward"] == 0.0


@pytest.mark.parametrize("source_image", ["lego"], indirect=True)
def test_aborted_scorer_cannot_forge_a_reward(source_image):
    """A same-UID child that forges the reward and kills its process group fails closed."""
    victim = PROTECTED / "untouched.txt"
    victim.write_text("private material\n")
    poison = (
        "import os, pathlib, signal\n"
        "pathlib.Path('/logs/verifier').mkdir(parents=True, exist_ok=True)\n"
        "pathlib.Path('/logs/verifier/reward.txt').write_text('1')\n"
        "pathlib.Path('/logs/verifier/reward.json').write_text('{\"reward\":1}')\n"
        "for name in ('reward.txt', 'verdict.json'):\n"
        "    path = pathlib.Path('/logs/verifier') / name\n"
        "    path.unlink(missing_ok=True)\n"
        "    path.symlink_to('/opt/taskcompendium-shared/untouched.txt')\n"
        "os.killpg(os.getpgrp(), signal.SIGKILL)\n"
    )
    program = f"import pathlib\npathlib.Path('/testbed/conftest.py').write_text({poison!r})\n"
    result = run_as_agent(program)
    assert result.returncode == 0, result.stderr
    verdict = run_grade(source_image.files)
    assert verdict["reward"] == 0.0
    assert verdict["detail"]["reason"] == "scorer_aborted"
    assert float((REPORTS / "reward.txt").read_text()) == 0.0
    assert not (REPORTS / "reward.json").exists()
    assert victim.read_text() == "private material\n"


@pytest.mark.parametrize("source_image", ["lego"], indirect=True)
def test_source_scorer_cannot_override_reward_with_harbor_json(source_image):
    poison = "from pathlib import Path\nPath('/logs/verifier/reward.json').write_text('{\"reward\":1}')\n"
    result = run_as_agent(f"from pathlib import Path;Path('/testbed/conftest.py').write_text({poison!r})")
    assert result.returncode == 0, result.stderr
    assert run_grade(source_image.files)["reward"] == 0
    assert not (REPORTS / "reward.json").exists()


@pytest.mark.parametrize("source_image", ["scale"], indirect=True)
def test_hijacked_scorer_interpreter_cannot_forge_a_reward(source_image):
    """Replacing the scorer's interpreter binary still fails closed when it aborts."""
    poison = (
        "#!/bin/sh\n"
        "mkdir -p /logs/verifier\n"
        "echo 1 > /logs/verifier/reward.txt\n"
        "kill -9 -\"$(cut -d ' ' -f5 /proc/self/stat)\"\n"
    )
    program = (
        "import pathlib\n"
        f"python = pathlib.Path('{SHIM}/python')\n"
        f"python.write_text({poison!r})\n"
        "python.chmod(0o755)\n"
    )
    result = run_as_agent(program)
    assert result.returncode == 0, result.stderr
    verdict = run_grade(source_image.files)
    assert verdict["reward"] == 0.0
    assert verdict["detail"]["reason"] == "scorer_aborted"


@pytest.mark.parametrize("source_image", ["scale"], indirect=True)
def test_agent_symlinked_tests_directory_cannot_reach_protected_state(source_image):
    program = (
        "import shutil, pathlib\n"
        "tests = pathlib.Path('/testbed/tests')\n"
        "shutil.rmtree(tests)\n"
        "tests.symlink_to('/opt/taskcompendium-shared/reference')\n"
    )
    result = run_as_agent(program)
    assert result.returncode == 0, result.stderr
    assert run_grade(source_image.files)["reward"] == 0.0
    restored = WORKSPACE / "tests"
    assert not restored.is_symlink()
    assert (restored / "test_app.py").is_file()


@pytest.mark.parametrize("source_image", ["swe-rebench"], indirect=True)
def test_swe_rebench_reward_channel_and_test_files_scope(source_image):
    """The SWE-rebench flow: verbatim-style test.sh writes the reward file; the
    declared test_files list joins the protected grading scope. The agent-phase
    tampering happens before the single grading, exactly as production runs."""
    # Agent phase: tamper with a declared test_files path and shadow the
    # patch-created test, on the unpatched buggy source.
    tamper = run_as_agent(
        "import pathlib\n"
        "base = pathlib.Path('/testbed')\n"
        "(base / 'tests' / 'test_app.py').write_text('def test_resolved():\\n    assert True\\n')\n"
        "(base / 'tests' / 'test_new.py').write_text('def test_resolved():\\n    assert True\\n')\n"
    )
    assert tamper.returncode == 0, tamper.stderr
    # Grading neutralizes both: the declared test_files path is restored to the
    # frozen reference bytes and the shadowing patch-path file is removed
    # before the source's test.sh reapplies the real patch. The unpatched
    # source fails the F2P test, so the reward stays 0.
    verdict = run_grade(source_image.files)
    assert verdict["reward"] == 0.0, verdict
    assert (WORKSPACE / "tests" / "test_app.py").read_text() == SCALE_TEST
    assert (WORKSPACE / "tests/test_new.py").is_file(), verdict
    assert "assert add(1, 2) == 3" in (WORKSPACE / "tests" / "test_new.py").read_text()


def contract_workspace_of(files: dict[str, bytes]) -> str:
    payload = json.loads(zlib.decompress(files[SOURCE_ARCHIVE]))
    return payload["contract"]["workspace"]


@pytest.mark.parametrize("source_image", ["scale-preowned"], indirect=True)
def test_source_git_setup_runs_after_build_time_uid_handover(source_image):
    assert (WORKSPACE / "public-ready.txt").read_text() == "prepared"
    result = run_as_agent(
        "import subprocess\n"
        "head = subprocess.check_output(['git', '-C', '/testbed', 'rev-parse', 'HEAD'], text=True).strip()\n"
        f"assert head == {source_image.base_commit!r}\n"
    )
    assert result.returncode == 0, result.stderr
    assert run_grade(source_image.files)["reward"] == 0.0
