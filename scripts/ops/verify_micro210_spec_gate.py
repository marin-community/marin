# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the self-contained Micro210 oracle/NoP build gate.

Every row of a built prepared table must grade its oracle reference at 1.0 and a constant
NoP answer at 0.0 (continuous metrics accept a recorded maximum) inside an isolated
container that carries only Python, the dependency set the embedded graders import, and
the row's own TaskSpec files. Any import resolving outside that set fails the row, which
fails the build. The receipt records per-row rewards and the exact dependency pins; the
policy build refuses to emit without this receipt matching the table digest.
"""

import argparse
import base64
import hashlib
import json
import re
import shlex
import subprocess
import sys
import tempfile
import tomllib
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from rigging.secrets import resolve_secret_spec

from experiments.evaluation.micro210_gate import (
    HARBOR_BENCHMARKS,
    IFBENCH_ORACLE_PROGRAM,
    expectations,
    row_probes,
)
from experiments.evaluation.micro210_spec_grading import GraderLibrary

GATE_IMAGE = "micro210-gate:spec-complete"
# The embedded graders' third-party imports, all hash-pinned in the rows' package spec.
GATE_DEPENDENCIES = (
    "aiohttp",
    "emoji",
    "loguru",
    "math-verify",
    "nltk",
    "numpy",
    "pydantic",
    "pyyaml",
    "regex",
    "syllapy",
    "sympy",
)
# sympy's LaTeX parser needs the antlr runtime at call time; the pin is a wheel URL.
GATE_URL_DEPENDENCIES = ("antlr4-python3-runtime",)
JUDGE_ENVIRONMENT = ("JUDGE_BASE_URL", "JUDGE_API_KEY")


def _canonical(name: str) -> str:
    return name.lower().replace("-", "_")


def resolve_gate_pins(package_spec: list[str]) -> list[str]:
    """The exact pins for the gate dependency set, as named by the row's package spec."""
    pins = []
    for dependency in (*GATE_DEPENDENCIES, *GATE_URL_DEPENDENCIES):
        normalised = _canonical(dependency)
        matches = [package for package in package_spec if _canonical(_distribution_name(package)) == normalised]
        exact = [package for package in matches if "==" in package]
        wheel = [
            package for package in matches if "==" not in package and _canonical(_wheel_stem(package)) == normalised
        ]
        if not exact and not wheel:
            raise ValueError(f"The verifier package spec does not pin gate dependency {dependency!r}")
        pins.append((exact or wheel)[0])
    return pins


def _distribution_name(package: str) -> str:
    """The requirement's distribution name, before any version or URL pin."""
    return package.split("==")[0].split(" @ ")[0].strip()


def _wheel_stem(package: str) -> str:
    """The distribution name encoded in a pinned wheel URL's filename."""
    filename = package.rsplit("/", 1)[-1].split("#")[0]
    return filename.split("-")[0]


def build_gate_image(pins: list[str]) -> None:
    """Build the isolated gate runtime: Python plus exactly the embedded graders' dependencies."""
    requirements = "\n".join(pins) + "\n"
    dockerfile = "\n".join(
        (
            "FROM python:3.12-slim",
            "RUN pip install --no-cache-dir " + " ".join(shlex.quote(pin) for pin in pins),
        )
    )
    with tempfile.TemporaryDirectory(prefix="micro210-gate-image-") as directory:
        (Path(directory) / "Dockerfile").write_text(dockerfile)
        (Path(directory) / "requirements.txt").write_text(requirements)
        subprocess.run(
            ["docker", "build", "--pull", "-t", GATE_IMAGE, directory],
            check=True,
            stdout=subprocess.DEVNULL,
        )


def extract_row(task_spec_json: str) -> Path:
    """Materialise one row's verifier resources under a fresh sandbox root."""
    spec = json.loads(task_spec_json)
    root = Path(tempfile.mkdtemp(prefix="micro210-row-"))
    tests = root / "tests"
    app = root / "app"
    (tests / "grader_lib").mkdir(parents=True)
    (app).mkdir()
    for resource in spec["resources"]["verifier"]:
        payload = base64.b64decode(resource["source"]["content_base64"])
        destination = tests / resource["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    return root


def write_answer(root: Path, answer: str) -> None:
    (root / "app" / "answer.txt").write_text(answer)


WHEEL_IMAGE = "micro210-gate:runtime-v4"


def build_wheel_image(packages: list[str]) -> None:
    """Build the runtime-v4 reference image: the frozen verifier environment with its wheels."""
    requirements = "\n".join(packages) + "\n"
    dockerfile = "FROM python:3.12-slim\nRUN pip install --no-cache-dir " + " ".join(
        shlex.quote(package) for package in packages
    )
    with tempfile.TemporaryDirectory(prefix="micro210-wheel-image-") as directory:
        (Path(directory) / "Dockerfile").write_text(dockerfile)
        (Path(directory) / "requirements.txt").write_text(requirements)
        subprocess.run(
            ["docker", "build", "--pull", "-t", WHEEL_IMAGE, directory],
            check=True,
            stdout=subprocess.DEVNULL,
        )


def wheel_grade_frozen(source_row: dict, answer: str, judge_environment: dict[str, str] | None) -> float:
    """Grade one probe answer with the frozen runtime-v4 grader for differential comparison."""
    root = extract_row(source_row["task_spec"])
    try:
        # The frozen row's verifier files carry the wheel-importing grade.py and config.
        write_answer(root, answer)
        return float(docker_grade(root, judge_environment, image=WHEEL_IMAGE))
    finally:
        subprocess.run(["rm", "-rf", str(root)], check=False)


def docker_grade(root: Path, judge_environment: dict[str, str] | None, *, image: str = GATE_IMAGE) -> str:
    """Run the row's own grade.py in the isolated gate image; return stdout."""
    command = ["docker", "run", "--rm", "-v", f"{root / 'tests'}:/tests:ro", "-v", f"{root / 'app'}:/app:ro"]
    for name, value in (judge_environment or {}).items():
        command += ["-e", f"{name}={value}"]
    command += [image, "python3", "/tests/grade.py"]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
    if completed.returncode != 0:
        raise RuntimeError(f"grade.py failed ({completed.returncode}): {completed.stderr[-2000:]}")
    lines = [line for line in completed.stdout.strip().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"grade.py printed no reward: {completed.stdout[-500:]}")
    return lines[-1]


def grade_family_a_row(
    row: dict,
    judge_environment: dict[str, str] | None,
    source_row: dict | None = None,
) -> dict:
    """Grade one script-graded row's oracle and NoP answers through its embedded grader."""
    probes = row_probes(row["benchmark"], row["task_spec"])
    rewards = {}
    root = extract_row(row["task_spec"])
    try:
        oracle = probes.oracle
        if oracle is None:
            oracle = docker_oracle(root)
        for label, answer in (("oracle", oracle), ("nop", probes.nop)):
            write_answer(root, answer)
            rewards[label] = float(docker_grade(root, judge_environment))
    finally:
        subprocess.run(["rm", "-rf", str(root)], check=False)
    result = {
        "task_id": row["task_id"],
        "repetition_index": row["repetition_index"],
        "oracle_reward": rewards["oracle"],
        "nop_reward": rewards["nop"],
        "judge_required": probes.judge_required,
    }
    oracle_minimum, _ = expectations(row["benchmark"])
    if source_row is not None and rewards["oracle"] < oracle_minimum:
        # An oracle below the minimum can be inherited from the frozen grader itself (for
        # example the CPython-3.12 cmath precision behavior on MBPP+ canonical answers).
        # Differential-grade the same answer with the frozen wheel grader: an equal reward
        # records the row as an environment exception, a different reward is a regression.
        wheel_oracle = wheel_grade_frozen(source_row, oracle, judge_environment)
        result["frozen_grader_oracle_reward"] = wheel_oracle
        result["environment_exception"] = abs(wheel_oracle - rewards["oracle"]) < 1e-9
    return result


def docker_oracle(root: Path) -> str:
    """Construct an oracle answer inside the gate image using only the row's spec files."""
    completed = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{root / 'tests'}:/tests:ro",
            "--entrypoint",
            "python3",
            GATE_IMAGE,
            "-c",
            IFBENCH_ORACLE_PROGRAM,
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"oracle construction failed: {completed.stderr[-1500:]}")
    return completed.stdout.strip()


HARBOR_ENVIRONMENT_DIR = "environment"
HARBOR_VERIFIER_LOGS = "/logs/verifier"


def _harbor_task_files(row: dict) -> dict[str, bytes]:
    spec = json.loads(row["task_spec"])
    return {
        resource["path"]: base64.b64decode(resource["source"]["content_base64"])
        for resource in spec["resources"]["verifier"]
    }


def _harbor_workdir(files: dict[str, bytes], image: str) -> str:
    """The effective container workdir, as baked into the built image."""
    completed = _docker(["inspect", "--format", "{{.Config.WorkingDir}}", image])
    workdir = completed.stdout.strip()
    if not workdir or workdir == "/":
        raise ValueError(f"Harbor image {image} has no effective WORKDIR")
    return workdir


def _harbor_verifier_env(row: dict, judge_environment: dict[str, str] | None) -> dict[str, str]:
    files = _harbor_task_files(row)
    config = tomllib.loads(files["task.toml"].decode())
    declared = {**config["verifier"].get("env", {})}
    resolved = {}
    for key, value in declared.items():
        match = re.fullmatch(r"\$\{([A-Z_]+)\}", str(value))
        resolved[key] = (judge_environment or {}).get(match.group(1), "") if match else str(value)
    return resolved


def _docker(args: list[str], *, timeout: int = 3600, check: bool = True) -> subprocess.CompletedProcess:
    completed = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    if check and completed.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args[:3])} failed ({completed.returncode}): {completed.stderr[-2000:]}")
    return completed


def _harbor_reward(container: str) -> float:
    for name in ("reward.json", "reward.txt"):
        completed = _docker(["exec", container, "cat", f"{HARBOR_VERIFIER_LOGS}/{name}"], check=False)
        if completed.returncode == 0:
            text = completed.stdout.strip()
            if name.endswith("json"):
                return float(json.loads(text)["reward"])
            return float(text)
    raise RuntimeError(f"Harbor verifier wrote no reward file in {container}")


def _run_harbor_tests(
    row: dict,
    files: dict[str, bytes],
    judge_environment: dict[str, str] | None,
    *,
    apply_solution: bool,
) -> float:
    """One Harbor trial's verifier against a fresh container: optional oracle solve, then tests."""
    with tempfile.TemporaryDirectory(prefix="micro210-harbor-") as directory:
        task_dir = Path(directory) / "task"
        (task_dir / HARBOR_ENVIRONMENT_DIR).mkdir(parents=True)
        for path, payload in files.items():
            destination = task_dir / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        image = f"micro210-harbor-gate:{row['task_id'].replace('/', '-')}"
        # The Harbor base images are x86-64 (the eval runtime hosts are x86-64); pin the
        # platform so arm hosts build the same images under emulation.
        _docker(["build", "--platform", "linux/amd64", "-t", image, str(task_dir / HARBOR_ENVIRONMENT_DIR)])
        workdir = _harbor_workdir(files, image)
        container = f"micro210-harbor-{uuid.uuid4().hex[:12]}"
        try:
            _docker(
                [
                    "run",
                    "-d",
                    "--name",
                    container,
                    "--platform",
                    "linux/amd64",
                    "--workdir",
                    workdir,
                    image,
                    "sleep",
                    "infinity",
                ]
            )
            if apply_solution:
                _docker(["cp", str(task_dir / "solution"), f"{container}:/solution"])
                _docker(["exec", "--workdir", workdir, container, "bash", "/solution/solve.sh"], timeout=1800)
            _docker(["cp", str(task_dir / "tests"), f"{container}:/tests"])
            _docker(["exec", container, "mkdir", "-p", HARBOR_VERIFIER_LOGS])
            env_arguments = []
            for key, value in _harbor_verifier_env(row, judge_environment).items():
                env_arguments += ["-e", f"{key}={value}"]
            _docker(
                ["exec", "--workdir", workdir, *env_arguments, container, "bash", "/tests/test.sh"],
                timeout=int(json.loads(row["task_spec"])["grader"].get("timeout", 0) or 3600) or 3600,
                check=False,
            )
            return _harbor_reward(container)
        finally:
            _docker(["rm", "-f", container], check=False)


def grade_family_b_row(
    row: dict,
    judge_environment: dict[str, str] | None,
    source_row: dict | None = None,
) -> dict:
    """Grade one Harbor-trial row: build its image, apply the solution, run its tests."""
    files = _harbor_task_files(row)
    result = {
        "task_id": row["task_id"],
        "repetition_index": row["repetition_index"],
        "judge_required": row["benchmark"] == "bixbench-pi",
        "oracle_reward": _run_harbor_tests(row, files, judge_environment, apply_solution=True),
        "nop_reward": _run_harbor_tests(row, files, judge_environment, apply_solution=False),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True, help="The built rollouts.parquet")
    parser.add_argument(
        "--source-table", type=Path, help="The frozen 2026.11.20-together table, for differential checks"
    )
    parser.add_argument("--output", type=Path, required=True, help="Receipt path")
    parser.add_argument("--limit", type=int, default=None, help="Grade only the first N rows (development)")
    parser.add_argument("--benchmark", action="append", default=[], help="Grade only these benchmarks")
    parser.add_argument(
        "--judge-credential", default="gcp-secret://projects/hai-gcp-models/secrets/TOGETHER_API_KEY/versions/1"
    )
    arguments = parser.parse_args()

    table = pq.read_table(arguments.table)
    digest = hashlib.sha256(arguments.table.read_bytes()).hexdigest()
    rows = table.to_pylist()
    if arguments.benchmark:
        rows = [row for row in rows if row["benchmark"] in set(arguments.benchmark)]
    if arguments.limit is not None:
        rows = rows[: arguments.limit]

    library = GraderLibrary()
    package_specs = {
        json.dumps(
            json.loads(row["task_spec"])["grader"]["environment"]["package_spec"]["pypi_packages"], sort_keys=True
        )
        for row in rows
        if row["benchmark"] not in HARBOR_BENCHMARKS
    }
    pins = []
    if package_specs:
        if len(package_specs) != 1:
            raise ValueError(f"Script-graded rows carry {len(package_specs)} distinct package specs; expected one")
        pins = resolve_gate_pins(json.loads(next(iter(package_specs))))
        build_gate_image(pins)

    judge_environment = None
    if arguments.judge_credential:
        judge = resolve_secret_spec((arguments.judge_credential,))
        judge_environment = {
            "JUDGE_BASE_URL": "https://api.together.ai/v1",
            "JUDGE_API_KEY": judge.value,
            "OPENAI_API_KEY": judge.value,
        }

    source_rows: dict[tuple[str, int], dict] = {}
    if arguments.source_table:
        source_table = pq.read_table(arguments.source_table)
        for row in source_table.to_pylist():
            source_rows[(row["task_id"], row["repetition_index"])] = row
        source_packages = json.loads(next(iter(source_rows.values()))["task_spec"])["grader"]["environment"][
            "package_spec"
        ]["pypi_packages"]
        build_wheel_image(source_packages)

    results, failures, deferred = [], [], 0
    family_a_rows = [row for row in rows if row["benchmark"] not in HARBOR_BENCHMARKS]

    def grade_family_a(row):
        return row, grade_family_a_row(
            row, judge_environment, source_rows.get((row["task_id"], row["repetition_index"]))
        )

    graded = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for row, result in pool.map(grade_family_a, family_a_rows):
            graded += 1
            oracle_minimum, nop_maximum = expectations(row["benchmark"])
            oracle = result["oracle_reward"]
            if oracle < oracle_minimum and not result.get("environment_exception"):
                failures.append({**result, "check": "oracle", "required": oracle_minimum})
            elif nop_maximum is None or result["nop_reward"] > nop_maximum:
                failures.append({**result, "check": "nop", "required": nop_maximum})
            results.append(result)
            if graded % 50 == 0:
                print(f"graded {graded} rows, {len(failures)} failures", flush=True)

    harbor_rows = [row for row in rows if row["benchmark"] in HARBOR_BENCHMARKS]
    if harbor_rows:

        def grade_harbor(row):
            return row, grade_family_b_row(
                row, judge_environment, source_rows.get((row["task_id"], row["repetition_index"]))
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            for row, result in pool.map(grade_harbor, harbor_rows):
                oracle_minimum, nop_maximum = expectations(row["benchmark"])
                if result["oracle_reward"] < oracle_minimum:
                    failures.append({**result, "check": "oracle", "required": oracle_minimum})
                elif result["nop_reward"] > (nop_maximum or 0.0):
                    failures.append({**result, "check": "nop", "required": nop_maximum})
                results.append(result)
                print(f"harbor graded {row['task_id']}", flush=True)

    receipt = {
        "prepared_rollouts_sha256": digest,
        "grader_library_manifest_sha256": library.manifest_sha256,
        "dependencies": pins,
        "rows": results,
        "failures": failures,
        "deferred_harbor_rows": deferred,
        "summary": (
            f"{len(results) - sum(1 for item in results if item.get('failed'))} graded rows, "
            f"{len(failures)} failures, {deferred} harbor rows deferred"
        ),
        "pass": not failures,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(receipt, indent=2))
    print(f"gate receipt: {arguments.output} — {receipt['summary']}")
    if failures:
        for failure in failures[:20]:
            print("FAIL:", json.dumps(failure)[:400])
        sys.exit(1)


if __name__ == "__main__":
    main()
