# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import ast
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import pytest
import yaml

REPO_ROOT = Path(__file__).parents[3]
DOCKERFILE = REPO_ROOT / "lib" / "iris" / "Dockerfile"
DOCKERIGNORE = REPO_ROOT / "lib" / "iris" / "Dockerfile.dockerignore"
PACKAGE_MANIFEST = REPO_ROOT / "lib" / "iris" / "images" / "h100-evidence-debian12-amd64.sha256"
NSYS_HELP_VALIDATOR = REPO_ROOT / "lib" / "iris" / "images" / "h100_evidence_nsys_help.py"
NVTX_SMOKE = REPO_ROOT / "lib" / "iris" / "images" / "h100_evidence_nvtx_smoke.py"
SASS_VALIDATOR = REPO_ROOT / "lib" / "iris" / "images" / "h100_evidence_sass_smoke.py"
H100_IMAGE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ops-h100-evidence-image.yaml"
BROAD_IMAGE_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ops-docker-images.yaml"
ROOT_PYPROJECT = REPO_ROOT / "pyproject.toml"
TILE_LIFETIME_PYPROJECT = REPO_ROOT / "lib" / "tile_lifetime" / "pyproject.toml"
UV_LOCK = REPO_ROOT / "uv.lock"
H100_RUNNER = REPO_ROOT / "lib" / "tile_lifetime" / "benchmarks" / "h100_contract_map_backend_runner.py"
NVTX_RANGE = REPO_ROOT / "lib" / "tile_lifetime" / "src" / "tile_lifetime" / "nvtx_range.py"
BACKEND_RESOURCES = REPO_ROOT / "lib" / "tile_lifetime" / "src" / "tile_lifetime" / "contract_map_backend_resources.py"
NVIDIA_DEBIAN12_AMD64_REPOSITORY = "https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64"

CUOBJDUMP_SASS = """\
        code for sm_90a
                Function : h100_evidence_smoke
        .headerflags    @"EF_CUDA_SM90A EF_CUDA_VIRTUAL_SM(EF_CUDA_SM90A)"
        /*0000*/                   LDC R1, c[0x0][0x37c] ;       /* 0x0000df00ff017b82 */
                                                                 /* 0x000fe20000000800 */
        /*0010*/              @!P0 S2R R9, SR_TID.X ;            /* 0x0000000000097919 */
                                                                 /* 0x000e2e0000002100 */
"""
NVDISASM_BODY = """\
        /*0000*/                   LDC R1, c[0x0][0x37c] ;
        /*0010*/                   S2R R9, SR_TID.X ;
"""
NVDISASM_SASS = (
    """\
        .section        .nv.constant0.h100_evidence_smoke,"a",@progbits
        /*0000*/ \t.byte\t0xff, 0xff, 0xff, 0xff, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff
        .section        .text.h100_evidence_smoke,"ax",@progbits
        .sectioninfo    @"SHI_REGISTERS=6"
        .global         h100_evidence_smoke
        .type           h100_evidence_smoke,@function
h100_evidence_smoke:
.text.h100_evidence_smoke:
"""
    + NVDISASM_BODY
)

EXPECTED_PACKAGES = {
    "cuda-cccl-13-2_13.2.86-1_amd64.deb",
    "cuda-crt-13-2_13.2.86-1_amd64.deb",
    "cuda-cudart-13-2_13.2.86-1_amd64.deb",
    "cuda-cudart-dev-13-2_13.2.86-1_amd64.deb",
    "cuda-culibos-dev-13-2_13.2.86-1_amd64.deb",
    "cuda-cuobjdump-13-2_13.2.86-1_amd64.deb",
    "cuda-driver-dev-13-2_13.2.86-1_amd64.deb",
    "cuda-nvcc-13-2_13.2.86-1_amd64.deb",
    "cuda-nvdisasm-13-2_13.2.86-1_amd64.deb",
    "cuda-nvtx-13-2_13.2.86-1_amd64.deb",
    "cuda-toolkit-13-2-config-common_13.2.86-1_all.deb",
    "cuda-toolkit-13-config-common_13.2.86-1_all.deb",
    "cuda-toolkit-config-common_13.2.86-1_all.deb",
    "libnvptxcompiler-13-2_13.2.86-1_amd64.deb",
    "libnvvm-13-2_13.2.86-1_amd64.deb",
    "nsight-compute-2026.1.1_2026.1.1.2-1_amd64.deb",
}

LEGACY_IMAGE_JOBS = {
    "iris-tags",
    "iris-images",
    "iris-manifests",
    "finelog-image",
    "marin-tpu-ci-images",
}
WEEKLY_IMAGE_JOBS = {"iris-tags", "iris-images", "iris-manifests", "finelog-image"}
MANUAL_ONLY_PATTERN = re.compile(
    r"github\.event_name == 'workflow_dispatch' && inputs\.image_set == '(?P<image_set>all|h100-evidence)'"
)
MANUAL_OR_SCHEDULE_PATTERN = re.compile(
    r"\(github\.event_name == 'workflow_dispatch' && inputs\.image_set == '(?P<image_set>all|h100-evidence)'\)"
    r"\s*\|\|\s*"
    r"\(github\.event_name == 'schedule' && github\.event\.schedule == '(?P<schedule>[^']+)'\)"
)


def _job_runs(job: dict, *, event_name: str, image_set: str = "", schedule: str = "") -> bool:
    condition = job["if"].strip()
    manual_only = MANUAL_ONLY_PATTERN.fullmatch(condition)
    if manual_only:
        return event_name == "workflow_dispatch" and image_set == manual_only.group("image_set")

    manual_or_schedule = MANUAL_OR_SCHEDULE_PATTERN.fullmatch(condition)
    if manual_or_schedule:
        manual = event_name == "workflow_dispatch" and image_set == manual_or_schedule.group("image_set")
        scheduled = event_name == "schedule" and schedule == manual_or_schedule.group("schedule")
        return manual or scheduled

    raise AssertionError(f"job has an unsupported condition: {condition}")


def _jobs_for(workflow: dict, *, event_name: str, image_set: str = "", schedule: str = "") -> set[str]:
    return {
        name
        for name, job in workflow["jobs"].items()
        if _job_runs(job, event_name=event_name, image_set=image_set, schedule=schedule)
    }


def _docker_context_includes(path: str, dockerignore: str) -> bool:
    included = True
    candidate = PurePosixPath(path)
    for raw_rule in dockerignore.splitlines():
        rule = raw_rule.strip()
        if not rule or rule.startswith("#"):
            continue

        reinclude = rule.startswith("!")
        pattern = rule.removeprefix("!").removeprefix("/")
        directory_prefix = pattern.removesuffix("/")
        matches = candidate.match(pattern) or (pattern.endswith("/") and path.startswith(f"{directory_prefix}/"))
        if matches:
            included = reinclude
    return included


def _docker_stage(name: str) -> str:
    dockerfile = DOCKERFILE.read_text()
    stage_start = re.search(rf"^FROM \S+ AS {re.escape(name)}$", dockerfile, re.MULTILINE)
    assert stage_start is not None
    next_stage = re.search(r"^FROM \S+ AS \S+$", dockerfile[stage_start.end() :], re.MULTILINE)
    if next_stage is None:
        return dockerfile[stage_start.end() :]
    return dockerfile[stage_start.end() : stage_start.end() + next_stage.start()]


def _locked_package(name: str) -> dict:
    lock = tomllib.loads(UV_LOCK.read_text())
    matches = [package for package in lock["package"] if package["name"] == name]
    assert len(matches) == 1
    return matches[0]


def _package_manifest_records() -> list[tuple[str, int, str]]:
    records = []
    for line in PACKAGE_MANIFEST.read_text().splitlines():
        digest, size, url = line.split()
        records.append((digest, int(size), url))
    return records


def _runner_cubin_disassemblers() -> set[str]:
    tree = ast.parse(BACKEND_RESOURCES.read_text())
    disassemblers = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "sass" for target in node.targets
        ):
            continue
        strings = {
            child.value
            for child in ast.walk(node.value)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
        }
        if "--dump-sass" not in strings:
            continue
        for child in ast.walk(node.value):
            if (
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "with_name"
                and len(child.args) == 1
                and isinstance(child.args[0], ast.Constant)
                and isinstance(child.args[0].value, str)
            ):
                disassemblers.add(child.args[0].value)
    return disassemblers


def _run_sass_validator(tmp_path: Path, *, output_format: str, text: str) -> subprocess.CompletedProcess[str]:
    artifact = tmp_path / f"{output_format}.sass"
    artifact.write_text(text)
    return subprocess.run(
        (
            sys.executable,
            str(SASS_VALIDATOR),
            "--format",
            output_format,
            "--expected-kernel",
            "h100_evidence_smoke",
            str(artifact),
        ),
        capture_output=True,
        check=False,
        text=True,
    )


def _run_sass_validator_bytes(tmp_path: Path, *, output_format: str, payload: bytes) -> subprocess.CompletedProcess[str]:
    artifact = tmp_path / f"{output_format}.sass"
    artifact.write_bytes(payload)
    return subprocess.run(
        (
            sys.executable,
            str(SASS_VALIDATOR),
            "--format",
            output_format,
            "--expected-kernel",
            "h100_evidence_smoke",
            str(artifact),
        ),
        capture_output=True,
        check=False,
        text=True,
    )


def test_h100_evidence_package_manifest_is_closed_and_hash_pinned():
    records = _package_manifest_records()
    filenames = {Path(urlparse(url).path).name for _, _, url in records}

    assert filenames == EXPECTED_PACKAGES
    assert len(records) == len(EXPECTED_PACKAGES)
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest, _, _ in records)
    assert all(size > 0 for _, size, _ in records)
    assert all(url == f"{NVIDIA_DEBIAN12_AMD64_REPOSITORY}/{Path(urlparse(url).path).name}" for _, _, url in records)
    assert (
        "9d3ba750108356723313fa6e42d396a50fff8b00fc6f092a7b098537ac430b79",
        4_284_630,
        f"{NVIDIA_DEBIAN12_AMD64_REPOSITORY}/cuda-nvdisasm-13-2_13.2.86-1_amd64.deb",
    ) in records

    target = _docker_stage("task-h100-evidence")
    assert "while read -r expected_sha256 expected_size package_url extra" in target
    assert 'test "${actual_size}" -eq "${expected_size}"' in target
    assert '"${package_url}"' in target
    assert "ARG NVIDIA_DEBIAN12_AMD64_REPOSITORY" not in target


def test_h100_evidence_manifest_is_in_the_real_nonempty_docker_context():
    target = DOCKERFILE.read_text().split("FROM task AS task-h100-evidence", maxsplit=1)[1]
    manifest_copy = re.search(r"^COPY (?P<source>\S+) /tmp/h100-evidence-debian12-amd64\.sha256$", target, re.MULTILINE)

    assert manifest_copy is not None
    source = manifest_copy.group("source")
    assert (REPO_ROOT / source).read_bytes()
    dockerignore = DOCKERIGNORE.read_text()
    assert _docker_context_includes(source, dockerignore)
    assert not _docker_context_includes(source, dockerignore.replace(f"!{source}\n", ""))


def test_h100_evidence_target_inherits_task_and_checks_every_required_tool():
    target = _docker_stage("task-h100-evidence")

    assert "h100-evidence-debian12-amd64.sha256" in target
    assert 'test "$(dpkg --print-architecture)" = amd64' in target
    for tool in ("nvcc", "ptxas", "cuobjdump", "nvdisasm", "ncu", "nsys"):
        assert re.search(rf"/[^ ;]*{tool} --version", target)


def test_h100_evidence_image_validates_the_pinned_nsys_profile_contract():
    dockerfile = DOCKERFILE.read_text()
    target = _docker_stage("task-h100-evidence")
    validator_source = NSYS_HELP_VALIDATOR.relative_to(REPO_ROOT).as_posix()

    assert "ARG NSYS_VERSION=2026.1.3" in dockerfile
    assert "ARG NSYS_BUILD=2026.1.3.425-1" in dockerfile
    assert (
        "RUN --mount=type=bind,source=lib/iris/images/h100_evidence_nsys_help.py,"
        "target=/tmp/h100-evidence-nsys-help.py,ro" in target
    )
    assert "/usr/local/bin/nsys profile --help > /tmp/h100-evidence-nsys-profile-help.txt" in target
    assert (
        "/opt/h100-evidence-runtime/bin/python /tmp/h100-evidence-nsys-help.py "
        "\\\n      /tmp/h100-evidence-nsys-profile-help.txt" in target
    )
    assert f"COPY {validator_source}" not in target
    assert (
        "nvidia-smi"
        not in target.split("# Bind the pinned CLI's own option contract", maxsplit=1)[1].split(
            "# Exercise the runner's exact compiler/disassembler dependency", maxsplit=1
        )[0]
    )

    dockerignore = DOCKERIGNORE.read_text()
    assert _docker_context_includes(validator_source, dockerignore)
    assert not _docker_context_includes(validator_source, dockerignore.replace(f"!{validator_source}\n", ""))


def test_h100_evidence_image_profiles_the_exact_runner_nvtx_range_without_a_device_query():
    target = _docker_stage("task-h100-evidence")
    smoke = target.split("# Exercise the runner's exact ctypes NVTX range", maxsplit=1)[1].split(
        "# Exercise the runner's exact compiler/disassembler dependency", maxsplit=1
    )[0]
    module_source = NVTX_RANGE.relative_to(REPO_ROOT).as_posix()
    harness_source = NVTX_SMOKE.relative_to(REPO_ROOT).as_posix()

    assert f"source={module_source},target=/tmp/h100-evidence-nvtx-range.py,ro" in smoke
    assert f"source={harness_source},target=/tmp/h100-evidence-nvtx-smoke.py,ro" in smoke
    assert "env NSYS_NVTX_PROFILER_REGISTER_ONLY=0" in smoke
    assert "/usr/local/bin/nsys profile --force-overwrite=true --trace=nvtx" in smoke
    assert "--capture-range=nvtx --capture-range-end=stop" in smoke
    assert "--nvtx-capture=h100-evidence-nvtx-smoke" in smoke
    assert "h100-evidence-nvtx-smoke.py emit" in smoke
    assert '--library "${CUDA_HOME}/lib64/libnvToolsExt.so"' in smoke
    assert "/usr/local/bin/nsys export --force-overwrite=true --type=sqlite" in smoke
    assert "h100-evidence-nvtx-smoke.py validate" in smoke
    assert "test -s /tmp/h100-evidence-nvtx.nsys-rep" in smoke
    assert "nvidia-smi" not in smoke
    assert "cudaProfiler" not in smoke
    assert f"COPY {module_source}" not in target
    assert f"COPY {harness_source}" not in target

    dockerignore = DOCKERIGNORE.read_text()
    for source in (module_source, harness_source):
        assert (REPO_ROOT / source).read_bytes()
        assert _docker_context_includes(source, dockerignore)
        assert not _docker_context_includes(source, dockerignore.replace(f"!{source}\n", ""))


def test_h100_evidence_image_executes_the_runner_cubin_disassembly_closure():
    target = _docker_stage("task-h100-evidence")
    smoke = target.split("# Exercise the runner's exact compiler/disassembler dependency", maxsplit=1)[1].split(
        "# The runner loads these libraries", maxsplit=1
    )[0]

    assert _runner_cubin_disassemblers() == {"cuobjdump"}
    assert "cuda-nvdisasm-13-2_13.2.86-1_amd64.deb" in EXPECTED_PACKAGES
    assert 'ENV NVDISASM_PATH="${CUDA_HOME}/bin"' in target
    assert '"${CUDA_HOME}/bin/nvcc" -arch=sm_90a --cubin' in smoke
    assert 'env PATH=/usr/bin:/bin NVDISASM_PATH="${CUDA_HOME}/bin"' in smoke
    assert '"${CUDA_HOME}/bin/cuobjdump" --dump-sass /tmp/h100-evidence-smoke.cubin' in smoke
    assert '"${CUDA_HOME}/bin/nvdisasm" /tmp/h100-evidence-smoke.cubin' in smoke
    assert smoke.count("/opt/h100-evidence-runtime/bin/python /tmp/h100-evidence-sass-smoke.py") == 2
    assert "--format cuobjdump --expected-kernel h100_evidence_smoke" in smoke
    assert "--format nvdisasm --expected-kernel h100_evidence_smoke" in smoke
    assert (
        "RUN --mount=type=bind,source=lib/iris/images/h100_evidence_sass_smoke.py,"
        "target=/tmp/h100-evidence-sass-smoke.py,ro" in target
    )
    assert "grep -Eq" not in smoke
    assert f"COPY {SASS_VALIDATOR.relative_to(REPO_ROOT).as_posix()}" not in target
    assert "nvidia-smi" not in smoke

    validator_source = SASS_VALIDATOR.relative_to(REPO_ROOT).as_posix()
    dockerignore = DOCKERIGNORE.read_text()
    assert _docker_context_includes(validator_source, dockerignore)
    assert not _docker_context_includes(validator_source, dockerignore.replace(f"!{validator_source}\n", ""))


@pytest.mark.parametrize(
    ("output_format", "sass"),
    (("cuobjdump", CUOBJDUMP_SASS), ("nvdisasm", NVDISASM_SASS)),
)
def test_h100_evidence_sass_validator_accepts_addressed_kernel_instructions(tmp_path, output_format, sass):
    result = _run_sass_validator(tmp_path, output_format=output_format, text=sass)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "format": output_format,
        "instruction_count": 2,
        "kernel": "h100_evidence_smoke",
    }


def test_h100_evidence_nvdisasm_validator_ignores_trailing_function_body(tmp_path):
    trailing = (
        NVDISASM_SASS
        + """\
        .section        .text.trailing,"ax",@progbits
        .global         trailing
        .type           trailing,@function
trailing:
.text.trailing:
        /*0000*/ \t.byte\t0xff, 0xff, 0xff, 0xff
"""
    )
    result = _run_sass_validator(tmp_path, output_format="nvdisasm", text=trailing)

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["instruction_count"] == 2


@pytest.mark.parametrize(
    "sass",
    (
        "",
        "warning: source unavailable\n" + CUOBJDUMP_SASS,
        "Function : h100_evidence_smoke\nplain text without instructions\n",
        "Function : h100_evidence_smoke\n/*address*/ MOV R1, R2 ;\n",
        "Function : h100_evidence_smoke\n/*0000*/ MOV R1, R2 ;\n/*0010*/ this is not SASS\n",
        "Function : h100_evidence_smoke\n/*0000*/ MOV R1, R2 ;\n/*0000*/ EXIT ;\n",
        "Function : different_kernel\n/*0000*/ MOV R1, R2 ;\n",
        "Function : h100_evidence_smoke\nFunction : h100_evidence_smoke\n/*0000*/ MOV R1, R2 ;\n",
        "Function : h100_evidence_smoke\n/*0010*/ MOV R1, R2 ;\n/*0000*/ EXIT ;\n",
    ),
)
def test_h100_evidence_sass_validator_rejects_unusable_cuobjdump_output(tmp_path, sass):
    result = _run_sass_validator(tmp_path, output_format="cuobjdump", text=sass)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.startswith("cuobjdump SASS validation failed:")


@pytest.mark.parametrize(
    "sass",
    (
        NVDISASM_SASS.replace(NVDISASM_BODY, "plain text without instructions\n"),
        NVDISASM_SASS.replace(
            NVDISASM_BODY,
            "        /*0000*/ MOV R1, R2 ;\n        /*0010*/ not SASS\n",
        ),
        NVDISASM_SASS.replace(NVDISASM_BODY, "        /*0000*/ .byte 0xff, 0xff\n"),
        "warning: source unavailable\n" + NVDISASM_SASS,
        NVDISASM_SASS.replace(".section        .text.h100_evidence_smoke", ".section        .text.wrong"),
        NVDISASM_SASS.replace("        .global         h100_evidence_smoke\n", ""),
        NVDISASM_SASS.replace("\nh100_evidence_smoke:\n", "\n"),
        NVDISASM_SASS.replace(".text.h100_evidence_smoke:\n", ""),
        NVDISASM_SASS.replace(
            "h100_evidence_smoke:\n.text.h100_evidence_smoke:\n",
            ".text.h100_evidence_smoke:\nh100_evidence_smoke:\n",
        ),
        NVDISASM_SASS.replace(
            ".text.h100_evidence_smoke:\n",
            ".text.h100_evidence_smoke:\n.text.h100_evidence_smoke:\n",
        ),
        NVDISASM_SASS.replace(".global         h100_evidence_smoke", ".global         different_kernel"),
        NVDISASM_SASS.replace(
            "        .global         h100_evidence_smoke\n",
            "        .global         h100_evidence_smoke\n        .global         h100_evidence_smoke\n",
        ),
        NVDISASM_SASS.replace("\nh100_evidence_smoke:\n", "\nh100_evidence_smoke:\nh100_evidence_smoke:\n"),
        NVDISASM_SASS.replace(
            NVDISASM_BODY,
            "        /*0010*/ MOV R1, R2 ;\n        /*0000*/ EXIT ;\n",
        ),
        NVDISASM_SASS.replace(NVDISASM_BODY, "plain text without instructions\n")
        + """\
        .section        .text.trailing,"ax",@progbits
        .global         trailing
trailing:
        /*0000*/ MOV R1, R2 ;
""",
    ),
)
def test_h100_evidence_sass_validator_rejects_unusable_nvdisasm_output(tmp_path, sass):
    result = _run_sass_validator(tmp_path, output_format="nvdisasm", text=sass)

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.startswith("nvdisasm SASS validation failed:")


def test_h100_evidence_sass_validator_rejects_non_utf8_artifact(tmp_path):
    result = _run_sass_validator_bytes(tmp_path, output_format="cuobjdump", payload=b"\xff")

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.startswith("cuobjdump SASS validation failed:")


def test_h100_evidence_sass_validator_rejects_oversized_artifact(tmp_path):
    artifact = tmp_path / "cuobjdump.sass"
    with artifact.open("wb") as stream:
        stream.seek(1 << 20)
        stream.write(b"x")
    result = subprocess.run(
        (
            sys.executable,
            str(SASS_VALIDATOR),
            "--format",
            "cuobjdump",
            "--expected-kernel",
            "h100_evidence_smoke",
            str(artifact),
        ),
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.startswith("cuobjdump SASS validation failed:")


def test_h100_evidence_sass_validator_replaces_the_failed_double_escaped_ere(tmp_path):
    previous_pattern = r"/\\*[[:xdigit:]]+\\*/.*[A-Z][A-Z0-9.]+"
    previous = subprocess.run(
        ("grep", "-Eq", previous_pattern),
        input=CUOBJDUMP_SASS,
        capture_output=True,
        check=False,
        text=True,
    )
    current = _run_sass_validator(tmp_path, output_format="cuobjdump", text=CUOBJDUMP_SASS)

    assert previous.returncode == 1
    assert current.returncode == 0, current.stderr


def test_h100_evidence_image_probes_every_runner_loaded_cuda_library():
    runner = H100_RUNNER.read_text()
    required_libraries = set(re.findall(r'cuda_toolkit_shared_library\(nvcc, "([A-Za-z0-9]+)"\)', runner))
    target = _docker_stage("task-h100-evidence")
    probe_match = re.search(r"probes = (?P<probes>\{.*?\n\})", target, re.DOTALL)

    assert required_libraries == {"cudart", "nvToolsExt"}
    assert probe_match is not None
    probes = ast.literal_eval(probe_match.group("probes"))
    assert set(probes) == required_libraries
    assert "cuda-cudart-13-2_13.2.86-1_amd64.deb" in EXPECTED_PACKAGES
    assert "cuda-nvtx-13-2_13.2.86-1_amd64.deb" in EXPECTED_PACKAGES
    assert "ln -s libnvtx3interop.so /usr/local/cuda-13.2/lib64/libnvToolsExt.so" in target
    assert "RUN /opt/h100-evidence-runtime/bin/python - <<'PY'" in target


def test_h100_evidence_runtime_uses_the_frozen_cuda13_workspace_closure():
    project = tomllib.loads(TILE_LIFETIME_PYPROJECT.read_text())["project"]
    assert {"jax==0.10.1", "marin-shuttle", "numpy>=2.0"} <= set(project["dependencies"])
    assert project["optional-dependencies"] == {"cuda13": ["jax[cuda13]==0.10.1"]}

    for distribution in ("jax", "jaxlib", "jax-cuda13-plugin", "jax-cuda13-pjrt"):
        assert _locked_package(distribution)["version"] == "0.10.1"
    assert _locked_package("marin-shuttle")["version"] == "0.1.0"
    assert _locked_package("jax")["optional-dependencies"]["cuda13"] == [
        {"name": "jax-cuda13-plugin", "extra": ["with-cuda"]},
        {"name": "jaxlib"},
    ]

    runtime = _docker_stage("h100-evidence-runtime")
    sync = " ".join(line.strip().removesuffix("\\").strip() for line in runtime.splitlines())
    assert "uv sync --frozen --package marin-tile-lifetime --extra cuda13" in sync
    assert "--no-default-groups --no-editable --no-install-project" in sync
    assert "UV_PROJECT_ENVIRONMENT=/opt/h100-evidence-runtime" in runtime


def test_h100_evidence_runtime_builder_has_complete_context_without_shipping_repo_source():
    runtime = _docker_stage("h100-evidence-runtime")
    final = _docker_stage("task-h100-evidence")
    copied_sources = {
        token.rstrip("/")
        for line in runtime.splitlines()
        if line.startswith("COPY ") and "--from=" not in line
        for token in line.split()[1:-1]
    }
    workspace_members = tomllib.loads(ROOT_PYPROJECT.read_text())["tool"]["uv"]["workspace"]["members"]
    assert {f"{member}/pyproject.toml" for member in workspace_members} <= copied_sources

    dockerignore = DOCKERIGNORE.read_text()
    for source in copied_sources:
        assert (REPO_ROOT / source).exists(), source
        assert _docker_context_includes(source, dockerignore), source

    assert "lib/shuttle/src" in copied_sources
    assert "lib/tile_lifetime/src" not in copied_sources
    assert "COPY --from=h100-evidence-runtime /opt/h100-evidence-runtime /opt/h100-evidence-runtime" in final
    assert "COPY lib/shuttle/" not in final
    assert "COPY lib/tile_lifetime/" not in final


def test_h100_evidence_runtime_smoke_is_cpu_only_and_matches_the_lock():
    final = _docker_stage("task-h100-evidence")
    smoke = final.split("RUN JAX_PLATFORMS=cpu /opt/h100-evidence-runtime/bin/python - <<'PY'", maxsplit=1)[1].split(
        "\nPY", maxsplit=1
    )[0]
    expected_match = re.search(r"expected = (?P<expected>\{.*?\n\})", smoke, re.DOTALL)
    assert expected_match is not None
    expected = ast.literal_eval(expected_match.group("expected"))
    assert expected == {
        "jax": "0.10.1",
        "jax-cuda13-pjrt": "0.10.1",
        "jax-cuda13-plugin": "0.10.1",
        "jaxlib": "0.10.1",
        "marin-shuttle": "0.1.0",
    }
    assert expected == {distribution: _locked_package(distribution)["version"] for distribution in expected}
    for imported_module in ("jax", "jaxlib", "numpy", "scipy"):
        assert re.search(rf"^import {imported_module}$", smoke, re.MULTILINE)
    assert "from shuttle.ir import DType" in smoke
    assert 'DType.BF16.value != "bf16"' in smoke
    assert "jax.devices(" not in smoke
    assert "nvidia-smi" not in smoke


def test_h100_evidence_workflow_dispatch_builds_one_exact_source_image():
    workflow = yaml.safe_load(H100_IMAGE_WORKFLOW.read_text())
    triggers = workflow[True]

    assert set(triggers) == {"workflow_dispatch", "workflow_call"}
    assert triggers["workflow_dispatch"]["inputs"] == {
        "ref": {"description": "Git ref to build", "required": True, "type": "string"}
    }
    assert triggers["workflow_call"] == {
        "inputs": {"ref": {"description": "Git ref to build", "required": True, "type": "string"}},
        "outputs": {
            "image_ref": {
                "description": "Full-SHA image reference with OCI digest",
                "value": "${{ jobs.build-h100-evidence-image.outputs.image_ref }}",
            }
        },
    }
    assert workflow["permissions"] == {"contents": "read", "packages": "write"}
    assert set(workflow["jobs"]) == {"build-h100-evidence-image"}

    job = workflow["jobs"]["build-h100-evidence-image"]
    assert "strategy" not in job
    checkout = next(step for step in job["steps"] if step.get("uses") == "actions/checkout@v5")
    source = next(step for step in job["steps"] if step.get("id") == "source")
    context = next(step for step in job["steps"] if step.get("id") == "context")
    build = next(step for step in job["steps"] if step.get("id") == "build")

    assert checkout["with"] == {"ref": "${{ inputs.ref }}", "persist-credentials": False}
    assert source["run"].count("git rev-parse HEAD") == 1
    assert "^[0-9a-f]{40}$" in source["run"]
    assert 'context_manifest="lib/iris/images/h100-evidence-debian12-amd64.sha256"' in context["run"]
    assert '[[ ! -s "$context_manifest" ]]' in context["run"]
    assert 'grep -Fxq "!$context_manifest" "$dockerignore"' in context["run"]
    assert job["steps"].index(context) < job["steps"].index(build)
    assert build["env"]["IMAGE"] == (
        "ghcr.io/marin-community/iris-task-h100-evidence:${{ steps.source.outputs.full_sha }}"
    )
    assert "--target task-h100-evidence" in build["run"]
    assert "--platform linux/amd64" in build["run"]
    assert build["run"].count("docker buildx build ") == 1
    assert re.findall(r"--tag ([^ ]+)", build["run"]) == ['"$IMAGE"']
    assert "containerimage.digest" in build["run"]
    assert 'docker buildx imagetools inspect "$image_ref"' in build["run"]
    assert ":latest" not in build["run"]
    assert "DATE_TAG" not in build["run"]
    assert "HASH_TAG" not in build["run"]


def test_broad_ops_manual_image_set_selects_legacy_or_h100_exclusively():
    workflow = yaml.safe_load(BROAD_IMAGE_WORKFLOW.read_text())
    dispatch = workflow[True]["workflow_dispatch"]
    bridge = workflow["jobs"]["h100-evidence-image"]

    assert dispatch["inputs"] == {
        "image_set": {
            "description": "Image set to build",
            "required": True,
            "default": "all",
            "type": "choice",
            "options": ["all", "h100-evidence"],
        }
    }
    assert set(workflow["jobs"]) == LEGACY_IMAGE_JOBS | {"h100-evidence-image"}
    assert _jobs_for(workflow, event_name="workflow_dispatch", image_set="all") == LEGACY_IMAGE_JOBS
    assert _jobs_for(workflow, event_name="workflow_dispatch", image_set="h100-evidence") == {"h100-evidence-image"}
    assert _jobs_for(workflow, event_name="schedule", schedule="0 2 * * 0") == WEEKLY_IMAGE_JOBS
    assert _jobs_for(workflow, event_name="schedule", schedule="0 3 * * *") == {"marin-tpu-ci-images"}
    assert bridge == {
        "if": "github.event_name == 'workflow_dispatch' && inputs.image_set == 'h100-evidence'",
        "permissions": {"contents": "read", "packages": "write"},
        "uses": "./.github/workflows/ops-h100-evidence-image.yaml",
        "with": {"ref": "${{ github.sha }}"},
    }
