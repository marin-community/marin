#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Flatten the marin monorepo from lib/*/src/* layout to src/* layout.

Moves source, test, and auxiliary directories from per-package lib/ subdirectories
into top-level src/, tests/, config/, examples/, scripts/, infra/, and docker/ dirs.
Generates a merged pyproject.toml, updates CI workflows, Dockerfiles, Makefile, and
AGENTS.md. Dry-run by default; pass --execute to perform changes.

Usage:
    uv run scripts/migrations/flatten_monorepo.py          # dry-run
    uv run scripts/migrations/flatten_monorepo.py --execute # actually move files
"""

import argparse
import logging
import os
import shutil
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_MOVES = {
    "lib/marin/src/marin": "src/marin",
    "lib/levanter/src/levanter": "src/levanter",
    "lib/haliax/src/haliax": "src/haliax",
    "lib/iris/src/iris": "src/iris",
    "lib/fray/src/fray": "src/fray",
    "lib/zephyr/src/zephyr": "src/zephyr",
}

TEST_MOVES = {
    "lib/levanter/tests": "tests/levanter",
    "lib/haliax/tests": "tests/haliax",
    "lib/iris/tests": "tests/iris",
    "lib/fray/tests": "tests/fray",
    "lib/zephyr/tests": "tests/zephyr",
}

AUX_MOVES = {
    "lib/levanter/config": "config/levanter",
    "lib/levanter/examples": "examples/levanter",
    "lib/levanter/scripts": "scripts/levanter",
    "lib/levanter/infra": "infra/levanter",
    "lib/levanter/docker": "docker/levanter",
    "lib/levanter/etc": "etc/levanter",
    "lib/haliax/etc": "etc/haliax",
    "lib/iris/examples": "examples/iris",
    "lib/marin/tools": "tools",
}

IRIS_DOCKER_FILE_MOVES = {
    "lib/iris/Dockerfile.controller": "docker/iris/Dockerfile.controller",
    "lib/iris/Dockerfile.worker": "docker/iris/Dockerfile.worker",
    "lib/iris/Dockerfile.task": "docker/iris/Dockerfile.task",
}

BUF_MOVES = {
    "lib/iris/buf.yaml": "buf.yaml",
    "lib/iris/buf.gen.yaml": "buf.gen.yaml",
}

# Packages that stay in lib/ (Rust/maturin and conflicting deps).
KEEP_IN_LIB = {"dupekit", "harbor"}

# Subproject directories to clean up after all moves.
CLEANUP_LIB_DIRS = ["lib/marin", "lib/levanter", "lib/haliax", "lib/iris", "lib/fray", "lib/zephyr"]

WORKFLOW_REPLACEMENTS = [
    # Working directories
    ("working-directory: lib/levanter", "working-directory: ."),
    ("working-directory: lib/haliax", "working-directory: ."),
    # Path trigger filters (quoted)
    ("- 'lib/haliax/**'", "- 'src/haliax/**'"),
    ("- 'lib/iris/**'", "- 'src/iris/**'"),
    ("- 'lib/fray/**'", "- 'src/fray/**'"),
    ("- 'lib/zephyr/**'", "- 'src/zephyr/**'"),
    # Path trigger filters (unquoted)
    ("- lib/haliax/**", "- src/haliax/**"),
    ("- lib/iris/**", "- src/iris/**"),
    ("- lib/fray/**", "- src/fray/**"),
    ("- lib/zephyr/**", "- src/zephyr/**"),
    # cd commands
    ("cd lib/iris && ", ""),
    ("cd lib/fray && ", ""),
    # Source file refs
    ("lib/marin/src/marin/", "src/marin/"),
    # Docker refs
    ("dockerfile: lib/iris/Dockerfile", "dockerfile: docker/iris/Dockerfile"),
    ("context: lib/iris", "context: ."),
    ("-f lib/iris/Dockerfile", "-f docker/iris/Dockerfile"),
    ("lib/levanter/docker/", "docker/levanter/"),
    ("-f lib/levanter/docker/", "-f docker/levanter/"),
    ("--file lib/levanter/docker/", "--file docker/levanter/"),
    # Test paths
    ("lib/levanter/tests", "tests/levanter"),
    ("lib/zephyr/tests/", "tests/zephyr/"),
    # Config/examples
    ("lib/iris/examples/", "examples/iris/"),
    ("config=lib/iris/examples/", "config=examples/iris/"),
    # Cache dependency glob
    ('cache-dependency-glob: "lib/iris/pyproject.toml"', 'cache-dependency-glob: "pyproject.toml"'),
    # Relative uv.lock
    ("../../uv.lock", "uv.lock"),
    # CodeQL paths
    ("- lib/marin/src", "- src/marin"),
]


# ---------------------------------------------------------------------------
# Merged pyproject.toml content
# ---------------------------------------------------------------------------

MERGED_PYPROJECT = """\
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "marin"
version = "0.1.0"
description = "Marin: Scalable Training, Data Processing, and Evaluation"
license = { file = "LICENSE" }
requires-python = ">=3.11"

# Base dependencies: lightweight distributed infra (iris + fray + zephyr).
# Iris Docker images use `uv sync` with just these deps.
dependencies = [
    # --- iris ---
    "click>=8.3.1",
    "cloudpickle>=3.1.2",
    "connect-python @ git+https://github.com/connectrpc/connect-python.git@5342eacecef85e52717604ee5ac7e03a1e16c7ac",
    "fsspec>=2025.3.0",
    "gcsfs>=2024.0.0",
    "s3fs>=2024.0.0",
    "grpcio>=1.76.0",
    "httpx>=0.28.1",
    "humanfriendly>=10.0",
    "pydantic>=2.12.5",
    "pyyaml>=6.0",
    "starlette>=0.50.0",
    "tabulate>=0.9.0",
    "typing-extensions>=4.0",
    "uvicorn[standard]>=0.23.0",
    # --- fray (above iris) ---
    "fastapi>=0.100.0",
    "flask",
    "mergedeep",
    "zstandard>=0.22.0",
    # --- zephyr (above fray) ---
    "braceexpand>=0.1.0",
    "msgspec>=0.18.0",
    "numpy>=2.0",
    "tqdm-loggable>=0.2",
    "vortex-data>=0.1.0",
    # --- root ---
    "watchdog",
]

[project.scripts]
iris = "iris.cli:iris"
iris-run = "iris.cli.run:run"
fray = "fray.v1.cli:main"
zephyr = "zephyr.cli:main"

[project.optional-dependencies]
# Training / ML dependencies (haliax + levanter + marin).
# Install with: uv sync --extra train
train = [
    # --- haliax ---
    "aqtp>=0.8.2",
    # --- levanter ---
    "async-lru~=2.0",
    "dataclasses-json~=0.6.4",
    "deepdiff",
    "draccus>=0.11.5",
    "einops",
    "equinox>=0.11.7,!=0.12.0",
    "filelock~=3.13",
    "google-api-python-client>=2.175.0",
    "google-auth",
    "google-cloud-storage",
    "google-cloud-storage-transfer",
    "immutabledict",
    "jaxtyping>=0.2.34",
    "jinja2",
    "jmp>=0.0.4",
    "lenses",
    "openai>=1.0.0",
    "optax>=0.1.9",
    "protobuf>=6,<7",
    "pyarrow>=22",
    "pytimeparse>=1.1.8",
    "ray[default]==2.53.0",
    "safetensors[numpy]>=0.4.2,<0.7.0",
    "tblib>=1.7.0,<4.0.0",
    "tensorstore>=0.1.73",
    "tokenizers>=0.15.2",
    "transformers>=4.57.1,<5.0",
    "wandb>=0.17.8",
    # --- marin ---
    "cryptography>=45",
    "datasets<4.0.0",
    "ddsketch",
    "fasteners>=0.19",
    "floret",
    "jax==0.8.0",
    "jaxopt>=0.8.3",
    "lxml[html_clean]",
    "lz4",
    "markdownify==0.12.1",
    "multiprocess==0.70.16",
    "pandas>=2.0",
    "plotly",
    "regex",
    "requests",
    "resiliparse>=0.17.2",
    "toml",
    "tqdm",
    "warcio",
]

# Hardware-specific JAX + PyTorch backends.
gpu = ["jax[cuda12]==0.8.0", "torch==2.9.0", "torchvision==0.24.0"]
tpu = ["jax[tpu]==0.8.0", "torch==2.9.0", "torchvision==0.24.0"]
cpu = [
    "jax==0.8.0",
    "torch==2.9.0",
    "torchvision==0.24.0+cpu; sys_platform == 'linux' and platform_machine == 'x86_64'",
    "torchvision==0.24.0+cpu; sys_platform == 'win32' and platform_machine == 'AMD64'",
    "torchvision==0.24.0; (sys_platform == 'linux' and platform_machine == 'aarch64')"
    " or (sys_platform == 'darwin' and platform_machine == 'arm64')",
]

# lm-eval harness (large dep tree, pinned to our fork). Isolated because
# it pulls in many transitive deps that most workflows don't need.
lm_eval = [
    "lm-eval[math,api]@git+https://github.com/stanford-crfm/lm-evaluation-harness@d5e3391f22cde186c827674d5c3ec7c5f4fe0cab",
]

# Evalchemy: reasoning-focused evals (AIME, MATH500, HumanEval+, etc.).
evalchemy = [
    "lm-eval[math]@git+https://github.com/stanford-crfm/lm-evaluation-harness@d5e3391f22cde186c827674d5c3ec7c5f4fe0cab",
    "sympy>=1.12.1,<1.14",
    "antlr4-python3-runtime==4.11",
    "bespokelabs-curator",
    "sqlalchemy",
    "fire",
]

# Reinforcement learning.
rl = [
    "prime",
    "sympy>=1.12.1,<1.14",
    "verifiers==0.1.5",
]

# vLLM inference on TPU.
vllm = [
    "vllm-tpu==0.13.2.post6",
    "triton==3.5.0; platform_system == 'Linux' and platform_machine == 'x86_64'",
]

# Profiling.
profiling = ["xprof", "tensorboard>=2.16", "tensorboardX>=2.6"]
kernels = ["tokamax"]
vizier = ["google-vizier[jax]"]
math = ["pylatexenc", "sympy>=1.12.1,<1.14"]

# Optional integrations with local lib/ packages.
harbor = ["harbor>=0.1.42"]
dedup = ["dupekit"]

[dependency-groups]
test = [
    "pytest>=8.3.2",
    "pytest-asyncio",
    "pytest-xdist",
    "pytest-timeout",
    "pytest-cov",
    "pytest-profiling>=1.8.1",
    "pytest-forked",
    "pytest-flakefinder>=1.1.0",
    "soundfile",
    "librosa>=0.11.0",
    "tensorboardX>=2.6",
    "chex>=0.1.86",
    "httpx>=0.28.1",
    "trackio>=0.5.0",
    "numba>=0.62.1",
    "psutil>=5.9.0",
    "pip",
    "openai-responses",
    "plotly",
    # Torch deps needed for levanter torch-interop tests.
    "torch>=2.7.0",
    "peft>=0.12.0",
]
lint = [
    "ruff==0.14.3",
    "black==25.9.0",
    "mypy>=1.4.1",
    "pyrefly==0.40.0",
    "types-PyYAML",
    "types-requests",
    "types-six",
    "click>=8.0",
    "pyyaml>=6.0",
]
docs = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.5.0",
    "mkdocstrings>=0.24.0",
    "mkdocstrings-python>=1.7.0",
    "pymdown-extensions>=10.0.0",
    "mkdocs-git-revision-date-localized-plugin>=1.2.0",
    "mkdocs-git-authors-plugin>=0.9.0",
    "mkdocs-minify-plugin>=0.7.0",
    "mkdocs-include-markdown-plugin>=7.1.5",
]
math = ["pylatexenc", "sympy>=1.12.1,<1.14"]
metrics = ["google-cloud-logging"]
dev = [
    { include-group = "test" },
    { include-group = "lint" },
    { include-group = "docs" },
    { include-group = "math" },
]

[tool.uv]
fork-strategy = "fewest"
override-dependencies = [
    "omegaconf>=2.4.0.dev4",
    "antlr4-python3-runtime==4.11",
    "python-multipart>=0.0.22",
    "wheel>=0.46.2",
    "datasets>=3.1.0,<4.0.0",
    "equinox>=0.11.10",
]
conflicts = [
    [
        { extra = "gpu" },
        { extra = "tpu" },
    ],
    [
        { extra = "gpu" },
        { extra = "cpu" },
    ],
    [
        { extra = "vllm" },
        { extra = "cpu" },
    ],
    [
        { extra = "vllm" },
        { extra = "cuda12" },
    ],
]

[tool.uv.sources]
# Non-workspace path deps (stay in lib/)
dupekit = { path = "lib/dupekit", editable = true }
harbor = { path = "lib/harbor", editable = true }
torchvision = [
    { index = "pytorch-cpu", extra = "cpu" },
    { index = "pytorch-cpu", extra = "tpu" },
    { index = "pytorch-cu128", extra = "gpu" },
]
resiliparse = { index = "marin-resiliparse" }
torch = [
    { index = "pytorch-cu128", extra = "gpu", marker = "sys_platform == 'linux'" },
    { index = "pytorch-cpu", extra = "cpu" },
    { index = "pytorch-cpu", extra = "tpu" },
]

[[tool.uv.index]]
name = "marin-resiliparse"
url = "https://marin-community.github.io/chatnoir-resiliparse/simple"

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.black]
line-length = 121
target-version = ["py310"]
preview = true
extend-exclude = \"\"\"
(
    scripts/
)
\"\"\"

[tool.ruff]
line-length = 121
target-version = "py310"
extend-exclude = ["scripts/"]

[tool.ruff.lint]
select = ["A", "B", "E", "F", "I", "NPY", "RUF", "UP", "W"]
ignore = ["F722", "B008", "UP015", "A005", "I001", "E741"]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["E402", "F401"]

[tool.mypy]
python_version = "3.10"
ignore_missing_imports = true
exclude = ["marin/", "scripts/"]

[tool.pyrefly]
project-includes = [
    "src/**/*.py",
]
search-path = ["src"]
disable-search-path-heuristics = true
skip-interpreter-query = true
use-ignore-files = false
disable-project-excludes-heuristics = true
project-excludes = [
    "experiments/**",
    "scripts/**",
    "tests/**",
    "examples/**",
    "src/**/crawl/**",
    "src/iris/rpc/*_pb2*",
    "**/node_modules",
    "**/__pycache__",
    "**/venv/**",
]

[tool.pyrefly.errors]
missing-import = false
unexpected-keyword = false
missing-attribute = false
deprecated = false
unknown-name = false
not-iterable = false
no-matching-overload = false
bad-index = false
bad-argument-type = false
bad-context-manager = false
missing-argument = false
bad-argument-count = false
unbound-name = false

[tool.hatch.metadata]
allow-direct-references = true

[tool.hatch.version]
path = "src/haliax/__about__.py"

[tool.hatch.build.targets.wheel]
packages = ["src/marin", "src/levanter", "src/haliax", "src/iris", "src/fray", "src/zephyr", "experiments"]

[tool.hatch.build.targets.sdist]
packages = ["src/marin", "src/levanter", "src/haliax", "src/iris", "src/fray", "src/zephyr", "experiments"]

[tool.pytest.ini_options]
timeout = 60
pythonpath = ["src", "tests"]
filterwarnings = ["ignore::DeprecationWarning"]
log_format = "%(asctime)s %(levelname)s %(message)s"
log_date_format = "%Y-%m-%d %H:%M:%S"
markers = [
    "slow: mark tests as slow for CI - use -m 'not slow'",
    "tpu_ci: mark tests that require a TPU",
    "entry: marks tests as entry point tests",
    "ray: marks tests that require Ray",
    "torch: mark tests that use Torch",
    "tpu: mark tests that require TPU",
    "docker: marks tests requiring Docker runtime",
    "e2e: end-to-end cluster tests",
]
testpaths = ["tests", "experiments"]
addopts = "--session-timeout=480 -m 'not tpu_ci and not slow'"
"""


# ---------------------------------------------------------------------------
# Dockerfile replacement blocks
# ---------------------------------------------------------------------------

CLUSTER_DOCKERFILE_OLD = """\
COPY --chown=ray pyproject.toml /tmp/marin/
COPY --chown=ray lib/ /tmp/marin/lib/
WORKDIR /tmp/marin
RUN ray --version >/tmp/marin/ray_version_before.txt \\
  && mkdir -p src/marin/\\
  && touch src/marin/__init__.py\\
  && touch README.md\\
  && touch LICENSE\\
  && uv pip install -e .[dev,tpu,gcp] --system --torch-backend=cpu\\
  && uv pip install -e lib/dupekit --system \\"""

CLUSTER_DOCKERFILE_NEW = """\
COPY --chown=ray pyproject.toml /tmp/marin/
COPY --chown=ray src/ /tmp/marin/src/
COPY --chown=ray lib/dupekit/ /tmp/marin/lib/dupekit/
COPY --chown=ray experiments/ /tmp/marin/experiments/
WORKDIR /tmp/marin
RUN ray --version >/tmp/marin/ray_version_before.txt \\
  && touch README.md\\
  && touch LICENSE\\
  && uv pip install -e .[train,dev,tpu] --system --torch-backend=cpu\\
  && uv pip install -e lib/dupekit --system \\"""

VLLM_DOCKERFILE_OLD = """\
COPY --chown=ray pyproject.toml /tmp/marin/
COPY --chown=ray lib/ /tmp/marin/lib/
WORKDIR /tmp/marin
RUN ray --version >/tmp/marin/ray_version_before.txt \\
  && mkdir -p src/marin/\\
  && touch src/marin/__init__.py\\
  && touch README.md\\
  && touch LICENSE\\
  && uv pip install -e .[dev,gcp,vllm] --system \\
  && uv pip install -e lib/dupekit --system \\"""

VLLM_DOCKERFILE_NEW = """\
COPY --chown=ray pyproject.toml /tmp/marin/
COPY --chown=ray src/ /tmp/marin/src/
COPY --chown=ray lib/dupekit/ /tmp/marin/lib/dupekit/
COPY --chown=ray experiments/ /tmp/marin/experiments/
WORKDIR /tmp/marin
RUN ray --version >/tmp/marin/ray_version_before.txt \\
  && touch README.md\\
  && touch LICENSE\\
  && uv pip install -e .[train,dev,vllm] --system \\
  && uv pip install -e lib/dupekit --system \\"""


# ---------------------------------------------------------------------------
# Migration implementation
# ---------------------------------------------------------------------------


class Migration:
    """Holds repo root and dry-run state, orchestrates all migration steps."""

    def __init__(self, repo_root: Path, dry_run: bool):
        self.repo_root = repo_root
        self.dry_run = dry_run

    def run_all(self) -> None:
        self.step_01_delete_stale_src_marin()
        self.step_02_move_source_dirs()
        self.step_03_move_test_dirs()
        self.step_04_move_aux_dirs()
        self.step_05_generate_pyproject()
        self.step_06_update_workflows()
        self.step_07_update_dockerfiles()
        self.step_08_update_makefile()
        self.step_09_update_python_path_refs()
        self.step_09b_update_remaining_lib_refs()
        self.step_10_merge_agents_md()
        self.step_11_cleanup_empty_lib_dirs()
        self.step_12_scan_remaining_refs()
        self.step_13_run_precommit()
        log.info("Migration %s.", "complete" if not self.dry_run else "dry-run complete")

    # -- helpers -----------------------------------------------------------

    def _git_mv(self, src: str, dst: str) -> None:
        src_path = self.repo_root / src
        dst_path = self.repo_root / dst
        if not src_path.exists():
            log.warning("Skipping git mv: source does not exist: %s", src)
            return
        if self.dry_run:
            log.info("[dry-run] git mv %s -> %s", src, dst)
            return
        os.makedirs(dst_path.parent, exist_ok=True)
        subprocess.run(["git", "mv", str(src_path), str(dst_path)], cwd=self.repo_root, check=True)
        log.info("git mv %s -> %s", src, dst)

    def _git_rm(self, path: str) -> None:
        full = self.repo_root / path
        if not full.exists():
            log.warning("Skipping git rm: does not exist: %s", path)
            return
        if self.dry_run:
            log.info("[dry-run] git rm -r %s", path)
            return
        subprocess.run(["git", "rm", "-rf", str(full)], cwd=self.repo_root, check=True)
        log.info("git rm -r %s", path)

    def _write_file(self, rel_path: str, content: str) -> None:
        full = self.repo_root / rel_path
        if self.dry_run:
            log.info("[dry-run] write %s (%d bytes)", rel_path, len(content))
            return
        os.makedirs(full.parent, exist_ok=True)
        full.write_text(content)
        log.info("Wrote %s (%d bytes)", rel_path, len(content))

    def _read_file(self, rel_path: str) -> str:
        return (self.repo_root / rel_path).read_text()

    def _replace_in_file(self, rel_path: str, old: str, new: str) -> bool:
        """Replace a substring in a file. Returns True if a replacement was made."""
        full = self.repo_root / rel_path
        if not full.exists():
            return False
        content = full.read_text()
        if old not in content:
            return False
        if self.dry_run:
            log.info("[dry-run] replace in %s: %r -> %r", rel_path, old[:80], new[:80])
            return True
        full.write_text(content.replace(old, new))
        log.info("Replaced in %s: %r -> %r", rel_path, old[:80], new[:80])
        return True

    # -- steps -------------------------------------------------------------

    def step_01_delete_stale_src_marin(self) -> None:
        """Delete the stale src/marin/ shadow copy at repo root (untracked __pycache__ dirs)."""
        log.info("=== Step 1: Delete stale src/marin/ ===")
        stale = self.repo_root / "src" / "marin"
        if not stale.exists():
            log.info("No stale src/marin/ found, skipping.")
            return
        if self.dry_run:
            log.info("[dry-run] shutil.rmtree(%s)", stale)
            return
        shutil.rmtree(stale)
        log.info("Deleted stale %s", stale)

    def step_02_move_source_dirs(self) -> None:
        log.info("=== Step 2: Move source directories ===")
        for src, dst in SOURCE_MOVES.items():
            self._git_mv(src, dst)

    def step_03_move_test_dirs(self) -> None:
        log.info("=== Step 3: Move test directories ===")
        for src, dst in TEST_MOVES.items():
            self._git_mv(src, dst)

    def step_04_move_aux_dirs(self) -> None:
        log.info("=== Step 4: Move auxiliary directories ===")
        for src, dst in AUX_MOVES.items():
            self._git_mv(src, dst)

        log.info("--- Iris Dockerfiles ---")
        for src, dst in IRIS_DOCKER_FILE_MOVES.items():
            self._git_mv(src, dst)

        log.info("--- buf config files ---")
        for src, dst in BUF_MOVES.items():
            self._git_mv(src, dst)

    def step_05_generate_pyproject(self) -> None:
        log.info("=== Step 5: Generate merged pyproject.toml ===")
        self._write_file("pyproject.toml", MERGED_PYPROJECT)

    def step_06_update_workflows(self) -> None:
        log.info("=== Step 6: Update GitHub workflow path references ===")
        workflow_dir = self.repo_root / ".github" / "workflows"
        if not workflow_dir.exists():
            log.warning("No .github/workflows/ directory found, skipping.")
            return
        for wf_path in sorted(workflow_dir.iterdir()):
            if wf_path.suffix not in (".yaml", ".yml"):
                continue
            rel = str(wf_path.relative_to(self.repo_root))
            for old, new in WORKFLOW_REPLACEMENTS:
                self._replace_in_file(rel, old, new)

    def step_07_update_dockerfiles(self) -> None:
        log.info("=== Step 7: Update Dockerfiles ===")

        # docker/marin/Dockerfile.cluster
        self._replace_in_file("docker/marin/Dockerfile.cluster", CLUSTER_DOCKERFILE_OLD, CLUSTER_DOCKERFILE_NEW)

        # docker/marin/Dockerfile.vllm
        self._replace_in_file("docker/marin/Dockerfile.vllm", VLLM_DOCKERFILE_OLD, VLLM_DOCKERFILE_NEW)

        # docker/iris/Dockerfile.controller (after move)
        self._replace_in_file(
            "docker/iris/Dockerfile.controller",
            "# Build from the iris directory:",
            "# Build from repo root:",
        )
        self._replace_in_file(
            "docker/iris/Dockerfile.controller",
            "COPY src/ ./src/",
            "COPY src/iris/ ./src/iris/",
        )

        # docker/iris/Dockerfile.worker (after move)
        self._replace_in_file(
            "docker/iris/Dockerfile.worker",
            "# Build from the iris directory:",
            "# Build from repo root:",
        )
        self._replace_in_file(
            "docker/iris/Dockerfile.worker",
            "COPY src/ ./src/",
            "COPY src/iris/ ./src/iris/",
        )

        # docker/iris/Dockerfile.task (after move)
        self._replace_in_file(
            "docker/iris/Dockerfile.task",
            "-f lib/iris/Dockerfile.task",
            "-f docker/iris/Dockerfile.task",
        )

    def step_08_update_makefile(self) -> None:
        log.info("=== Step 8: Update Makefile ===")
        self._replace_in_file("Makefile", "lib/marin/src/marin/cluster/config.py", "src/marin/cluster/config.py")

    def step_09_update_python_path_refs(self) -> None:
        log.info("=== Step 9: Update Python cross-package path references ===")
        # tests/zephyr/conftest.py: IRIS_CONFIG path resolution changed because
        # the file moved from lib/zephyr/tests/ (3 levels below repo root) to
        # tests/zephyr/ (2 levels below). The old code walked up to lib/zephyr,
        # then across to lib/iris/examples/. Now we walk up to repo root and
        # into examples/iris/.
        conftest = "tests/zephyr/conftest.py"
        self._replace_in_file(
            conftest,
            'IRIS_CONFIG = Path(__file__).resolve().parents[2] / "iris" / "examples" / "demo.yaml"',
            'IRIS_CONFIG = Path(__file__).resolve().parents[2] / "examples" / "iris" / "demo.yaml"',
        )

        # Fix intra-test imports: each subproject's tests used `from tests.xxx`
        # to refer to their own test tree when `tests/` was the local root.
        # Now they live under `tests/{subproject}/`, so imports must be prefixed.
        for subproject in ("iris", "levanter", "haliax", "fray", "zephyr"):
            test_dir = self.repo_root / "tests" / subproject
            if not test_dir.exists():
                continue
            for py_file in test_dir.rglob("*.py"):
                rel = str(py_file.relative_to(self.repo_root))
                # Replace `from tests.` with `from tests.{subproject}.` but
                # avoid double-prefixing if already correct.
                self._replace_in_file(
                    rel,
                    "from tests.",
                    f"from tests.{subproject}.",
                )
                # Also fix pytest_plugins references like ["tests.test_utils"]
                self._replace_in_file(
                    rel,
                    '"tests.',
                    f'"tests.{subproject}.',
                )

        # Fix Path(__file__).parents references in iris tests.
        # These were relative to lib/iris/ — now need to resolve to repo root
        # and use examples/iris/ instead of examples/.
        #
        # IRIS_ROOT was lib/iris (parents[2] from tests/iris/e2e/*, parents[1] from tests/iris/*)
        # Now tests are one level deeper in the tree, so we bump parent count by 1
        # and adjust the path from there.

        # tests/iris/cluster/platform/test_scaling_group.py: parents[3] was lib/iris
        self._replace_in_file(
            "tests/iris/cluster/platform/test_scaling_group.py",
            'Path(__file__).parents[3] / "examples" / "coreweave.yaml"',
            'Path(__file__).parents[4] / "examples" / "iris" / "coreweave.yaml"',
        )

        # tests/iris/e2e/conftest.py: IRIS_ROOT = parents[2]  # lib/iris
        # tests/iris/e2e/test_building_backpressure.py: IRIS_ROOT = parents[2]
        # tests/iris/cluster/test_attempt_logs.py: IRIS_ROOT = parents[2]  # lib/iris
        # These all want the repo root now (parents[3] from tests/iris/e2e/ or tests/iris/cluster/)
        for f in [
            "tests/iris/e2e/conftest.py",
            "tests/iris/e2e/test_building_backpressure.py",
            "tests/iris/cluster/test_attempt_logs.py",
        ]:
            self._replace_in_file(f, ".parents[2]", ".parents[3]")
            # IRIS_ROOT / "examples" / "demo.yaml" → IRIS_ROOT / "examples" / "iris" / "demo.yaml"
            self._replace_in_file(
                f,
                'IRIS_ROOT / "examples" / "demo.yaml"',
                'IRIS_ROOT / "examples" / "iris" / "demo.yaml"',
            )

        # tests/iris/e2e/test_coreweave_live_kubernetes_runtime.py: parents[2] / "examples" / ...
        self._replace_in_file(
            "tests/iris/e2e/test_coreweave_live_kubernetes_runtime.py",
            '.parents[2] / "examples"',
            '.parents[3] / "examples" / "iris"',
        )

        # tests/iris/test_iris_run.py: parents[1] was lib/iris
        self._replace_in_file(
            "tests/iris/test_iris_run.py",
            ".parents[1]",
            ".parents[2]",
        )

        # tests/iris/examples/test_demo_notebook_submit.py: parents[3] was repo root
        # After move, this is at tests/iris/examples/ so parents[3] = tests/ (not repo root)
        self._replace_in_file(
            "tests/iris/examples/test_demo_notebook_submit.py",
            ".parents[3]",
            ".parents[4]",
        )

        # Fix test_marin_fs.py: several tests expect no MARIN_PREFIX but root
        # conftest.py sets one via autouse fixture. Add an autouse fixture to the
        # iris conftest that clears it for iris tests.
        iris_conftest = "tests/iris/conftest.py"
        iris_conftest_path = self.repo_root / iris_conftest
        if iris_conftest_path.exists():
            content = iris_conftest_path.read_text()
            fixture_code = (
                "\n\n@pytest.fixture(autouse=True)\n"
                "def _clear_marin_prefix(monkeypatch):\n"
                '    """Iris tests should not inherit MARIN_PREFIX from the root conftest."""\n'
                '    monkeypatch.delenv("MARIN_PREFIX", raising=False)\n'
            )
            if "_clear_marin_prefix" not in content:
                if "import pytest" not in content:
                    content = "import pytest\n" + content
                if self.dry_run:
                    log.info("[dry-run] Add _clear_marin_prefix fixture to %s", iris_conftest)
                else:
                    iris_conftest_path.write_text(content + fixture_code)
                    log.info("Added _clear_marin_prefix fixture to %s", iris_conftest)

        # Fix hardcoded lib/*/src PYTHONPATH entries in fray Ray deps files.
        # These are used at runtime to set PYTHONPATH for Ray workers.
        # v1 has harbor, v2 does not. Both use 8-space indentation.
        self._replace_in_file(
            "src/fray/v1/cluster/ray/deps.py",
            '        "lib/harbor/src",\n'
            '        "lib/fray/src",\n'
            '        "lib/haliax/src",\n'
            '        "lib/iris/src",\n'
            '        "lib/levanter/src",\n'
            '        "lib/marin/src",\n'
            '        "lib/zephyr/src",',
            '        "src",',
        )
        self._replace_in_file(
            "src/fray/v2/ray_backend/deps.py",
            '        "lib/fray/src",\n'
            '        "lib/haliax/src",\n'
            '        "lib/iris/src",\n'
            '        "lib/levanter/src",\n'
            '        "lib/marin/src",\n'
            '        "lib/zephyr/src",',
            '        "src",',
        )

        # Fix test_integration_test.py: use sys.executable instead of bare
        # "python" (which may not exist in venv-only setups), and pass cwd so
        # that integration_test.py's relative path ./tests/quickstart-data
        # resolves correctly.
        self._replace_in_file(
            "tests/test_integration_test.py",
            "import os\nimport subprocess\nimport tempfile",
            "import os\nimport subprocess\nimport sys\nimport tempfile",
        )
        self._replace_in_file(
            "tests/test_integration_test.py",
            '            ["python", os.path.join(MARIN_ROOT, "tests/integration_test.py"), "--prefix", temp_dir],\n'
            "            capture_output=False,\n"
            "            text=True,",
            '            [sys.executable, os.path.join(MARIN_ROOT, "tests/integration_test.py"),\n'
            '             "--prefix", temp_dir],\n'
            "            cwd=MARIN_ROOT,\n"
            '            env={**os.environ, "PYTHONPATH": os.pathsep.join([\n'
            '                os.path.join(MARIN_ROOT, "src"),\n'
            '                os.path.join(MARIN_ROOT, "tests"),\n'
            "            ])},\n"
            "            capture_output=False,\n"
            "            text=True,",
        )

    def step_09b_update_remaining_lib_refs(self) -> None:
        """Bulk-update remaining lib/ path references across docs, configs, scripts, and source."""
        log.info("=== Step 9b: Update remaining lib/ references ===")

        # Map of old-path-prefix -> new-path-prefix for simple text replacement.
        # These cover the vast majority of references in docs, comments, and configs.
        PATH_REWRITES = [
            # Source paths: lib/X/src/X/ -> src/X/
            ("lib/marin/src/marin/", "src/marin/"),
            ("lib/levanter/src/levanter/", "src/levanter/"),
            ("lib/haliax/src/haliax/", "src/haliax/"),
            ("lib/iris/src/iris/", "src/iris/"),
            ("lib/fray/src/fray/", "src/fray/"),
            ("lib/zephyr/src/zephyr/", "src/zephyr/"),
            # Test paths
            ("lib/levanter/tests/", "tests/levanter/"),
            ("lib/haliax/tests/", "tests/haliax/"),
            ("lib/iris/tests/", "tests/iris/"),
            ("lib/fray/tests/", "tests/fray/"),
            ("lib/zephyr/tests/", "tests/zephyr/"),
            # Scripts/infra/config/examples/docker
            ("lib/levanter/scripts/", "scripts/levanter/"),
            ("lib/levanter/config/", "config/levanter/"),
            ("lib/levanter/examples/", "examples/levanter/"),
            ("lib/levanter/infra/", "infra/levanter/"),
            ("lib/levanter/docker/", "docker/levanter/"),
            ("lib/iris/examples/", "examples/iris/"),
            ("lib/iris/Dockerfile", "docker/iris/Dockerfile"),
            # Doc paths
            ("lib/levanter/docs/", "docs/levanter/"),
            ("lib/zephyr/docs/", "docs/zephyr/"),
            ("lib/zephyr/README.md", "docs/zephyr/README.md"),
            ("lib/fray/docs/", "docs/fray/"),
            # Pyproject refs
            ("lib/levanter/pyproject.toml", "pyproject.toml"),
            ("lib/haliax/pyproject.toml", "pyproject.toml"),
            ("lib/marin/pyproject.toml", "pyproject.toml"),
            ("lib/iris/pyproject.toml", "pyproject.toml"),
            # License headers (these files don't move; we'll handle below)
            ("lib/levanter/etc/", "etc/levanter/"),
            ("lib/haliax/etc/", "etc/haliax/"),
        ]

        # Scan all tracked text files and apply bulk replacements.
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            log.warning("Could not list git files, skipping bulk lib/ rewrite.")
            return

        tracked_files = result.stdout.strip().splitlines()
        skip_suffixes = (".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".ttf", ".lock")
        skip_files = {"scripts/migrations/flatten_monorepo.py"}

        for rel_path in tracked_files:
            if rel_path in skip_files:
                continue
            full = self.repo_root / rel_path
            if not full.exists() or full.suffix in skip_suffixes:
                continue
            try:
                content = full.read_text(errors="ignore")
            except (OSError, UnicodeDecodeError):
                continue

            new_content = content
            for old, new in PATH_REWRITES:
                new_content = new_content.replace(old, new)

            if new_content != content:
                if self.dry_run:
                    log.info("[dry-run] bulk-rewrite lib/ refs in %s", rel_path)
                else:
                    full.write_text(new_content)
                    log.info("Bulk-rewrote lib/ refs in %s", rel_path)

        # --- Targeted fixes for references that don't follow the simple pattern ---

        # infra/pre-commit.py: glob patterns like lib/levanter/**/*.py -> src/levanter/**/*.py
        for old_pat, new_pat in [
            ('"lib/levanter/**/*.py"', '"src/levanter/**/*.py"'),
            ('"lib/haliax/**/*.py"', '"src/haliax/**/*.py"'),
            ('"lib/levanter/**"', '"src/levanter/**"'),
            ('"lib/haliax/**"', '"src/haliax/**"'),
            ('"lib/**/vendor/**"', '"**/vendor/**"'),
            ('"lib/marin/src/**/*.py"', '"src/marin/**/*.py"'),
            ('"lib/fray/src/**/*.py"', '"src/fray/**/*.py"'),
            ('"lib/iris/src/**/*.py"', '"src/iris/**/*.py"'),
            ('"lib/zephyr/src/**/*.py"', '"src/zephyr/**/*.py"'),
        ]:
            self._replace_in_file("infra/pre-commit.py", old_pat, new_pat)

        # src/iris/cli/build.py: repo root detection
        self._replace_in_file(
            "src/iris/cli/build.py",
            "Find the marin monorepo root (contains pyproject.toml + lib/iris).",
            "Find the marin monorepo root (contains pyproject.toml + src/iris).",
        )
        self._replace_in_file(
            "src/iris/cli/build.py",
            "lib/iris",
            "src/iris",
        )

        # src/marin/run/ray_run.py: excludes list
        self._replace_in_file(
            "src/marin/run/ray_run.py",
            '"lib/levanter/docs"',
            '"docs/levanter"',
        )

        # mkdocs.yml: search paths
        self._replace_in_file("mkdocs.yml", "lib/marin/src/", "src/marin/")
        self._replace_in_file("mkdocs.yml", '"lib/marin/src"', '"src"')

        # experiments/speedrun shell scripts
        self._replace_in_file(
            "experiments/speedrun/mixtral/scripts/run_maxtext_mixtral_profile.sh",
            "uv pip install -e lib/marin >/dev/null\nuv pip install -e lib/levanter >/dev/null",
            "# Packages are installed from the unified pyproject.toml",
        )
        self._replace_in_file(
            "experiments/speedrun/mixtral/scripts/run_maxtext_mixtral_profile.sh",
            "${REPO_ROOT}/lib/marin/src:${REPO_ROOT}/lib/levanter/src:${REPO_ROOT}/experiments",
            "${REPO_ROOT}/src:${REPO_ROOT}/experiments",
        )

        # tests/iris/e2e/_docker_cluster.py: workspace path
        self._replace_in_file(
            "tests/iris/e2e/_docker_cluster.py",
            "workspace=Path(__file__).parent.parent.parent,  # lib/iris",
            "workspace=Path(__file__).parent.parent.parent.parent,  # repo root",
        )

        # tests/iris/e2e/benchmark_controller.py: usage comments
        for old_cmd, new_cmd in [
            (
                "uv run python lib/iris/tests/e2e/benchmark_controller.py",
                "uv run python tests/iris/e2e/benchmark_controller.py",
            ),
        ]:
            self._replace_in_file("tests/iris/e2e/benchmark_controller.py", old_cmd, new_cmd)

        # Stale comments referencing old lib/ layout
        self._replace_in_file(
            "tests/iris/e2e/conftest.py",
            "IRIS_ROOT = Path(__file__).resolve().parents[3]  # lib/iris",
            "IRIS_ROOT = Path(__file__).resolve().parents[3]  # repo root",
        )
        self._replace_in_file(
            "tests/iris/cluster/test_attempt_logs.py",
            "IRIS_ROOT = Path(__file__).resolve().parents[3]  # lib/iris",
            "IRIS_ROOT = Path(__file__).resolve().parents[3]  # repo root",
        )
        self._replace_in_file(
            "tests/zephyr/conftest.py",
            "# Path to zephyr root (from tests/conftest.py -> tests -> lib/zephyr)",
            "# Path to zephyr root",
        )

        # Workflow comment remnant
        self._replace_in_file(
            ".github/workflows/zephyr-unit-tests.yaml",
            "# important: don't cd into lib/zephyr so that ray uv integration doesn't freak out",
            "# important: don't cd into a subdir so that ray uv integration doesn't freak out",
        )

        # Workflow fray TPU test (still references lib/fray path in docker)
        self._replace_in_file(
            ".github/workflows/fray-unit-tests.yaml",
            "cd /workspace/lib/fray && ",
            "cd /workspace && ",
        )

        # uv run --package X commands need updating (no more separate packages)
        for wf_path in sorted((self.repo_root / ".github" / "workflows").iterdir()):
            if wf_path.suffix not in (".yaml", ".yml"):
                continue
            rel = str(wf_path.relative_to(self.repo_root))
            self._replace_in_file(rel, "--package levanter ", "")
            self._replace_in_file(rel, "--package marin ", "")

        # docs/dev-guide: update command examples
        self._replace_in_file(
            "docs/dev-guide/guidelines-internal.md",
            "uv run lib/marin/src/marin/run/ray_run.py",
            "uv run src/marin/run/ray_run.py",
        )

        # infra/levanter scripts that reference docker paths
        self._replace_in_file(
            "infra/levanter/launch_on_ray.py",
            'repo_root / "lib/levanter/docker/tpu/Dockerfile.incremental"',
            'repo_root / "docker/levanter/tpu/Dockerfile.incremental"',
        )
        self._replace_in_file(
            "infra/levanter/launch.py",
            'repo_root / "lib/levanter/docker/tpu/Dockerfile.incremental"',
            'repo_root / "docker/levanter/tpu/Dockerfile.incremental"',
        )
        self._replace_in_file(
            "infra/levanter/push_docker.py",
            '"lib/levanter/docker/tpu/Dockerfile.base"',
            '"docker/levanter/tpu/Dockerfile.base"',
        )

        # infra/levanter setup scripts
        self._replace_in_file(
            "infra/levanter/helpers/setup-tpu-vm-tests.sh",
            "lib/levanter/infra/venv_path.txt",
            "infra/levanter/venv_path.txt",
        )
        self._replace_in_file(
            "infra/levanter/helpers/setup-tpu-vm.sh",
            "lib/levanter/infra/venv_path.txt",
            "infra/levanter/venv_path.txt",
        )

        # tools/ path (moved from lib/marin/tools/)
        for old_t, new_t in [
            ("lib/marin/tools/", "tools/"),
        ]:
            for f in [
                "docs/recipes/add_dataset.md",
                "docs/recipes/agent_profiling.md",
                "experiments/exp1880_sft_baseline.py",
            ]:
                self._replace_in_file(f, old_t, new_t)

        # docs/harbor-integration.md: cd lib/marin -> cd .  (repo root)
        self._replace_in_file("docs/harbor-integration.md", "cd lib/marin\n", "cd .\n")

        # docs/recipes/multi-stage.md: AGENTS.md references
        self._replace_in_file(
            "docs/recipes/multi-stage.md",
            "@lib/iris/AGENTS.md @lib/iris/README.md",
            "@AGENTS.md",
        )

        # examples/iris/demo_cluster.py
        self._replace_in_file(
            "examples/iris/demo_cluster.py",
            "# The iris project root (lib/iris/) - used as workspace for the example",
            "# The repo root - used as workspace for the example",
        )
        self._replace_in_file(
            "examples/iris/demo_cluster.py",
            'help="Workspace directory (default: lib/iris)"',
            'help="Workspace directory (default: repo root)"',
        )

        # infra/pre-commit.py: remaining src patterns that used full lib path
        self._replace_in_file(
            "infra/pre-commit.py",
            '"lib/levanter/src/**/*.py"',
            '"src/levanter/**/*.py"',
        )
        self._replace_in_file(
            "infra/pre-commit.py",
            '"lib/haliax/src/**/*.py"',
            '"src/haliax/**/*.py"',
        )

        # src/iris/cluster/k8s/kubectl.py: log parsing example (just a docstring)
        self._replace_in_file(
            "src/iris/cluster/k8s/kubectl.py",
            "file:///app/lib/haliax",
            "file:///app/src/haliax",
        )

    def step_10_merge_agents_md(self) -> None:
        log.info("=== Step 10: Merge AGENTS.md files ===")
        agents_files = {
            "lib/levanter/AGENTS.md": "Levanter",
            "lib/marin/AGENTS.md": "Marin",
            "lib/iris/AGENTS.md": "Iris",
            "lib/haliax/AGENTS.md": "Haliax",
        }

        # Collect content from subproject AGENTS.md files before they get removed.
        sections: list[str] = []
        for rel_path, section_name in agents_files.items():
            full = self.repo_root / rel_path
            if not full.exists():
                log.warning("AGENTS.md not found: %s", rel_path)
                continue
            content = full.read_text().strip()
            sections.append(f"\n\n## {section_name}\n\n{content}")

        if not sections:
            log.info("No subproject AGENTS.md files found, skipping merge.")
            return

        # Update cross-references in root AGENTS.md
        root_agents = self.repo_root / "AGENTS.md"
        if not root_agents.exists():
            log.warning("Root AGENTS.md not found, skipping.")
            return

        root_content = root_agents.read_text()

        ref_replacements = [
            (
                "`lib/levanter/AGENTS.md` for Levanter-specific conventions.",
                "the **Levanter** section below for Levanter-specific conventions.",
            ),
            (
                "`lib/marin/AGENTS.md` for Marin-specific conventions",
                "the **Marin** section below for Marin-specific conventions",
            ),
            (
                "`lib/iris/AGENTS.md` for Iris-specific conventions",
                "the **Iris** section below for Iris-specific conventions",
            ),
        ]
        for old, new in ref_replacements:
            root_content = root_content.replace(old, new)

        merged = root_content.rstrip() + "\n" + "".join(sections) + "\n"

        # Rewrite lib/ paths in the merged content.
        agents_rewrites = [
            ("lib/levanter", "src/levanter"),
            ("lib/haliax", "src/haliax"),
            ("/lib/marin/src/", "/src/marin/"),
            ("/lib/marin`", "/src/marin`"),
            ("/lib/marin/pyproject.toml", "/pyproject.toml"),
            ("lib/marin/tests", "tests"),
            ("lib/marin", "src/marin"),
            ("../../AGENTS.md", "the root AGENTS.md"),
            ("uv run --package marin pytest", "uv run pytest"),
        ]
        for old, new in agents_rewrites:
            merged = merged.replace(old, new)

        if self.dry_run:
            log.info("[dry-run] Would append %d subproject sections to AGENTS.md", len(sections))
            return

        root_agents.write_text(merged)
        log.info("Merged %d subproject AGENTS.md sections into root AGENTS.md", len(sections))

    def step_11_cleanup_empty_lib_dirs(self) -> None:
        log.info("=== Step 11: Clean up empty lib/ directories ===")
        for d in CLEANUP_LIB_DIRS:
            self._git_rm(d)

    def step_12_scan_remaining_refs(self) -> None:
        """Scan for any remaining references to the old lib/ layout and log warnings."""
        log.info("=== Step 12: Scan for remaining lib/ references ===")
        patterns = [
            "lib/marin",
            "lib/levanter",
            "lib/haliax",
            "lib/iris",
            "lib/fray",
            "lib/zephyr",
        ]
        # Only scan tracked text files to avoid binary noise.
        try:
            result = subprocess.run(
                ["git", "ls-files"],
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            log.warning("Could not list git files, skipping remaining-ref scan.")
            return

        tracked_files = result.stdout.strip().splitlines()
        hit_count = 0
        for rel_path in tracked_files:
            full = self.repo_root / rel_path
            # Skip binary / non-text and this script itself.
            if not full.exists() or full.suffix in (
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".ico",
                ".woff",
                ".woff2",
                ".ttf",
                ".lock",
            ):
                continue
            if rel_path == "scripts/migrations/flatten_monorepo.py":
                continue
            try:
                content = full.read_text(errors="ignore")
            except (OSError, UnicodeDecodeError):
                continue
            for pat in patterns:
                if pat in content:
                    log.warning("Remaining reference to %r in %s", pat, rel_path)
                    hit_count += 1
        if hit_count == 0:
            log.info("No remaining lib/ references found.")
        else:
            log.info("Found %d file(s) with remaining lib/ references. Review manually.", hit_count)

    def step_13_run_precommit(self) -> None:
        """Run pre-commit with --fix to auto-format all files after the migration."""
        log.info("=== Step 13: Run pre-commit (auto-fix formatting) ===")
        if self.dry_run:
            log.info("[dry-run] Would run: ./infra/pre-commit.py --all-files --fix")
            return
        result = subprocess.run(
            ["python", str(self.repo_root / "infra" / "pre-commit.py"), "--all-files", "--fix"],
            cwd=self.repo_root,
        )
        if result.returncode != 0:
            log.warning("Pre-commit returned non-zero exit code %d. Review output above.", result.returncode)
        else:
            log.info("Pre-commit passed.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def find_repo_root() -> Path:
    """Walk up from cwd to find the git repository root."""
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flatten the marin monorepo from lib/*/src/* to src/* layout.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform the migration. Without this flag, the script only prints what it would do.",
    )
    args = parser.parse_args()

    repo_root = find_repo_root()
    dry_run = not args.execute

    if dry_run:
        log.info("DRY RUN — no changes will be made. Pass --execute to apply.")
    else:
        log.info("EXECUTING migration on %s", repo_root)

    migration = Migration(repo_root=repo_root, dry_run=dry_run)
    migration.run_all()


if __name__ == "__main__":
    main()
