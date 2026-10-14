# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Grader environments shared by the catalog's declarations.

Exact pins carry the versions earlier grader images were validated with; the build compiles them to a
hash lock, so changing a pin here changes the environment's identity and rebuilds it.
"""

from dataclasses import replace
from pathlib import Path

from experiments.post_training.task_curation.environment import Archive, Environment

HERE = Path(__file__).resolve().parent
VERIFYIT_PACKAGE = HERE.parents[3] / "lib/verifyit"

SCIENTIFIC_STACK = (
    # Shared by the vendored scorers (Nemotron Ultra, SkyRL code and SQL, ARC).
    "numpy==2.3.5",
    "pandas==2.2.2",
    "scipy==1.17.0",
)
MATH_SCORERS = (
    # TaskTrove math uses verifyit's symbolic comparator in the grader sandbox.
    "math-verify==0.9.0",
    "sympy==1.13.3",
    "antlr4-python3-runtime==4.11.0",
    "rdkit==2025.9.3",
    "reasoning-gym==0.1.25",
)
STRUCTURED_OUTPUT_SCORERS = (
    # Structured-output, tool-call and instruction-following scorers.
    "jsonschema==4.25.1",
    "pyyaml==6.0.3",
    "xmltodict==0.14.2",
    "openapi-schema-validator==0.6.3",
    "langdetect==1.0.9",
    "nltk==3.9.2",
    "absl-py==2.3.1",
    "immutabledict==4.3.0",
    "verifiable-instructions @ https://github.com/abukharin-nv/verifiable-instructions/archive/f46a5ac87b1400a4f8973039844b6be9b56e3faf.tar.gz",
)
NVARC_SANDBOX_CLIENT = (
    # The vendored NVARC sandbox client (arc/scorers/.../sandbox.py) imports requests.
    "requests==2.33.1",
)
PYTHON_TEST_GRADERS = (
    # Python test graders, and verifyit's own dependencies (verifyit is copied into the environment).
    "pytest==9.1.1",
    "pytest-json-report==1.5.0",
    "hypothesis==6.168.5",
    "tomli-w==1.2.0",
)
STACK_PYTEST_TASKS = (
    # The Stack Overflow Python-test tasks (tasktrove-stack_pytest): their hidden tests import mock, typer,
    # freezegun, fastapi, flask, scikit-learn and others. These are the packages, at the versions, of the
    # agent image those tasks are solved in, so a submission that runs there imports here too. Starlette's
    # TestClient now requires httpx2. Flask keeps the pin the grader image was validated with (the agent
    # image has 3.1.3).
    "aiohttp==3.14.4",
    "beautifulsoup4==4.15.0",
    "botocore==1.43.108",
    "cryptography==50.0.2",
    "dask==2026.8.0",
    "django==6.1.1",
    "faker==40.41.0",
    "fastapi==0.142.2",
    "httpx2==2.13.0",
    "flask==3.1.2",
    "freezegun==1.5.5",
    "importlib-metadata==9.0.1",
    "marshmallow==4.3.1",
    "mock==5.2.0",
    "networkx==3.7",
    "opencv-python-headless==5.0.0.93",
    "pathlib2==2.3.7.post1",
    "redis==8.1.0",
    "responses==0.26.3",
    "scikit-learn==1.9.1",
    "setuptools==84.0.0",
    "sqlalchemy==2.1.3",
    "typer==0.27.2",
    "xarray==2026.9.0",
)

PUNKT_TAB = Archive(
    "https://raw.githubusercontent.com/nltk/nltk_data/4f15a3d89eefe9748ec1c05be495d91289197155/packages/tokenizers/punkt_tab.zip",
    "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106",
    "punkt_tab/english",
)
"""The English Punkt sentence tokenizer the instruction-following scorers load through NLTK."""

GRADER_PACKAGES = Environment(
    requirements=(
        *SCIENTIFIC_STACK,
        *MATH_SCORERS,
        *STRUCTURED_OUTPUT_SCORERS,
        *NVARC_SANDBOX_CLIENT,
        *PYTHON_TEST_GRADERS,
        *STACK_PYTEST_TASKS,
    ),
    files={"/srv/nltk_data/tokenizers/punkt_tab/english": PUNKT_TAB},
    env={"NLTK_DATA": "/srv/nltk_data"},
)
"""The packages every grade script in the catalog imports, and the NLTK data they load.

NLTK reads ``NLTK_DATA`` for its data directory. Its default, ``/usr/local/share/nltk_data``, is
read-only inside the worker sandbox, so the data lives under ``/srv`` in both placements.
"""

COMPILER_GRADER_PACKAGES = replace(GRADER_PACKAGES, apt=("build-essential",))
"""The grader packages with a C++ toolchain, for graders that compile submissions."""

LEAN_GRADER_PACKAGES = replace(GRADER_PACKAGES, dockerfile=HERE / "nemotron_ultra/lean.Dockerfile")
"""The grader packages with Lean 4 and Mathlib, for the Nemotron Ultra Lean proof grader."""

BUILDABLE_ENVIRONMENTS = (GRADER_PACKAGES, COMPILER_GRADER_PACKAGES, LEAN_GRADER_PACKAGES)
"""Explicit environment registrations for the images build CLI."""
