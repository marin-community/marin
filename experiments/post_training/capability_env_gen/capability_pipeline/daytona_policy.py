"""Credential-free, hashable policy for Daytona private verifier snapshots."""

from __future__ import annotations

import hashlib
import shlex

from .image_runtime_metadata import derive_daytona_recipe

# Complete Linux dependency closure of the TaskCompendium project's core
# dependencies and TaskTrove's answer/schema/judge extras. Versions come from
# the staged dc6b501 uv.lock. The two source packages themselves are uploaded
# after snapshot creation. --no-deps makes future registry resolution unable to
# change this environment behind the recorded bootstrap hash.
VERIFIER_REQUIREMENTS = (
    "annotated-types==0.8.0",
    "antlr4-python3-runtime==4.13.2",
    "anyio==4.15.1",
    "attrs==26.1.0",
    "certifi==2026.7.22",
    "distro==1.9.0",
    "fsspec==2026.7.0",
    "h11==0.16.0",
    "httpcore==1.0.9",
    "httpx==0.28.1",
    "idna==3.19",
    "jiter==0.16.0",
    "jsonschema==4.26.0",
    "jsonschema-specifications==2025.9.1",
    "latex2sympy2-extended==1.11.0",
    "math-verify==0.9.0",
    "mpmath==1.3.0",
    "msgspec==0.21.1",
    "numpy==2.5.3",
    "openai==2.54.0",
    "pandas==3.0.5",
    "pyarrow==25.0.1",
    "pydantic==2.13.5",
    "pydantic-core==2.46.5",
    "python-dateutil==2.9.0.post0",
    "pyyaml==6.0.3",
    "referencing==0.37.0",
    "rpds-py==2026.6.3",
    "six==1.17.0",
    "sniffio==1.3.1",
    "sympy==1.14.0",
    "tomlkit==0.15.1",
    "tqdm==4.70.1",
    "typing-extensions==4.16.0",
    "typing-inspection==0.4.4",
)
VERIFIER_BOOTSTRAP = (
    "RUN python3 -m pip install --no-cache-dir --no-deps "
    + " ".join(VERIFIER_REQUIREMENTS)
    + "\n"
)


def verifier_bootstrap(supervisor_python: str = "python3") -> str:
    """Install supervisor dependencies into the interpreter that will import them.

    Even the default command binds pip to the supervisor interpreter. This
    deliberately changes the recipe hash from legacy bare-pip snapshots.
    """
    if supervisor_python == "python3":
        return VERIFIER_BOOTSTRAP
    if not supervisor_python or "\n" in supervisor_python or "\r" in supervisor_python:
        raise ValueError("supervisor Python must be a nonempty command path")
    return (
        f"RUN {shlex.quote(supervisor_python)} -m pip install "
        "--no-cache-dir --no-deps "
        + " ".join(VERIFIER_REQUIREMENTS)
        + "\n"
    )


def verifier_bootstrap_sha256(supervisor_python: str = "python3") -> str:
    return hashlib.sha256(verifier_bootstrap(supervisor_python).encode()).hexdigest()


def verifier_snapshot_recipe(image: str, supervisor_python: str = "python3") -> str:
    return (
        derive_daytona_recipe(image)
        + "USER root\n"
        + verifier_bootstrap(supervisor_python)
    )
