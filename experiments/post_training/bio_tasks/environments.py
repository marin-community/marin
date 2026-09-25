# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reusable image contexts built from successfully checked native package locks."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from experiments.post_training.tasktrove.taskbinary import TaskFiles

SOURCE = Path(__file__).parent


@dataclass(frozen=True)
class EnvironmentProfile:
    repository_index: int
    evidence: str
    memory_mb: int
    blas_core: str | None = None


PROFILES = {
    "real-heme-pocket-burial": EnvironmentProfile(19, "native_validation_runs/1d21faa8f7b0.json", 2048),
    "real-fastq-pair-filter": EnvironmentProfile(35, "native_validation_runs/fd1f4c3a31f1.json", 4096),
    "real-fastq-quality-yield": EnvironmentProfile(15, "native_validation_runs/fd1f4c3a31f1.json", 4096),
    "real-protein-alignment": EnvironmentProfile(43, "native_validation_runs/035bb3a42edc.json", 2048),
    "real-rnaseq-differential-expression": EnvironmentProfile(
        5, "native_validation_runs/e39ec1c86f62.json", 4096, "HASWELL"
    ),
    "real-rnaseq-population-interaction": EnvironmentProfile(
        5, "native_validation_runs/5436d9bd3811.json", 4096, "HASWELL"
    ),
    "real-rnaseq-shrinkage-enrichment-audit": EnvironmentProfile(
        5, "native_validation_runs/42fbd427ac27.json", 8192, "HASWELL"
    ),
    "real-rnaseq-go-enrichment": EnvironmentProfile(5, "native_validation_runs/e39ec1c86f62.json", 4096, "HASWELL"),
    "real-cox1-tree-comparison": EnvironmentProfile(28, "native_validation_runs/e2f647365a6f.json", 2048),
    "real-singlecell-read-qc": EnvironmentProfile(23, "native_validation_runs/40ed93019b52.json", 4096),
    "real-singlecell-representation-audit": EnvironmentProfile(23, "native_validation_runs/b1cfab727472.json", 8192),
    "real-proteome-domain-search": EnvironmentProfile(11, "native_validation_runs/a619ee74f5b8.json", 2048),
    "real-proteome-clustering": EnvironmentProfile(42, "native_validation_runs/9d28dd15e082.json", 2048),
    "real-phix-assembly": EnvironmentProfile(27, "native_validation_runs/cd4c5520958b.json", 4096),
    "real-phix-bam-read-structure": EnvironmentProfile(3, "native_validation_runs/81c78459a53b.json", 2048),
}


def environment_files(recipe: str, base_image: str) -> TaskFiles:
    """Keep variable task observations out of Harbor's environment snapshot hash."""
    dockerfile = f"FROM {base_image}\nWORKDIR /app\nRUN ln -s /setup_files/inputs /app/inputs\n"
    profile = PROFILES.get(recipe)
    if profile is None:
        return TaskFiles({"environment/Dockerfile": dockerfile.encode()})
    index = json.loads((SOURCE / "native_validation.json").read_text())
    record = next(run for run in index["runs"] if run["checks_file"] == profile.evidence)
    raw = (SOURCE / profile.evidence).read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["checks_sha256"]:
        raise ValueError(f"Changed environment validation evidence: {profile.evidence}")
    check = next(row for row in json.loads(raw)["checks"] if row["repository_index"] == profile.repository_index)
    if check["status"] != "passed":
        raise ValueError(f"Environment has no passing reference check: {recipe}")
    packages = check["environment"]["resolved_packages"]
    dockerfile += (
        "COPY packages.lock.json install_environment.py /opt/bio-build/\n"
        "RUN python3 /opt/bio-build/install_environment.py --lock /opt/bio-build/packages.lock.json "
        "--prefix /opt/bio && rm /opt/bio-build/install_environment.py\n"
        "ENV PATH=/opt/bio/bin:$PATH\n"
        "RUN printf '%s\\n' 'export PATH=/opt/bio/bin:$PATH' > /etc/profile.d/bio.sh\n"
        "ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 "
        "POLARS_MAX_THREADS=1 RAYON_NUM_THREADS=1 NUMBA_NUM_THREADS=1\n"
    )
    if profile.blas_core is not None:
        dockerfile += f"ENV OPENBLAS_CORETYPE={profile.blas_core}\n"
    return TaskFiles(
        {
            "environment/Dockerfile": dockerfile.encode(),
            "environment/packages.lock.json": (json.dumps(packages, indent=2) + "\n").encode(),
            "environment/install_environment.py": (SOURCE / "install_environment.py").read_bytes(),
        }
    )
