# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the regional worker's exact argv locally with metadata-only storage stubs."""

import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.post_training.task_curation.ingest_stages.graph import IngestExecution
from experiments.post_training.task_curation.ingest_stages.guard import SourceWitness
from experiments.post_training.task_curation.ingest_stages.worker import (
    REPORT_FILENAME,
    REQUEST_FILENAME,
    WorkerPhase,
    WorkerRequest,
)
from experiments.post_training.task_curation.release_stages.preflight import type_check_module


def worker_preflight(execution: IngestExecution, witness: SourceWitness, repo_root: Path) -> dict:
    """Exercise parsing, imports, source attestation, eligibility and graph lowering.

    The worker refuses remote metadata references in this phase. No provider or
    data-plane client is created; production stage behavior has separate tests.
    """
    type_check_module("experiments/post_training/task_curation/ingest_stages", repo_root)
    with TemporaryDirectory(prefix="atlas-ingest-preflight-") as temporary:
        scratch = Path(temporary)
        (scratch / ".venv").symlink_to(repo_root / ".venv", target_is_directory=True)
        request = WorkerRequest(phase=WorkerPhase.PREFLIGHT, execution=execution, witness=witness)
        (scratch / REQUEST_FILENAME).write_text(request.model_dump_json(indent=2) + "\n")
        paths = [repo_root, *sorted(repo_root.glob("lib/*/src")), repo_root / "infra/marina/src"]
        result = subprocess.run(
            execution.worker_argv,
            cwd=scratch,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(map(str, paths))},
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise ValueError(f"exact worker argv preflight failed:\n{result.stderr[-4000:]}")
        return json.loads((scratch / REPORT_FILENAME).read_text())
