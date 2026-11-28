# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Goedel-Lean RL V2: Lean 4 revision/completion RL data with a kernel-replay checker.

The dataset (Goedel-LM/RL_dataset_V2) carries multi-turn chat transcripts: the user
provides incomplete or incorrect Lean 4 code, the assistant produces a proof attempt,
and the user provides compiler-error feedback. Two subpopulations by ``subset``:
``lean4_revision`` (revise an incorrect proof) and ``lean4_completion`` (complete a
partial proof). The proofs are model-generated attempts, not verified golden proofs —
the recipe declares ``oracle_solutions=False``. The Lean kernel-replay grader
(``lean/lean_grade.py``) fails closed: a proof that doesn't kernel-check scores 0.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finestore.rl.curation import TaskFormat
from finestore.rl.status_taxonomy import DataRestriction
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.models import ImportFailureKind, ImportRejection, IntendedUse, RawRow

from experiments.post_training.task_curation.datasets.environments import LEAN_KERNEL_GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.lean.proofs import (
    LEAN_GRADER_MEMORY_MB,
    proof_task,
    statement_contract,
)
from experiments.post_training.task_curation.pipeline import CurationRecipe, HfSource, SourceFormat, process_rows
from experiments.post_training.task_curation.source import RlDataSource, SourceInfo, SourceOrigin

HERE = Path(__file__).parent
LOGGER = logging.getLogger(__name__)

GOEDEL_REVISION = "5c99dce377f387bd271d726f205e69380b792a9d"
GOEDEL_REPO = "Goedel-LM/RL_dataset_V2"
GOEDEL_FILES = (
    "data/train-00000-of-00003.parquet",
    "data/train-00001-of-00003.parquet",
    "data/train-00002-of-00003.parquet",
)
GOEDEL_GRADER = LEAN_KERNEL_GRADER_PACKAGES
GOEDEL_MEMORY = LEAN_GRADER_MEMORY_MB


def _extract_last_assistant(messages: list[dict[str, Any]]) -> str:
    """The last assistant message's content, which holds the proof attempt."""
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


def _extract_lean_block(content: str) -> str | None:
    """The first ```lean4 fenced block in the message, stripped."""
    marker = "```lean4\n"
    start = content.find(marker)
    if start < 0:
        return None
    end = content.find("```", start + len(marker))
    if end < 0:
        return None
    return content[start + len(marker) : end].strip()


def _is_valid_row(row: dict[str, Any]) -> bool:
    subset = row.get("subset", "")
    if subset not in ("lean4_revision", "lean4_completion"):
        return False
    messages = row.get("messages", [])
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    last_assistant = _extract_last_assistant(messages)
    if not last_assistant or _extract_lean_block(last_assistant) is None:
        return False
    return True


def convert_goedel(row: RawRow, context) -> TaskSpec | type:
    """Convert one Goedel-Lean row into a Lean proof TaskSpec."""
    data = row.data
    subset = data.get("subset", "")
    messages = data.get("messages", [])
    if subset not in ("lean4_revision", "lean4_completion"):
        return _reject("unsupported_subset", f"Unknown subset: {subset}")

    last_assistant = _extract_last_assistant(messages)
    lean_code = _extract_lean_block(last_assistant)
    if not lean_code:
        return _reject("missing_lean_block", "No ```lean4 fenced block in the last assistant message")

    # The prompt is the full multi-turn context: the user's request, any prior
    # assistant attempts, and the compiler-error feedback.
    prompt_parts = []
    for message in messages:
        role = message.get("role", "")
        content = message.get("content", "")
        prompt_parts.append(f"### {role}:\n{content}")
    prompt = "\n\n".join(prompt_parts)

    # The task's public statement is the user's first message (the problem request).
    first_user = next((m.get("content", "") for m in messages if m.get("role") == "user"), prompt)

    # The answer reference is the model's last proof attempt (diagnostics only,
    # never used as an oracle). The grader runs its own Lean kernel check.
    evidence = {"subset": subset, "source": "goedel-lean-rl-v2"}

    contract = statement_contract(lean_code, "", lean_code)
    if isinstance(contract, type):
        return contract

    return proof_task(
        row,
        context,
        contract,
        first_user,
        {**evidence, "proof": lean_code},
        memory_mb=LEAN_GRADER_MEMORY_MB,
    )


def _reject(kind: str, reason: str):
    """An ImportRejection-like typed rejection."""
    return ImportRejection(kind=ImportFailureKind.UNSUPPORTED, reason=f"{kind}: {reason}")


@dataclass(frozen=True)
class GoedelSubset:
    name: str
    subset: str


SUBSETS = (
    GoedelSubset("goedel_lean_revision", "lean4_revision"),
    GoedelSubset("goedel_lean_completion", "lean4_completion"),
)


def sources() -> list[RlDataSource]:
    """Two RlDataSource declarations (one per subset) for the Goedel-Lean RL V2 dataset."""
    out = []
    for subset in SUBSETS:
        out.append(
            RlDataSource(
                pipeline=process_rows,
                info=SourceInfo(
                    data_restriction=DataRestriction.OPEN,
                    origin=SourceOrigin.GOED_LM,
                    upstream_dataset="RL_dataset_V2",
                    upstream_name=f"Goedel-LM/RL_dataset_V2 · {subset.subset}",
                    build_path="Lean kernel replay",
                    task_format=TaskFormat.TASKSPEC,
                    subset=subset.subset,
                    family="math-proof",
                    tags=("lean", "theorem-proving", "rl-training", "license:check-subset"),
                    notes=(
                        "Lean 4 RL revision/completion data; the proofs are model-generated "
                        "attempts with compiler-error feedback. No verified golden oracle."
                    ),
                ),
                config=CurationRecipe(
                    name=subset.name,
                    source=HfSource(
                        GOEDEL_REPO,
                        GOEDEL_REVISION,
                        GOEDEL_FILES,
                        SourceFormat.PARQUET,
                    ),
                    convert=convert_goedel,
                    version="1",
                    intended_use=IntendedUse.TRAIN,
                    oracle_solutions=False,
                    grader=GOEDEL_GRADER,
                ),
            )
        )
    return out
