# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""nebius/SWE-rebench-openhands-trajectories dataset download and transform.

OpenHands agent trajectories on SWE-rebench tasks. Each row contains a
multi-turn conversation (system, assistant, user, tool roles) along with
a resolved flag indicating whether the trajectory solved the issue.
"""

from zephyr import counters

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import (
    load_parquet_batched,
    render_role_message,
    run_document_transform,
    text_document,
)
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "nebius/SWE-rebench-openhands-trajectories"
HF_REVISION = "3545538"


def resolved_to_tag(resolved: int | None) -> str | None:
    if resolved is None:
        return None
    if resolved >= 1:
        return "This trajectory solved the task successfully."
    return "This trajectory failed to solve the task."


def row_to_doc(row: dict) -> list[dict]:
    trajectory = row.get("trajectory")
    if not trajectory:
        counters.increment("swe_rebench_openhands/dropped")
        return []

    tag = resolved_to_tag(row.get("resolved"))
    rendered = "\n\n".join(render_role_message(m) for m in trajectory)
    text = f"{tag}\n\n{rendered}" if tag else rendered

    counters.increment("swe_rebench_openhands/kept")
    return [text_document(text, "nebius/SWE-rebench-openhands-trajectories")]


def transform(input_path: str, output_path: str) -> None:
    run_document_transform(
        input_path=input_path,
        output_path=output_path,
        row_to_doc=row_to_doc,
        name="swe-rebench-openhands-transform",
        ram="32g",
        loader=load_parquet_batched,
    )


def download_swe_rebench_openhands_step() -> StepSpec:
    """Download and transform SWE-rebench-openhands-trajectories into JSONL documents."""
    dl = download_hf_step(
        "raw/swe-rebench-openhands-trajectories",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
    )

    return StepSpec(
        name="processed/swe-rebench-openhands-trajectories",
        deps=[dl],
        fn=lambda output_path: transform(
            input_path=dl.output_path,
            output_path=output_path,
        ),
        hash_attrs={"version": "v2"},
    )


def swe_rebench_openhands_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the full ``(download+transform, normalize)`` chain for swe-rebench-openhands."""
    processed = download_swe_rebench_openhands_step()
    return (
        processed,
        normalize_step(name="normalized/swe-rebench-openhands", download=processed),
    )
