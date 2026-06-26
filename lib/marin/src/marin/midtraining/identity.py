# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run identity for midtraining launches.

A :class:`RunIdentity` carries one explicit name, one output path, and one
W&B run id. All training-side namespaces (``RUN_ID``, ``WANDB_RUN_ID``,
permanent checkpoints, temporary checkpoints, manifest path) derive from
this object.

Two layers of identity:

- ``logical_cell_id`` — stable across fresh attempts; analysis joins
  attempts through this key.
- ``run_id`` (= ``basename(output_path)``) — unique per attempt; equals
  the Levanter ``RUN_ID`` and W&B run id, so a fresh attempt is always a
  distinct W&B row and the monotonic-step rejection bug cannot recur.

See ``.agents/logbooks/midtraining_redesign.md`` § "Attempts and W&B
identity" for the rationale.
"""

import os
import re
from dataclasses import dataclass

GCS_PREFIX = "gs://"

# Marin's regional buckets follow the convention ``gs://marin-<region>/...``
# (e.g. ``marin-us-east5``, ``marin-us-central1``). The redesign requires
# output paths to encode exactly one region, so a regex match here is the
# canonical region resolver.
_MARIN_BUCKET_PATTERN = re.compile(r"^marin-([a-z0-9-]+)$")

# Attempt suffix on a run id. Always rendered with three zero-padded digits
# so lexicographic sort matches numeric order across attempt files.
_ATTEMPT_SUFFIX_PATTERN = re.compile(r"-a(\d{3})$")
_W_AND_B_NAME_MAX_LEN = 64


@dataclass(frozen=True)
class RunIdentity:
    """One run namespace shared by Levanter, W&B, GCS, and the manifest.

    Args:
        logical_cell_id: Stable identifier for the experiment cell. Two
            fresh attempts of the same cell share this id.
        attempt: 1-indexed attempt number. Increments per fresh restart;
            same-run resume keeps the same attempt.
        output_path: Concrete ``gs://...`` output namespace. Must include
            the zero-padded ``-aNNN`` attempt suffix.
        wandb_project: W&B project name. ``"delphi-midtraining"`` for the
            Delphi sweeps.
        wandb_entity: W&B entity (team). Marin community default.
    """

    logical_cell_id: str
    attempt: int
    output_path: str
    wandb_project: str
    wandb_entity: str = "marin-community"

    def __post_init__(self) -> None:
        if not self.logical_cell_id:
            raise ValueError("RunIdentity.logical_cell_id must be non-empty")
        if self.attempt <= 0:
            raise ValueError(f"RunIdentity.attempt must be positive, got {self.attempt!r}")
        if not self.output_path.startswith(GCS_PREFIX):
            raise ValueError(f"RunIdentity.output_path must be a gs:// URI, got {self.output_path!r}")
        if self.output_path.endswith("/"):
            raise ValueError(f"RunIdentity.output_path must not end with '/', got {self.output_path!r}")
        if "/checkpoints/step-" in self.output_path:
            raise ValueError(
                f"RunIdentity.output_path must be a run root, not a concrete checkpoint: {self.output_path!r}"
            )

        basename = self.run_id
        suffix_match = _ATTEMPT_SUFFIX_PATTERN.search(basename)
        if suffix_match is None:
            raise ValueError(
                f"RunIdentity.output_path basename {basename!r} must end with attempt suffix "
                "'-aNNN' (e.g. '-a001'); use build_run_identity() to construct the path."
            )
        encoded_attempt = int(suffix_match.group(1))
        if encoded_attempt != self.attempt:
            raise ValueError(f"Output basename encodes attempt {encoded_attempt} but RunIdentity.attempt={self.attempt}")

        if len(basename) > _W_AND_B_NAME_MAX_LEN:
            raise ValueError(f"RunIdentity.run_id {basename!r} exceeds W&B safe length ({_W_AND_B_NAME_MAX_LEN})")
        if not self.wandb_project:
            raise ValueError("RunIdentity.wandb_project must be non-empty")
        if not self.wandb_entity:
            raise ValueError("RunIdentity.wandb_entity must be non-empty")

    @property
    def run_id(self) -> str:
        """``basename(output_path)``; equals Levanter ``RUN_ID`` and W&B run id."""
        return os.path.basename(self.output_path)

    @property
    def output_region(self) -> str:
        """Region resolved from the ``gs://marin-<region>/...`` output bucket."""
        return output_region(self.output_path)

    @property
    def manifest_uri(self) -> str:
        """Authoritative manifest path inside the run directory."""
        return f"{self.output_path}/midtrain_manifest.json"

    @property
    def train_config_uri(self) -> str:
        """Rendered Levanter ``TrainLmConfig`` YAML alongside the manifest."""
        return f"{self.output_path}/train_lm_config.yaml"

    @property
    def launch_command_uri(self) -> str:
        """Recorded launch command for audit."""
        return f"{self.output_path}/launch_command.txt"

    @property
    def permanent_checkpoints_uri(self) -> str:
        """Run-permanent checkpoint directory."""
        return f"{self.output_path}/checkpoints"


def output_region(output_path: str) -> str:
    """Return the GCP region encoded in a Marin ``gs://marin-<region>/...`` URI."""
    if not output_path.startswith(GCS_PREFIX):
        raise ValueError(f"Expected gs:// URI, got {output_path!r}")
    rest = output_path.removeprefix(GCS_PREFIX)
    bucket = rest.split("/", 1)[0]
    match = _MARIN_BUCKET_PATTERN.match(bucket)
    if match is None:
        raise ValueError(f"Cannot resolve region from bucket {bucket!r}; Marin buckets follow 'marin-<region>'.")
    return match.group(1)


def build_run_identity(
    *,
    logical_cell_id: str,
    attempt: int,
    output_region_name: str,
    wandb_project: str,
    wandb_entity: str = "marin-community",
    output_root: str = "checkpoints",
) -> RunIdentity:
    """Construct a :class:`RunIdentity` with the canonical output-path layout.

    ``output_path = gs://marin-<region>/<output_root>/<logical_cell_id>-aNNN``.
    Use this helper instead of building the path manually so the attempt
    suffix is always rendered consistently.
    """
    if attempt <= 0:
        raise ValueError(f"attempt must be positive, got {attempt!r}")
    if "-a" in logical_cell_id and _ATTEMPT_SUFFIX_PATTERN.search(logical_cell_id):
        raise ValueError(
            f"logical_cell_id {logical_cell_id!r} already contains an attempt suffix; pass just the cell id."
        )
    attempt_suffix = f"-a{attempt:03d}"
    run_id = f"{logical_cell_id}{attempt_suffix}"
    output_path = f"{GCS_PREFIX}marin-{output_region_name}/{output_root.strip('/')}/{run_id}"
    return RunIdentity(
        logical_cell_id=logical_cell_id,
        attempt=attempt,
        output_path=output_path,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
    )


def attempt_group_manifest_uri(*, logical_cell_id: str, region: str) -> str:
    """Return the attempt-group manifest path joining all attempts of one cell.

    Analysis joins attempt manifests through this file rather than through
    W&B display names.
    """
    return f"{GCS_PREFIX}marin-{region}/midtrain-manifests/runs/{logical_cell_id}.json"


def expected_run_env(identity: RunIdentity) -> dict[str, str]:
    """Environment-variable mapping that pins the training child's identity.

    Marin's training submitter already enforces that
    ``RUN_ID == basename(output_path)``; setting these explicitly makes the
    intent visible in launch logs and the manifest.
    """
    return {"RUN_ID": identity.run_id, "WANDB_RUN_ID": identity.run_id, "WANDB_PROJECT": identity.wandb_project}
