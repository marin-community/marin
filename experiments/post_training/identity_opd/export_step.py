# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Re-run one run's HF export from its saved native checkpoint, as a separate Iris job.

The export job is derived from the run's resolved launch document (``resolved-launch.yaml`` under the run's output
root) and the checkpoint's ``hf_export_request.json``, through ``cloud.iris.export_hf_checkpoint``. It runs on the
saved policy geometry, in the production band with no retries, and is bounded by ``EXPORT_TIMEOUT_SECONDS``.

Run from the SkyRL environment, at the worktree root; the worktree is the code the job runs::

    python -m experiments.post_training.identity_opd.export_step \\
        --run-root s3://marin-rno2a/marin/users/karan/checkpoints/snowball-identity-opd-cd16/dry/2026.10.26.4 \\
        --checkpoint s3://marin-rno2a/tmp/ttl=14d/skyrl/.../checkpoints/global_step_2 --dry-run
"""

import json
import sys
import tempfile
from pathlib import Path

import click
import yaml
from cloud.iris import export_hf_checkpoint
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.identity_opd.launch import CLUSTER, IRIS_HUB_CLUSTER_CONFIG

# 40 minutes on the 4 x 8 H100 policy geometry: 21.3 H100-hours.
EXPORT_TIMEOUT_SECONDS = 2400


def training_launch_doc(resolved: dict, destination: Path) -> Path:
    """Write the composed training launch document held in a resolved-launch record."""
    destination.write_text(yaml.safe_dump(resolved["config"], sort_keys=False))
    return destination


def export_args(launch_doc: Path, checkpoint: str, *, dry_run: bool) -> list[str]:
    """Arguments for ``cloud.iris.export_hf_checkpoint`` from the training launch document."""
    allocation = yaml.safe_load(launch_doc.read_text())["iris"]["allocation"]
    args = [
        "--request",
        checkpoint,
        "--launch-config",
        str(launch_doc),
        "--timeout",
        str(EXPORT_TIMEOUT_SECONDS),
        "--cluster",
        CLUSTER,
        "--cluster-config",
        f"lib/iris/config/{CLUSTER}.yaml",
        "--target-cluster",
        CLUSTER,
        "--parent-cluster-config",
        IRIS_HUB_CLUSTER_CONFIG,
        "--gpu-variant",
        str(allocation["gpu_variant"]),
        "--allocation-gpus-per-node",
        str(allocation["gpus_per_node"]),
        "--cpu",
        str(allocation["cpu"]),
        "--memory",
        str(allocation["memory"]),
        "--disk",
        str(allocation["disk"]),
        "--priority",
        "production",
    ]
    return [*args, "--dry-run"] if dry_run else args


@click.command(help=__doc__)
@click.option("--run-root", required=True, help="the run's output root, holding resolved-launch.yaml")
@click.option("--checkpoint", required=True, help="the global_step_N directory holding hf_export_request.json")
@click.option("--dry-run", is_flag=True, help="derive the export job and print its command without submitting")
def main(run_root: str, checkpoint: str, dry_run: bool) -> None:
    # The export request is read and written through process-wide fsspec, which off-cluster is not CoreWeave.
    configure_coreweave_s3()
    fs, path = filesystem_for(f"{run_root.rstrip('/')}/resolved-launch.yaml")
    launch_doc = training_launch_doc(json.loads(fs.cat(path)), Path(tempfile.mkdtemp()) / "training-launch.yaml")
    sys.argv = ["export_hf_checkpoint", *export_args(launch_doc, checkpoint, dry_run=dry_run)]
    export_hf_checkpoint.main()


if __name__ == "__main__":
    main()
