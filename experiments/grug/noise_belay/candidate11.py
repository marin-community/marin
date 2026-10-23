# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load and audit the recorded Belay candidate 11 configuration.

Run this script from the historical checkout created by ``prepare.py``. This
performs shape inference only; it does not allocate model weights or train.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import draccus
import equinox as eqx
import jax
from jax.sharding import AbstractMesh, AxisType, use_abstract_mesh

from experiments.grug.fast_track import model as reference_model

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "candidate11-reference.json"


def load_probe_manifest() -> dict:
    """Verify every staged experiment file before running a measurement."""
    manifest = json.loads((ROOT / "gpu-probe-manifest.json").read_text())
    for name, digest in manifest["experiment_files"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Experiment source mismatch: {name}")
    return manifest


def load_model_config(*, source_sha256: str | None = None) -> reference_model.GrugModelConfig:
    """Decode recorded fields after checking checkout revision or shipped source hash.

    Iris bundles omit Git metadata. GPU probes pass the source digest recorded
    while staging the pinned checkout, including its explicit runtime patch.
    """
    record = json.loads(REFERENCE.read_text())
    source = Path(reference_model.__file__).resolve()
    if source_sha256 is not None:
        actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if actual_hash != source_sha256:
            raise ValueError(f"Model source hash mismatch: {actual_hash}")
    else:
        revision = subprocess.check_output(["git", "-C", str(source.parents[3]), "rev-parse", "HEAD"], text=True).strip()
        if revision != record["source_revision"]:
            raise ValueError(f"Expected source {record['source_revision']}, got {revision}")
    return draccus.decode(reference_model.GrugModelConfig, record["config"]["model"])


def parameter_audit() -> dict:
    """Infer every parameter shape and compare the total to the recorded run."""
    record = json.loads(REFERENCE.read_text())
    config = load_model_config()
    mesh = AbstractMesh(
        axis_sizes=(1, 1, 8, 1),
        axis_names=("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    key = jax.random.PRNGKey(record["config"]["trainer"]["trainer"]["seed"])
    with use_abstract_mesh(mesh):
        shape = eqx.filter_eval_shape(reference_model.Transformer.init, config, key=key)
    parameters = [
        {
            "path": jax.tree_util.keystr(path),
            "shape": list(leaf.shape),
            "size": leaf.size,
        }
        for path, leaf in jax.tree_util.tree_flatten_with_path(shape)[0]
        if isinstance(leaf, jax.ShapeDtypeStruct)
    ]
    count = sum(entry["size"] for entry in parameters)
    recorded_count = int(record["reference_summary"]["parameter_count"])
    if count != recorded_count:
        raise ValueError(f"Parameter mismatch: constructed {count}, recorded {recorded_count}")
    return {
        "source_revision": record["source_revision"],
        "reference_run": record["url"],
        "reference_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        "parameter_count": count,
        "layer_stack_indices": shape.stack_layer_indices(),
        "parameters": parameters,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = parameter_audit()
    args.output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({k: v for k, v in audit.items() if k != "parameters"}))


if __name__ == "__main__":
    main()
