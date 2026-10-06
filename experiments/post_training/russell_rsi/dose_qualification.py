# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify the completed four-update source for the separate dose experiment."""

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import yaml
from levanter.compat import MODEL_MANIFEST_FILENAME
from marin.execution.artifact import artifact_record_identity
from marin.external_dependencies import MARIN_SKYRL
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.russell_rsi.repair_tasks import pinned_bytes
from experiments.post_training.russell_rsi.sources import compact_json_sha256

QUALIFIED_RUNTIME_COMMIT = "f124f258383763e10766cff5e5af4c433cad4b1a"

# These are the pinned task_runtime._write_final_config and training_driver.run transforms.
# The exact comparison includes every setting in the terminal and resolved configs.
_COMPILE_SOURCE = """
import json, os, sys, tempfile
from pathlib import Path
from omegaconf import OmegaConf
from cloud.iris.launch_config import load_launch_config
from cloud.iris.rl_config_translation import TaskLocalSkyRLValues, apply_task_local_values
from marinskyrl.hf_model import immutable_model_cache_key
from marinskyrl.resource_locator import is_cloud_uri
from marinskyrl.task_sources import DirectoryDataSource, data_source, data_source_dict
root = load_launch_config(Path(sys.argv[1]))
for key in ("train_data", "validation_data"):
    sources = [data_source(item) for item in root.inputs[key]]
    if not all(isinstance(item, DirectoryDataSource) for item in sources):
        raise ValueError("Dose qualification supports only the sealed directory task sources")
    root.inputs[key] = [data_source_dict(item) for item in sources]
terminal = OmegaConf.to_container(root, resolve=True)
model = root.inputs.model
if not is_cloud_uri(str(model.uri)) or str(model.chat_template or ""):
    raise ValueError("Dose source requires the unchanged artifact model without a template rewrite")
if root.skyrl.generator.get("speculative_decoding") is not None:
    raise ValueError("Dose source does not allow speculative decoding")
model_path = os.path.join(
    "/tmp", "marinskyrl", "model_metadata", immutable_model_cache_key(str(model.uri), str(model.identity))
)
tokenizer_path = os.path.join(
    "/tmp", "marinskyrl", "model_metadata",
    immutable_model_cache_key(str(model.tokenizer_uri), str(model.tokenizer_revision))
)
original_model_path = str(root.skyrl.trainer.policy.model.path)
root.skyrl = apply_task_local_values(root.skyrl, TaskLocalSkyRLValues((), (), (), policy_model_path=model_path))
for key, value in {
    "trainer.policy.model.source_uri": str(model.uri),
    "trainer.policy.model.source_identity": str(model.identity),
    "generator.engine_init_kwargs.served_model_name": original_model_path.rstrip("/").rsplit("/", 1)[-1],
}.items():
    OmegaConf.update(root.skyrl, key, value, force_add=True)
for role in ("policy", "ref"):
    OmegaConf.update(root.skyrl, f"trainer.{role}.model.tokenizer_path", tokenizer_path, force_add=True)
    OmegaConf.update(root.skyrl, f"trainer.{role}.model.tokenizer_revision", None, force_add=True)
root.skyrl = apply_task_local_values(root.skyrl, TaskLocalSkyRLValues(
    tuple(data_source(item).resolved_path() for item in root.inputs.train_data),
    tuple(data_source(item).resolved_path() for item in root.inputs.validation_data), (),
))
resolved = {"config": OmegaConf.to_container(root, resolve=True),
            "train_data_sources": terminal["inputs"]["train_data"],
            "val_data_sources": terminal["inputs"]["validation_data"]}
print(json.dumps({"terminal": terminal, "resolved": resolved}, sort_keys=True))
"""


def compiled_source_configs(requested_yaml: str) -> dict:
    """Compile source settings on the CPU without submission, staging, or model access."""
    if MARIN_SKYRL.commit != QUALIFIED_RUNTIME_COMMIT:
        raise ValueError("Dose source task transforms require the declared runtime commit")
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "requested.yaml"
        source.write_text(requested_yaml)
        result = subprocess.check_output(
            [
                "uv",
                "run",
                "--quiet",
                "--isolated",
                "--no-project",
                "--prerelease=allow",
                "--python",
                "3.12",
                "--with",
                MARIN_SKYRL.requirement(),
                "python",
                "-c",
                _COMPILE_SOURCE,
                str(source),
            ],
            text=True,
        )
    return json.loads(result)


@dataclass(frozen=True)
class QualifiedDoseSource:
    requested: dict
    rl_identity: str
    reload_identity: str
    export_uri: str
    source_replay: dict
    evidence_sha256: str


def qualified_dose_source(evidence: dict) -> QualifiedDoseSource:
    """Read hash-pinned completed source records and require exact execution linkage."""
    files = evidence["files"]

    def read(role: str, expected_uri: str | None = None) -> bytes:
        pin = files[role]
        if expected_uri is not None and pin["uri"] != expected_uri:
            raise ValueError(f"Dose qualification {role} points at a different source file")
        return pinned_bytes(pin["uri"], pin["sha256"])

    def artifact(role: str) -> dict:
        value = json.loads(read(role))
        if files[role]["uri"] != prefix_join(value["output_path"], ".artifact.json"):
            raise ValueError(f"Dose qualification {role} points at a different source file")
        if read(f"{role}_status", prefix_join(value["output_path"], ".executor_status")).decode().strip() != "SUCCESS":
            raise ValueError(f"Dose source {role} is incomplete or failed")
        if not value["config"] or not value["fingerprint"]:
            raise ValueError(f"Dose source {role} lacks its original recipe record")
        return value

    rl, optimizer, reload = (artifact(role) for role in ("rl", "optimizer", "reload"))
    rl_dependency = f"{rl['name']}@{rl['version']}"
    optimizer_dependency = f"{optimizer['name']}@{optimizer['version']}"
    rl_identity = f"{rl_dependency}:{rl['fingerprint']}"
    reload_identity = artifact_record_identity(reload)
    requested = yaml.safe_load(rl["config"]["launch_config_yaml"])
    terminal = json.loads(read("terminal", requested["artifacts"]["terminal_manifest_uri"]))
    resolved = json.loads(read("resolved", requested["artifacts"]["resolved_config_uri"]))
    compiled = compiled_source_configs(rl["config"]["launch_config_yaml"])
    if terminal["config"] != compiled["terminal"] or resolved != compiled["resolved"]:
        raise ValueError("Dose source executed settings differ from its pinned runtime compilation")
    if len(requested["inputs"]["train_data"]) != 1:
        raise ValueError("Dose qualification requires the one frozen replay input")
    source_replay = json.loads(
        read("source_replay", prefix_join(requested["inputs"]["train_data"][0]["uri"], "replay-plan.json"))
    )
    result = rl["result"]
    finished = terminal["result"]
    exported = finished["model"]
    export_uri = result["hf_model_uri"]
    model = requested["inputs"]["model"]
    if (
        finished["state"] != "succeeded"
        or finished["iris_job_state"] != "succeeded"
        or finished["failure"] is not None
        or finished["launcher_commit"] != MARIN_SKYRL.commit
        or finished["runtime_profile"] != requested["runtime"]["profile"]
        or finished["run_id"] != requested["run"]["id"]
        or finished["attempt_id"] != requested["run"]["attempt_id"]
        or result["global_step"] != 4
        or exported["global_step"] != 4
        or result["iris_job_id"] != finished["iris_job_id"]
        or exported["policy_export_uri"] != export_uri
        or export_uri != prefix_join(requested["artifacts"]["export_root"], "global_step_4/policy")
        or result["checkpoint_root"] != requested["artifacts"]["checkpoint_root"]
        or result["terminal_manifest_uri"] != requested["artifacts"]["terminal_manifest_uri"]
        or exported["checkpoint_root"] != result["checkpoint_root"]
        or exported["terminal_manifest_uri"] != result["terminal_manifest_uri"]
        or any(
            result[key] != model[key] or exported[key] != model[key] for key in ("tokenizer_uri", "tokenizer_revision")
        )
    ):
        raise ValueError("Dose source lacks a successful exact four-update published export")
    if (
        optimizer["deps"] != [rl_dependency]
        or optimizer["dep_paths"] != []
        or optimizer["config"] != {"expected_updates": 4, "actual_updates": 4, "export_uri": export_uri}
        or reload["deps"] != [rl_dependency, optimizer_dependency]
        or reload["dep_paths"] != []
        or reload["config"]["model"]["location"] != export_uri
        or reload["config"]["model"]["identity"] != rl_identity
        or reload["config"]["evals"] != "mmlu-smoke"
        or reload["config"]["limit"] != 1
        or not reload["result"]["results_paths"]
    ):
        raise ValueError("Dose source optimizer, reload, or export dependency linkage differs")
    manifest = json.loads(read("export_manifest", prefix_join(export_uri, MODEL_MANIFEST_FILENAME)))
    index_bytes = read("export_index", prefix_join(export_uri, "model.safetensors.index.json"))
    index = json.loads(index_bytes)
    published = {item["path"]: item for item in manifest["files"]}
    if (
        not index["weight_map"]
        or not set(index["weight_map"].values()) <= published.keys()
        or published["model.safetensors.index.json"]["sha256"] != hashlib.sha256(index_bytes).hexdigest()
    ):
        raise ValueError("Dose source export index differs from its publication manifest")
    return QualifiedDoseSource(
        requested, rl_identity, reload_identity, export_uri, source_replay, compact_json_sha256(evidence)
    )
