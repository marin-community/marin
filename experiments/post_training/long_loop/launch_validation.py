# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate concrete graph configs before allocating remote workers."""

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import yaml
from marin.rl.skyrl import skyrl_cpu_command
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

from experiments.post_training.long_loop.serving import serving_release


def validate_launch(document: str, directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "launch.yaml"
    source.write_text(document)
    compiled = directory / "compiled.yaml"
    command = skyrl_cpu_command(
        "python", "-m", "experiments.post_training.long_loop.validate_launch", str(source), str(compiled)
    )
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"):
        environment.pop(key, None)
    result = subprocess.run(command, env=environment, text=True, capture_output=True)
    (directory / "validator.log").write_text(result.stderr)
    if result.returncode:
        raise ValueError(f"Strict SkyRL composition failed before allocation:\n{result.stderr}")
    receipt = json.loads(result.stdout)
    receipt["command"] = command
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def validate_before_allocation(document: str) -> dict:
    with tempfile.TemporaryDirectory(prefix="long-loop-validation-") as directory:
        return validate_launch(document, Path(directory))


def validate_selected_context(document: str) -> dict:
    """Validate the actual export config on CPU before allocating any judge GPUs."""
    launch = yaml.safe_load(document)
    model = launch["inputs"]["model"]
    generator = launch["skyrl"]["generator"]
    overrides = generator.get("engine_init_kwargs", {}).get("hf_overrides", {})
    requested = launch["skyrl"]["context_budget"]["request_window_tokens"]
    fs, key = filesystem_for(prefix_join(model["uri"], "config.json"))
    content = fs.cat_file(key)
    with tempfile.TemporaryDirectory(prefix="long-loop-selected-context-") as temporary:
        root = Path(temporary)
        (root / "config.json").write_bytes(content)
        overrides_path = root / "overrides.json"
        overrides_path.write_text(json.dumps(overrides))
        receipt_path = root / "context.json"
        command = skyrl_cpu_command(
            "python",
            "-m",
            "experiments.post_training.long_loop.selected_context_probe",
            str(root),
            str(overrides_path),
            str(requested),
            str(receipt_path),
        )
        # Isolate GPU-wheel dependencies from the ordinary CPU launcher environment.
        command[command.index("--extra") + 1] = "vllm"
        environment = dict(os.environ)
        for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "VLLM_ALLOW_LONG_MAX_MODEL_LEN"):
            environment.pop(name, None)
        environment["UV_PROJECT_ENVIRONMENT"] = str(
            Path(tempfile.gettempdir()) / f"long-loop-context-{serving_release().source_commit}"
        )
        result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=1800)
        if result.returncode:
            raise ValueError(f"Selected checkpoint context failed before judge allocation:\n{result.stderr}")
        receipt = json.loads(receipt_path.read_text())
    if receipt["config_sha256"] != hashlib.sha256(content).hexdigest():
        raise ValueError("Selected checkpoint context receipt refers to different config bytes")
    if (
        receipt["requested_window"] != requested
        or receipt["overrides"] != overrides
        or receipt["vllm_version"] != serving_release().version
        or not receipt["accepted"]
        or receipt["max_model_len"] < requested
    ):
        raise ValueError("Selected checkpoint context receipt does not match the launch")
    receipt["model_uri"] = model["uri"]
    receipt["model_identity"] = model["identity"]
    return receipt
