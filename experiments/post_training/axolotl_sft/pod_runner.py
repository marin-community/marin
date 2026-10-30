# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pod-side Axolotl SFT runner, executed inside the pinned Axolotl task image.

The image carries torch, CUDA and an upstream Axolotl install; it does not carry
Marin's environment, so this script uses only the standard library, PyYAML, psutil (an
accelerate dependency) and the image's ``aws`` CLI. It reads one JSON payload
(``AXOLOTL_PAYLOAD``) built by
:mod:`experiments.post_training.axolotl_sft.launcher` and:

1. clones the ``marin-community/axolotl`` fork at the pinned revision and installs it
   over the image's Axolotl with the fork's own pins (transformers 5.12.1 and the
   matching ``kernels``), keeping the image's torch; it then checks the transformers pin;
2. stages the CL12 data file and, for a chained stage, the previous stage's export
   from object storage onto local disk;
3. trains through ``torchrun -m axolotl.cli.train`` across the job's nodes (one Iris task
   per node; node 0 publishes its address through object storage), with node 0 syncing
   the output directory every few minutes so a lost gang keeps its latest checkpoint;
4. on node 0, checks the saved tokenizer embeds the configured chat template byte for byte,
   records the realized token counts, and uploads the output plus a run manifest.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import psutil
import yaml

logger = logging.getLogger("axolotl_pod_runner")

PAYLOAD_ENV = "AXOLOTL_PAYLOAD"
# Set to run only Axolotl's config validation and dataset preprocessing (a CPU check).
PREPROCESS_ONLY_ENV = "AXOLOTL_PREPROCESS_ONLY"
WORK_ROOT = Path("/tmp/axolotl-run")
SYNC_INTERVAL = 600
FORK_REPO = "https://github.com/marin-community/axolotl"
IMAGE_PYTHON = "/workspace/axolotl-venv/bin/python"
# The fork's pinned Cut Cross Entropy build (scripts/cutcrossentropy_install.py): it computes the
# loss without materializing [tokens x vocab] logits, which alone exceed an H100 at 32k tokens.
CUT_CROSS_ENTROPY = (
    "cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@5f0c7a7"
)
RUN_MANIFEST = "axolotl_run_manifest.json"
RENDEZVOUS_PORT = 29500
RENDEZVOUS_TIMEOUT = 1800
# The plugins this directory ships (the fp32-master plugin) import from here.
PLUGIN_DIR = Path(__file__).resolve().parent


def _run(command: list[str], **kwargs) -> None:
    logger.info("+ %s", " ".join(command))
    subprocess.run(command, check=True, **kwargs)


def _aws_environment() -> dict[str, str]:
    # CoreWeave object storage requires virtual-hosted addressing.
    config = WORK_ROOT / "aws-config"
    config.write_text("[default]\ns3 =\n    addressing_style = virtual\n")
    return os.environ | {"AWS_CONFIG_FILE": str(config), "AWS_MAX_ATTEMPTS": "20", "AWS_RETRY_MODE": "adaptive"}


def _s3(arguments: list[str]) -> None:
    _run(["aws", "s3", *arguments, "--only-show-errors"], env=_aws_environment())


def _install_fork(revision: str, transformers_pin: str) -> None:
    fork = WORK_ROOT / "axolotl-fork"
    _run(["git", "clone", "--quiet", FORK_REPO, str(fork)])
    _run(["git", "-C", str(fork), "checkout", "--quiet", revision])
    # Re-pinning transformers alone breaks against the image's newer `kernels`; the fork's
    # own dependency set is consistent.
    _run(["uv", "pip", "install", "--python", IMAGE_PYTHON, "-e", str(fork), CUT_CROSS_ENTROPY])
    installed = subprocess.run(
        [IMAGE_PYTHON, "-c", "import transformers; print(transformers.__version__)"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if installed != transformers_pin:
        raise RuntimeError(f"the fork installed transformers {installed}, not the pinned {transformers_pin}")


def _stage_inputs(payload: dict) -> dict:
    """Download the data and any init checkpoint; return the config with local paths."""
    config = dict(payload["config"])
    data_dir = WORK_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    local_data = data_dir / payload["data_uri"].rsplit("/", 1)[-1]
    _s3(["cp", payload["data_uri"], str(local_data)])
    config["datasets"] = [dict(config["datasets"][0], path=str(local_data))]
    if payload["init_uri"] is not None:
        init_dir = WORK_ROOT / "init"
        _s3(["sync", payload["init_uri"].rstrip("/"), str(init_dir), "--exclude", "checkpoint-*"])
        config["base_model"] = str(init_dir)
    config["output_dir"] = str(WORK_ROOT / "output")
    config["dataset_prepared_path"] = str(WORK_ROOT / "prepared")
    return config


class _PeriodicSync:
    """Sync the output directory to object storage every ``SYNC_INTERVAL`` seconds."""

    def __init__(self, source: Path, target: str):
        self._source, self._target = source, target
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.wait(SYNC_INTERVAL):
            if self._source.exists():
                try:
                    _s3(["sync", str(self._source), self._target])
                except subprocess.CalledProcessError:
                    # The next tick retries; the final sync after training is checked.
                    logger.exception("periodic sync failed")

    def __enter__(self) -> _PeriodicSync:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


def _task_position() -> tuple[str, int, int]:
    """(job id, node rank, node count) from Iris: IRIS_TASK_ID is ``<job id>/<index>:<attempt>``."""
    task_id = os.environ["IRIS_TASK_ID"]
    job_id, _, index_and_attempt = task_id.rpartition("/")
    return job_id, int(index_and_attempt.split(":")[0]), int(os.environ["IRIS_NUM_TASKS"])


def _master_address(output_uri: str, job_id: str, node_rank: int) -> str:
    """Node 0 publishes its host under the output URI, keyed by this job; the others wait for it."""
    marker = f"{output_uri}/_rendezvous/{job_id.strip('/').replace('/', '--')}"
    if node_rank == 0:
        host = os.environ["IRIS_ADVERTISE_HOST"]
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as handle:
            handle.write(host)
        _s3(["cp", handle.name, marker])
        return host
    deadline = time.monotonic() + RENDEZVOUS_TIMEOUT
    while time.monotonic() < deadline:
        result = subprocess.run(["aws", "s3", "cp", marker, "-"], capture_output=True, text=True, env=_aws_environment())
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
        time.sleep(10)
    raise TimeoutError(f"node 0 never published {marker}")


def _interface_for(address: str) -> str:
    """The network interface carrying ``address`` (the image has no ``ip`` command)."""
    for name, addresses in psutil.net_if_addrs().items():
        if any(entry.address == address for entry in addresses):
            return name
    raise ValueError(f"no interface carries {address}")


def _check_template(output_dir: Path, expected: str) -> None:
    """The fork's template_integrity contract, re-checked outside the save path."""
    with open(output_dir / "tokenizer_config.json") as handle:
        embedded = json.load(handle).get("chat_template")
    if embedded != expected:
        raise ValueError(f"{output_dir}/tokenizer_config.json does not embed the configured chat template")
    # Multimodal shells (Qwen3.5-MoE) also pre-save a processor, which writes chat_template.jinja;
    # transformers loads that file ahead of tokenizer_config.json, so it must match too.
    jinja = output_dir / "chat_template.jinja"
    if jinja.exists() and jinja.read_text() != expected:
        raise ValueError(f"{jinja} differs from the configured chat template")


def _realized_tokens(output_dir: Path) -> dict:
    """Token totals from the last ``trainer_state.json`` (Axolotl logs them per step)."""
    states = sorted(glob.glob(str(output_dir / "**" / "trainer_state.json"), recursive=True))
    if not states:
        return {"source": None, "counts": {}, "global_step": None}
    with open(states[-1]) as handle:
        state = json.load(handle)
    counts: dict[str, int] = {}
    for entry in state.get("log_history", []):
        for key, value in entry.items():
            if "token" in key and "per_second" not in key and isinstance(value, (int, float)):
                counts[key] = max(counts.get(key, 0), int(value))
    return {"source": states[-1], "counts": counts, "global_step": state.get("global_step")}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", stream=sys.stdout)
    payload = json.loads(os.environ[PAYLOAD_ENV])
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    _install_fork(payload["fork_revision"], payload["transformers_pin"])
    config = _stage_inputs(payload)
    config_path = WORK_ROOT / "axolotl_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    output_dir = Path(config["output_dir"])
    output_uri = payload["output_uri"].rstrip("/")

    environment = os.environ | {"PYTHONPATH": os.pathsep.join((str(PLUGIN_DIR), str(PLUGIN_DIR.parents[2])))}
    if os.environ.get(PREPROCESS_ONLY_ENV):
        # Validate the config and tokenize the data on CPU, before any GPU job.
        _run([IMAGE_PYTHON, "-m", "axolotl.cli.preprocess", str(config_path)], env=environment, cwd=WORK_ROOT)
        logger.info("preprocess succeeded")
        return
    job_id, node_rank, nodes = _task_position()
    if nodes != payload["nodes"]:
        raise ValueError(f"the job has {nodes} tasks but the payload trains on {payload['nodes']} nodes")
    # Gloo (Axolotl's CPU process group for multipack) binds the address the pod hostname
    # resolves to, which is loopback here; peers on the other node then cannot connect.
    environment["GLOO_SOCKET_IFNAME"] = _interface_for(os.environ["IRIS_ADVERTISE_HOST"])
    command = [
        "torchrun",
        f"--nnodes={nodes}",
        f"--node-rank={node_rank}",
        f"--nproc-per-node={payload['gpus']}",
        f"--master-addr={_master_address(output_uri, job_id, node_rank)}",
        f"--master-port={RENDEZVOUS_PORT}",
        "-m",
        "axolotl.cli.train",
        str(config_path),
    ]
    # Gloo (multipack's CPU collectives) otherwise binds the hostname's loopback address.
    environment["GLOO_SOCKET_IFNAME"] = _interface_for(os.environ["IRIS_ADVERTISE_HOST"])
    if node_rank != 0:
        _run(command, env=environment, cwd=WORK_ROOT)
        return
    with _PeriodicSync(output_dir, output_uri):
        _run(command, env=environment, cwd=WORK_ROOT)

    _check_template(output_dir, config["chat_template_jinja"])
    manifest = dict(payload["manifest"], realized_tokens=_realized_tokens(output_dir))
    (output_dir / RUN_MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (output_dir / "axolotl_config.yaml").write_text(config_path.read_text())
    _s3(["sync", str(output_dir), output_uri])
    logger.info("uploaded %s to %s", output_dir, output_uri)


if __name__ == "__main__":
    main()
