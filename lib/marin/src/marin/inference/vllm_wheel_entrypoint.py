# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify a Marin vLLM wheel before invoking its normal CLI."""

import importlib
import importlib.metadata
import json
import os
from pathlib import Path
from urllib.parse import unquote

_PROVENANCE_ENV_VAR = "MARIN_VLLM_WHEEL_PROVENANCE"
_SELECTED_SENTINEL = "MARIN_VLLM_WHEEL_SELECTED="
_VERIFIED_SENTINEL = "MARIN_VLLM_WHEEL_VERIFIED="


def main() -> None:
    expected = json.loads(os.environ[_PROVENANCE_ENV_VAR])
    print(f"{_SELECTED_SENTINEL}{json.dumps(expected, sort_keys=True)}", flush=True)

    distribution = importlib.metadata.distribution("vllm")
    installed_version = distribution.version
    if installed_version != expected["version"]:
        raise RuntimeError(
            f"Installed vLLM version {installed_version} does not match verified wheel {expected['version']}"
        )

    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError("Installed vLLM does not record direct wheel provenance")
    direct_url = json.loads(direct_url_text)
    if unquote(direct_url["url"]) != unquote(expected["wheel_url"]):
        raise RuntimeError(f"Installed vLLM URL {direct_url['url']} does not match {expected['wheel_url']}")
    installed_sha256 = direct_url["archive_info"]["hashes"]["sha256"]
    if installed_sha256 != expected["wheel_sha256"]:
        raise RuntimeError(
            f"Installed vLLM SHA-256 {installed_sha256} does not match {expected['wheel_sha256']}"
        )

    torch = importlib.import_module("torch")
    major, minor = torch.cuda.get_device_capability()
    compute_capability = f"{major}.{minor}"
    if compute_capability not in expected["sm_targets"]:
        raise RuntimeError(
            f"GPU compute capability {compute_capability} is not supported by verified wheel "
            f"targets {expected['sm_targets']}"
        )

    vllm_extension = importlib.import_module("vllm._C")
    extension_path = vllm_extension.__file__
    assert extension_path is not None
    resolved_extension_path = Path(extension_path).resolve()
    distribution_root = Path(distribution.locate_file("")).resolve()
    if not resolved_extension_path.is_relative_to(distribution_root):
        raise RuntimeError(f"vllm._C loaded outside the verified distribution: {resolved_extension_path}")

    provenance = {
        **expected,
        "compute_capability": compute_capability,
        "extension_path": str(resolved_extension_path),
    }
    print(f"{_VERIFIED_SENTINEL}{json.dumps(provenance, sort_keys=True)}", flush=True)
    vllm_cli = importlib.import_module("vllm.entrypoints.cli.main")
    vllm_cli.main()


if __name__ == "__main__":
    main()
