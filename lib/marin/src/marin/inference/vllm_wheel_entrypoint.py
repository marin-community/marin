# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify a Marin vLLM wheel before invoking its normal CLI."""

import dataclasses
import importlib
import importlib.metadata
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit, urlunsplit

_SELECTED_SENTINEL = "MARIN_VLLM_WHEEL_SELECTED="
_VERIFIED_SENTINEL = "MARIN_VLLM_WHEEL_VERIFIED="


@dataclass(frozen=True)
class _WheelProvenance:
    release_tag: str
    sm_targets: tuple[str, ...]
    source_commit: str
    version: str
    wheel_sha256: str
    wheel_url: str

    @classmethod
    def from_json(cls, value: str) -> "_WheelProvenance":
        payload = json.loads(value)
        return cls(
            release_tag=payload["release_tag"],
            sm_targets=tuple(payload["sm_targets"]),
            source_commit=payload["source_commit"],
            version=payload["version"],
            wheel_sha256=payload["wheel_sha256"],
            wheel_url=payload["wheel_url"],
        )

    def record(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def main() -> None:
    expected = _WheelProvenance.from_json(sys.argv.pop(1))
    print(f"{_SELECTED_SENTINEL}{json.dumps(expected.record(), sort_keys=True)}", flush=True)

    distribution = importlib.metadata.distribution("vllm")
    installed_version = distribution.version
    if installed_version != expected.version:
        raise RuntimeError(
            f"Installed vLLM version {installed_version} does not match verified wheel {expected.version}"
        )

    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError("Installed vLLM does not record direct wheel provenance")
    direct_url = json.loads(direct_url_text)
    installed_url = urlsplit(direct_url["url"])
    installed_url_without_fragment = urlunsplit(installed_url._replace(fragment=""))
    if unquote(installed_url_without_fragment) != unquote(expected.wheel_url):
        raise RuntimeError(f"Installed vLLM URL {direct_url['url']} does not match {expected.wheel_url}")

    installed_sha256 = set(parse_qs(installed_url.fragment).get("sha256", []))
    archive_sha256 = direct_url.get("archive_info", {}).get("hashes", {}).get("sha256")
    if archive_sha256 is not None:
        installed_sha256.add(archive_sha256)
    # uv enforces the requirement's SHA-256 fragment while installing, but does not
    # retain that fragment in direct_url.json. Reject contradictory metadata when present.
    if installed_sha256 and installed_sha256 != {expected.wheel_sha256}:
        raise RuntimeError(
            f"Installed vLLM SHA-256 values {sorted(installed_sha256)} do not match {expected.wheel_sha256}"
        )

    torch = importlib.import_module("torch")
    major, minor = torch.cuda.get_device_capability()
    compute_capability = f"{major}.{minor}"
    if compute_capability not in expected.sm_targets:
        raise RuntimeError(
            f"GPU compute capability {compute_capability} is not supported by verified wheel "
            f"targets {expected.sm_targets}"
        )

    vllm_extension = importlib.import_module("vllm._C")
    extension_path = vllm_extension.__file__
    assert extension_path is not None
    resolved_extension_path = Path(extension_path).resolve()
    distribution_root = Path(distribution.locate_file("")).resolve()
    if not resolved_extension_path.is_relative_to(distribution_root):
        raise RuntimeError(f"vllm._C loaded outside the verified distribution: {resolved_extension_path}")

    provenance = {
        **expected.record(),
        "compute_capability": compute_capability,
        "extension_path": str(resolved_extension_path),
    }
    print(f"{_VERIFIED_SENTINEL}{json.dumps(provenance, sort_keys=True)}", flush=True)
    vllm_cli = importlib.import_module("vllm.entrypoints.cli.main")
    vllm_cli.main()


if __name__ == "__main__":
    main()
