# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify a Marin vLLM wheel before invoking its normal CLI.

The launcher already names one promoted wheel URL, so startup only checks what building that command
cannot settle: the PEP 610 record identifies the promoted wheel, vLLM's compiled extension comes from
inside that distribution, and the GPU this process was scheduled onto is one the wheel was compiled for.
The first two are a pair. A leaked ``PYTHONPATH`` entry holding a complete vLLM install would satisfy the
extension check on its own, because its metadata and its extension agree with each other.

The extension is discovered from the wheel's own ``RECORD`` rather than hardcoded: vLLM renames its
core extension across CUDA toolchains (``vllm._C`` on CUDA 12, ``vllm._C_stable_libtorch`` on CUDA 13),
and a hardcoded name breaks the check when it next moves.
"""

import dataclasses
import importlib
import importlib.metadata
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit, urlunsplit

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


def installed_wheel_url_matches(direct_url: dict, expected_wheel_url: str) -> bool:
    """Whether a PEP 610 record identifies ``expected_wheel_url``.

    The URL is the only artifact identity available at runtime: uv drops the ``#sha256=`` fragment of
    a remote direct reference and writes an empty ``archive_info``, so there is no installed digest
    to compare against. Do not add a digest check here.
    """
    installed_url = urlsplit(direct_url["url"])
    installed_url_without_fragment = urlunsplit(installed_url._replace(fragment=""))
    return unquote(installed_url_without_fragment) == unquote(expected_wheel_url)


def _core_extension_module(distribution: importlib.metadata.Distribution) -> str:
    """The dotted module name of vLLM's core compiled extension, read from the wheel's ``RECORD``.

    Selecting the ``_C`` family by shape rather than a literal name keeps the verifier working across
    the toolchain rename (``_C`` -> ``_C_stable_libtorch``) that broke a hardcoded check once.
    """
    stems = {
        file.name.split(".", 1)[0]
        for file in distribution.files or ()
        if file.parts[:-1] == ("vllm",)
        and file.suffix in (".so", ".pyd", ".py")
        and (file.name.split(".", 1)[0] == "_C" or file.name.split(".", 1)[0].startswith("_C_"))
    }
    if len(stems) != 1:
        raise RuntimeError(f"Expected exactly one vLLM core (_C) extension under vllm/, found {sorted(stems)}")
    return f"vllm.{stems.pop()}"


def main() -> None:
    expected = _WheelProvenance.from_json(sys.argv.pop(1))
    print(f"{_SELECTED_SENTINEL}{json.dumps(expected.record(), sort_keys=True)}", flush=True)

    # The installed version is not checked separately: promotion requires the release URL to carry the
    # declared version in its filename, so a distribution installed from that URL has that version.
    distribution = importlib.metadata.distribution("vllm")
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise RuntimeError("Installed vLLM does not record direct wheel provenance")
    direct_url = json.loads(direct_url_text)
    if not installed_wheel_url_matches(direct_url, expected.wheel_url):
        raise RuntimeError(f"Installed vLLM URL {direct_url['url']} does not match {expected.wheel_url}")

    torch = importlib.import_module("torch")
    major, minor = torch.cuda.get_device_capability()
    compute_capability = f"{major}.{minor}"
    if compute_capability not in expected.sm_targets:
        raise RuntimeError(
            f"GPU compute capability {compute_capability} is not supported by verified wheel "
            f"targets {expected.sm_targets}"
        )

    distribution_root = Path(distribution.locate_file("")).resolve()
    extension_module = _core_extension_module(distribution)
    vllm_extension = importlib.import_module(extension_module)
    extension_path = vllm_extension.__file__
    assert extension_path is not None
    resolved_extension_path = Path(extension_path).resolve()
    if not resolved_extension_path.is_relative_to(distribution_root):
        raise RuntimeError(f"{extension_module} loaded outside the verified distribution: {resolved_extension_path}")

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
