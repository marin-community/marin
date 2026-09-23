# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The pinned quality scorer, and the checks that bind its bytes to an output path.

A quality leaf's directory name carries the identity of the scorer that wrote it, so
a step reads the digests here before it writes anything: a model or a calibration
file that digests to something else must not land under a path that claims this
pin. The model artifacts and the calibration file are digested separately because
they change independently. A refit of the cutpoints moves the bucketed dataset
without moving the raw scores it is computed from.
"""

import hashlib
import json
from dataclasses import dataclass

from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.storage_path import StoragePath, prefix_join

CALIBRATION_FILE = "calib_bme.json"
# The deployed ``.eqx`` is 158 MB, so the digest streams each file.
DIGEST_CHUNK_BYTES = 8 << 20


@dataclass(frozen=True)
class QualityPin:
    """A quality scorer's identity: where its bytes live and what they digest to."""

    name: str
    """Scorer tag, folded into the step hashes."""

    model_path: str
    """Model directory relative to ``MARIN_PREFIX``. It holds one ``<stem>.eqx`` with
    its ``<stem>_remap.json`` and ``<stem>_meta.json``, plus :data:`CALIBRATION_FILE`."""

    model_sha256: str
    """:func:`model_sha256` over the three scorer artifacts."""

    calibration_sha256: str
    """:func:`calibration_sha256` over the calibration file."""

    tokenizer: str
    """HuggingFace name of the corpus tokenizer whose ids the scorer reads."""


def quality_model_dir(pin: QualityPin) -> str:
    """Resolve the pin's model directory against the active ``MARIN_PREFIX``."""
    return prefix_join(marin_prefix(), pin.model_path)


def artifact_names(stem: str) -> tuple[str, str, str]:
    """The (.eqx, remap.json, meta.json) artifact filenames for a model stem."""
    return f"{stem}.eqx", f"{stem}_remap.json", f"{stem}_meta.json"


def model_stem(model_dir: str) -> str:
    """Return the sole ``.eqx`` artifact stem under ``model_dir``."""
    found = sorted(path.name for path in (StoragePath(model_dir) / "*.eqx").glob())
    if not found:
        raise ValueError(f"no .eqx artifact under {model_dir}")
    if len(found) > 1:
        raise ValueError(f"{model_dir} holds several models ({', '.join(found)}); it must hold one")
    return found[0][: -len(".eqx")]


def _file_digest(path: StoragePath) -> bytes:
    content = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(DIGEST_CHUNK_BYTES):
            content.update(chunk)
    return content.digest()


def _named_digest(paths: dict[str, StoragePath]) -> str:
    """Fold ``name + NUL + sha256(content)`` in name order into one digest.

    Names are inside the hash, so a renamed artifact is a different model, and only
    basenames are used, so the directory can be copied without changing it.
    """
    digest = hashlib.sha256()
    for name in sorted(paths):
        digest.update(name.encode() + b"\0" + _file_digest(paths[name]))
    return digest.hexdigest()


def model_sha256(model_dir: str) -> str:
    """Digest the scorer artifacts under ``model_dir``: the ``.eqx``, remap and meta files."""
    root = StoragePath(model_dir)
    return _named_digest({name: root / name for name in artifact_names(model_stem(model_dir))})


def calibration_sha256(model_dir: str) -> str:
    """Digest the calibration file under ``model_dir``."""
    return _named_digest({CALIBRATION_FILE: StoragePath(model_dir) / CALIBRATION_FILE})


def require_pinned_model(pin: QualityPin, model_dir: str) -> str:
    """Return ``model_dir``'s model digest, refusing bytes that are not ``pin``'s."""
    digest = model_sha256(model_dir)
    if digest != pin.model_sha256:
        raise ValueError(
            f"{model_dir} digests to {digest}, but {pin.name} pins {pin.model_sha256}; "
            f"its scores would be written to a path that claims {pin.name}"
        )
    return digest


def require_pinned_calibration(pin: QualityPin, model_dir: str) -> str:
    """Return ``model_dir``'s calibration digest, refusing a file that is not ``pin``'s."""
    digest = calibration_sha256(model_dir)
    if digest != pin.calibration_sha256:
        raise ValueError(
            f"{model_dir}/{CALIBRATION_FILE} digests to {digest}, but {pin.name} pins "
            f"{pin.calibration_sha256}; the bucketed path claims a different calibration"
        )
    return digest


def load_calibration(model_dir: str) -> dict:
    """Read the calibration knots: ``{xk, yk}`` or the per-type ``{default, types}`` form."""
    with (StoragePath(model_dir) / CALIBRATION_FILE).open("r") as fh:
        return json.loads(fh.read())
