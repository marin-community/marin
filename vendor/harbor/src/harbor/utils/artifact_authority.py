"""Validate launcher-provided authority over a local artifact tree."""

import json
import logging
import os
from pathlib import Path

from upath import UPath

from harbor.models.trial.config import ArtifactAuthorityConfig

logger = logging.getLogger(__name__)

_LOCAL_PROTOCOLS = ("", "file", "local")


class ArtifactAuthorityError(RuntimeError):
    """The current process does not hold the expected artifact authority."""


def validate_artifact_authority(
    root: Path | UPath,
    config: ArtifactAuthorityConfig | None,
    *,
    operation: str,
) -> None:
    """Validate a pre-created authority record before artifact-tree access.

    The launcher must put the record at the protected root or one of its
    lexical ancestors. Paths are made absolute without resolving symlinks so
    the mount identity supplied by the launcher remains intact.
    """
    if config is None:
        return

    protocol = root.protocol if isinstance(root, UPath) else ""
    if isinstance(protocol, tuple):
        protocol = protocol[0] if protocol else ""
    if protocol not in _LOCAL_PROTOCOLS:
        return _authority_violation(
            config,
            operation,
            root,
            "authority validation supports only local artifact roots",
        )

    root_path = Path(os.path.abspath(str(root)))
    record_path = config.record_path
    if not record_path.is_absolute():
        return _authority_violation(
            config,
            operation,
            root_path,
            "record_path must be absolute",
        )
    record_path = Path(os.path.abspath(record_path))
    if record_path.parent != root_path and record_path.parent not in root_path.parents:
        return _authority_violation(
            config,
            operation,
            root_path,
            "record_path must be at the protected root or a lexical ancestor",
        )
    if record_path.is_symlink():
        return _authority_violation(
            config,
            operation,
            root_path,
            "authority record must not be a symlink",
        )

    try:
        record = json.loads(record_path.read_text())
    except FileNotFoundError:
        return _authority_violation(
            config,
            operation,
            root_path,
            "authority record is missing",
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return _authority_violation(
            config,
            operation,
            root_path,
            f"authority record is unreadable: {exc}",
        )

    if not isinstance(record, dict) or not isinstance(record.get("authority_id"), str):
        return _authority_violation(
            config,
            operation,
            root_path,
            "authority record must contain a string authority_id",
        )
    if record["authority_id"] != config.authority_id:
        return _authority_violation(
            config,
            operation,
            root_path,
            "authority record does not match the expected authority_id",
        )
    return


def _authority_violation(
    config: ArtifactAuthorityConfig,
    operation: str,
    root: Path | UPath,
    reason: str,
) -> None:
    message = (
        f"Artifact authority validation failed before {operation}: {reason}. "
        f"root={root} record_path={config.record_path}"
    )
    if config.mode == "warn":
        logger.warning(message)
        return
    raise ArtifactAuthorityError(message)
