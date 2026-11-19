# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lossless transport of text containing binary data through Postgres JSONB and TEXT."""

import base64
import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

CATALOG_TEXT_CODEC = "atlas-text-v1"
PREFIX = "@" + CATALOG_TEXT_CODEC + ":"
BINARY_TEXT_MARKER = "Binary content withheld from catalog text."
NON_TEXT = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ud800-\udfff]")


class CatalogTextKind(StrEnum):
    UTF8 = "utf8"
    BYTES = "bytes"


HEADER = re.compile(
    "^" + re.escape(PREFIX) + "(" + "|".join(kind.value for kind in CatalogTextKind) + r"):([0-9a-f]{64}):"
)


def needs_encoding(value: str | bytes) -> bool:
    return isinstance(value, bytes) or bool(NON_TEXT.search(value)) or value.startswith(PREFIX)


@dataclass(frozen=True)
class EncodedCatalogText:
    kind: CatalogTextKind
    sha256: str
    base64_data: str

    @classmethod
    def encode(cls, value: str | bytes) -> "EncodedCatalogText":
        kind = CatalogTextKind.BYTES if isinstance(value, bytes) else CatalogTextKind.UTF8
        raw = value if isinstance(value, bytes) else value.encode("utf-8", errors="surrogatepass")
        return cls(kind, hashlib.sha256(raw).hexdigest(), base64.b64encode(raw).decode("ascii"))

    def storage_text(self) -> str:
        return f"{PREFIX}{self.kind}:{self.sha256}:{self.base64_data}"

    def decode(self) -> str | bytes:
        raw = base64.b64decode(self.base64_data, validate=True)
        if hashlib.sha256(raw).hexdigest() != self.sha256:
            raise ValueError("Catalog text encoding digest differs from its payload")
        return raw if self.kind is CatalogTextKind.BYTES else raw.decode("utf-8", errors="surrogatepass")


def encode_catalog_value(value: Any) -> Any:
    """Encode unsupported text or bytes while preserving JSON container structure."""
    if isinstance(value, (str, bytes)):
        return EncodedCatalogText.encode(value).storage_text() if needs_encoding(value) else value
    if isinstance(value, dict):
        return {encode_catalog_value(key): encode_catalog_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [encode_catalog_value(item) for item in value]
    return value


def decode_catalog_value(value: Any) -> Any:
    """Recover exactly one transport layer, including reserved-prefix literals."""
    if isinstance(value, str) and value.startswith(PREFIX):
        _, kind, sha256, payload = value.split(":", 3)
        return EncodedCatalogText(CatalogTextKind(kind), sha256, payload).decode()
    if isinstance(value, dict):
        return {decode_catalog_value(key): decode_catalog_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_catalog_value(item) for item in value]
    return value


def display_catalog_value(value: Any) -> Any:
    """Show a marker when the full header survives, even with a truncated payload."""
    if isinstance(value, str) and (match := HEADER.match(value)):
        return f"{BINARY_TEXT_MARKER} SHA-256: {match[2]}"
    if isinstance(value, dict):
        return {key: display_catalog_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [display_catalog_value(item) for item in value]
    return value
