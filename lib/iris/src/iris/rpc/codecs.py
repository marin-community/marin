# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compact JSON codec for connectrpc.

The connectrpc default ``ProtoJSONCodec`` calls ``MessageToJson(message)``
without an ``indent`` argument. ``MessageToJson`` defaults to ``indent=2``,
so every JSON response is pretty-printed with newlines and 2-space indentation
— that's the formatting the dashboard sees in DevTools. The indentation also
forces the server to do extra string work on every response.

Importing this module installs a compact replacement (``indent=None``) into
the connectrpc codec registry. The dashboard polling RPCs (ListJobs,
ListWorkers, GetSchedulerState) are the dominant JSON traffic; this is a
no-behavior wire-size + CPU win.

Idempotent. Safe to import from multiple modules.
"""

from __future__ import annotations

import threading

from connectrpc import _codec
from connectrpc._codec import (
    CODEC_NAME_JSON,
    CODEC_NAME_JSON_CHARSET_UTF8,
    ProtoJSONCodec,
)
from google.protobuf.json_format import MessageToJson
from google.protobuf.message import Message

_install_lock = threading.Lock()
_installed = False


class CompactProtoJSONCodec(ProtoJSONCodec):
    """``ProtoJSONCodec`` with ``MessageToJson(indent=None)``."""

    def encode(self, message: Message) -> bytes:
        return MessageToJson(message, indent=None).encode()


def install_compact_json_codec() -> None:
    """Replace the connectrpc JSON codec singleton with the compact variant.

    Connectrpc looks codecs up by name from a module-level dict; replacing
    the dict entries (and the exported singleton) is enough to redirect every
    server and client in this process to the compact encoder.
    """
    global _installed
    with _install_lock:
        if _installed:
            return
        codec = CompactProtoJSONCodec()
        _codec._proto_json_codec = codec
        _codec._codecs[CODEC_NAME_JSON] = codec
        _codec._codecs[CODEC_NAME_JSON_CHARSET_UTF8] = codec
        _installed = True


install_compact_json_codec()
