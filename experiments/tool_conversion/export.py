# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Lossless token export and assistant-only masks for converted traces."""

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from tokenizers import Tokenizer

from experiments.tool_conversion.convert import HEADER, Conversion, Disposition, parse_trace


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


@dataclass
class Export:
    files: dict[str, bytes]
    manifest: dict


def export_conversion(conversion: Conversion, tokenizer: Tokenizer, tokenizer_sha256: str) -> Export:
    """Export complete text, structured messages, IDs and assistant-only loss mask."""
    if conversion.disposition != Disposition.VERIFIED:
        raise ValueError("Only verified, modified conversions can be exported")
    encoding = tokenizer.encode(conversion.text, add_special_tokens=False)
    ids = np.asarray(encoding.ids, dtype="<i4")
    if len(ids) > 262144:
        raise ValueError("Complete converted trace exceeds token budget")
    if tokenizer.decode(encoding.ids, skip_special_tokens=False) != conversion.text:
        raise ValueError("Token serialization does not round trip")
    mask = np.zeros(len(ids), dtype="u1")
    heads = list(HEADER.finditer(conversion.text))
    offsets = np.asarray(encoding.offsets)
    for i, head in enumerate(heads):
        if head[1] != "assistant":
            continue
        end = heads[i + 1].start() if i + 1 < len(heads) else len(conversion.text)
        selected = (offsets[:, 0] >= head.end()) & (offsets[:, 1] <= end) & (offsets[:, 1] > offsets[:, 0])
        mask[selected] = 1
    if not mask.any():
        raise ValueError("No assistant supervision")
    parse_trace(conversion.text)
    files = {
        "text.txt": conversion.text.encode(),
        "messages.json": json.dumps(conversion.messages, ensure_ascii=False).encode(),
        "tools.json": json.dumps(conversion.tools, ensure_ascii=False).encode(),
        "input_ids.i32": ids.tobytes(),
        "loss_mask.u8": mask.tobytes(),
    }
    manifest = {
        "tokens": len(ids),
        "supervised_tokens": int(mask.sum()),
        "tokenizer_sha256": tokenizer_sha256,
        "template": "source-preserving-token-surgery-v1",
        "files": {name: digest(content) for name, content in files.items()},
    }
    files["manifest.json"] = json.dumps(manifest, sort_keys=True).encode()
    return Export(files, manifest)
