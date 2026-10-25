# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen case/separator token classes for the case-folded CPLM copy head.

A vocabulary token belongs to a class with every other token whose decoded
bytes are equal after stripping leading spaces, then leading underscores, and
ASCII lowercasing, provided the remainder is nonempty ASCII alphanumeric with
at least one letter (``View``, `` view``, ``_VIEW``). Every other token, including
special/added tokens, is its own singleton class. Built only from the frozen
tokenizer file; no corpus statistics.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _byte_decoder():
    # GPT-2 byte-level BPE byte<->unicode map used by the frozen tokenizer.
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs, strict=True)}


def case_key(raw):
    """Class key of decoded token bytes, or None for a singleton token."""
    if raw is None:
        return None
    text = raw.lstrip(b" ").lstrip(b"_")
    if text and text.isascii() and text.isalnum() and any(chr(c).isalpha() for c in text):
        return text.lower()
    return None


def build(tokenizer_json, vocab_size, width=12):
    """Return ``(classes[V], members[V, width])``; members padded with -1.

    ``classes[v]`` is the smallest id in v's class. ``members[v]`` lists v's class
    in increasing id order and therefore always contains v.
    """
    tokenizer = json.loads(Path(tokenizer_json).read_text(encoding="utf-8"))
    decoder = _byte_decoder()
    groups = defaultdict(list)
    for piece, index in tokenizer["model"]["vocab"].items():
        if index >= vocab_size:
            continue
        try:
            raw = bytes(decoder[c] for c in piece)
        except KeyError:
            raw = None
        key = case_key(raw)
        if key is not None:
            groups[key].append(index)
    classes = np.arange(vocab_size, dtype=np.int32)
    members = np.full((vocab_size, width), -1, dtype=np.int32)
    members[:, 0] = classes
    for ids in groups.values():
        ids = sorted(ids)
        if len(ids) > width:
            raise ValueError(f"case class larger than {width}: {len(ids)}")
        for i in ids:
            classes[i] = ids[0]
            members[i, : len(ids)] = ids
    return classes, members


def table_hash(classes, members):
    h = hashlib.sha256()
    for a in (classes, members):
        a = np.ascontiguousarray(a, dtype=np.int32)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def summary(classes, members):
    sizes = (members >= 0).sum(-1)
    return {
        "vocabulary": len(classes),
        "width": int(members.shape[1]),
        "tokens_in_shared_classes": int((sizes > 1).sum()),
        "shared_classes": len(np.unique(classes[sizes > 1])),
        "max_class_size": int(sizes.max()),
        "sha256": table_hash(classes, members),
    }
