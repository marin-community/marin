# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
import os
import re
import string
from collections.abc import Iterator, Sequence

import msgspec

from marin.utils import fsspec_exists, fsspec_glob, fsspec_isdir

_TOKENIZER_PATTERN = re.compile(rf"[\s{re.escape(string.punctuation)}]+")
_EVAL_HASH_SUFFIX = re.compile(r"^(?P<name>.+)-[0-9a-f]{6}$")
_EVAL_DOLMA_SUFFIX = "-dolma"
_TRAINING_EXTENSIONS = (
    ".json",
    ".json.gz",
    ".json.zst",
    ".jsonl",
    ".jsonl.gz",
    ".jsonl.zst",
    ".parquet",
)


class DefaultTokenizer:
    """Tokenize with lowercase + whitespace/punctuation splitting."""

    def __init__(self, lowercase: bool = True) -> None:
        self._lowercase = lowercase

    def tokenize(self, text: str) -> list[str]:
        if self._lowercase:
            text = text.lower()
        return _TOKENIZER_PATTERN.split(text)


class WhitespaceTokenizer:
    """Tokenize on whitespace only."""

    def __init__(self, lowercase: bool = False) -> None:
        self._lowercase = lowercase

    def tokenize(self, text: str) -> list[str]:
        if self._lowercase:
            text = text.lower()
        return text.split()


def get_tokenizer(name: str) -> DefaultTokenizer | WhitespaceTokenizer:
    """Return a tokenizer instance by name."""
    if name == "default":
        return DefaultTokenizer()
    if name == "no_lowercase":
        return DefaultTokenizer(lowercase=False)
    if name == "whitespace":
        return WhitespaceTokenizer()
    if name == "whitespace_lower":
        return WhitespaceTokenizer(lowercase=True)
    raise ValueError(f"Unknown tokenizer: {name}")


def iter_ngrams(tokens: list[str], n: int, stride: int) -> Iterator[str]:
    """Yield n-grams as space-joined strings."""
    step = stride + 1
    for i in range(0, len(tokens) - n + 1, step):
        yield " ".join(tokens[i : i + n])


def stable_hash(value: str | bytes) -> int:
    """Return a deterministic 64-bit hash using blake2b."""
    if isinstance(value, str):
        value = value.encode()
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "big")


def record_id(record: dict) -> str:
    """Return record id if present, else a deterministic hash of the record."""
    if "id" in record and record["id"] is not None:
        return str(record["id"])
    payload = msgspec.msgpack.encode(record, order="deterministic")
    return str(stable_hash(payload))


def parse_eval_dataset_name(path: str) -> str:
    """Extract dataset name from an eval dataset output path."""
    cleaned = path.rstrip("/")
    basename = os.path.basename(cleaned)
    if basename.endswith((".jsonl", ".jsonl.gz", ".jsonl.zst")):
        basename = os.path.basename(os.path.dirname(cleaned))
    match = _EVAL_HASH_SUFFIX.match(basename)
    if match:
        basename = match.group("name")
    if basename.endswith(_EVAL_DOLMA_SUFFIX):
        basename = basename[: -len(_EVAL_DOLMA_SUFFIX)]
    return basename


def _normalize_paths(paths: str | Sequence[str]) -> list[str]:
    if isinstance(paths, str):
        return [paths]
    return [str(path) for path in paths]


def collect_input_files(input_path: str | Sequence[str]) -> list[str]:
    """Collect training data files from file or directory inputs."""
    input_paths = _normalize_paths(input_path)
    all_files: list[str] = []
    for path in input_paths:
        if fsspec_exists(path) and not fsspec_isdir(path):
            all_files.append(path)
            continue
        extensions = ",".join(ext.lstrip(".") for ext in _TRAINING_EXTENSIONS)
        files = fsspec_glob(f"{path.rstrip('/')}/**/*.{{{extensions}}}")
        if files:
            all_files.extend(files)
        else:
            if not path.endswith(_TRAINING_EXTENSIONS):
                raise FileNotFoundError(f"No files found in path: {path}")
            all_files.append(path)
    if not all_files:
        raise FileNotFoundError("No input files found for overlap computation.")
    return sorted(all_files)


def collect_eval_file_specs(eval_paths: str | Sequence[str]) -> list[dict]:
    """Collect eval JSONL files and attach dataset names."""
    specs: list[dict] = []
    for eval_path in _normalize_paths(eval_paths):
        eval_dataset = parse_eval_dataset_name(eval_path)
        files = fsspec_glob(f"{eval_path.rstrip('/')}/**/*.jsonl*")
        if not files and eval_path.endswith((".jsonl", ".jsonl.gz", ".jsonl.zst")):
            files = [eval_path]
        if not files:
            raise FileNotFoundError(f"No eval jsonl files found in path: {eval_path}")
        for file_path in sorted(files):
            specs.append({"path": file_path, "eval_dataset": eval_dataset})
    return specs
