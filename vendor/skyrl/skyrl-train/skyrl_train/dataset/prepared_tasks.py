"""Persist validated canonical task rows and their unfiltered prompt lengths."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
from finestore.rl.strata import read_selection_definition
from loguru import logger
from rigging.filesystem.buckets import MissingCredentials, filesystem_for
from rigging.filesystem.storage_path import StoragePath
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils_tokenizers import TokenizersBackend

# Bump when selection validation, canonical task lowering, or prompt length semantics change.
_CACHE_VERSION = 1
_CACHE_KEY_METADATA = b"skyrl.prepared_task_key"
PROMPT_LENGTH_COLUMN = "_prompt_token_count"


@dataclass(frozen=True)
class PreparedTaskCacheSource:
    """An immutable selection staged locally and its shared cache directory."""

    path: str
    identity: str
    cache_dir: str

    def cache_key(
        self, tokenizer: PreTrainedTokenizerBase, task_settings: Mapping, chat_template_kwargs: dict
    ) -> str | None:
        source = Path(self.path)
        if source.suffix.lower() != ".parquet":
            return None
        definition = read_selection_definition(str(source.parent))
        if definition is None or source.name not in definition.active_files:
            return None
        tokenizer_key = _tokenizer_key(tokenizer, chat_template_kwargs)
        if tokenizer_key is None:
            return None
        # The manifest's validation covers both populations, so even files not used by this launch affect the key.
        inputs = sorted(
            {source.name, "summary.json", "task_manifest.parquet", *definition.task_files, *definition.active_files}
        )
        directory = StoragePath(str(source.parent))
        digests = {}
        for name in inputs:
            with (directory / name).open("rb") as stream:
                digests[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        provenance = {
            "version": _CACHE_VERSION,
            "dependencies": [version(package) for package in ("transformers", "tokenizers", "jinja2", "pydantic")],
            "selection": self.identity,
            "source": source.name,
            "files": digests,
            "tokenizer": tokenizer_key,
            "chat_template_kwargs": chat_template_kwargs,
            "task_settings": task_settings,
        }
        # Preserve the order of template kwargs and nested tool schemas as seen by Jinja.
        try:
            encoded = json.dumps(provenance, separators=(",", ":"), allow_nan=False).encode()
        except (TypeError, ValueError):
            return None
        return hashlib.sha256(encoded).hexdigest()

    def cache_path(self, key: str) -> StoragePath:
        return StoragePath(self.cache_dir) / f"{key}.parquet"


def _tokenizer_key(tokenizer: PreTrainedTokenizerBase, chat_template_kwargs: dict) -> dict | None:
    # Built-in and remote-code subclasses can add preprocessing state outside the serialized backend.
    if type(tokenizer) is not TokenizersBackend:
        return None
    backend = tokenizer.backend_tokenizer
    if isinstance(backend.model, BPE) and backend.model.dropout:
        return None
    template = tokenizer.chat_template
    templates = list(template.values()) if isinstance(template, dict) else [template]
    templates.append(chat_template_kwargs.get("chat_template"))
    if any(text and any(name in text for name in ("strftime_now", "random", "lipsum")) for text in templates):
        return None
    cls = type(tokenizer)
    return {
        "class": f"{cls.__module__}.{cls.__qualname__}",
        "backend": hashlib.sha256(backend.to_str().encode()).hexdigest(),
        "template": template,
        "special_tokens": tokenizer.special_tokens_map,
        "model_max_length": tokenizer.model_max_length,
        "model_input_names": tokenizer.model_input_names,
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "split_special_tokens": tokenizer.split_special_tokens,
    }


def load_prepared_tasks(path: StoragePath, key: str, columns: set[str], row_count: int) -> pa.Table | None:
    """Read a complete cache object, returning a miss for unavailable or damaged objects."""
    try:
        filesystem, filename = filesystem_for(str(path))
        with filesystem.open(filename, "rb") as stream:
            table = pq.read_table(stream, page_checksum_verification=True)
        if (
            (table.schema.metadata or {}).get(_CACHE_KEY_METADATA) != key.encode()
            or set(table.column_names) != columns
            or len(table) != row_count
        ):
            raise ValueError("prepared task cache does not match the source or expected schema")
        lengths = table.column(PROMPT_LENGTH_COLUMN)
        if lengths.type != pa.int64() or lengths.null_count:
            raise ValueError("prepared task cache contains invalid prompt lengths")
    except FileNotFoundError:
        return None
    except (OSError, ValueError, MissingCredentials) as error:
        logger.warning("Could not read prepared task cache {}: {}", path, error)
        return None
    return table


def write_prepared_tasks(path: StoragePath, key: str, table: pa.Table) -> None:
    """Publish a complete file so concurrent writers and retries cannot expose partial rows."""
    try:
        filesystem, filename = filesystem_for(str(path))
        parent = Path(filename).parent if path.is_local else None
        if parent is not None:
            parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=parent) as directory:
            temporary = Path(directory) / "prepared.parquet"
            metadata = {**(table.schema.metadata or {}), _CACHE_KEY_METADATA: key.encode()}
            pq.write_table(
                table.replace_schema_metadata(metadata), temporary, compression="zstd", write_page_checksum=True
            )
            if parent is None:
                filesystem.put_file(str(temporary), filename)
            else:
                temporary.chmod(0o600)
                temporary.replace(filename)
    except (OSError, ValueError, MissingCredentials) as error:
        logger.warning("Could not write prepared task cache {}: {}", path, error)
