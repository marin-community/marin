"""Persist validated canonical task rows and their unfiltered prompt lengths."""

import hashlib
import json
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset
from finestore.rl.strata import SELECTION_MANIFEST_FILENAME, SELECTION_SUMMARY_FILENAME, read_selection_definition
from loguru import logger
from marinskyrl.task_sources import PreparedTaskCacheSource
from rigging.filesystem.buckets import MissingCredentials, filesystem_for
from rigging.filesystem.storage_path import StoragePath
from tokenizers.models import BPE
from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils_tokenizers import TokenizersBackend

_CACHE_KEY_METADATA = b"skyrl.prepared_task_key"
PREPARED_TASK_BATCH_SIZE = 1024
PROMPT_LENGTH_COLUMN = "_prompt_token_count"

# Include the installed preparation code so experimental branches can share the cache safely.
_PREPARATION_MODULES = (
    "finestore.rl.strata",
    "rigging.filesystem.path_validation",
    "rolloutengine",
    "shellbox",
    "skyrl_gym",
    "skyrl_train.config.rollout_validation",
    "skyrl_train.dataset",
    "skyrl_train.inference_engines.chat_template",
    "skyrl_train.task_inputs",
    "taskcompendium",
    "verifyit",
)


def _preparation_code_digest() -> str | None:
    digest = hashlib.sha256()
    for name in _PREPARATION_MODULES:
        spec = find_spec(name)
        if spec is None or spec.origin is None or not spec.origin.endswith(".py"):
            logger.debug("Cannot fingerprint prepared task implementation module {}", name)
            return None
        origin = Path(spec.origin)
        paths = sorted(origin.parent.rglob("*.py")) if spec.submodule_search_locations is not None else [origin]
        if not paths:
            return None
        for path in paths:
            try:
                with path.open("rb") as stream:
                    file_digest = hashlib.file_digest(stream, "sha256").digest()
            except OSError as error:
                logger.debug("Cannot fingerprint prepared task implementation file {}: {}", path, error)
                return None
            digest.update(f"{name}/{path.relative_to(origin.parent).as_posix()}\0".encode())
            digest.update(file_digest)
    return digest.hexdigest()


def prepared_task_cache_key(
    source: PreparedTaskCacheSource,
    tokenizer: PreTrainedTokenizerBase,
    task_settings: Mapping,
    chat_template_kwargs: dict,
) -> str | None:
    """Fingerprint one prepared selection, or return None if the inputs are ineligible for caching."""
    path = Path(source.path)
    if path.suffix.lower() != ".parquet":
        return None
    definition = read_selection_definition(str(path.parent))
    if definition is None or path.name not in definition.active_files:
        return None
    tokenizer_key = _tokenizer_key(tokenizer, chat_template_kwargs)
    if tokenizer_key is None:
        return None
    preparation_code_digest = _preparation_code_digest()
    if preparation_code_digest is None:
        return None
    # The manifest validates both populations, including files outside the launched split.
    inputs = sorted(
        {SELECTION_SUMMARY_FILENAME, SELECTION_MANIFEST_FILENAME, *definition.task_files, *definition.active_files}
    )
    directory = StoragePath(str(path.parent))
    digests = {}
    for name in inputs:
        with (directory / name).open("rb") as stream:
            digests[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    provenance = {
        "preparation_code": preparation_code_digest,
        "dependencies": [version(package) for package in ("transformers", "tokenizers", "jinja2", "pydantic")],
        "selection": source.identity,
        "source": path.name,
        "files": digests,
        "tokenizer": tokenizer_key,
        "chat_template_kwargs": chat_template_kwargs,
        "task_settings": task_settings,
    }
    # Preserve the order of template kwargs and nested tool schemas as seen by Jinja.
    try:
        encoded = json.dumps(provenance, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        logger.debug("Skipping prepared task cache for {}: {}", path, error)
        return None
    return hashlib.sha256(encoded).hexdigest()


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


def load_prepared_tasks(
    path: StoragePath, key: str, required_columns: set[str], row_count: int, arrow_path: Path
) -> Dataset | None:
    """Decode a complete cache object into a local memory-mapped dataset in batches."""
    try:
        filesystem, filename = filesystem_for(str(path))
        with nullcontext() if path.is_local else TemporaryDirectory() as directory:
            if directory is not None:
                local_path = str(Path(directory) / "prepared.parquet")
                filesystem.get_file(filename, local_path)
                filename = local_path
            with pq.ParquetFile(filename, page_checksum_verification=True) as parquet:
                schema = parquet.schema_arrow
                if (
                    (schema.metadata or {}).get(_CACHE_KEY_METADATA) != key.encode()
                    or not required_columns.issubset(schema.names)
                    or parquet.metadata.num_rows != row_count
                ):
                    raise ValueError("prepared task cache does not match the source or expected schema")
                lengths_index = schema.get_field_index(PROMPT_LENGTH_COLUMN)
                if schema.field(lengths_index).type != pa.int64():
                    raise ValueError("prepared task cache contains invalid prompt lengths")
                with pa.ipc.new_stream(arrow_path, schema) as writer:
                    for batch in parquet.iter_batches(batch_size=PREPARED_TASK_BATCH_SIZE):
                        if batch.column(lengths_index).null_count:
                            raise ValueError("prepared task cache contains null prompt lengths")
                        writer.write_batch(batch)
        return Dataset.from_file(str(arrow_path), in_memory=False)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, MissingCredentials) as error:
        logger.warning("Could not read prepared task cache {}: {}", path, error)
        arrow_path.unlink(missing_ok=True)
        return None


def write_prepared_tasks(path: StoragePath, key: str, schema: pa.Schema, arrow_shards: Iterable[str]) -> None:
    """Publish a complete file so concurrent writers and retries cannot expose partial rows."""
    try:
        filesystem, filename = filesystem_for(str(path))
        parent = Path(filename).parent if path.is_local else None
        if parent is not None:
            parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=parent) as directory:
            temporary = Path(directory) / "prepared.parquet"
            metadata = {**(schema.metadata or {}), _CACHE_KEY_METADATA: key.encode()}
            with pq.ParquetWriter(
                temporary,
                schema.with_metadata(metadata),
                compression="zstd",
                write_page_checksum=True,
            ) as writer:
                for shard in arrow_shards:
                    # Read one batch at a time so the export does not fault in all the mapped rows.
                    with pa.input_stream(shard) as stream, pa.ipc.open_stream(stream) as reader:
                        for batch in reader:
                            writer.write_batch(batch)
            if parent is None:
                filesystem.put_file(str(temporary), filename)
            else:
                temporary.chmod(0o600)
                temporary.replace(filename)
    except (OSError, ValueError, MissingCredentials) as error:
        logger.warning("Could not write prepared task cache {}: {}", path, error)
