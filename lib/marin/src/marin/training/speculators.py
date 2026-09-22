# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Concrete operations for offline draft-model training with Speculators."""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finestore.eval import ARCHIVE_ROLLOUTS_TABLE, ParticipantType
from finestore.reader import ReadView
from levanter.model_cache import cache_hf_model
from rigging.filesystem.storage_path import StoragePath, prefix_join

from marin.inference.backend import OPENAI_API_SUFFIX, ModelSpec
from marin.inference.config import VllmEngineConfig, VllmLauncherType, VllmSource
from marin.inference.vllm_backend import VllmBackend

logger = logging.getLogger(__name__)

SPECULATORS_DATA_FILENAME = "conversations.jsonl"
SPECULATORS_PREPARED_DATA_DIR = "data"
SPECULATORS_HIDDEN_STATES_DIR = "hidden_states"
_OBJECT_STORE_BATCH_SIZE = 16

_VERIFIER_TENSORS = (
    "model.embed_tokens.weight",
    "lm_head.weight",
    "model.norm.weight",
    "model.final_gated_norm.down_proj.weight",
    "model.final_gated_norm.up_proj.weight",
)
_CHECKPOINT_INDEX = "model.safetensors.index.json"
_CONFIG_FILENAME = "config.json"
_TOKENIZER_FILES = (
    "added_tokens.json",
    "chat_template.jinja",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
)


@dataclass(frozen=True)
class RolloutConversationConfig:
    source_archives: tuple[str, ...]
    output_path: str
    expected_conversations: int


@dataclass(frozen=True)
class HfSnapshotConfig:
    repo_id: str
    revision: str
    output_path: str


@dataclass(frozen=True)
class VerifierViewConfig:
    source_model: str
    transformers_model_type: str
    output_path: str


@dataclass(frozen=True)
class HiddenStateCaptureConfig:
    dataset_path: str
    target_model: str
    processor_model: str
    output_path: str
    target_layer_ids: tuple[int, ...]
    verifier_num_hidden_layers: int
    sequence_length: int
    data_parallel_size: int
    concurrency: int
    max_samples: int | None
    minimum_valid_tokens: int | None
    gpu_memory_utilization: float


@dataclass(frozen=True)
class DraftTrainingConfig:
    captured_data_path: str
    verifier_path: str
    initial_draft_path: str
    output_path: str
    target_layer_ids: tuple[int, ...]
    sequence_length: int
    epochs: int
    learning_rate: float
    muon_learning_rate: float
    num_processes: int
    train_data_ratio: float
    save_best: bool


def _rollout_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return str(row["task"]), str(row["doc_id"]), str(row.get("trial_id") or "")


def _message_role(participant: str) -> str:
    if participant == ParticipantType.SYSTEM:
        return "system"
    if participant == ParticipantType.USER:
        return "user"
    if participant == ParticipantType.ASSISTANT:
        return "assistant"
    if participant in (ParticipantType.TOOL, ParticipantType.ENVIRONMENT):
        return "tool"
    raise ValueError(f"unsupported rollout participant type: {participant}")


def rollout_conversations(rows: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    """Convert primary-key-ordered FineStore rollout rows to conversation records."""
    current_key: tuple[str, str, str] | None = None
    messages: list[dict[str, str]] = []
    has_assistant = False
    has_prompt = False

    for row in rows:
        key = _rollout_key(row)
        if current_key is not None and key != current_key:
            if messages and has_assistant:
                if not has_prompt:
                    raise ValueError(f"rollout {current_key} has an assistant response but no prompt")
                yield {"conversations": messages}
            messages = []
            has_assistant = False
            has_prompt = False
        current_key = key

        content = row.get("content")
        if not isinstance(content, str) or not content:
            continue
        participant = str(row["participant_type"])
        role = _message_role(participant)
        messages.append({"role": role, "content": content})
        has_assistant = has_assistant or role == "assistant"
        has_prompt = has_prompt or role != "assistant"

    if messages and has_assistant:
        if not has_prompt:
            raise ValueError(f"rollout {current_key} has an assistant response but no prompt")
        yield {"conversations": messages}


def interleave_conversation_streams(
    streams: Iterable[Iterator[dict[str, Any]]],
) -> Iterator[dict[str, Any]]:
    """Round-robin conversation streams until every source is exhausted."""
    active = deque(streams)
    while active:
        stream = active.popleft()
        try:
            yield next(stream)
        except StopIteration:
            continue
        active.append(stream)


def write_rollout_conversations(config: RolloutConversationConfig) -> None:
    """Write source-interleaved on-policy conversations from FineStore archives."""
    destination = StoragePath(prefix_join(config.output_path, SPECULATORS_DATA_FILENAME))
    streams: list[Iterator[dict[str, Any]]] = []
    for archive in config.source_archives:
        reader = ReadView(archive)
        if not reader.list_shards(ARCHIVE_ROLLOUTS_TABLE):
            raise ValueError(f"evaluation archive has no {ARCHIVE_ROLLOUTS_TABLE} table: {archive}")
        streams.append(rollout_conversations(reader.iter_rows(ARCHIVE_ROLLOUTS_TABLE)))

    count = 0
    with destination.open("w") as output:
        for record in interleave_conversation_streams(streams):
            output.write(json.dumps(record, separators=(",", ":")))
            output.write("\n")
            count += 1
    if count == 0:
        raise ValueError("evaluation archives contain no assistant responses")
    if count != config.expected_conversations:
        raise ValueError(f"expected {config.expected_conversations} conversations, wrote {count}")
    logger.info("Wrote %d on-policy conversations to %s", count, destination)


def mirror_hf_snapshot(config: HfSnapshotConfig) -> None:
    cache_hf_model(config.output_path, config.repo_id, revision=config.revision)


def verifier_config_for_transformers(config: Mapping[str, Any], model_type: str) -> dict[str, Any]:
    """Return a Transformers-loadable verifier config while preserving architecture identity."""
    rewritten = dict(config)
    rewritten["model_type"] = model_type
    return rewritten


def build_verifier_view(config: VerifierViewConfig) -> None:
    """Publish only the verifier tensors and metadata needed by Speculators."""
    source = StoragePath(config.source_model)
    index = json.loads((source / _CHECKPOINT_INDEX).read_text())
    weight_map = index["weight_map"]
    missing = [name for name in _VERIFIER_TENSORS if name not in weight_map]
    if missing:
        raise ValueError(f"verifier checkpoint is missing required tensors: {missing}")

    with tempfile.TemporaryDirectory() as workdir:
        work_path = Path(workdir)
        selected_shards = sorted({weight_map[name] for name in _VERIFIER_TENSORS})
        for shard in selected_shards:
            local_shard = work_path / shard
            (source / shard).download_to(str(local_shard))
        selected_index = {
            "metadata": index.get("metadata", {}),
            "weight_map": {name: weight_map[name] for name in _VERIFIER_TENSORS},
        }
        (work_path / _CHECKPOINT_INDEX).write_text(json.dumps(selected_index, indent=2, sort_keys=True) + "\n")

        source_config = json.loads((source / _CONFIG_FILENAME).read_text())
        transformed = verifier_config_for_transformers(source_config, config.transformers_model_type)
        (work_path / _CONFIG_FILENAME).write_text(json.dumps(transformed, indent=2, sort_keys=True) + "\n")

        for filename in _TOKENIZER_FILES:
            source_file = source / filename
            if source_file.exists():
                source_file.download_to(str(work_path / filename))
        _publish_directory(work_path, config.output_path)


def _run_command(command: Sequence[str], *, environment: Mapping[str, str] | None = None) -> None:
    logger.info("Running %s", shlex.join(command))
    subprocess.run(command, check=True, env=None if environment is None else dict(environment))


def _publish_directory(source: Path, destination: str) -> None:
    """Publish missing or size-mismatched files and retain completed output."""
    destination_path = StoragePath(destination)
    for source_file in sorted(path for path in source.rglob("*") if path.is_file()):
        relative_path = source_file.relative_to(source).as_posix()
        output_file = destination_path / relative_path
        source_size = source_file.stat().st_size
        if output_file.exists() and output_file.size() == source_size:
            logger.info("Already published %s", output_file)
            continue

        logger.info("Publishing %s (%d bytes)", output_file, source_size)
        if output_file.is_local:
            output_file.parent.mkdirs()
        output_file.upload_from(str(source_file))
        uploaded_size = output_file.size()
        if uploaded_size != source_size:
            raise RuntimeError(
                f"Published file has the wrong size: {output_file} ({uploaded_size} bytes, expected {source_size})"
            )


def _restore_directory(source: str, destination: Path) -> None:
    """Restore a previously published directory when retrying a remote task."""
    source_path = StoragePath(source)
    if not source_path.exists():
        return

    logger.info("Restoring capture progress from %s", source_path)
    source_path.download_to(str(destination), recursive=True, batch_size=_OBJECT_STORE_BATCH_SIZE)


def _capture_vllm_args(config: HiddenStateCaptureConfig, hidden_states_path: Path) -> list[str]:
    layer_ids = [*config.target_layer_ids, config.verifier_num_hidden_layers]
    speculative_config = {
        "method": "extract_hidden_states",
        "num_speculative_tokens": 1,
        "draft_model_config": {"hf_config": {"eagle_aux_hidden_state_layer_ids": layer_ids}},
    }
    transfer_config = {
        "kv_connector": "ExampleHiddenStatesConnector",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {"shared_storage_path": str(hidden_states_path)},
    }
    return [
        "--enforce-eager",
        "--no-enable-flashinfer-autotune",
        "--data-parallel-size",
        str(config.data_parallel_size),
        "--enable-expert-parallel",
        "--api-server-count",
        str(config.data_parallel_size),
        "--renderer-num-workers",
        "2",
        "--gpu-memory-utilization",
        str(config.gpu_memory_utilization),
        "--speculative-config",
        json.dumps(speculative_config, separators=(",", ":")),
        "--kv-transfer-config",
        json.dumps(transfer_config, separators=(",", ":")),
    ]


def _prepare_capture_data(
    config: HiddenStateCaptureConfig, raw_data: Path, prepared_data: Path, render_endpoint: str
) -> None:
    command = [
        sys.executable,
        "-m",
        "speculators",
        "prepare-data",
        "--model",
        config.processor_model,
        "--data",
        str(raw_data),
        "--output",
        str(prepared_data),
        "--seq-length",
        str(config.sequence_length),
        "--render-endpoint",
        render_endpoint,
    ]
    if config.max_samples is not None:
        command.extend(("--max-samples", str(config.max_samples)))
    if config.minimum_valid_tokens is not None:
        command.extend(("--minimum-valid-tokens", str(config.minimum_valid_tokens)))
    _run_command(command)


def _generate_hidden_states(
    config: HiddenStateCaptureConfig, prepared_data: Path, hidden_states: Path, endpoint: str
) -> None:
    command = [
        sys.executable,
        "-m",
        "speculators",
        "generate-offline-data",
        "--endpoint",
        endpoint,
        "--preprocessed-data",
        str(prepared_data),
        "--output",
        str(hidden_states),
        "--concurrency",
        str(config.concurrency),
        "--validate-outputs",
        "--fail-on-error",
    ]
    if config.max_samples is not None:
        command.extend(("--max-samples", str(config.max_samples)))
    _run_command(command)


def capture_hidden_states(config: HiddenStateCaptureConfig) -> None:
    """Prepare conversations and publish their cached verifier hidden states."""
    with tempfile.TemporaryDirectory() as workdir:
        work_path = Path(workdir)
        raw_data = work_path / SPECULATORS_DATA_FILENAME
        prepared_data = work_path / SPECULATORS_PREPARED_DATA_DIR
        hidden_states = prepared_data / SPECULATORS_HIDDEN_STATES_DIR
        connector_staging = work_path / "connector"
        target_model = work_path / "target_model"
        StoragePath(config.dataset_path).download_to(str(raw_data))
        StoragePath(config.target_model).download_to(
            str(target_model), recursive=True, batch_size=_OBJECT_STORE_BATCH_SIZE
        )
        _restore_directory(config.output_path, prepared_data)
        connector_staging.mkdir()

        backend = VllmBackend(
            VllmEngineConfig(
                launcher=VllmLauncherType.CUDA,
                source=VllmSource.MARIN_FORK,
                startup_timeout_seconds=3600,
                max_num_batched_tokens=config.sequence_length,
            )
        )
        model = ModelSpec(
            weights=str(target_model),
            api_model="verifier",
            num_chips=config.data_parallel_size,
            tensor_parallel_size=1,
            dtype="bfloat16",
            max_model_len=config.sequence_length,
            chat_template_content=None,
        )
        capture_complete = False
        prepared_complete = False
        try:
            with backend.start(
                model,
                extra_args=_capture_vllm_args(config, connector_staging),
                subprocess_env={"VLLM_ENABLE_SCALE_OUT_ENDPOINTS": "1"},
            ) as environment:
                environment.wait_until_ready()
                render_endpoint = environment.server_url.removesuffix(OPENAI_API_SUFFIX)
                _prepare_capture_data(config, raw_data, prepared_data, render_endpoint)
                prepared_complete = True
                hidden_states.mkdir(exist_ok=True)
                _generate_hidden_states(config, prepared_data, hidden_states, environment.server_url)
                capture_complete = True
        except Exception:
            if not capture_complete:
                raise
            logger.warning("Ignoring vLLM teardown failure after hidden-state capture completed", exc_info=True)
        finally:
            if prepared_complete:
                _publish_directory(prepared_data, config.output_path)


def _latest_checkpoint(checkpoints: Path) -> Path:
    candidates = sorted(
        (path for path in checkpoints.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if not candidates:
        raise ValueError(f"Speculators training wrote no checkpoints under {checkpoints}")
    return candidates[-1]


def _make_checkpoint_portable(checkpoint: Path) -> None:
    config_path = checkpoint / _CONFIG_FILENAME
    config = json.loads(config_path.read_text())
    config["speculators_config"]["verifier"]["name_or_path"] = None
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def _safetensors_tensor_names(path: Path) -> set[str]:
    with path.open("rb") as checkpoint:
        header_size_bytes = checkpoint.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"Invalid safetensors header: {path}")
        header_size = int.from_bytes(header_size_bytes, "little")
        header = json.loads(checkpoint.read(header_size))
    return set(header) - {"__metadata__"}


def _validate_draft_checkpoint(checkpoint: Path) -> None:
    config = json.loads((checkpoint / _CONFIG_FILENAME).read_text())
    if config.get("embed_requires_grad", False):
        return

    for weights_path in checkpoint.glob("*.safetensors"):
        if "embed_tokens.weight" in _safetensors_tensor_names(weights_path):
            raise ValueError(
                "Frozen EAGLE embeddings must be omitted so vLLM shares the verifier embedding: "
                f"{weights_path} contains embed_tokens.weight"
            )


def train_draft(config: DraftTrainingConfig) -> None:
    """Train and publish a draft checkpoint from cached verifier states."""
    with tempfile.TemporaryDirectory() as workdir:
        work_path = Path(workdir)
        data = work_path / "data"
        verifier = work_path / "verifier"
        initial_draft = work_path / "initial_draft"
        checkpoints = work_path / "checkpoints"
        published = work_path / "published"
        StoragePath(config.captured_data_path).download_to(str(data), recursive=True)
        StoragePath(config.verifier_path).download_to(str(verifier), recursive=True)
        StoragePath(config.initial_draft_path).download_to(str(initial_draft), recursive=True)

        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node",
            str(config.num_processes),
            "-m",
            "speculators.train",
            "--verifier-name-or-path",
            str(verifier),
            "--from-pretrained",
            str(initial_draft),
            "--data-path",
            str(data),
            "--hidden-states-path",
            str(data / SPECULATORS_HIDDEN_STATES_DIR),
            "--save-path",
            str(checkpoints),
            "--on-missing",
            "raise",
            "--target-layer-ids",
            *(str(layer_id) for layer_id in config.target_layer_ids),
            "--total-seq-len",
            str(config.sequence_length),
            "--epochs",
            str(config.epochs),
            "--optimizer",
            "muon",
            "--lr",
            str(config.learning_rate),
            "--muon-lr",
            str(config.muon_learning_rate),
            "--train-data-ratio",
            str(config.train_data_ratio),
            "--checkpoint-freq",
            "1",
        ]
        if config.save_best:
            command.append("--save-best")
        _run_command(command, environment=os.environ | {"TOKENIZERS_PARALLELISM": "false"})

        if config.save_best:
            checkpoint = checkpoints / "checkpoint_best"
            if not checkpoint.exists():
                raise ValueError(f"Speculators training wrote no best checkpoint under {checkpoints}")
            checkpoint = checkpoint.resolve()
        else:
            checkpoint = _latest_checkpoint(checkpoints)

        shutil.copytree(checkpoint, published)
        _make_checkpoint_portable(published)
        _validate_draft_checkpoint(published)
        _publish_directory(published, config.output_path)
