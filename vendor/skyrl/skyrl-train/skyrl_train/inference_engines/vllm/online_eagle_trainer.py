"""Single-GPU online EAGLE-3 updates from sealed vLLM rollout captures."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
import copy
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any, Iterator, Mapping
from uuid import uuid4

from safetensors.torch import load_file
import torch
from torch import nn

from marinskyrl.resource_locator import join_resource_path
from marinskyrl.speculative_decoding import (
    SpeculatorModelConfig,
    SpeculatorModelSourceKind,
    SpeculatorTrainingConfig,
    runai_model_uri,
)
from skyrl_train.hf_model_io import HF_WEIGHT_FILENAME
from skyrl_train.io import io


_CANDIDATE_FORMAT = "marinskyrl-online-eagle-candidate"
_CAPTURE_FORMAT = "vllm-online-eagle-capture"
_TRAINER_STATE_FORMAT = "marinskyrl-online-eagle-trainer-state"
_TRAINER_STATE_VERSION = 2
ONLINE_EAGLE_MANIFEST_FILENAME = "manifest.json"
ONLINE_EAGLE_MERGED_CAPTURE_DIRECTORY = "merged"
_SKYRL_REQUEST_PREFIX = "skyrl-group-"
ONLINE_EAGLE_TARGET_CONFIG_FILENAME = "target-config.json"
ONLINE_EAGLE_TARGET_WEIGHTS_FILENAME = "target.safetensors"
TRAINER_STATE_FILENAME = "trainer_state.pt"


def capture_rank_name(worker_rank: int) -> str:
    """Return the stable per-rank capture object name."""
    return f"rank-{worker_rank:05d}"


def capture_rank_directory(capture_root: Path, worker_rank: int) -> Path:
    """Return the shared per-DP-rank capture directory."""
    return capture_root / capture_rank_name(worker_rank)


@dataclass(frozen=True)
class OnlineEagleTrainingJob:
    """One bounded draft update assembled inside DraftTrainer."""

    step: int
    capture_dir: str
    initial_draft_model: SpeculatorModelConfig
    parent_draft_revision: str
    target_revision: str
    output_dir: str
    num_speculative_tokens: int
    seed: int
    training: SpeculatorTrainingConfig


@dataclass(frozen=True)
class OnlineEagleCaptureConfig:
    """One bounded capture interval passed from the trainer to vLLM."""

    step: int
    max_tokens: int
    max_window_tokens: int
    target_revision: str
    draft_revision: str
    reserved_gpu_memory_gib: float

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def request_group_from_id(request_id: str) -> str:
    """Return SkyRL's prompt-group digest, or the ungrouped request ID."""
    if request_id.startswith(_SKYRL_REQUEST_PREFIX):
        remainder = request_id[len(_SKYRL_REQUEST_PREFIX) :]
        group, separator, _attempt = remainder.partition("-")
        if separator and group:
            return group
    return request_id


def request_id_for_group(group_id: object) -> str:
    """Return a unique request ID carrying SkyRL's prompt-group digest."""
    group_digest = hashlib.sha256(str(group_id).encode()).hexdigest()[:16]
    return f"{_SKYRL_REQUEST_PREFIX}{group_digest}-{uuid4().hex}"


def _window_group_id(window: Mapping[str, Any]) -> str:
    group_id = window.get("group_id")
    return str(group_id) if group_id is not None else request_group_from_id(str(window["request_id"]))


def per_worker_capture_token_credit(
    *,
    global_max_tokens: int,
    max_window_tokens: int,
    worker_count: int,
    worker_index: int,
) -> int:
    """Bound local reservation while leaving room for one packing-fragment window.

    vLLM must reserve a request's maximum possible window before generation, but
    the global selector charges the much smaller realized window. Giving each
    rank only its exact share can therefore admit just one request even when the
    realized captures would fit comfortably. One additional maximum-size window
    bounds aggregate host staging by ``global + workers * max_window`` while the
    merged cloud capture remains strictly bounded by ``global``.
    """
    if global_max_tokens <= 0 or max_window_tokens <= 0 or worker_count <= 0:
        raise ValueError("Online EAGLE capture credit bounds must be positive")
    if worker_index < 0 or worker_index >= worker_count:
        raise ValueError(f"Online EAGLE worker index is out of range: {worker_index}")
    token_credit, remainder = divmod(global_max_tokens, worker_count)
    return token_credit + int(worker_index < remainder) + max_window_tokens


@dataclass(frozen=True)
class OnlineEagleEvaluation:
    """Mean EAGLE loss and next-token agreement on one window set."""

    mean_loss: float
    agreement: float


@dataclass(frozen=True)
class OnlineEagleUpdateResult:
    """Typed result envelope shared by DraftTrainer and its coordinator."""

    accepted: bool
    step: int | None = None
    parent_draft_revision: str | None = None
    trained_against_target_revision: str | None = None
    train_sequences: int | None = None
    holdout_sequences: int | None = None
    train_loss: float | None = None
    incumbent_holdout_loss: float | None = None
    candidate_holdout_loss: float | None = None
    incumbent_holdout_agreement: float | None = None
    candidate_holdout_agreement: float | None = None
    holdout_loss_increase: float | None = None
    holdout_agreement_decrease: float | None = None
    max_validation_loss_increase: float | None = None
    max_validation_agreement_decrease: float | None = None
    duration_seconds: float | None = None
    candidate_uri: str | None = None
    draft_revision: str | None = None
    error: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        return {name: value for name, value in asdict(self).items() if value is not None}


def partition_capture_windows(
    windows: list[dict[str, Any]],
    *,
    step: int,
    holdout_fraction: float,
    min_train_sequences: int,
    min_holdout_sequences: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return a deterministic partition that keeps every prompt group in one split."""
    required = min_train_sequences + min_holdout_sequences
    if len(windows) < required:
        raise ValueError(f"Online EAGLE update needs at least {required} captured windows, got {len(windows)}")
    ordered_windows = sorted(windows, key=lambda window: str(window["request_id"]))
    groups: dict[str, list[dict[str, Any]]] = {}
    for window in ordered_windows:
        group_id = _window_group_id(window)
        groups.setdefault(group_id, []).append(window)
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: hashlib.sha256(f"{step}:{item[0]}".encode()).digest(),
    )
    target_holdout_count = max(min_holdout_sequences, math.ceil(len(windows) * holdout_fraction))
    holdout_groups: set[str] = set()
    holdout_count = 0
    for group_id, group_windows in ordered_groups:
        if holdout_count >= target_holdout_count:
            break
        if len(windows) - holdout_count - len(group_windows) < min_train_sequences:
            continue
        holdout_groups.add(group_id)
        holdout_count += len(group_windows)
    if holdout_count < min_holdout_sequences:
        raise ValueError(
            "Online EAGLE capture cannot form group-disjoint train and holdout splits with "
            f"at least {min_train_sequences} and {min_holdout_sequences} sequences"
        )
    holdout = [window for window in ordered_windows if _window_group_id(window) in holdout_groups]
    train = [window for window in ordered_windows if _window_group_id(window) not in holdout_groups]
    return train, holdout


def _capture_file(capture_dir: Path, relative_path: object) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("Online EAGLE capture path must be a nonempty string")
    path = (capture_dir / relative_path).resolve()
    if path.parent != capture_dir.resolve() or not path.is_file():
        raise ValueError(f"Online EAGLE capture file is missing or escaped its rank directory: {relative_path}")
    return path


def _load_and_validate_capture(capture_dir: Path) -> dict[str, Any]:
    manifest_path = capture_dir / ONLINE_EAGLE_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("format") != _CAPTURE_FORMAT:
        raise ValueError(f"Invalid online EAGLE capture manifest: {manifest_path}")
    for window in manifest["windows"]:
        _capture_file(capture_dir, window["path"])
    target = manifest.get("target")
    if target is not None:
        for key in ("weights", "config"):
            _capture_file(capture_dir, target[f"{key}_path"])
    return manifest


def merge_online_eagle_captures(
    capture_root: Path,
    output_dir: Path,
    *,
    expected_step: int,
    max_tokens: int,
    max_sequences_per_prompt_group: int,
    max_window_tokens: int,
) -> dict[str, Any]:
    """Merge cloud-downloaded rank captures into one bounded trainer input."""
    if max_tokens <= 0 or max_sequences_per_prompt_group <= 0 or max_window_tokens <= 0:
        raise ValueError("Online EAGLE capture bounds must be positive")
    rank_directories = sorted(path for path in capture_root.glob("rank-*") if path.is_dir())
    if not rank_directories:
        raise ValueError(f"Online EAGLE capture has no rank directories: {capture_root}")
    captures = [(directory, _load_and_validate_capture(directory)) for directory in rank_directories]

    identity_fields = (
        "format",
        "format_version",
        "step",
        "target_revision",
        "draft_revision",
        "aux_layer_ids",
        "head_input_semantics",
    )
    baseline_directory, baseline = captures[0]
    if baseline.get("step") != expected_step:
        raise ValueError(f"Online EAGLE capture step mismatch: expected {expected_step}, got {baseline.get('step')!r}")
    baseline_identity = {field: baseline.get(field) for field in identity_fields}
    baseline_target = baseline["target"]
    if baseline_target is None:
        raise ValueError("Online EAGLE rank 0 did not capture the target snapshot")
    candidates: list[tuple[Path, dict[str, Any]]] = []
    request_ids: set[str] = set()
    for directory, manifest in captures:
        if {field: manifest.get(field) for field in identity_fields} != baseline_identity:
            raise ValueError("Online EAGLE rank captures do not share one target/draft identity")
        for window in manifest["windows"]:
            request_id = str(window["request_id"])
            if request_id in request_ids:
                raise ValueError(f"Duplicate online EAGLE request across ranks: {request_id}")
            request_ids.add(request_id)
            candidates.append((directory, window))

    step = int(baseline["step"])
    candidates.sort(
        key=lambda item: hashlib.sha256(f"{step}:{_window_group_id(item[1])}:{item[1]['request_id']}".encode()).digest()
    )
    selected = []
    group_counts: dict[str, int] = {}
    selected_tokens = 0
    oversized_windows = 0
    for directory, window in candidates:
        group_id = _window_group_id(window)
        tokens = window.get("tokens")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            raise ValueError(f"Invalid online EAGLE captured-window token count: {tokens!r}")
        if _window_forward_tokens(window) > max_window_tokens:
            oversized_windows += 1
            continue
        if group_counts.get(group_id, 0) >= max_sequences_per_prompt_group:
            continue
        if selected_tokens + tokens > max_tokens:
            continue
        selected.append((directory, window))
        selected_tokens += tokens
        group_counts[group_id] = group_counts.get(group_id, 0) + 1

    output_dir.mkdir(parents=True, exist_ok=False)
    windows = []
    for index, (directory, window) in enumerate(selected):
        destination_path = f"window-{index:06d}.safetensors"
        shutil.copyfile(_capture_file(directory, window["path"]), output_dir / destination_path)
        windows.append({**window, "path": destination_path})
    shutil.copyfile(
        _capture_file(baseline_directory, baseline_target["weights_path"]),
        output_dir / ONLINE_EAGLE_TARGET_WEIGHTS_FILENAME,
    )
    shutil.copyfile(
        _capture_file(baseline_directory, baseline_target["config_path"]),
        output_dir / ONLINE_EAGLE_TARGET_CONFIG_FILENAME,
    )
    manifest = {
        **baseline_identity,
        "worker_rank": 0,
        "worker_ranks": [int(manifest["worker_rank"]) for _, manifest in captures],
        "windows": windows,
        "captured_rows": selected_tokens,
        "source_captured_rows": sum(int(manifest.get("captured_rows", 0)) for _, manifest in captures),
        "source_windows": len(candidates),
        "dropped_requests": sum(int(manifest.get("dropped_requests", 0)) for _, manifest in captures),
        "dropped_windows": sum(int(manifest.get("dropped_windows", 0)) for _, manifest in captures),
        "oversized_windows": oversized_windows,
        "unselected_windows": len(candidates) - len(selected),
        "target": {
            **baseline_target,
            "weights_path": ONLINE_EAGLE_TARGET_WEIGHTS_FILENAME,
            "config_path": ONLINE_EAGLE_TARGET_CONFIG_FILENAME,
        },
    }
    (output_dir / ONLINE_EAGLE_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _prepare_model(
    draft_model: SpeculatorModelConfig,
    capture_dir: Path,
    device: torch.device,
):
    from speculators.models.eagle3 import (  # noqa: PLC0415
        Eagle3DraftModel,
        Eagle3SpeculatorConfig,
    )
    from transformers import PreTrainedModel  # noqa: PLC0415

    pretrained_path = None
    revision = None
    if draft_model.source_kind is SpeculatorModelSourceKind.LOCAL:
        pretrained_path = draft_model.local_source_path
    elif draft_model.source_kind is SpeculatorModelSourceKind.HUGGING_FACE:
        pretrained_path = draft_model.hugging_face_repo_id
        revision = draft_model.source_identity
    if pretrained_path is not None:
        config = Eagle3SpeculatorConfig.from_pretrained(pretrained_path, revision=revision)
    else:
        config = Eagle3SpeculatorConfig.from_dict(
            json.loads(io.read_bytes(join_resource_path(draft_model.source_uri, "config.json")))
        )
    # Speculators' FlexAttention mask is block-padded, but online captures retain
    # their exact sequence length. EAGLE's time-shift extension can therefore
    # produce real Q/KV dimensions smaller than the padded BlockMask contract.
    # SDPA accepts the exact dense mask without materializing eager attention's
    # heads x queries x keys score tensor. At an 8k context and three TTT passes,
    # that eager tensor alone is about 16 GiB for the Snowball draft.
    _configure_exact_mask_attention(config)
    # The public embedding-free Snowball checkpoint intentionally has no verifier
    # path. Bypass Speculators' generic post-load verifier hook because the exact
    # target-owned tensors come from this sealed rollout, not a second HF model.
    if pretrained_path is not None:
        model = PreTrainedModel.from_pretrained.__func__(
            Eagle3DraftModel,
            pretrained_path,
            config=config,
            revision=revision,
        )
    else:
        from vllm.model_executor.model_loader.weight_utils import (  # noqa: PLC0415
            runai_safetensors_weights_iterator,
        )
        from vllm.transformers_utils.runai_utils import list_safetensors  # noqa: PLC0415

        model = Eagle3DraftModel(config)
        weight_files = list_safetensors(runai_model_uri(draft_model.source_uri))
        if not weight_files:
            raise FileNotFoundError(f"Draft checkpoint has no safetensors files: {draft_model.source_uri}")
        model.load_state_dict(
            dict(runai_safetensors_weights_iterator(weight_files, use_tqdm_on_load=False)),
            strict=False,
        )
    _refresh_target_owned_weights(model, capture_dir)
    for name, parameter in model.named_parameters():
        target_owned = name == "embed_tokens.weight" or name == "lm_head.weight" or name.startswith("verifier_")
        parameter.requires_grad_(not target_owned)
    return model.to(device=device, dtype=_serving_dtype(device))


def _refresh_target_owned_weights(model: nn.Module, capture_dir: Path) -> None:
    """Refresh exact target-owned tensors without reconstructing the draft."""
    target = load_file(capture_dir / ONLINE_EAGLE_TARGET_WEIGHTS_FILENAME)
    embedding = target["model.embed_tokens.weight"]
    head = target["lm_head.weight"]
    if model.t2d is None or not torch.any(model.t2d):
        raise ValueError("The EAGLE checkpoint has no target-to-draft vocabulary map")
    draft_rows = int(model.t2d.to(dtype=torch.bool).sum())
    if head.shape[0] == draft_rows:
        draft_head = head
    elif head.shape[0] == model.t2d.numel():
        draft_head = head[model.t2d.to(dtype=torch.bool)]
    else:
        raise ValueError("Captured target head has neither target nor draft vocabulary rows")
    with torch.no_grad():
        model.embed_tokens.weight.copy_(embedding)
        model.lm_head.weight.copy_(draft_head)
        model.verifier_lm_head.weight.copy_(draft_head)
    model.verifier_norm = nn.Identity()
    model.verifier_gate_down = None
    model.verifier_gate_up = None


def _configure_exact_mask_attention(config: Any) -> None:
    """Select the backend that accepts Speculators' exact dense boolean mask."""
    config.transformer_layer_config._attn_implementation = "sdpa"  # noqa: SLF001


def _serving_dtype(device: torch.device) -> torch.dtype:
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def _forward_context(device: torch.device) -> AbstractContextManager[Any]:
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _sdpa_kernel_context(device: torch.device) -> AbstractContextManager[Any]:
    """Require fused exact-mask SDPA instead of permitting a quadratic fallback."""
    if device.type == "cuda":
        from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: PLC0415

        return sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
    return nullcontext()


def _require_finite_tensor(tensor: torch.Tensor, *, label: str) -> None:
    if (tensor.is_floating_point() or tensor.is_complex()) and not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"Online EAGLE found a non-finite tensor: {label}")


def _require_finite_trainable_state(model: nn.Module, optimizer: torch.optim.Optimizer | None = None) -> None:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            _require_finite_tensor(parameter, label=f"parameter {name}")
    if optimizer is not None:
        for parameter_index, state in enumerate(optimizer.state.values()):
            for name, value in state.items():
                if isinstance(value, torch.Tensor):
                    _require_finite_tensor(value, label=f"optimizer parameter {parameter_index} state {name}")


def _convert_trainable_parameters(model: nn.Module, dtype: torch.dtype) -> None:
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.to(dtype=dtype)
    _require_finite_trainable_state(model)


def _capture_trainable_master_state(model: nn.Module) -> dict[str, torch.Tensor]:
    state = {
        name: parameter.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not state:
        raise ValueError("Online EAGLE model has no trainable master parameters")
    return state


def _restore_trainable_master_state(
    model: nn.Module,
    state: Mapping[str, torch.Tensor],
    *,
    serving_dtype: torch.dtype,
) -> None:
    parameters = _validate_trainable_master_state(model, state)
    with torch.no_grad():
        for name, parameter in parameters.items():
            master = state[name]
            device_master = master.to(device=parameter.device)
            if not torch.equal(device_master.to(dtype=serving_dtype), parameter.detach().to(dtype=serving_dtype)):
                raise ValueError(f"Online EAGLE FP32 master does not round to the served tensor: {name}")
            parameter.copy_(device_master)
    _require_finite_trainable_state(model)


def _validate_trainable_master_state(model: nn.Module, state: Mapping[str, torch.Tensor]) -> dict[str, nn.Parameter]:
    parameters = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(state) != set(parameters):
        raise ValueError("Online EAGLE FP32 master tensor inventory does not match the draft model")
    for name, parameter in parameters.items():
        master = state[name]
        if master.dtype != torch.float32 or master.shape != parameter.shape:
            raise ValueError(f"Invalid online EAGLE FP32 master tensor: {name}")
        _require_finite_tensor(master, label=f"FP32 master tensor {name}")
    return parameters


def _load_trainable_master_state(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Load an internal FP32 snapshot without treating current weights as a checkpoint fence."""
    parameters = _validate_trainable_master_state(model, state)
    with torch.no_grad():
        for name, parameter in parameters.items():
            master = state[name]
            parameter.copy_(master.to(device=parameter.device))
    _require_finite_trainable_state(model)


def _load_batch(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    window = load_file(path)
    input_ids = window["input_ids"]
    if input_ids.shape[0] < 2:
        raise ValueError(f"Online EAGLE window is too short: {path}")
    batch = {
        "input_ids": input_ids[1:].unsqueeze(0),
        "hidden_states": window["hidden_states"][:-1].unsqueeze(0),
        "verifier_last_hidden_states": window["head_input_hidden_states"][1:].unsqueeze(0),
        "loss_mask": window["loss_mask"][1:].unsqueeze(0),
        "position_ids": window["position_ids"][1:].unsqueeze(0),
    }
    batch["document_ids"] = torch.zeros_like(batch["input_ids"])
    for name, value in batch.items():
        _require_finite_tensor(value, label=f"capture {path} tensor {name}")
    return {name: value.to(device) for name, value in batch.items()}


def _window_forward_tokens(window: Mapping[str, Any]) -> int:
    return max(1, int(window["tokens"]) - 1)


def _pack_windows(
    windows: list[dict[str, Any]],
    max_tokens: int,
    max_window_tokens: int,
) -> list[list[dict[str, Any]]]:
    """Pack forwards to a token target while preserving request-local histories."""
    batches: list[list[dict[str, Any]]] = []
    batch: list[dict[str, Any]] = []
    batch_tokens = 0
    for window in windows:
        window_tokens = _window_forward_tokens(window)
        if window_tokens > max_window_tokens:
            raise ValueError(
                "Online EAGLE window exceeds max_window_tokens: "
                f"request={window.get('request_id')!r} tokens={window_tokens} limit={max_window_tokens}"
            )
        if batch and batch_tokens + window_tokens > max_tokens:
            batches.append(batch)
            batch = []
            batch_tokens = 0
        batch.append(window)
        batch_tokens += window_tokens
    if batch:
        batches.append(batch)
    return batches


def _offload_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """Release gradients and move optimizer state to CPU."""
    optimizer.zero_grad(set_to_none=True)
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                state[name] = value.detach().cpu()


def _load_window_group(
    windows: list[dict[str, Any]],
    capture_dir: Path,
    device: torch.device,
) -> Iterator[tuple[dict[str, Any], dict[str, torch.Tensor]]]:
    """Load independent histories without crossing teacher-forcing boundaries."""
    for window in windows:
        yield window, _load_batch(capture_dir / window["path"], device)


def _evaluate(
    model: nn.Module,
    capture_dir: Path,
    windows: list[dict[str, Any]],
    num_speculative_tokens: int,
    max_tokens_per_micro_batch: int,
    max_window_tokens: int,
    loss_config,
) -> OnlineEagleEvaluation:
    """Evaluate mean loss and next-token agreement on the supplied windows."""
    weighted_loss = 0.0
    loss_tokens = 0.0
    correct = 0.0
    total = 0.0
    model.eval()
    with torch.no_grad():
        for window_group in _pack_windows(windows, max_tokens_per_micro_batch, max_window_tokens):
            device = next(model.parameters()).device
            for window, batch in _load_window_group(window_group, capture_dir, device):
                path = capture_dir / window["path"]
                with _forward_context(device), _sdpa_kernel_context(device):
                    _, loss, metrics = model(
                        **batch,
                        ttt_steps=num_speculative_tokens,
                        loss_config=loss_config,
                    )
                _require_finite_tensor(loss, label=f"evaluation loss for {path}")
                supervised_tokens = float(batch["loss_mask"].sum())
                weighted_loss += float(loss) * supervised_tokens
                loss_tokens += supervised_tokens
                for index in range(num_speculative_tokens):
                    correct_value = metrics[f"full_acc_{index}_sum"]
                    total_value = metrics[f"full_acc_{index}_total"]
                    _require_finite_tensor(correct_value, label=f"evaluation agreement numerator for {path}")
                    _require_finite_tensor(total_value, label=f"evaluation agreement denominator for {path}")
                    correct += float(correct_value)
                    total += float(total_value)
    if loss_tokens == 0:
        raise ValueError("Online EAGLE holdout contains no supervised tokens")
    return OnlineEagleEvaluation(
        mean_loss=weighted_loss / loss_tokens,
        agreement=correct / total if total else 0.0,
    )


def _candidate_state(model: nn.Module, *, serving_dtype: torch.dtype) -> dict[str, torch.Tensor]:
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    state = {}
    for name, value in model.state_dict().items():
        if name in trainable_names:
            if value.is_floating_point() and value.dtype != serving_dtype:
                raise ValueError(
                    f"Online EAGLE candidate tensor {name} has dtype {value.dtype}, expected {serving_dtype}"
                )
            _require_finite_tensor(value, label=f"candidate tensor {name}")
            state[name] = value.detach().cpu().contiguous()
    return state


def candidate_is_acceptable(
    *,
    incumbent_loss: float,
    candidate_loss: float,
    incumbent_agreement: float,
    candidate_agreement: float,
    max_loss_increase: float,
    max_agreement_decrease: float,
) -> bool:
    """Apply the declared same-holdout loss and agreement tolerances."""
    values = (incumbent_loss, candidate_loss, incumbent_agreement, candidate_agreement)
    return (
        all(math.isfinite(value) for value in values)
        and candidate_loss <= incumbent_loss + max_loss_increase
        and candidate_agreement >= incumbent_agreement - max_agreement_decrease
    )


def _restore_rng_states(saved_state: Mapping[str, Any], device: torch.device) -> None:
    """Restore trainer RNG state after a device-mapped checkpoint load."""
    torch.set_rng_state(saved_state["torch_rng_state"].cpu())
    cuda_rng_state = saved_state.get("cuda_rng_state")
    if device.type == "cuda" and cuda_rng_state is not None:
        # CUDA generator state is represented by a CPU ByteTensor. ``torch.load``
        # maps it to ``device`` with the optimizer tensors, so move it back first.
        torch.cuda.set_rng_state(cuda_rng_state.cpu(), device=device)
    python_rng_state = saved_state.get("python_rng_state")
    if python_rng_state is not None:
        random.setstate(python_rng_state)


def _save_candidate(
    model,
    optimizer: torch.optim.Optimizer,
    output_dir: Path,
    lineage: dict[str, Any],
    *,
    device: torch.device,
    serving_dtype: torch.dtype,
    master_parameters: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.tmp-{uuid4().hex}")
    staging.mkdir()
    try:
        model.save_pretrained(
            staging,
            state_dict=_candidate_state(model, serving_dtype=serving_dtype),
            safe_serialization=True,
            max_shard_size="5GB",
        )
        weights_path = staging / HF_WEIGHT_FILENAME
        if not weights_path.exists():
            raise RuntimeError("Online EAGLE candidate unexpectedly produced sharded weights")
        trainer_state_path = staging / TRAINER_STATE_FILENAME
        torch.save(
            {
                "format": _TRAINER_STATE_FORMAT,
                "format_version": _TRAINER_STATE_VERSION,
                "master_parameter_dtype": str(torch.float32),
                "served_parameter_dtype": str(serving_dtype),
                "master_parameters": dict(master_parameters),
                "optimizer": optimizer.state_dict(),
                **_capture_rng_states(device),
            },
            trainer_state_path,
        )
        manifest = {
            "format": _CANDIDATE_FORMAT,
            "format_version": 1,
            "complete": True,
            **lineage,
            "weights_path": weights_path.name,
            "trainer_state_path": trainer_state_path.name,
        }
        (staging / ONLINE_EAGLE_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))
        if output_dir.exists():
            raise FileExistsError(f"Online EAGLE candidate already exists: {output_dir}")
        os.replace(staging, output_dir)
        return {**manifest, "path": str(output_dir / ONLINE_EAGLE_MANIFEST_FILENAME)}
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor) and value.device != device:
                state[name] = value.to(device=device)


def _capture_rng_states(device: torch.device) -> dict[str, Any]:
    return {
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "python_rng_state": random.getstate(),
    }


class OnlineEagleTrainerRuntime:
    """Persistent accepted draft model, FP32 masters, optimizer, and RNG state."""

    def __init__(self, job: OnlineEagleTrainingJob, capture_dir: Path):
        seed = job.seed + job.step
        random.seed(seed)
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.serving_dtype = _serving_dtype(self.device)
        self.model = _prepare_model(
            job.initial_draft_model,
            capture_dir,
            self.device,
        )
        _convert_trainable_parameters(self.model, torch.float32)
        self.trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = torch.optim.AdamW(self.trainable, lr=job.training.learning_rate)
        from speculators.losses import resolve_loss_config  # noqa: PLC0415

        self.loss_config = resolve_loss_config("kl_div", "fused" if self.device.type == "cuda" else "eager")
        _offload_optimizer_state(self.optimizer)
        _require_finite_trainable_state(self.model, self.optimizer)

    def restore(self, checkpoint_dir: Path) -> None:
        """Restore the last published draft and private optimizer state."""
        trainer_state_path = checkpoint_dir / TRAINER_STATE_FILENAME
        saved_state = torch.load(trainer_state_path, map_location=self.device, weights_only=False)
        if (
            saved_state.get("format") != _TRAINER_STATE_FORMAT
            or saved_state.get("format_version") != _TRAINER_STATE_VERSION
            or saved_state.get("master_parameter_dtype") != str(torch.float32)
            or saved_state.get("served_parameter_dtype") != str(self.serving_dtype)
        ):
            raise ValueError(f"Incompatible online EAGLE trainer state: {trainer_state_path}")
        master_parameters = saved_state.get("master_parameters")
        if not isinstance(master_parameters, Mapping):
            raise ValueError(f"Online EAGLE trainer state has no FP32 masters: {trainer_state_path}")
        self.model.load_state_dict(load_file(checkpoint_dir / HF_WEIGHT_FILENAME), strict=False)
        _restore_trainable_master_state(
            self.model,
            master_parameters,
            serving_dtype=self.serving_dtype,
        )
        self.optimizer.load_state_dict(saved_state["optimizer"])
        _offload_optimizer_state(self.optimizer)
        _restore_rng_states(saved_state, self.device)
        _require_finite_trainable_state(self.model, self.optimizer)

    def _snapshot(self) -> dict[str, Any]:
        return {
            "master_parameters": _capture_trainable_master_state(self.model),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            **_capture_rng_states(self.device),
        }

    def _restore_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        _convert_trainable_parameters(self.model, torch.float32)
        _load_trainable_master_state(self.model, snapshot["master_parameters"])
        self.optimizer.load_state_dict(snapshot["optimizer"])
        _offload_optimizer_state(self.optimizer)
        _restore_rng_states(snapshot, self.device)
        _require_finite_trainable_state(self.model, self.optimizer)

    def _evaluate_serving_weights(
        self,
        capture_dir: Path,
        windows: list[dict[str, Any]],
        job: OnlineEagleTrainingJob,
    ) -> OnlineEagleEvaluation:
        masters = _capture_trainable_master_state(self.model)
        _convert_trainable_parameters(self.model, self.serving_dtype)
        try:
            return _evaluate(
                self.model,
                capture_dir,
                windows,
                job.num_speculative_tokens,
                job.training.max_tokens_per_micro_batch,
                job.training.max_window_tokens,
                self.loss_config,
            )
        finally:
            _convert_trainable_parameters(self.model, torch.float32)
            _restore_trainable_master_state(self.model, masters, serving_dtype=self.serving_dtype)

    def update(self, job: OnlineEagleTrainingJob) -> OnlineEagleUpdateResult:
        """Train and accept one candidate, restoring the incumbent on local rejection or failure."""
        started_at = time.perf_counter()
        capture_dir = Path(job.capture_dir)
        output_dir = Path(job.output_dir)
        manifest = _load_and_validate_capture(capture_dir)
        if manifest["target_revision"] != job.target_revision:
            raise ValueError(
                "Online EAGLE capture target mismatch: "
                f"expected {job.target_revision}, got {manifest['target_revision']}"
            )
        training = job.training
        train_windows, holdout_windows = partition_capture_windows(
            manifest["windows"],
            step=job.step,
            holdout_fraction=training.holdout_fraction,
            min_train_sequences=training.min_train_sequences,
            min_holdout_sequences=training.min_holdout_sequences,
        )
        incumbent_snapshot = self._snapshot()
        try:
            _refresh_target_owned_weights(self.model, capture_dir)
            incumbent = self._evaluate_serving_weights(capture_dir, holdout_windows, job)
            _move_optimizer_state(self.optimizer, self.device)
            for parameter_group in self.optimizer.param_groups:
                parameter_group["lr"] = training.learning_rate
            weighted_train_loss = 0.0
            train_loss_tokens = 0.0
            self.model.train()
            for epoch in range(training.epochs_per_update):
                epoch_windows = list(train_windows)
                random.shuffle(epoch_windows)
                packed_windows = _pack_windows(
                    epoch_windows,
                    training.max_tokens_per_micro_batch,
                    training.max_window_tokens,
                )
                for group_index, window_group in enumerate(packed_windows):
                    group_supervised_tokens = sum(int(window["supervised_tokens"]) for window in window_group)
                    if group_supervised_tokens <= 0:
                        raise ValueError("Online EAGLE training microbatch contains no supervised tokens")
                    self.optimizer.zero_grad(set_to_none=True)
                    for window, batch in _load_window_group(window_group, capture_dir, self.device):
                        path = capture_dir / window["path"]
                        supervised_tokens = int(batch["loss_mask"].sum())
                        if supervised_tokens != int(window["supervised_tokens"]):
                            raise ValueError(
                                "Online EAGLE supervised-token count does not match its manifest: "
                                f"{path} has {supervised_tokens}, manifest says {window['supervised_tokens']}"
                            )
                        with _forward_context(self.device), _sdpa_kernel_context(self.device):
                            _, loss, _metrics = self.model(
                                **batch,
                                ttt_steps=job.num_speculative_tokens,
                                loss_config=self.loss_config,
                            )
                        _require_finite_tensor(loss, label=f"training loss for {path}")
                        (loss * (supervised_tokens / group_supervised_tokens)).backward()
                        weighted_train_loss += float(loss.detach()) * supervised_tokens
                        train_loss_tokens += supervised_tokens
                        print(
                            json.dumps(
                                {
                                    "kind": "online_eagle_microbatch",
                                    "step": job.step,
                                    "epoch": epoch,
                                    "group": group_index,
                                    "request_id": window["request_id"],
                                    "path": str(path),
                                    "supervised_tokens": supervised_tokens,
                                    "loss": float(loss.detach()),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        self.trainable,
                        max_norm=1.0,
                        error_if_nonfinite=True,
                    )
                    _require_finite_tensor(gradient_norm, label=f"gradient norm for group {group_index}")
                    self.optimizer.step()
                    _require_finite_trainable_state(self.model, self.optimizer)
                    print(
                        json.dumps(
                            {
                                "kind": "online_eagle_optimizer_step",
                                "step": job.step,
                                "epoch": epoch,
                                "group": group_index,
                                "supervised_tokens": group_supervised_tokens,
                                "gradient_norm": float(gradient_norm),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            if train_loss_tokens == 0:
                raise ValueError("Online EAGLE training partition contains no supervised tokens")
            _offload_optimizer_state(self.optimizer)
            master_parameters = _capture_trainable_master_state(self.model)
            candidate = self._evaluate_serving_weights(capture_dir, holdout_windows, job)
            max_loss_increase = training.max_validation_loss_increase
            max_agreement_decrease = training.max_validation_agreement_decrease
            accepted = candidate_is_acceptable(
                incumbent_loss=incumbent.mean_loss,
                candidate_loss=candidate.mean_loss,
                incumbent_agreement=incumbent.agreement,
                candidate_agreement=candidate.agreement,
                max_loss_increase=max_loss_increase,
                max_agreement_decrease=max_agreement_decrease,
            )
            result = OnlineEagleUpdateResult(
                accepted=accepted,
                step=job.step,
                parent_draft_revision=job.parent_draft_revision,
                trained_against_target_revision=manifest["target_revision"],
                train_sequences=len(train_windows),
                holdout_sequences=len(holdout_windows),
                train_loss=weighted_train_loss / train_loss_tokens,
                incumbent_holdout_loss=incumbent.mean_loss,
                candidate_holdout_loss=candidate.mean_loss,
                incumbent_holdout_agreement=incumbent.agreement,
                candidate_holdout_agreement=candidate.agreement,
                holdout_loss_increase=candidate.mean_loss - incumbent.mean_loss,
                holdout_agreement_decrease=incumbent.agreement - candidate.agreement,
                max_validation_loss_increase=max_loss_increase,
                max_validation_agreement_decrease=max_agreement_decrease,
                duration_seconds=time.perf_counter() - started_at,
            )
            if not accepted:
                self._restore_snapshot(incumbent_snapshot)
                return result

            draft_revision = f"draft-step-{job.step}"
            _convert_trainable_parameters(self.model, self.serving_dtype)
            try:
                _save_candidate(
                    self.model,
                    self.optimizer,
                    output_dir,
                    {
                        "draft_revision": draft_revision,
                        "initial_source_identity": job.initial_draft_model.source_identity,
                        "parent_draft_revision": job.parent_draft_revision,
                        "trained_against_target_revision": manifest["target_revision"],
                        "training": asdict(training),
                        "metrics": result.to_mapping(),
                    },
                    device=self.device,
                    serving_dtype=self.serving_dtype,
                    master_parameters=master_parameters,
                )
            finally:
                _convert_trainable_parameters(self.model, torch.float32)
                _restore_trainable_master_state(
                    self.model,
                    master_parameters,
                    serving_dtype=self.serving_dtype,
                )
            return replace(result, draft_revision=draft_revision)
        except BaseException:
            self._restore_snapshot(incumbent_snapshot)
            raise
