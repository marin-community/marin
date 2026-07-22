# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Snowball model identity and representative inference goldens."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rigging.filesystem import StoragePath

from tests.cluster.vllm.backend_parity import ParityContract, ParityDiscovery, TokenScore

BATCH_SIZE = 8
TOP_K = 25
PROMPT_BUCKET_MAX_TOKENS = (256, 1024, 4096, 16384, 32768)
VLLM_MAX_MODEL_LEN = 32768
VLLM_MAX_NUM_BATCHED_TOKENS = 512
VLLM_HTTP_TIMEOUT = (30.0, 5 * 60.0)
TOKENIZER_FILE_PATTERNS = ("tokenizer*", "special_tokens*", "added_tokens*", "chat_template*")
# Shared serving-path bound; repeated-run measurements are recorded in #7354.
MAX_PROBABILITY_ERROR = 0.075

_RESOURCES = Path(__file__).parent / "resources"
# Frozen artifacts retain the identifiers of the June 67B training lineage.
_REPRESENTATIVE_GOLDEN_PATH = _RESOURCES / "june_tpu_67b_a2b_step_42150_representative_eval_golden.json"
_NATIVE_TPU_GOLDEN_PATH = _RESOURCES / "snowball_native_levanter_tpu_v2_golden.json"
# v1 remains checked in because its predeclared 0.08 holdout rejected. v2 was
# frozen from the final capture graph before its independent holdout.
_NATIVE_TPU_CONTRACT_PATH = _RESOURCES / "snowball_native_levanter_tpu_contract_v2.json"
_EXPORTED_LEVANTER_GPU_CONTRACT_PATH = _RESOURCES / "snowball_exported_levanter_gpu_contract_v1.json"
_EXPORTED_LEVANTER_TPU_CONTRACT_PATH = _RESOURCES / "snowball_exported_levanter_tpu_contract_v1.json"
# v1 was measured without Fray's production v6e scoped-VMEM policy. v2 is the
# production-policy snapshot and contract qualified through the standing path.
_VLLM_TPU_GOLDEN_PATH = _RESOURCES / "snowball_vllm_tpu_v2_golden.json"
_VLLM_TPU_CONTRACT_PATH = _RESOURCES / "snowball_vllm_tpu_contract_v2.json"
REPRESENTATIVE_GOLDEN_SHA256 = "d695624cc411d7bf79a6c2e28f34538437985a9e07612d2a07a5095128ff4b2d"
NATIVE_TPU_GOLDEN_SHA256 = "c45738171293b85d70db20fe5f2534fbbc29ddec6c59c84bdb004e9053fae0bb"
NATIVE_TPU_CONTRACT_SHA256 = "9329a6cc507777fc6d7e9609b1342c22921b5edddb10e93e81f580fe58195329"
EXPORTED_LEVANTER_GPU_CONTRACT_SHA256 = "cc4dd0a0d9db68dee7183c8678396cfdee52b5040aa97e8da9f743c955bc31b3"
EXPORTED_LEVANTER_TPU_CONTRACT_SHA256 = "d05e2527e6f7d6114dbede477fbb596a38d6a8f278ccc6ae8faf90cb7f24a60a"
VLLM_TPU_GOLDEN_SHA256 = "bca5486c824e1b7c4f83bf709722c0ccb16bbf28292766c058fb96932ac5b2e6"
VLLM_TPU_CONTRACT_SHA256 = "2c321540bf883c5de9451b952dd97df33a066c782e668022803f61197ff2bfc7"
PROMPT_FIXTURE_SHA256 = "47863868cbfe336739c8097535f113f4d2dae4954f772eb91511c911433596e8"
PROMPT_FIXTURE_URL = (
    "https://storage.googleapis.com/marin-public/test-data/vllm/e2e/representative-eval-prompts/"
    f"{PROMPT_FIXTURE_SHA256}.json"
)
TPU_PROMPT_FIXTURE_URI = (
    f"gs://marin-us-east5/test-data/vllm/e2e/representative-eval-prompts/{PROMPT_FIXTURE_SHA256}.json"
)


@dataclass(frozen=True)
class ModelLineage:
    """Immutable model identity shared by every platform-local copy."""

    checkpoint_step: int
    export_sha256: str

    @property
    def model_name(self) -> str:
        return f"snowball-step-{self.checkpoint_step}-bf16"


@dataclass(frozen=True)
class ModelLocation:
    """Platform-local inputs and outputs for one model lineage."""

    name: str
    lineage: ModelLineage
    run_root: str
    prompt_fixture_uri: str
    compilation_cache_dir: str
    export_uri: str | None

    @property
    def executor_info_path(self) -> str:
        return str(StoragePath(self.run_root) / ".executor_info")

    @property
    def checkpoint_path(self) -> str:
        return str(StoragePath(self.run_root) / "checkpoints" / f"step-{self.lineage.checkpoint_step}")


@dataclass(frozen=True)
class NativeLevanterCell:
    """One native Levanter platform configuration in the parity matrix."""

    location: ModelLocation
    parameter_dtype: str | None
    requested_attention: str
    effective_attention: str
    requested_moe: str
    effective_moe: str


@dataclass(frozen=True)
class ExportedLevanterCell:
    """One HF-export-loaded Levanter platform configuration."""

    location: ModelLocation
    requested_attention: str
    effective_attention: str
    requested_moe: str
    effective_moe: str


@dataclass(frozen=True)
class VllmCell:
    """One exported-checkpoint vLLM serving configuration."""

    location: ModelLocation
    tensor_parallel_size: int
    data_parallel_size: int
    requested_attention: str
    effective_attention: str
    requested_moe: str
    effective_moe: str


SNOWBALL = ModelLineage(
    checkpoint_step=42150,
    export_sha256="d819cbc63780bd866a942e47f9283cbd7932bbb237b52df527edd750c65be8f0",
)

SNOWBALL_GPU = ModelLocation(
    name="gpu",
    lineage=SNOWBALL,
    run_root=(
        "s3://marin-us-east-02a/marin/grug/"
        "moe_67b_a2b_d2560_ep1_rep8_bs1024_seq65536_sw2k_v4_2048_muon_cooldown_step39k-79ebf3"
    ),
    prompt_fixture_uri=PROMPT_FIXTURE_URL,
    compilation_cache_dir=(
        "s3://marin-us-east-02a/tmp/ttl=30d/compilation-cache/june-tpu-67b-a2b-step-42150-sonic-fa4-representative-v2"
    ),
    export_uri="s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b/step-42150/hf-bf16-vllm/d819cbc63780bd86/",
)

SNOWBALL_TPU = ModelLocation(
    name="tpu",
    lineage=SNOWBALL,
    run_root=(
        "gs://marin-us-east5/grug/moe_67b_a2b_d2560_ep1_rep8_bs1024_seq65536_sw2k_v4_2048_muon_cooldown_step39k-79ebf3"
    ),
    prompt_fixture_uri=TPU_PROMPT_FIXTURE_URI,
    compilation_cache_dir=("gs://marin-us-east5/tmp/ttl=30d/compilation-cache/snowball-step-42150-levanter-native-v1"),
    export_uri=(
        "gs://marin-us-east5/marin/exports/grug/june-67b-a2b/step-42150/hf-bf16-vllm/"
        "d819cbc63780bd866a942e47f9283cbd7932bbb237b52df527edd750c65be8f0"
    ),
)

SNOWBALL_NATIVE_GPU = NativeLevanterCell(
    location=SNOWBALL_GPU,
    parameter_dtype=None,
    requested_attention="gpu_fa4_cute",
    effective_attention="gpu_fa4_cute",
    requested_moe="sonic",
    effective_moe="sonic",
)

SNOWBALL_NATIVE_TPU = NativeLevanterCell(
    location=SNOWBALL_TPU,
    parameter_dtype="bfloat16",
    requested_attention="tpu_splash",
    effective_attention="tpu_splash",
    requested_moe="ring",
    # compact_grug_mesh() has expert=1, so non-local MoE implementations use local scatter.
    effective_moe="scatter",
)

SNOWBALL_EXPORTED_GPU = ExportedLevanterCell(
    location=SNOWBALL_GPU,
    requested_attention="gpu_fa4_cute",
    effective_attention="gpu_fa4_cute",
    requested_moe="sonic",
    effective_moe="sonic",
)

SNOWBALL_EXPORTED_TPU = ExportedLevanterCell(
    location=SNOWBALL_TPU,
    requested_attention="tpu_splash",
    effective_attention="tpu_splash",
    requested_moe="ring",
    # The serving mesh has no multi-device expert axis, so dispatch is local.
    effective_moe="scatter",
)

SNOWBALL_VLLM_TPU = VllmCell(
    location=SNOWBALL_TPU,
    # The 67B export cannot be replicated on 32-GiB chips; production TPU
    # serving shards one model across the complete v6e-8 slice.
    tensor_parallel_size=8,
    data_parallel_size=1,
    requested_attention="production",
    effective_attention="production",
    requested_moe="tpu-inference-grug",
    effective_moe="tpu-inference-grug",
)


@dataclass(frozen=True)
class RepresentativeGolden:
    id: str
    top_logprobs: tuple[TokenScore, ...]


@dataclass(frozen=True)
class RepresentativeCase:
    id: str
    prompt_token_ids: tuple[int, ...]
    top_logprobs: tuple[TokenScore, ...]


@dataclass(frozen=True)
class PromptBatch:
    max_tokens: int
    cases: tuple[RepresentativeCase, ...]


@dataclass(frozen=True)
class RepresentativePromptFixture:
    tokenizer: str
    tokenizer_revision: str
    cases: tuple[RepresentativeCase, ...]
    batches: tuple[PromptBatch, ...]


def read_representative_goldens() -> tuple[RepresentativeGolden, ...]:
    return _read_goldens(
        _REPRESENTATIVE_GOLDEN_PATH,
        expected_sha256=REPRESENTATIVE_GOLDEN_SHA256,
        label="Representative golden",
    )


def read_native_tpu_goldens() -> tuple[RepresentativeGolden, ...]:
    """Read the exact TPU snapshot created only after cross-slice stability."""
    return _read_goldens(
        _NATIVE_TPU_GOLDEN_PATH,
        expected_sha256=NATIVE_TPU_GOLDEN_SHA256,
        label="Native TPU golden",
    )


def read_vllm_tpu_goldens() -> tuple[RepresentativeGolden, ...]:
    """Read the exact TPU-vLLM snapshot created after clean-process stability."""
    return _read_goldens(
        _VLLM_TPU_GOLDEN_PATH,
        expected_sha256=VLLM_TPU_GOLDEN_SHA256,
        label="TPU vLLM golden",
    )


def _read_goldens(path: Path, *, expected_sha256: str, label: str) -> tuple[RepresentativeGolden, ...]:
    golden_bytes = path.read_bytes()
    actual_sha256 = hashlib.sha256(golden_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}")
    payload = json.loads(golden_bytes)
    return tuple(
        RepresentativeGolden(
            id=raw_case["id"],
            top_logprobs=tuple(
                TokenScore(logprob=float(score["logprob"]), token_id=score["token_id"])
                for score in raw_case["top_logprobs"]
            ),
        )
        for raw_case in payload["cases"]
    )


def read_native_tpu_contract() -> ParityContract:
    """Read the content-addressed contract frozen before the TPU holdout run."""
    contract_bytes = _NATIVE_TPU_CONTRACT_PATH.read_bytes()
    actual_sha256 = hashlib.sha256(contract_bytes).hexdigest()
    if actual_sha256 != NATIVE_TPU_CONTRACT_SHA256:
        raise ValueError(
            f"Native TPU contract SHA-256 mismatch: expected {NATIVE_TPU_CONTRACT_SHA256}, got {actual_sha256}"
        )
    payload = json.loads(contract_bytes)
    if payload["schema_version"] != 1:
        raise ValueError(f"Unsupported native TPU contract schema {payload['schema_version']}")
    discovery = ParityDiscovery(**payload.pop("discovery"))
    return ParityContract(
        schema_version=payload["schema_version"],
        name=payload["name"],
        backend="levanter-native",
        platform="tpu",
        max_probability_error=payload["max_probability_error"],
        parameter_digest=payload["logical_bf16_parameters_sha256"],
        model_config_digest=payload["model_config_digest"],
        prompt_fixture_digest=payload["prompt_fixture_sha256"],
        canonical_golden_digest=payload["canonical_golden_sha256"],
        requested_attention=payload["requested_attention"],
        effective_attention=payload["effective_attention"],
        requested_moe=payload["requested_moe"],
        effective_moe=payload["effective_moe"],
        mesh_shape=(("data", BATCH_SIZE),),
        discovery=discovery,
    )


def _read_standard_parity_contract(path: Path, *, expected_sha256: str, label: str) -> ParityContract:
    contract_bytes = path.read_bytes()
    actual_sha256 = hashlib.sha256(contract_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}")
    payload = json.loads(contract_bytes)
    if payload["schema_version"] != 1:
        raise ValueError(f"Unsupported {label.lower()} schema {payload['schema_version']}")
    discovery = ParityDiscovery(**payload.pop("discovery"))
    mesh_shape = tuple((str(axis), int(size)) for axis, size in payload.pop("mesh_shape"))
    bucket_bounds = tuple(
        (int(bucket), float(bound)) for bucket, bound in payload.pop("max_probability_error_by_bucket", ())
    )
    fork_source_revisions = tuple(
        (str(package), str(revision)) for package, revision in payload.pop("fork_source_revisions", ())
    )
    return ParityContract(
        discovery=discovery,
        mesh_shape=mesh_shape,
        max_probability_error_by_bucket=bucket_bounds,
        fork_source_revisions=fork_source_revisions,
        **payload,
    )


def read_exported_levanter_gpu_contract() -> ParityContract:
    """Read the exported-Levanter GPU contract frozen before its holdout."""
    return _read_standard_parity_contract(
        _EXPORTED_LEVANTER_GPU_CONTRACT_PATH,
        expected_sha256=EXPORTED_LEVANTER_GPU_CONTRACT_SHA256,
        label="Exported Levanter GPU contract",
    )


def read_exported_levanter_tpu_contract() -> ParityContract:
    """Read the exported-Levanter TPU contract frozen before its holdout."""
    return _read_standard_parity_contract(
        _EXPORTED_LEVANTER_TPU_CONTRACT_PATH,
        expected_sha256=EXPORTED_LEVANTER_TPU_CONTRACT_SHA256,
        label="Exported Levanter TPU contract",
    )


def read_vllm_tpu_contract() -> ParityContract:
    """Read the TPU-vLLM contract frozen before its distinct-slice holdout."""
    return _read_standard_parity_contract(
        _VLLM_TPU_CONTRACT_PATH,
        expected_sha256=VLLM_TPU_CONTRACT_SHA256,
        label="TPU vLLM contract",
    )


def read_prompt_fixture(
    expected_cases: tuple[RepresentativeGolden, ...],
    *,
    fixture_uri: str = PROMPT_FIXTURE_URL,
    expected_sha256: str = PROMPT_FIXTURE_SHA256,
) -> RepresentativePromptFixture:
    fixture_bytes = StoragePath(fixture_uri).read_bytes()
    actual_sha256 = hashlib.sha256(fixture_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Prompt fixture SHA-256 mismatch for {fixture_uri}: expected {expected_sha256}, got {actual_sha256}"
        )
    payload = json.loads(fixture_bytes)
    expected_by_id = {case.id: case for case in expected_cases}
    assert {case["id"] for case in payload["cases"]} == expected_by_id.keys()
    cases = tuple(
        RepresentativeCase(
            id=raw_case["id"],
            prompt_token_ids=tuple(raw_case["prompt_token_ids"]),
            top_logprobs=expected_by_id[raw_case["id"]].top_logprobs,
        )
        for raw_case in payload["cases"]
    )
    return RepresentativePromptFixture(
        tokenizer=payload["tokenizer"],
        tokenizer_revision=payload["tokenizer_revision"],
        cases=cases,
        batches=_prompt_batches(cases),
    )


def _prompt_batches(cases: tuple[RepresentativeCase, ...]) -> tuple[PromptBatch, ...]:
    batches = []
    remaining_cases = cases
    for max_tokens in PROMPT_BUCKET_MAX_TOKENS:
        bucket = tuple(
            sorted(
                (case for case in remaining_cases if len(case.prompt_token_ids) <= max_tokens), key=lambda case: case.id
            )
        )
        remaining_cases = tuple(case for case in remaining_cases if len(case.prompt_token_ids) > max_tokens)
        if len(bucket) % BATCH_SIZE:
            raise ValueError(
                f"Prompt bucket <= {max_tokens} has {len(bucket)} cases; expected full batches of {BATCH_SIZE}"
            )
        batches.extend(
            PromptBatch(max_tokens=max_tokens, cases=bucket[start : start + BATCH_SIZE])
            for start in range(0, len(bucket), BATCH_SIZE)
        )

    if remaining_cases:
        raise ValueError(f"Prompts exceed {PROMPT_BUCKET_MAX_TOKENS[-1]} tokens")
    return tuple(batches)


def pad_prompt_batch(batch: PromptBatch, eos_token_id: int) -> tuple[np.ndarray, np.ndarray]:
    token_ids = np.full((BATCH_SIZE, batch.max_tokens), eos_token_id, dtype=np.int32)
    last_token_indices = np.empty(BATCH_SIZE, dtype=np.int32)
    for row, case in enumerate(batch.cases):
        token_ids[row, : len(case.prompt_token_ids)] = case.prompt_token_ids
        last_token_indices[row] = len(case.prompt_token_ids) - 1
    return token_ids, last_token_indices
