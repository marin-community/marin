# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture production-shaped Snowball TPU-vLLM greedy behavior."""

import argparse
import dataclasses
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from marin.inference.serving_backend import OPENAI_API_SUFFIX, ModelSpec, VllmBackend
from marin.inference.tpu_vllm_pins import fork_source_revision, tpu_inference_fork_ref, vllm_fork_ref
from marin.inference.vllm_server import IsolatedTpuVllm, VllmRuntimeFingerprint
from rigging.filesystem import StoragePath

from tests.cluster.vllm import snowball as snowball_module
from tests.cluster.vllm.backend_parity import source_digest
from tests.cluster.vllm.snowball import (
    PROMPT_FIXTURE_SHA256,
    SNOWBALL,
    SNOWBALL_VLLM_TPU,
    VLLM_HTTP_TIMEOUT,
    VLLM_MAX_MODEL_LEN,
    VLLM_MAX_NUM_BATCHED_TOKENS,
    RepresentativeGolden,
    VllmCell,
    read_prompt_fixture,
    read_representative_goldens,
)

MAX_NUM_SEQS = 8
CONTINUATION_TOKENS = 8
SEQUENTIAL_REPEATS = 3
CONCURRENT_WAVES = 2
ORACLE_CASE_ID = "code-humaneval-01"
CAPTURE_SOURCE_DIGEST = source_digest(snowball_module.__file__, __file__)
PRODUCTION_BEHAVIOR_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ProductionCompletion:
    case_id: str
    wave: int
    token_ids: tuple[int, ...]
    cached_prompt_tokens: int


@dataclass(frozen=True)
class _CompletionResult:
    token_ids: tuple[int, ...]
    cached_prompt_tokens: int


@dataclass(frozen=True)
class ProductionBehaviorReport:
    parameter_digest: str
    model_config_digest: str
    prompt_fixture_digest: str
    code_digest: str
    prefix_caching: bool
    max_num_seqs: int
    tensor_parallel_size: int
    data_parallel_size: int
    fork_source_revisions: tuple[tuple[str, str], ...]
    runtime_fingerprint: VllmRuntimeFingerprint
    sequential: tuple[ProductionCompletion, ...]
    concurrent: tuple[ProductionCompletion, ...]
    schema_version: int = PRODUCTION_BEHAVIOR_SCHEMA_VERSION

    def to_json_bytes(self) -> bytes:
        return (
            json.dumps(dataclasses.asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()

    @classmethod
    def from_json_bytes(cls, payload: bytes) -> "ProductionBehaviorReport":
        raw = json.loads(payload)
        if raw["schema_version"] != PRODUCTION_BEHAVIOR_SCHEMA_VERSION:
            raise ValueError(f"Unsupported production behavior schema {raw['schema_version']}")
        return cls(
            parameter_digest=raw["parameter_digest"],
            model_config_digest=raw["model_config_digest"],
            prompt_fixture_digest=raw["prompt_fixture_digest"],
            code_digest=raw["code_digest"],
            prefix_caching=bool(raw["prefix_caching"]),
            max_num_seqs=int(raw["max_num_seqs"]),
            tensor_parallel_size=int(raw["tensor_parallel_size"]),
            data_parallel_size=int(raw["data_parallel_size"]),
            fork_source_revisions=tuple(tuple(revision) for revision in raw["fork_source_revisions"]),
            runtime_fingerprint=VllmRuntimeFingerprint.from_dict(raw["runtime_fingerprint"]),
            sequential=tuple(_completion_from_json(item) for item in raw["sequential"]),
            concurrent=tuple(_completion_from_json(item) for item in raw["concurrent"]),
            schema_version=int(raw["schema_version"]),
        )


def _completion_from_json(item: dict) -> ProductionCompletion:
    return ProductionCompletion(
        case_id=item["case_id"],
        wave=int(item["wave"]),
        token_ids=tuple(int(token_id) for token_id in item["token_ids"]),
        cached_prompt_tokens=int(item["cached_prompt_tokens"]),
    )


def _request_completion(
    completions_url: str,
    model_id: str,
    *,
    case_id: str,
    prompt_token_ids: tuple[int, ...],
    max_tokens: int,
    request_id: str,
) -> _CompletionResult:
    import requests  # noqa: PLC0415

    try:
        response = requests.post(
            completions_url,
            headers={"X-Request-Id": request_id},
            json={
                "model": model_id,
                "prompt": list(prompt_token_ids),
                "add_special_tokens": False,
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "return_token_ids": True,
            },
            timeout=VLLM_HTTP_TIMEOUT,
        )
        response.raise_for_status()
        payload = response.json()
        (choice,) = payload["choices"]
        assert choice["prompt_token_ids"] == list(prompt_token_ids), case_id
        token_ids = tuple(int(token_id) for token_id in choice["token_ids"])
        assert len(token_ids) == max_tokens, (case_id, token_ids)
        prompt_token_details = payload["usage"].get("prompt_tokens_details")
        if prompt_token_details is None or prompt_token_details.get("cached_tokens") is None:
            raise ValueError("vLLM did not report cached prompt tokens")
        return _CompletionResult(
            token_ids=token_ids,
            cached_prompt_tokens=int(prompt_token_details["cached_tokens"]),
        )
    except Exception as error:
        error.add_note(f"case={case_id} request={request_id}")
        raise


def capture_production_behavior(
    cell: VllmCell,
    *,
    goldens: tuple[RepresentativeGolden, ...] | None = None,
    vllm_ref: str | None = None,
    tpu_inference_ref: str | None = None,
) -> ProductionBehaviorReport:
    """Exercise cache miss/hit and concurrent production-shaped requests."""
    if cell.location.name != "tpu":
        raise ValueError(f"Only the TPU vLLM cell is supported, got {cell.location.name}")
    if cell.location.export_uri is None:
        raise ValueError("TPU vLLM requires the verified regional export")

    goldens = read_representative_goldens() if goldens is None else goldens
    prompt_fixture = read_prompt_fixture(goldens, fixture_uri=cell.location.prompt_fixture_uri)
    short_batch = prompt_fixture.batches[0]
    if short_batch.max_tokens != 256 or len(short_batch.cases) != MAX_NUM_SEQS:
        raise ValueError("The production concurrency oracle requires the complete 256-token batch")
    case_by_id = {case.id: case for case in short_batch.cases}
    oracle_case = case_by_id[ORACLE_CASE_ID]

    config_bytes = (StoragePath(cell.location.export_uri) / "config.json").read_bytes()
    spec = ModelSpec(
        model=SNOWBALL.model_name,
        model_path=cell.location.export_uri,
        num_chips=cell.tensor_parallel_size * cell.data_parallel_size,
        tensor_parallel_size=cell.tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=VLLM_MAX_MODEL_LEN,
        chat_template_content=None,
    )
    vllm_requirement = vllm_fork_ref() if vllm_ref is None else vllm_ref
    tpu_inference_requirement = tpu_inference_fork_ref() if tpu_inference_ref is None else tpu_inference_ref
    backend = VllmBackend(
        launcher=IsolatedTpuVllm(
            vllm_ref=vllm_requirement,
            tpu_inference_ref=tpu_inference_requirement,
        ),
        max_num_batched_tokens=VLLM_MAX_NUM_BATCHED_TOKENS,
        extra_args=(
            "--max-num-seqs",
            str(MAX_NUM_SEQS),
            "--enable-prefix-caching",
            "--enable-prompt-tokens-details",
        ),
    )

    sequential = []
    concurrent = []
    runtime_fingerprint = None
    with backend.serve(spec) as served:
        runtime_fingerprint = served.environment.runtime_fingerprint()
        completions_url = f"{served.base_url}{OPENAI_API_SUFFIX}/completions"
        for repeat in range(SEQUENTIAL_REPEATS):
            result = _request_completion(
                completions_url,
                served.model_id,
                case_id=oracle_case.id,
                prompt_token_ids=oracle_case.prompt_token_ids,
                max_tokens=CONTINUATION_TOKENS,
                request_id=f"sequential-{repeat}-{oracle_case.id}",
            )
            sequential.append(
                ProductionCompletion(
                    case_id=oracle_case.id,
                    wave=repeat,
                    token_ids=result.token_ids,
                    cached_prompt_tokens=result.cached_prompt_tokens,
                )
            )

        with ThreadPoolExecutor(max_workers=MAX_NUM_SEQS) as executor:
            for wave in range(CONCURRENT_WAVES):
                futures = [
                    (
                        case,
                        executor.submit(
                            _request_completion,
                            completions_url,
                            served.model_id,
                            case_id=case.id,
                            prompt_token_ids=case.prompt_token_ids,
                            max_tokens=1,
                            request_id=f"concurrent-{wave}-{case.id}",
                        ),
                    )
                    for case in short_batch.cases
                ]
                for case, future in futures:
                    result = future.result()
                    concurrent.append(
                        ProductionCompletion(
                            case_id=case.id,
                            wave=wave,
                            token_ids=result.token_ids,
                            cached_prompt_tokens=result.cached_prompt_tokens,
                        )
                    )

    assert runtime_fingerprint is not None
    observed_prefix_caching = sequential[0].cached_prompt_tokens == 0 and all(
        completion.cached_prompt_tokens > 0 for completion in sequential[1:]
    )

    return ProductionBehaviorReport(
        parameter_digest=SNOWBALL.export_sha256,
        model_config_digest=hashlib.sha256(config_bytes).hexdigest(),
        prompt_fixture_digest=PROMPT_FIXTURE_SHA256,
        code_digest=CAPTURE_SOURCE_DIGEST,
        prefix_caching=observed_prefix_caching,
        max_num_seqs=MAX_NUM_SEQS,
        tensor_parallel_size=cell.tensor_parallel_size,
        data_parallel_size=cell.data_parallel_size,
        fork_source_revisions=(
            ("vllm", fork_source_revision(vllm_requirement, package="vllm")),
            (
                "tpu-inference",
                fork_source_revision(tpu_inference_requirement, package="tpu-inference"),
            ),
        ),
        runtime_fingerprint=runtime_fingerprint,
        sequential=tuple(sequential),
        concurrent=tuple(concurrent),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--vllm-ref")
    parser.add_argument("--tpu-inference-ref")
    args = parser.parse_args()

    report = capture_production_behavior(
        SNOWBALL_VLLM_TPU,
        vllm_ref=args.vllm_ref,
        tpu_inference_ref=args.tpu_inference_ref,
    )
    StoragePath(args.output).write_bytes(report.to_json_bytes())


if __name__ == "__main__":
    main()
