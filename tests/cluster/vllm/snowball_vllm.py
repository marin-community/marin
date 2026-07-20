# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Capture Snowball observations through the standard vLLM serving path."""

import argparse
import hashlib
import uuid

from marin.inference.serving_backend import OPENAI_API_SUFFIX, ModelSpec, VllmBackend
from marin.inference.tpu_vllm_pins import fork_source_revision, tpu_inference_fork_ref, vllm_fork_ref
from marin.inference.vllm_server import IsolatedTpuVllm
from rigging.filesystem import StoragePath

from tests.cluster.vllm import backend_parity as backend_parity_module
from tests.cluster.vllm import snowball as snowball_module
from tests.cluster.vllm.backend_parity import (
    ObservationReport,
    RunProvenance,
    request_next_token_observation,
    source_digest,
)
from tests.cluster.vllm.snowball import (
    PROMPT_FIXTURE_SHA256,
    REPRESENTATIVE_GOLDEN_SHA256,
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

# Keep enough returned candidates to evaluate every token in the canonical GPU
# top-25 at long context lengths. The initial top-50 smoke exposed legitimate
# candidates falling outside that window at 16K and 32K; one 16K canonical
# token remained below rank 256 in the complete discovery run.
OBSERVATION_TOP_K = 1024
CAPTURE_SOURCE_DIGEST = source_digest(backend_parity_module.__file__, snowball_module.__file__, __file__)


def capture_vllm(
    cell: VllmCell,
    *,
    same_process_repeats: int = 1,
    batch_indices: tuple[int, ...] | None = None,
    case_ids: tuple[str, ...] | None = None,
    goldens: tuple[RepresentativeGolden, ...] | None = None,
    vllm_ref: str | None = None,
    tpu_inference_ref: str | None = None,
) -> ObservationReport:
    """Start the pinned TPU stack and capture selected OpenAI API requests."""
    if same_process_repeats <= 0:
        raise ValueError("same_process_repeats must be positive")
    if cell.location.name != "tpu":
        raise ValueError(f"Only the TPU vLLM cell is currently supported, got {cell.location.name}")
    if cell.location.export_uri is None:
        raise ValueError("TPU vLLM requires the verified regional export")

    goldens = read_representative_goldens() if goldens is None else goldens
    prompt_fixture = read_prompt_fixture(goldens, fixture_uri=cell.location.prompt_fixture_uri)
    selected_indices = tuple(range(len(prompt_fixture.batches))) if batch_indices is None else batch_indices
    if len(set(selected_indices)) != len(selected_indices) or any(
        index < 0 or index >= len(prompt_fixture.batches) for index in selected_indices
    ):
        raise ValueError(f"Invalid batch indices {selected_indices}")
    selected_case_ids = None if case_ids is None else set(case_ids)
    if selected_case_ids is not None:
        if len(selected_case_ids) != len(case_ids):
            raise ValueError(f"Duplicate case ids {case_ids}")
        known_case_ids = {case.id for case in prompt_fixture.cases}
        unknown_case_ids = selected_case_ids - known_case_ids
        if unknown_case_ids:
            raise ValueError(f"Unknown case ids {sorted(unknown_case_ids)}")

    config_bytes = (StoragePath(cell.location.export_uri) / "config.json").read_bytes()
    num_chips = cell.tensor_parallel_size * cell.data_parallel_size
    spec = ModelSpec(
        model=SNOWBALL.model_name,
        model_path=cell.location.export_uri,
        num_chips=num_chips,
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
            "1",
            # Numerical discovery must execute every prompt. Prefix-cache
            # behavior is a separate production-serving invariant; mixing a
            # cache miss with later hits makes same-process repeatability
            # measure two execution modes rather than kernel noise.
            "--no-enable-prefix-caching",
            "--max-logprobs",
            str(OBSERVATION_TOP_K),
        ),
    )

    observations = []
    runtime_fingerprint = None
    with backend.serve(spec) as served:
        runtime_fingerprint = served.environment.runtime_fingerprint()
        completions_url = f"{served.base_url}{OPENAI_API_SUFFIX}/completions"
        for repeat_index in range(same_process_repeats):
            for batch_index in selected_indices:
                batch = prompt_fixture.batches[batch_index]
                for case in batch.cases:
                    if selected_case_ids is not None and case.id not in selected_case_ids:
                        continue
                    observations.append(
                        request_next_token_observation(
                            completions_url,
                            served.model_id,
                            case_id=case.id,
                            prompt_token_ids=case.prompt_token_ids,
                            expected_top_logprobs=case.top_logprobs,
                            bucket_max_tokens=batch.max_tokens,
                            repeat_index=repeat_index,
                            backend_index=0,
                            returned_logprobs=OBSERVATION_TOP_K,
                            timeout=VLLM_HTTP_TIMEOUT,
                        )
                    )

    assert runtime_fingerprint is not None
    return ObservationReport(
        provenance=RunProvenance(
            backend="vllm",
            platform="tpu",
            process_id=uuid.uuid4().hex,
            code_digest=CAPTURE_SOURCE_DIGEST,
            parameter_digest=SNOWBALL.export_sha256,
            model_config_digest=hashlib.sha256(config_bytes).hexdigest(),
            prompt_fixture_digest=PROMPT_FIXTURE_SHA256,
            requested_attention=cell.requested_attention,
            effective_attention=cell.effective_attention,
            requested_moe=cell.requested_moe,
            effective_moe=cell.effective_moe,
            mesh_shape=(
                ("tensor", cell.tensor_parallel_size),
                ("data", cell.data_parallel_size),
                ("expert", 1),
            ),
            device_kind="v6e-8",
            golden_digest=REPRESENTATIVE_GOLDEN_SHA256,
            fork_source_revisions=(
                ("vllm", fork_source_revision(vllm_requirement, package="vllm")),
                (
                    "tpu-inference",
                    fork_source_revision(tpu_inference_requirement, package="tpu-inference"),
                ),
            ),
            runtime_fingerprint=runtime_fingerprint,
        ),
        observations=tuple(observations),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--same-process-repeats", type=int, default=1)
    parser.add_argument("--batch-index", action="append", type=int)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--vllm-ref")
    parser.add_argument("--tpu-inference-ref")
    args = parser.parse_args()

    report = capture_vllm(
        SNOWBALL_VLLM_TPU,
        same_process_repeats=args.same_process_repeats,
        batch_indices=None if args.batch_index is None else tuple(args.batch_index),
        case_ids=None if args.case_id is None else tuple(args.case_id),
        vllm_ref=args.vllm_ref,
        tpu_inference_ref=args.tpu_inference_ref,
    )
    StoragePath(args.output).write_bytes(report.to_json_bytes())


if __name__ == "__main__":
    main()
