# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from tests.cluster.vllm.snowball import (
    SNOWBALL,
    SNOWBALL_NATIVE_TPU,
    RepresentativeGolden,
    read_exported_levanter_tpu_contract,
    read_native_tpu_contract,
    read_native_tpu_goldens,
    read_prompt_fixture,
    read_representative_goldens,
    read_vllm_tpu_contract,
    read_vllm_tpu_goldens,
)


def test_native_tpu_contract_is_frozen_and_matches_cell() -> None:
    contract = read_native_tpu_contract()

    assert contract.schema_version == 1
    assert contract.name == "snowball-native-levanter-tpu-v2"
    assert contract.max_probability_error == 0.2
    assert contract.prompt_fixture_digest == "47863868cbfe336739c8097535f113f4d2dae4954f772eb91511c911433596e8"
    assert contract.canonical_golden_digest == "d695624cc411d7bf79a6c2e28f34538437985a9e07612d2a07a5095128ff4b2d"
    assert contract.requested_attention == SNOWBALL_NATIVE_TPU.requested_attention
    assert contract.effective_attention == SNOWBALL_NATIVE_TPU.effective_attention
    assert contract.requested_moe == SNOWBALL_NATIVE_TPU.requested_moe
    assert contract.effective_moe == SNOWBALL_NATIVE_TPU.effective_moe
    assert contract.discovery.clean_process_count == 3
    assert contract.discovery.same_process_repeat_count == 3
    assert contract.discovery.repeatability_probability_error == 0.0
    assert contract.discovery.max_probability_error == pytest.approx(0.09182652544810627)
    assert contract.discovery.max_probability_error < contract.max_probability_error


def test_exported_levanter_tpu_contract_is_independently_frozen() -> None:
    contract = read_exported_levanter_tpu_contract()

    assert contract.name == "snowball-exported-levanter-tpu-v1"
    assert contract.backend == "levanter-exported"
    assert contract.platform == "tpu"
    assert contract.parameter_digest == SNOWBALL.export_sha256
    assert contract.max_probability_error == 0.2
    assert contract.discovery.clean_process_count == 3
    assert contract.discovery.same_process_repeat_count == 3
    assert contract.discovery.repeatability_probability_error == 0.0
    assert contract.discovery.max_probability_error == pytest.approx(0.09182652544810627)
    assert contract.discovery.summary_sha256 == "7512ab4e899454865fe992cda22664b36807dadd591775c08121a1e58398ab4f"


def test_vllm_tpu_contract_is_frozen_for_the_production_compiler_policy() -> None:
    contract = read_vllm_tpu_contract()

    assert contract.name == "snowball-vllm-tpu-v2"
    assert dict(contract.max_probability_error_by_bucket) == {
        256: 0.03,
        1024: 0.06,
        4096: 0.09,
        16384: 0.7,
        32768: 0.5,
    }
    assert contract.discovery.clean_process_count == 3
    assert contract.discovery.same_process_repeat_count == 3
    assert contract.discovery.repeatability_probability_error == 0.0
    assert contract.discovery.max_probability_error == pytest.approx(0.5840894216142387)
    assert contract.fork_source_revisions == (
        ("vllm", "40cfab43c208dd9e762e6752fca887bdde69d1c9"),
        ("tpu-inference", "e9a360537e2c9f8cffaecdbba124685061630550"),
    )
    assert contract.discovery.summary_sha256 == "3a15113cdef58d24ca383c17026df82c959501e7324852666fc7e22539786164"
    assert contract.runtime_fingerprint_digest == "b39059fe5a26fa9888a79525de7055dd0efb9dd26bd108242afeeebe4fc2ef8a"


def test_vllm_tpu_golden_is_frozen_in_the_shared_schema() -> None:
    gpu_goldens = read_representative_goldens()
    tpu_goldens = read_vllm_tpu_goldens()

    assert len(tpu_goldens) == len(gpu_goldens) == 64
    gpu_by_id = {golden.id: golden for golden in gpu_goldens}
    tpu_by_id = {golden.id: golden for golden in tpu_goldens}
    assert tpu_by_id.keys() == gpu_by_id.keys()
    assert all(len(golden.top_logprobs) == 25 for golden in tpu_goldens)
    assert any(tpu_by_id[case_id].top_logprobs != gpu_by_id[case_id].top_logprobs for case_id in gpu_by_id)


def test_native_tpu_golden_is_frozen_in_the_shared_schema() -> None:
    gpu_goldens = read_representative_goldens()
    tpu_goldens = read_native_tpu_goldens()

    assert len(tpu_goldens) == len(gpu_goldens) == 64
    gpu_by_id = {golden.id: golden for golden in gpu_goldens}
    tpu_by_id = {golden.id: golden for golden in tpu_goldens}
    assert tpu_by_id.keys() == gpu_by_id.keys()
    assert all(len(golden.top_logprobs) == 25 for golden in tpu_goldens)
    assert any(tpu_by_id[case_id].top_logprobs != gpu_by_id[case_id].top_logprobs for case_id in gpu_by_id)


def test_read_prompt_fixture_verifies_content_digest(tmp_path) -> None:
    goldens = tuple(RepresentativeGolden(id=f"case-{index}", top_logprobs=()) for index in range(8))
    payload = {
        "tokenizer": "test-tokenizer",
        "tokenizer_revision": "revision",
        "cases": [{"id": golden.id, "prompt_token_ids": [index + 1]} for index, golden in enumerate(goldens)],
    }
    fixture_bytes = (json.dumps(payload, sort_keys=True) + "\n").encode()
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_bytes(fixture_bytes)
    digest = hashlib.sha256(fixture_bytes).hexdigest()

    fixture = read_prompt_fixture(goldens, fixture_uri=str(fixture_path), expected_sha256=digest)

    assert fixture.tokenizer == "test-tokenizer"
    assert len(fixture.cases) == 8
    assert len(fixture.batches) == 1

    with pytest.raises(ValueError, match="Prompt fixture SHA-256 mismatch"):
        read_prompt_fixture(goldens, fixture_uri=str(fixture_path), expected_sha256="0" * 64)
