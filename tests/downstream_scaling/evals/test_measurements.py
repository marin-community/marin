# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import math
from pathlib import Path

import pytest

from experiments.downstream_scaling.evals.algorithms import xtok_selection
from experiments.downstream_scaling.evals.measurements import entropy, gate_hits, kl, topk_logprobs


class FakeTokenizer:
    def encode(self, prompt: str) -> list[int]:
        assert prompt == "prompt"
        return [10, 11]


class FakeFlatLogprobs:
    def __init__(self, positions: list[list[tuple[int, float]]]) -> None:
        self.start_indices: list[int] = []
        self.end_indices: list[int] = []
        self.token_ids: list[int] = []
        self.logprobs: list[float] = []
        for position in positions:
            self.start_indices.append(len(self.token_ids))
            for token_id, logprob in position:
                self.token_ids.append(token_id)
                self.logprobs.append(logprob)
            self.end_indices.append(len(self.token_ids))


class FailingLLM:
    def generate(self, *_args, **_kwargs):
        raise RuntimeError("inference failed")


class FakeTokensPrompt:
    def __init__(self, *, prompt_token_ids: list[int]) -> None:
        self.prompt_token_ids = prompt_token_ids


TOKEN_PATH_ROW: topk_logprobs.TokenPathRow = {
    "id": "prompt-0",
    "completion_index": 0,
    "steps": [
        {
            "bytes_hex": "61",
            "tokens_a": [20, 21],
            "tokens_b": [30],
        },
        {
            "bytes_hex": "",
            "tokens_a": [2],
            "tokens_b": [3, 4],
        },
    ],
}


def _flat_logprobs() -> FakeFlatLogprobs:
    return FakeFlatLogprobs(
        [
            [],
            [(11, -0.1), (50, -0.2), (51, -0.3)],
            [(20, -0.2), (20, -0.2), (60, -0.4)],
            [(21, -0.3), (61, -0.5), (62, -0.7)],
            [(2, -0.4), (70, -0.6), (71, -0.8)],
        ]
    )


def _write_measurement_inputs(tmp_path: Path) -> tuple[str, str]:
    prompts_path = tmp_path / "prompts.jsonl.gz"
    token_paths_path = tmp_path / "token_paths.jsonl.gz"
    with gzip.open(prompts_path, "wt") as f:
        f.write(json.dumps({"id": "prompt-0", "prompt": "prompt"}) + "\n")
    with gzip.open(token_paths_path, "wt") as f:
        f.write(json.dumps(TOKEN_PATH_ROW) + "\n")
    return str(prompts_path), str(token_paths_path)


def _failing_vllm(*_args):
    return FailingLLM(), dict, FakeTokensPrompt, FakeTokenizer()


def test_entropy_chunk_inference_failure_leaves_no_output(tmp_path, monkeypatch):
    prompts_path, token_paths_path = _write_measurement_inputs(tmp_path)
    output_path = tmp_path / "entropy.jsonl.gz"
    monkeypatch.setattr(entropy, "_load_vllm", _failing_vllm)

    with pytest.raises(RuntimeError, match="inference failed"):
        entropy._run_entropy_chunk(
            entropy.EntropyChunkSpec(chunk_id=0, chunk_start=0, chunk_end=1, output_path=str(output_path)),
            model_path="model",
            prompts_path=prompts_path,
            token_paths_path=token_paths_path,
            model=entropy._EngineConfig(
                max_model_len=None,
                gpu_memory_utilization=None,
                apply_rpa_block_size_patch=False,
            ),
            side=entropy.TokenPathSide.A,
            k=2,
            microbatch_size=1,
            tensor_parallel_size=1,
        )

    assert not output_path.exists()


def test_topk_chunk_inference_failure_leaves_no_output(tmp_path, monkeypatch):
    prompts_path, token_paths_path = _write_measurement_inputs(tmp_path)
    output_path = tmp_path / "topk.jsonl.gz"
    monkeypatch.setattr(topk_logprobs, "_load_vllm", _failing_vllm)

    with pytest.raises(RuntimeError, match="inference failed"):
        topk_logprobs._run_topk_logprobs_chunk(
            topk_logprobs.TopkLogprobsChunkSpec(
                chunk_id=0,
                chunk_start=0,
                chunk_end=1,
                output_path=str(output_path),
            ),
            model_path="model",
            prompts_path=prompts_path,
            token_paths_path=token_paths_path,
            model=topk_logprobs._EngineConfig(
                max_model_len=None,
                gpu_memory_utilization=None,
                apply_rpa_block_size_patch=False,
            ),
            side=topk_logprobs.TokenPathSide.A,
            k=2,
            microbatch_size=1,
            tensor_parallel_size=1,
        )

    assert not output_path.exists()


@pytest.mark.parametrize(
    ("side", "expected_ids", "expected_lengths"),
    [
        (topk_logprobs.TokenPathSide.A, [10, 11, 20, 21, 2], (2, 1)),
        (topk_logprobs.TokenPathSide.B, [10, 11, 30, 3, 4], (1, 2)),
    ],
)
def test_scoring_request_preserves_recorded_side_tokens(side, expected_ids, expected_lengths):
    request = topk_logprobs._prepare_scoring_request(TOKEN_PATH_ROW, "prompt", FakeTokenizer(), side)

    assert request.prompt_length == 2
    assert request.token_ids == expected_ids
    assert request.step_lengths == expected_lengths


def test_topk_value_uses_step_offsets_and_skips_actual_token_entry():
    request = topk_logprobs._prepare_scoring_request(
        TOKEN_PATH_ROW,
        "prompt",
        FakeTokenizer(),
        topk_logprobs.TokenPathSide.A,
    )

    value = topk_logprobs._topk_value(request, _flat_logprobs(), k=2)

    assert value == {
        "steps": [
            {
                "topk_ids": [[20, 60], [61, 62]],
                "topk_logprobs": [[-0.2, -0.4], [-0.5, -0.7]],
            },
            {
                "topk_ids": [[70, 71]],
                "topk_logprobs": [[-0.6, -0.8]],
            },
        ]
    }


def test_zero_step_completion_produces_empty_values():
    row: topk_logprobs.TokenPathRow = {
        "id": "prompt-0",
        "completion_index": 0,
        "steps": [],
    }
    request = topk_logprobs._prepare_scoring_request(
        row,
        "prompt",
        FakeTokenizer(),
        topk_logprobs.TokenPathSide.A,
    )
    prompt_logprobs = FakeFlatLogprobs(
        [
            [],
            [(11, -0.1), (50, -0.2), (51, -0.3)],
        ]
    )

    assert topk_logprobs._topk_value(request, prompt_logprobs, k=2) == {"steps": []}


def test_shannon_entropy_normalizes_topk_logprobs():
    value = entropy.shannon_entropy([math.log(0.25), math.log(0.75)])

    assert value == pytest.approx(-(0.25 * math.log(0.25) + 0.75 * math.log(0.75)))


def test_direct_and_chained_entropy_match_on_token_path_fixture():
    entropy_row: entropy.TokenPathRow = {
        "id": TOKEN_PATH_ROW["id"],
        "completion_index": TOKEN_PATH_ROW["completion_index"],
        "steps": TOKEN_PATH_ROW["steps"],
    }
    entropy_request = entropy._prepare_scoring_request(
        entropy_row,
        "prompt",
        FakeTokenizer(),
        entropy.TokenPathSide.A,
    )
    topk_request = topk_logprobs._prepare_scoring_request(
        TOKEN_PATH_ROW,
        "prompt",
        FakeTokenizer(),
        topk_logprobs.TokenPathSide.A,
    )
    prompt_logprobs = _flat_logprobs()

    direct = entropy._entropy_value(entropy_request, prompt_logprobs, k=2)
    chained = entropy._entropy_from_topk_value(topk_logprobs._topk_value(topk_request, prompt_logprobs, k=2))

    assert direct == chained


@pytest.mark.parametrize(
    "records",
    [
        [
            {"completion_index": 0, "value": {"entropy": [0.1]}},
            {"completion_index": 2, "value": {"entropy": [0.2]}},
        ],
        [
            {"completion_index": 0, "value": {"entropy": [0.1]}},
            {"completion_index": 0, "value": {"entropy": [0.2]}},
        ],
    ],
)
def test_statistic_aggregation_rejects_incomplete_completion_indices(records):
    with pytest.raises(ValueError, match="completion indices"):
        entropy._aggregate_statistic_row("prompt-0", iter(records), metadata={})


def _gate_hit_row(signals: list[float]) -> dict:
    return {
        "id": "prompt-0",
        "completion_index": 0,
        "steps": [{"bytes_hex": "61", "tokens_a": [1], "tokens_b": [2], "kl": signal} for signal in signals],
    }


@pytest.mark.parametrize(
    ("threshold", "expected_below"),
    [
        # 0.0 is the pure-advisor anchor: no signal is below it.
        (0.0, 0),
        # The gate hands a signal equal to the threshold to the advisor, so
        # only 0.25 counts here.
        (0.5, 1),
        # 1e9 is the pure-decoder anchor for any finite signal.
        (1e9, 3),
    ],
)
def test_gate_hits_counts_signals_strictly_below_threshold(threshold, expected_below):
    record = gate_hits._gate_hit_record(_gate_hit_row([0.25, 0.5, 0.75]), field="kl", threshold=threshold)

    assert record["value"] == {"n_steps": 3, "n_below": expected_below}


def test_gate_hits_zero_step_completion_counts_zero():
    record = gate_hits._gate_hit_record(_gate_hit_row([]), field="kl", threshold=1.0)

    assert record["value"] == {"n_steps": 0, "n_below": 0}


def test_gate_hits_aggregation_rejects_incomplete_completion_indices():
    records = [
        {"completion_index": 0, "value": {"n_steps": 2, "n_below": 1}},
        {"completion_index": 2, "value": {"n_steps": 2, "n_below": 0}},
    ]

    with pytest.raises(ValueError, match="completion indices"):
        gate_hits._aggregate_statistic_row("prompt-0", iter(records), metadata={})


@pytest.mark.parametrize("side", [kl.TokenPathSide.A, kl.TokenPathSide.B])
def test_kl_boundary_topk_steps_match_topk_statistic_boundaries(side):
    kl_row: kl.TokenPathRow = {
        "id": TOKEN_PATH_ROW["id"],
        "completion_index": TOKEN_PATH_ROW["completion_index"],
        "steps": TOKEN_PATH_ROW["steps"],
    }
    kl_request = kl._prepare_scoring_request(kl_row, "prompt", FakeTokenizer(), side)
    topk_request = topk_logprobs._prepare_scoring_request(
        TOKEN_PATH_ROW,
        "prompt",
        FakeTokenizer(),
        topk_logprobs.TokenPathSide(side.value),
    )

    boundary_steps = kl._boundary_topk_steps(kl_request, _flat_logprobs(), k=2)
    topk_steps = topk_logprobs._topk_value(topk_request, _flat_logprobs(), k=2)["steps"]

    assert boundary_steps == [
        {"topk_ids": step["topk_ids"][0], "topk_logprobs": step["topk_logprobs"][0]} for step in topk_steps
    ]


def test_step_kl_matches_hand_computed_union_floor_value():
    step_a = {"topk_ids": [1, 2], "topk_logprobs": [-0.5, -1.5]}
    step_b = {"topk_ids": [2, 3], "topk_logprobs": [-0.2, -2.0]}

    # Union {1, 2, 3} with floors a=-1.5 and b=-2.0:
    # KL(log_softmax([-0.5, -1.5, -1.5]) || log_softmax([-2.0, -0.2, -2.0])).
    assert kl._step_kl(step_a, step_b, 1.0) == pytest.approx(0.7288056643539165)
    assert kl._step_kl(step_a, step_a, 1.0) == 0.0
    # Same inputs with both sides sharpened by 1/0.7 before the softmax.
    assert kl._step_kl(step_a, step_b, 0.7) == pytest.approx(1.4426539179206337)


def _topk_entries(step: dict) -> list[dict]:
    return [
        {"token_id": token_id, "logit": logprob}
        for token_id, logprob in zip(step["topk_ids"], step["topk_logprobs"], strict=True)
    ]


def test_step_kl_matches_kl_bytes_union_on_shared_byte_vocab():
    # Ids below 256 are their own single byte, so byte keys and token ids coincide.
    vocab = xtok_selection.Vocab(
        token_bytes=(*(bytes([value]) for value in range(256)), None),
        piece_ids={bytes([value]): value for value in range(256)},
        max_piece_len=1,
        eos_id=256,
    )
    step_a = {"topk_ids": [65, 66, 67], "topk_logprobs": [-0.3, -1.2, -2.5]}
    step_b = {"topk_ids": [66, 68], "topk_logprobs": [-0.1, -2.4]}

    expected = xtok_selection.kl_bytes_union(
        xtok_selection.candidates(vocab, _topk_entries(step_a)),
        xtok_selection.candidates(vocab, _topk_entries(step_b)),
    )

    assert kl._step_kl(step_a, step_b, 1.0) == pytest.approx(expected)
