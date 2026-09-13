# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import math
import random
from dataclasses import replace
from types import SimpleNamespace

import pytest
from thalas.execution.context import executor_context
from thalas.execution.executor import compute_output_path

from experiments.downstream_scaling.evals import (
    run_delphi_gsm8k_joint_decode_avg_kl_unnormalized_add_llama as legacy_runner,
)
from experiments.downstream_scaling.evals import run_delphi_gsm8k_joint_decode_replay_unnormalized_add_llama as runner
from experiments.downstream_scaling.evals.algorithms.xtok_selection import Vocab
from experiments.downstream_scaling.evals.measurements import joint_decode_replay as replay

VOCAB_A = Vocab(
    token_bytes=(b"a", b"b", b"c", b"ab", None),
    piece_ids={b"a": 0, b"b": 1, b"c": 2, b"ab": 3},
    max_piece_len=2,
    eos_id=4,
)
VOCAB_B = Vocab(
    token_bytes=(b"a", b"b", b"c", None),
    piece_ids={b"a": 0, b"b": 1, b"c": 2},
    max_piece_len=1,
    eos_id=3,
)
# At temperature 2, the floor-filled union distributions are (4,1,1)/6
# and (1,9,1)/11. Logit averaging at w=1/2 gives (2,3,1)/6.
A_TOPK = [{"token_id": 0, "logit": 2 * math.log(4)}, {"token_id": 1, "logit": 0.0}]
B_TOPK = [{"token_id": 1, "logit": 2 * math.log(9)}, {"token_id": 2, "logit": 0.0}]
PATH_STEP = {"bytes_hex": "61", "tokens_a": [0], "tokens_b": [0]}
ALL_STATISTICS = tuple(replay.ReplayStatistic)
REPLAY_VALUES = {
    "kl": (2 / 3) * math.log(2) - math.log(3) / 6,
    "reverse_kl": -math.log(2) / 3 + math.log(3) / 2,
    "total_variation": 1 / 3,
    "jensen_shannon": 0.5 * ((2 / 3) * math.log(4 / 3) + math.log(0.5) / 6 + math.log(2 / 3) / 3 + math.log(1.5) / 2),
    "hellinger_squared": 1 - math.sqrt(2) / 3 - math.sqrt(3) / 6 - 1 / 6,
}


def write_jsonl(path, rows):
    with gzip.open(path, "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


@pytest.mark.parametrize(
    ("rule", "weight", "expected"),
    [
        (replay.XtokSelectionRule.AVG_LOGITS, 0.5, {0: 1 / 3, 1: 1 / 2, 2: 1 / 6}),
        (replay.XtokSelectionRule.BYTES_UNION, 0.5, {b"a": 1 / 3, b"b": 1 / 2, b"c": 1 / 6}),
        (replay.XtokSelectionRule.UNNORMALIZED_ADD, 2.0, {0: 4 / 86, 1: 81 / 86, 2: 1 / 86}),
        (replay.XtokSelectionRule.AVG_PROBS, 0.5, {0: 50 / 132, 1: 65 / 132, 2: 17 / 132}),
        (replay.XtokSelectionRule.AVG_PROBS, 0.0, {0: 4 / 6, 1: 1 / 6, 2: 1 / 6}),
        (replay.XtokSelectionRule.AVG_PROBS, 1.0, {0: 1 / 11, 1: 9 / 11, 2: 1 / 11}),
    ],
)
def test_sampling_logprobs_matches_hand_calculated_distribution(rule, weight, expected):
    a_topk = A_TOPK
    if rule is replay.XtokSelectionRule.BYTES_UNION:
        # The byte rule must use the shared candidate filter before finding floors.
        a_topk = [*a_topk, {"token_id": 99, "logit": 100.0}, {"token_id": 2, "logit": -math.inf}]
    actual = replay._sampling_logprobs(
        rule.value,
        a_topk,
        B_TOPK,
        advisor_weight=weight,
        temperature=2.0,
        vocab_a=VOCAB_A,
        vocab_b=VOCAB_B,
        prefix_credit=1.0,
    )
    assert actual == pytest.approx({key: math.log(probability) for key, probability in expected.items()}, abs=1e-12)


def test_anchored_logprobs_uses_prefix_mass_and_missing_mass_floor():
    # B assigns a=2/3, ab=1/3. At credit=1/2 its masses are ab=2/3,
    # a=1, c=0 (floor 1/3). Mixing with A's logits at w=1/2 gives
    # weights proportional to ab=2*sqrt(2), a=sqrt(3), c=1.
    total = 2 * math.sqrt(2) + math.sqrt(3) + 1
    actual = replay._sampling_logprobs(
        replay.XtokSelectionRule.ANCHORED_PREFIX_MASS.value,
        [{"token_id": 3, "logit": math.log(4)}, {"token_id": 0, "logit": 0.0}, {"token_id": 2, "logit": 0.0}],
        [{"token_id": 0, "logit": math.log(2)}, {"token_id": 3, "logit": 0.0}],
        advisor_weight=0.5,
        temperature=1.0,
        vocab_a=VOCAB_A,
        vocab_b=VOCAB_A,
        prefix_credit=0.5,
    )
    assert actual == pytest.approx(
        {
            b"ab": math.log(2 * math.sqrt(2) / total),
            b"a": math.log(math.sqrt(3) / total),
            b"c": math.log(1 / total),
        },
        abs=1e-12,
    )


@pytest.mark.parametrize(
    ("log_p", "log_q", "expected"),
    [
        (
            {0: math.log(0.75), 1: math.log(0.25)},
            {0: math.log(0.5), 1: math.log(0.5)},
            0.75 * math.log(1.5) + 0.25 * math.log(0.5),
        ),
        ({0: -math.log(2), 1: -math.log(2)}, {0: 0.0, 1: -1000.0}, 500 - math.log(2)),
    ],
)
def test_kl_matches_forward_value_even_when_destination_probability_underflows(log_p, log_q, expected):
    assert replay._statistic_values(log_p, log_q, (replay.ReplayStatistic.KL,)) == pytest.approx(
        {"kl": expected}, abs=1e-12
    )


class ReplayDecoder:
    def __init__(self, select_token, batches):
        self.select_token = select_token
        self.batches = iter(batches)
        self.prompt_batches = []
        self.forced_batches = []

    def generate(self, prompts_a, prompts_b):
        self.prompt_batches.append((prompts_a, prompts_b))
        order, texts = next(self.batches)
        forced = []
        for index in order:
            tokens_a, tokens_b = self.select_token(A_TOPK, B_TOPK, rng=random.Random(0), request_index=index)
            forced.append((index, tokens_a, tokens_b))
        self.forced_batches.append(forced)
        return [SimpleNamespace(text=text) for text in texts]


def replay_sampling():
    return replay.JointDecodeReplaySamplingConfig(
        max_tokens=8,
        advisor_max_tokens=16,
        top_k_a=2,
        top_k_b=2,
        seed=0,
        selection_rule=replay.XtokSelectionRule.BYTES_UNION,
        temperature=2.0,
    )


@pytest.mark.parametrize(
    "statistics",
    [ALL_STATISTICS, (replay.ReplayStatistic.KL,), (replay.ReplayStatistic.JENSEN_SHANNON,)],
    ids=["all", "kl-only", "without-kl"],
)
def test_replay_chunks_keep_interleaved_weights_paths_and_outputs_aligned(tmp_path, statistics):
    token_paths = tmp_path / replay.TOKEN_PATHS_FILENAME
    write_jsonl(
        token_paths,
        [
            {
                "id": "p1",
                "completion_index": 1,
                "advisor_weight": 0.5,
                "steps": [
                    {"bytes_hex": "6162", "tokens_a": [3], "tokens_b": [0, 1]},
                    {"bytes_hex": "", "tokens_a": [4], "tokens_b": [3]},
                ],
            },
            {
                "id": "p0",
                "completion_index": 0,
                "advisor_weight": 0.0,
                "steps": [{"bytes_hex": "63", "tokens_a": [2], "tokens_b": [2]}],
            },
            {"id": "p1", "completion_index": 0, "advisor_weight": 0.0, "steps": [PATH_STEP]},
        ],
    )
    write_jsonl(
        tmp_path / "completions.jsonl.gz",
        [
            {"id": "p0", "completions": [{"text": "c"}]},
            {"id": "p1", "completions": [{"text": "a"}, {"text": "ab"}]},
        ],
    )
    requests = {}
    sampling = replay_sampling()
    decoder = ReplayDecoder(
        replay._make_replay_select_token(sampling, statistics, VOCAB_A, VOCAB_B, requests),
        [((0, 1, 0), ["ab", "c"]), ((0,), ["a"])],
    )
    num_records = replay._validate_token_paths(str(tmp_path), sampling.selection_rule)
    chunks = replay._chunk_specs(str(tmp_path), num_records, 2)
    for chunk in chunks:
        replay.write_replay_chunk(
            chunk,
            decoder=decoder,
            prompts={"p0": "A0", "p1": "A1"},
            advisor_prompts={"p0": "B0", "p1": "B1"},
            completions={("p0", 0): "c", ("p1", 0): "a", ("p1", 1): "ab"},
            token_paths_path=str(token_paths),
            selection_rule=sampling.selection_rule,
            statistics=statistics,
            requests=requests,
        )
    records = [row for chunk in chunks for row in read_jsonl(chunk.output_path)]
    assert [(row["id"], row["completion_index"]) for row in records] == [("p1", 1), ("p0", 0), ("p1", 0)]
    expected_keys = {statistic.value for statistic in statistics}
    assert all(record["value"].keys() == expected_keys for record in records)
    for statistic in statistics:
        assert records[0]["value"][statistic.value] == pytest.approx([REPLAY_VALUES[statistic.value]] * 2, abs=1e-12)
        assert [row["value"][statistic.value] for row in records[1:]] == [[0.0], [0.0]]
    row = replay._aggregate_statistic_row("p1", iter([records[2], records[0]]), metadata={})
    assert row["values"] == [records[2]["value"], records[0]["value"]]
    assert decoder.prompt_batches == [(["A1", "A0"], ["B1", "B0"]), (["A1"], ["B1"])]
    assert decoder.forced_batches == [
        [(0, [3], [0, 1]), (1, [2], [2]), (0, [4], [3])],
        [(0, [0], [0])],
    ]


@pytest.mark.parametrize(
    ("order", "text"),
    [
        ((0,), "a"),
        ((0, 0, 0), "a"),
        ((0, 0), "wrong text"),
    ],
    ids=["ended-early", "ran-past-path", "text-mismatch"],
)
def test_replay_mismatch_does_not_write_chunk(tmp_path, order, text):
    statistics = ALL_STATISTICS
    token_paths = tmp_path / replay.TOKEN_PATHS_FILENAME
    write_jsonl(
        token_paths,
        [
            {
                "id": "p0",
                "completion_index": 0,
                "advisor_weight": 0.5,
                "steps": [PATH_STEP, {"bytes_hex": "", "tokens_a": [4], "tokens_b": [3]}],
            }
        ],
    )
    requests = {}
    sampling = replay_sampling()
    decoder = ReplayDecoder(
        replay._make_replay_select_token(sampling, statistics, VOCAB_A, VOCAB_B, requests),
        [(order, [text])],
    )
    output_path = tmp_path / "chunk.jsonl.gz"
    with pytest.raises(RuntimeError):
        replay.write_replay_chunk(
            replay.ReplayChunkSpec(0, 0, 1, str(output_path)),
            decoder=decoder,
            prompts={"p0": "A0"},
            advisor_prompts={"p0": "B0"},
            completions={("p0", 0): "a"},
            token_paths_path=str(token_paths),
            selection_rule=sampling.selection_rule,
            statistics=statistics,
            requests=requests,
        )
    assert not output_path.exists()


@pytest.mark.parametrize(
    ("rule", "fields", "error"),
    [
        (replay.XtokSelectionRule.AVG_LOGITS, {}, TypeError),
        (replay.XtokSelectionRule.AVG_LOGITS, {"advisor_weight": math.nan}, ValueError),
        (replay.XtokSelectionRule.AVG_LOGITS, {"advisor_weight": 1.5}, ValueError),
        (replay.XtokSelectionRule.UNNORMALIZED_ADD, {"advisor_weight": -0.1}, ValueError),
    ],
    ids=["missing", "nonfinite", "averaging-out-of-range", "negative-alpha"],
)
def test_token_path_reader_rejects_invalid_recorded_weight(tmp_path, rule, fields, error):
    path = tmp_path / replay.TOKEN_PATHS_FILENAME
    write_jsonl(path, [{"id": "p0", "completion_index": 0, "steps": [PATH_STEP], **fields}])
    with pytest.raises(error):
        list(replay.read_token_path_rows(str(path), rule))


@pytest.mark.parametrize("indices", [(0, 0, 1), (0,)], ids=["duplicate", "missing-final-completion"])
def test_source_validation_rejects_incomplete_or_duplicate_sidecar(tmp_path, indices):
    write_jsonl(
        tmp_path / "completions.jsonl.gz",
        [{"id": "p0", "completions": [{"text": "a"}, {"text": "a"}]}],
    )
    write_jsonl(
        tmp_path / replay.TOKEN_PATHS_FILENAME,
        [{"id": "p0", "completion_index": index, "advisor_weight": 0.5, "steps": [PATH_STEP]} for index in indices],
    )
    with pytest.raises(ValueError):
        replay._validate_token_paths(str(tmp_path), replay.XtokSelectionRule.AVG_LOGITS)


def test_statistics_match_two_outcome_values_and_symmetry():
    log_p = {0: math.log(0.75), 1: math.log(0.25)}
    log_q = {0: math.log(0.5), 1: math.log(0.5)}
    expected = {
        "kl": 0.75 * math.log(1.5) + 0.25 * math.log(0.5),
        "reverse_kl": 0.5 * math.log(2 / 3) + 0.5 * math.log(2),
        "total_variation": 0.25,
        "jensen_shannon": (
            0.5 * (0.75 * math.log(6 / 5) + 0.25 * math.log(2 / 3) + 0.5 * math.log(4 / 5) + 0.5 * math.log(4 / 3))
        ),
        "hellinger_squared": 1 - math.sqrt(0.375) - math.sqrt(0.125),
    }
    assert replay._statistic_values(log_p, log_q, ALL_STATISTICS) == pytest.approx(expected, abs=1e-12)
    assert replay._statistic_values(log_q, log_p, ALL_STATISTICS) == pytest.approx(
        {**expected, "kl": expected["reverse_kl"], "reverse_kl": expected["kl"]}, abs=1e-12
    )
    assert replay._statistic_values(log_p, log_p, ALL_STATISTICS) == pytest.approx(
        dict.fromkeys(expected, 0.0), abs=1e-12
    )


def test_all_statistics_remain_finite_when_probabilities_underflow():
    actual = replay._statistic_values({0: -math.log(2), 1: -math.log(2)}, {0: 0.0, 1: -1000.0}, ALL_STATISTICS)
    assert actual == pytest.approx(
        {
            "kl": 500 - math.log(2),
            "reverse_kl": math.log(2),
            "total_variation": 0.5,
            "jensen_shannon": 0.75 * math.log(4 / 3),
            "hellinger_squared": 1 - 1 / math.sqrt(2),
        },
        abs=1e-12,
    )


def test_kl_clamps_roundoff_but_rejects_larger_negative_values():
    # A one-outcome distribution perturbed just below its exact log probability, zero.
    assert replay._statistic_values({0: -1e-13}, {0: 0.0}, (replay.ReplayStatistic.KL,)) == {"kl": 0.0}
    with pytest.raises(ValueError, match="outside"):
        replay._statistic_values({0: -1e-5}, {0: 0.0}, (replay.ReplayStatistic.KL,))


def test_empty_recorded_path_writes_empty_statistic_arrays(tmp_path):
    token_paths = tmp_path / replay.TOKEN_PATHS_FILENAME
    write_jsonl(token_paths, [{"id": "p0", "completion_index": 0, "advisor_weight": 0.0, "steps": []}])
    requests = {}
    decoder = ReplayDecoder(
        replay._make_replay_select_token(replay_sampling(), ALL_STATISTICS, VOCAB_A, VOCAB_B, requests),
        [((), [""])],
    )
    chunk = replay.ReplayChunkSpec(0, 0, 1, str(tmp_path / "chunk.jsonl.gz"))
    replay.write_replay_chunk(
        chunk,
        decoder=decoder,
        prompts={"p0": "A0"},
        advisor_prompts={"p0": "B0"},
        completions={("p0", 0): ""},
        token_paths_path=str(token_paths),
        selection_rule=replay.XtokSelectionRule.BYTES_UNION,
        statistics=ALL_STATISTICS,
        requests=requests,
    )
    assert read_jsonl(chunk.output_path) == [
        {"id": "p0", "completion_index": 0, "value": {statistic.value: [] for statistic in ALL_STATISTICS}}
    ]


def test_worker_json_preserves_selection_and_rejects_unknown_statistics(tmp_path):
    config = replay.JointDecodeLocalWorkerConfig(
        decoder_model_path="/decoder",
        advisor_model_path="/advisor",
        prompts_path="/prompts",
        advisor_prompts_path="/advisor_prompts",
        alg_output_path="/completions",
        token_paths_path="/token_paths",
        sampling=replace(replay_sampling(), stop=("stop",)),
        statistics=(replay.ReplayStatistic.JENSEN_SHANNON, replay.ReplayStatistic.TOTAL_VARIATION),
        decoder_model=replay.JointDecodeModelConfig(),
        advisor_model=replay.JointDecodeModelConfig(),
        ledger_path="/ledger",
        poll_backoff=0.5,
        microbatch_size=4,
        barrier_timeout_s=60.0,
        owner="worker-0",
        placement=replay.joint_decode_tp1_placements(2)[0],
    )
    path = replay._write_child_config(tmp_path, config)
    assert replay._child_config_from_file(str(path)) == config
    with path.open() as f:
        data = json.load(f)
    data["statistics"] = ["unknown"]
    with path.open("w") as f:
        json.dump(data, f)
    with pytest.raises(ValueError):
        replay._child_config_from_file(str(path))


def measurement_step(statistics):
    with executor_context():
        return replay.JointDecodeReplay(
            decoder_model_path="/decoder",
            config=replay.JointDecodeReplayConfig(
                sampling=replay_sampling(),
                advisor_model_path="/advisor",
                decoder_model=replay.JointDecodeModelConfig(),
                advisor_model=replay.JointDecodeModelConfig(),
                execution=replay.JointDecodeExecutionConfig(worker_pools=()),
                statistics=statistics,
            ),
        ).make_statistic_step(name="test/joint_decode_replay", prompts_path="/prompts", alg_output_path="/completions")


def step_path(step, prefix):
    return compute_output_path(step.name, step.config, prefix=str(prefix))


def test_measurement_identity_versions_selection_independent_of_order(tmp_path):
    forward = (replay.ReplayStatistic.KL,)
    reverse = (replay.ReplayStatistic.REVERSE_KL,)
    together = forward + reverse
    paths = [step_path(measurement_step(statistics), tmp_path) for statistics in (forward, reverse, together)]
    assert len(set(paths)) == 3
    assert step_path(measurement_step(tuple(reversed(together))), tmp_path) == paths[-1]


@pytest.mark.parametrize("statistics", [(), (replay.ReplayStatistic.KL, replay.ReplayStatistic.KL)])
def test_measurement_rejects_empty_or_duplicate_selections(statistics):
    with pytest.raises(ValueError, match=r"nonempty.*duplicates"):
        measurement_step(statistics)


def test_runner_reuses_each_generation_batch_and_separates_measurement_outputs(tmp_path, monkeypatch):
    batches = ((*(i / 10.0 for i in range(13)), 2.0), (1.6, 3.0, 4.0, 5.0, 7.0, 10.0))
    monkeypatch.setattr(runner, "ALPHA_SWEEPS", batches)
    regions = ["us-central2"]
    with executor_context():
        actual = runner.build_run_steps(regions)
        pools = runner.sweep.make_worker_pools(
            tpu_types=list(runner.sweep.TPU_TYPES),
            worker_regions=regions,
            num_workers=runner.sweep.WORKERS_PER_TPU_TYPE,
        )
        expected_source_paths = set()
        legacy_paths = set()
        for alphas in batches:
            with monkeypatch.context() as source_config:
                source_config.setattr(runner.sweep, "ALPHAS", alphas)
                for grade in runner.sweep.build_run_steps(pools):
                    expected_source_paths.add(step_path(grade.config.completions_path.step, tmp_path))
                legacy_paths.update(step_path(step, tmp_path) for step in legacy_runner.build_run_steps(regions))
        monkeypatch.setattr(runner, "STATISTICS", (replay.ReplayStatistic.KL,))
        kl_only = runner.build_run_steps(regions)
    assert len(actual) == len(expected_source_paths)
    assert {step_path(step.config.alg_output_path.step, tmp_path) for step in actual} == expected_source_paths
    assert {step_path(step.config.alg_output_path.step, tmp_path) for step in kl_only} == expected_source_paths
    assert all(step.config.alg_output_path.block_on_step for step in actual)
    assert {step_path(step, tmp_path) for step in actual}.isdisjoint(step_path(step, tmp_path) for step in kl_only)
    assert {step_path(step, tmp_path) for step in kl_only}.isdisjoint(legacy_paths)
