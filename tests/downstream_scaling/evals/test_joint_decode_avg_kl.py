# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import math
import random
from types import SimpleNamespace

import pytest

from experiments.downstream_scaling.evals.algorithms.xtok_selection import Vocab
from experiments.downstream_scaling.evals.measurements import joint_decode_avg_kl as kl

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
REPLAY_KL = (2 / 3) * math.log(2) - math.log(3) / 6


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
        (kl.XtokSelectionRule.AVG_LOGITS, 0.5, {0: 1 / 3, 1: 1 / 2, 2: 1 / 6}),
        (kl.XtokSelectionRule.BYTES_UNION, 0.5, {b"a": 1 / 3, b"b": 1 / 2, b"c": 1 / 6}),
        (kl.XtokSelectionRule.UNNORMALIZED_ADD, 2.0, {0: 4 / 86, 1: 81 / 86, 2: 1 / 86}),
        (kl.XtokSelectionRule.AVG_PROBS, 0.5, {0: 50 / 132, 1: 65 / 132, 2: 17 / 132}),
        (kl.XtokSelectionRule.AVG_PROBS, 0.0, {0: 4 / 6, 1: 1 / 6, 2: 1 / 6}),
        (kl.XtokSelectionRule.AVG_PROBS, 1.0, {0: 1 / 11, 1: 9 / 11, 2: 1 / 11}),
    ],
)
def test_sampling_logprobs_matches_hand_calculated_distribution(rule, weight, expected):
    a_topk = A_TOPK
    if rule is kl.XtokSelectionRule.BYTES_UNION:
        # The byte rule must use the shared candidate filter before finding floors.
        a_topk = [*a_topk, {"token_id": 99, "logit": 100.0}, {"token_id": 2, "logit": -math.inf}]
    actual = kl._sampling_logprobs(
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
    actual = kl._sampling_logprobs(
        kl.XtokSelectionRule.ANCHORED_PREFIX_MASS.value,
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
    assert kl._kl(log_p, log_q) == pytest.approx(expected, abs=1e-12)


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
    return kl.JointDecodeReplaySamplingConfig(
        max_tokens=8,
        advisor_max_tokens=16,
        top_k_a=2,
        top_k_b=2,
        seed=0,
        selection_rule=kl.XtokSelectionRule.BYTES_UNION,
        temperature=2.0,
    )


def test_replay_chunks_keep_interleaved_weights_paths_and_outputs_aligned(tmp_path):
    token_paths = tmp_path / kl.TOKEN_PATHS_FILENAME
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
        kl._make_replay_select_token(sampling, VOCAB_A, VOCAB_B, requests),
        [((0, 1, 0), ["ab", "c"]), ((0,), ["a"])],
    )
    num_records = kl._validate_token_paths(str(tmp_path), sampling.selection_rule)
    chunks = kl._chunk_specs(str(tmp_path), num_records, 2)
    for chunk in chunks:
        kl.write_replay_chunk(
            chunk,
            decoder=decoder,
            prompts={"p0": "A0", "p1": "A1"},
            advisor_prompts={"p0": "B0", "p1": "B1"},
            completions={("p0", 0): "c", ("p1", 0): "a", ("p1", 1): "ab"},
            token_paths_path=str(token_paths),
            selection_rule=sampling.selection_rule,
            requests=requests,
        )
    records = [row for chunk in chunks for row in read_jsonl(chunk.output_path)]
    assert [(row["id"], row["completion_index"]) for row in records] == [("p1", 1), ("p0", 0), ("p1", 0)]
    assert records[0]["value"]["kl"] == pytest.approx([REPLAY_KL, REPLAY_KL], abs=1e-12)
    assert [row["value"]["kl"] for row in records[1:]] == [[0.0], [0.0]]
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
    token_paths = tmp_path / kl.TOKEN_PATHS_FILENAME
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
        kl._make_replay_select_token(sampling, VOCAB_A, VOCAB_B, requests),
        [(order, [text])],
    )
    output_path = tmp_path / "chunk.jsonl.gz"
    with pytest.raises(RuntimeError):
        kl.write_replay_chunk(
            kl.ReplayChunkSpec(0, 0, 1, str(output_path)),
            decoder=decoder,
            prompts={"p0": "A0"},
            advisor_prompts={"p0": "B0"},
            completions={("p0", 0): "a"},
            token_paths_path=str(token_paths),
            selection_rule=sampling.selection_rule,
            requests=requests,
        )
    assert not output_path.exists()


@pytest.mark.parametrize(
    ("rule", "fields", "error"),
    [
        (kl.XtokSelectionRule.AVG_LOGITS, {}, TypeError),
        (kl.XtokSelectionRule.AVG_LOGITS, {"advisor_weight": math.nan}, ValueError),
        (kl.XtokSelectionRule.AVG_LOGITS, {"advisor_weight": 1.5}, ValueError),
        (kl.XtokSelectionRule.UNNORMALIZED_ADD, {"advisor_weight": -0.1}, ValueError),
    ],
    ids=["missing", "nonfinite", "averaging-out-of-range", "negative-alpha"],
)
def test_token_path_reader_rejects_invalid_recorded_weight(tmp_path, rule, fields, error):
    path = tmp_path / kl.TOKEN_PATHS_FILENAME
    write_jsonl(path, [{"id": "p0", "completion_index": 0, "steps": [PATH_STEP], **fields}])
    with pytest.raises(error):
        list(kl.read_token_path_rows(str(path), rule))


@pytest.mark.parametrize("indices", [(0, 0, 1), (0,)], ids=["duplicate", "missing-final-completion"])
def test_source_validation_rejects_incomplete_or_duplicate_sidecar(tmp_path, indices):
    write_jsonl(
        tmp_path / "completions.jsonl.gz",
        [{"id": "p0", "completions": [{"text": "a"}, {"text": "a"}]}],
    )
    write_jsonl(
        tmp_path / kl.TOKEN_PATHS_FILENAME,
        [{"id": "p0", "completion_index": index, "advisor_weight": 0.5, "steps": [PATH_STEP]} for index in indices],
    )
    with pytest.raises(ValueError):
        kl._validate_token_paths(str(tmp_path), kl.XtokSelectionRule.AVG_LOGITS)
