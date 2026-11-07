import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from skyrl_train.callbacks.base import TrainerState
from skyrl_train.callbacks.builtin import create_callback_from_config
from skyrl_train.callbacks.reference_likelihood import ReferenceExample, score_reference


class ScoringEngine:
    async def generate(self, request):
        # A deterministic serving-boundary fake: prompt tokens have a different
        # likelihood from answer tokens, so an offset error changes the result.
        probabilities = {1: -9.0, 2: -8.0, 10: -1.0, 11: -3.0, 12: -6.0}
        return {
            "prompt_logprobs": [
                [None, *[{token: probabilities[token], 99: -0.01} for token in tokens[1:]]]
                for tokens in request["prompt_token_ids"]
            ]
        }


@pytest.mark.asyncio
async def test_reference_callback_preserves_token_evidence_and_distinct_averages(tmp_path):
    source = tmp_path / "references.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"id": "short", "validation_domain": "facts", "reference_token_ids": [1, 2, 12], "reference_start": 2},
                {"id": "long", "validation_domain": "facts", "reference_token_ids": [1, 10, 11], "reference_start": 1},
            ]
        ),
        source,
    )
    callback = create_callback_from_config(
        {
            "type": "reference_likelihood",
            "reference_paths": [str(source)],
            "output_path": str(tmp_path / "scores"),
            "batch_size": 1,
        }
    )
    state = TrainerState(global_step=5, epoch=0, total_steps=20, num_steps_per_epoch=20)
    metrics = await callback.collect_evaluation(
        state,
        trainer=SimpleNamespace(inference_engine_client=ScoringEngine()),
        pool=None,
    )
    assert metrics["eval/reference/facts/example_mean_nll"] == 4.0
    assert metrics["eval/reference/facts/token_mean_nll"] == pytest.approx(10 / 3)
    records = [json.loads(line) for line in (tmp_path / "scores/step-5.jsonl").read_text().splitlines()]
    assert records == [
        {
            "step": 5,
            "id": "short",
            "domain": "facts",
            "token_ids": [1, 2, 12],
            "reference_start": 2,
            "token_logprobs": [-6.0],
            "nll": 6.0,
        },
        {
            "step": 5,
            "id": "long",
            "domain": "facts",
            "token_ids": [1, 10, 11],
            "reference_start": 1,
            "token_logprobs": [-1.0, -3.0],
            "nll": 2.0,
        },
    ]


@pytest.mark.parametrize("scores", [[None, {9: -1.0}], [None], [None, {2: float("nan")}]])
def test_reference_scoring_rejects_missing_or_misaligned_evidence(scores):
    with pytest.raises(ValueError):
        score_reference(ReferenceExample("example", "facts", (1, 2), 1), scores)
