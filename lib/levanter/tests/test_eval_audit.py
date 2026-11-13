# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import haliax as hax
import jax
import jax.numpy as jnp
import pytest

from levanter.data.dataset import ListAsyncDataset
from levanter.eval import TaggedEvaluator, eval_model, lm_eval_loss
from levanter.eval_audit import evaluation_input_audit, initial_eval_match
from levanter.models.lm_model import LmExample
from levanter.testing.helpers import use_test_mesh
from levanter.testing.toy_lm import ToyLmConfig, ToyLmHeadModel


def test_reference_match_survives_batch_changes_and_rejects_changed_tokens():
    with use_test_mesh(tensor_parallelism=1) as mesh:
        pos = hax.Axis("position", 8)
        examples = [LmExample.causal(hax.named(jnp.arange(i, i + 8, dtype=jnp.int32), pos)) for i in range(4)]
        model = ToyLmHeadModel.init(hax.Axis("vocab", 32), ToyLmConfig(), key=jax.random.key(0))
        results = []
        for batch_size in (1, 2):
            batch = hax.Axis("batch", batch_size)
            evaluator = TaggedEvaluator(
                EvalBatch=batch,
                tagged_eval_sets=[(ListAsyncDataset(examples), ["heldout"])],
                loss_fn=partial(lm_eval_loss, EvalBatch=batch, mp=None),
                device_mesh=mesh,
                axis_mapping={"batch": "data"},
            )
            results.append(
                {
                    "metrics": eval_model(evaluator, model, prefix="eval"),
                    "input_audit": evaluation_input_audit(evaluator, (7,)),
                }
            )
        match = initial_eval_match(results[0], results[1]["metrics"], results[1]["input_audit"], 1e-5)
        assert match["passed"]
        assert match["input_audit"]["special_token_counts"] == {"7": 4}
        changed_audit = dict(results[1]["input_audit"], tokens_sha256="different-corpus")
        with pytest.raises(ValueError, match="inputs do not match"):
            initial_eval_match(results[0], results[1]["metrics"], changed_audit, 1e-5)
        with pytest.raises(ValueError, match="loss mismatch"):
            initial_eval_match(
                results[0], {"eval/loss": results[0]["metrics"]["eval/loss"] + 0.1}, results[1]["input_audit"], 1e-5
            )
