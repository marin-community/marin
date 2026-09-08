# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate document loss logging on the frozen Snowball 67B-A2B export.

Run on one H100x8 node in cw-us-east-02a, colocated with the checkpoint::

    uv run iris --cluster=marin job run --no-wait --enable-extra-resources \
      --target-cluster cw-us-east-02a --gpu H100x8 --cpu 32 --memory 512g --disk 256g \
      --extra gpu --sync-package marin-core --sync-package marin-levanter \
      --timeout 3600 --job-name snowball-document-losses \
      -- python -m experiments.evaluation.snowball_document_losses

Nine fixed prompt prefixes exercise a padded final batch. Snowball currently ignores
segmented attention masks, so each example contains one document window.
"""

import dataclasses
import json
import logging
import os
from pathlib import Path

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
from levanter.data.dataset import ListAsyncDataset
from levanter.eval import TaggedEvaluator
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.lm_model import LmExample, LmHeadModel
from levanter.models.snowball import SnowballConfig
from levanter.utils.tree_utils import inference_mode
from marin.testing.inference.snowball import SNOWBALL, read_prompt_fixture, read_representative_goldens

logger = logging.getLogger(__name__)
BATCH = hax.Axis("batch", 8)
POSITION = hax.Axis("position", 256)
AXIS_MAPPING = {"batch": ("replica_dcn", "data", "expert")}


def loss_fn(model: LmHeadModel, example: LmExample):
    model = inference_mode(model, True)
    unweighted = dataclasses.replace(example, loss_weight=hax.ones_like(example.loss_weight))
    losses = model.compute_next_token_loss(unweighted, reduction=None, reduction_axis=()).array
    return losses, example.loss_weight.array, jnp.roll(example.tokens.array, -1, axis=-1)


def main():
    logging.basicConfig(level=logging.INFO)
    output_dir = Path(os.environ["IRIS_OUTPUT_DIR"])
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture = read_prompt_fixture(read_representative_goldens())
    cases = sorted(fixture.cases, key=lambda case: case.id)[:9]
    examples = []
    for index, case in enumerate(cases):
        token_ids = np.asarray(case.prompt_token_ids[: POSITION.size], dtype=np.int32)
        tokens = np.zeros(POSITION.size, dtype=np.int32)
        tokens[: len(token_ids)] = token_ids
        weights = np.zeros(POSITION.size, dtype=np.float32)
        weights[: len(token_ids) - 1] = 1.0
        # Exercise completion-only masking and fractional scoring weights.
        weights[: min(8, len(token_ids) // 2)] = 0.0
        if index == 1:
            weights *= 0.5
        examples.append(
            LmExample.causal(
                hax.named(jnp.asarray(tokens), POSITION),
                loss_weight=hax.named(jnp.asarray(weights), POSITION),
            )
        )

    converter = SnowballConfig(
        reference_checkpoint=SNOWBALL.export_uri, tokenizer=fixture.tokenizer
    ).hf_checkpoint_converter()
    config = dataclasses.replace(
        converter.config_from_hf_config(converter.default_hf_config),
        moe_implementation="sonic",
    )
    mesh = compact_grug_mesh()
    with jax.set_mesh(mesh), hax.axis_mapping(AXIS_MAPPING):
        logger.info("Loading Snowball from %s", SNOWBALL.export_uri)
        model = converter.load_pretrained(
            config.model_type, config=config, axis_mapping=AXIS_MAPPING, dtype=jnp.bfloat16
        )
        jax.block_until_ready(model)
        dataset = ListAsyncDataset(examples)
        kwargs = dict(
            EvalBatch=BATCH,
            tagged_eval_sets=[(dataset, ["snowball/frozen-prompts"])],
            loss_fn=loss_fn,
            axis_mapping=AXIS_MAPPING,
            device_mesh=mesh,
        )
        baseline = TaggedEvaluator(**kwargs).evaluate(model)
        output_path = output_dir / "document-losses.jsonl"
        logged = TaggedEvaluator(**kwargs, document_losses_path=str(output_path), shuffle=True).evaluate(model)
        np.testing.assert_allclose(logged.micro_avg_loss, baseline.micro_avg_loss, rtol=1e-5, atol=1e-5)
        rows = [json.loads(line) for line in output_path.read_text().splitlines()]
        assert len(rows) == len(examples), (len(rows), len(examples))
        by_index = {row["example_index"]: row for row in rows}
        assert set(by_index) == set(range(len(examples)))

        # Compare every stored record against the unreduced model loss, outside
        # TaggedEvaluator's aggregation and serialization path.
        compute = hax.named_jit(loss_fn, axis_resources=AXIS_MAPPING)
        for start in range(0, len(examples), BATCH.size):
            batch_examples = examples[start : start + BATCH.size]
            real_count = len(batch_examples)
            batch_examples += [examples[0]] * (BATCH.size - real_count)
            batch = jax.tree.map(lambda *xs: hax.stack(BATCH, xs), *batch_examples, is_leaf=hax.is_named_array)
            losses, weights, _ = compute(model, batch)
            losses, weights = np.asarray(losses), np.asarray(weights)
            for offset in range(real_count):
                row = by_index[start + offset]
                expected_sum = float(np.sum(losses[offset] * weights[offset], dtype=np.float64))
                expected_weight = float(np.sum(weights[offset], dtype=np.float64))
                assert row["dataset_index"] == 0
                assert row["dataset_tags"] == ["snowball/frozen-prompts"]
                assert row["segment_index"] == 0
                assert row["token_count"] == int(np.count_nonzero(weights[offset] > 0))
                np.testing.assert_allclose(row["token_weight"], expected_weight, rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(row["loss_sum"], expected_sum, rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(row["mean_loss"], expected_sum / expected_weight, rtol=1e-5, atol=1e-5)

        recovered_loss = sum(row["loss_sum"] for row in rows) / sum(row["token_weight"] for row in rows)
        np.testing.assert_allclose(recovered_loss, baseline.micro_avg_loss, rtol=1e-5, atol=1e-5)
        summary = {
            "checkpoint": SNOWBALL.export_uri,
            "prompt_ids": [case.id for case in cases],
            "tokenizer": fixture.tokenizer,
            "tokenizer_revision": fixture.tokenizer_revision,
            "max_tokens": POSITION.size,
            "documents": len(rows),
            "baseline_loss": baseline.micro_avg_loss,
            "logged_loss": logged.micro_avg_loss,
            "recovered_loss": recovered_loss,
            "validation": "passed",
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        logger.info("SNOWBALL_DOCUMENT_LOSSES_VALIDATED %s", json.dumps(summary))


if __name__ == "__main__":
    main()
