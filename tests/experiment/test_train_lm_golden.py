# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The inline-protocol DCLM experiment must resolve to the same *materialized
config* as the ``default_train`` version, so readable inline code and the executed
config cannot drift.

We compare the fully-resolved config (every ``InputName`` / ``THIS_OUTPUT_PATH`` /
``VersionedValue`` placeholder substituted), with content-hash path segments
normalized to ``-<H>``. This is stronger than comparing the content-version hash
*and* survives the move to explicit ``name@version`` addressing: the path scheme
may change, but the computation the training job receives must not.
"""

import json
import re

from fray.cluster import ResourceConfig
from marin.execution.executor import Executor, executor_context
from marin.utilities.json_encoder import CustomJsonEncoder

from experiments.defaults import default_train
from experiments.pretraining_datasets.dclm import dclm_mixture_config_llama3
from experiments.simple_train_config import SimpleTrainConfig
from experiments.tutorials.dclm_1b_1x_inline import build, llama_1_4b_dclm

# A content-version hash is ``{name}-{md5[:6]}``; normalize the 6-hex suffix so the
# snapshot is stable across the identity change (content hash -> explicit version).
_HASH_SUFFIX = re.compile(r"-[0-9a-f]{6}(?=[/\"]|$)")


def _materialized_config(step) -> str:
    """Resolve a step's config through the executor and return it as a
    path-hash-normalized JSON string."""
    prefix = "gs://marin-golden"
    executor = Executor(prefix=prefix, executor_info_base_path=f"{prefix}/experiments")
    executor.compute_version(step, is_pseudo_dep=False)
    resolved = executor.configs[step]
    as_json = json.dumps(resolved, sort_keys=True, cls=CustomJsonEncoder)
    return _HASH_SUFFIX.sub("-<H>", as_json)


def _old_dclm_step():
    training_config = SimpleTrainConfig(
        train_seq_len=2048,
        resources=ResourceConfig.with_tpu("v4-128"),
        train_batch_size=256,
        num_train_steps=int(28.8e9) // (256 * 2048),
        learning_rate=3e-3,
        weight_decay=0.033,
        min_lr_ratio=0.1,
        warmup=5000,
        z_loss_weight=1e-4,
    )
    return default_train(
        name="dclm_1b_1x_how_to",
        tokenized=dclm_mixture_config_llama3,
        model_config=llama_1_4b_dclm,
        train_config=training_config,
        tags=["HOWTOS", "DCLM_1B_1X"],
    )


def test_inline_dclm_matches_default_train(monkeypatch):
    # W&B group is a runtime binding: default_train reads it from the env and folds
    # it into the config, the inline helper does not. Clear it so the comparison
    # is on the protocol axes only.
    monkeypatch.delenv("WANDB_GROUP", raising=False)

    with executor_context():
        old = _old_dclm_step()
        new = build()

    assert old.name == new.name, f"\n old: {old.name}\n new: {new.name}"
    old_config = _materialized_config(old)
    new_config = _materialized_config(new)
    assert old_config == new_config, f"\n old: {old_config}\n new: {new_config}"
