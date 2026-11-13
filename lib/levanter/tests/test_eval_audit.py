# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
from functools import partial

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import optax
from iris.cluster.platforms.types import find_free_port
from iris.cluster.types import Entrypoint
from iris.jax.multigpu import IRIS_MULTIGPU_PROCESS_INDEX_ENV, MultiGpuHook
from jax.sharding import Mesh
from rigging.tunnel import terminate_process_group
from rigging.filesystem.storage_path import StoragePath

from levanter.data.dataset import ListAsyncDataset
from levanter.eval import TaggedEvaluator, eval_model, lm_eval_loss
from levanter.eval_audit import (
    InitialEvalMatchConfig,
    evaluation_input_audit,
    initial_eval_match,
    write_evaluation_result,
)
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.main.train_lm import TrainLmConfig, _load_lm_model_from_configured_source, check_initial_evaluation
from levanter.models.lm_model import LmExample
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.models.snowball_sft import SnowballSftConfig, reseed_snowball_head
from levanter.trainer import Trainer, TrainerConfig
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.utils.mesh import MeshConfig
from levanter.testing.helpers import use_test_mesh
from levanter.testing.toy_lm import ToyLmConfig, ToyLmHeadModel


def test_raw_gate_precedes_reseed_and_post_gate_records_the_intentional_shift(tmp_path):
    pos, batch, vocab = hax.Axis("position", 8), hax.Axis("batch", 1), hax.Axis("vocab", 32)
    examples = [GrugLmExample.causal(jnp.arange(8, dtype=jnp.int32))]
    data = LmDataConfig(
        tokenizer="passthrough",
        vocab_size=32,
        components={"heldout": DirectDatasetComponent(datasets={"validation": ListAsyncDataset(examples)})},
    )
    model_config = SnowballConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=8,
        sliding_window=4,
        initializer_std=0.3,
        attention_implementation="reference",
    )
    controls = SnowballSftConfig((30, 31), (2, 3))
    trainer_config = TrainerConfig(
        id="dual-initialization-gate",
        train_batch_size=1,
        per_device_parallelism=1,
        per_device_eval_parallelism=1,
        use_explicit_mesh_axes=True,
        mesh=MeshConfig(axes={"data": 1, "model": 1, "replica": 1, "context": 1, "expert": 1}),
        tracker=JsonFileTrackerConfig(output_path=str(tmp_path)),
        require_accelerator=False,
        log_jaxprs=False,
        log_xla_hlo=False,
    )
    key = jax.random.key(123)
    with Trainer(
        trainer_config, optax.sgd(0), SnowballLMHeadModel.compute_next_token_loss, add_default_hooks=False
    ) as trainer:
        model = model_config.build(vocab, key=key)
        evaluator = TaggedEvaluator(
            EvalBatch=batch,
            tagged_eval_sets=data.tagged_eval_sets(pos),
            loss_fn=partial(lm_eval_loss, EvalBatch=batch, mp=trainer.mp),
            device_mesh=trainer.device_mesh,
            axis_mapping=trainer.compute_axis_mapping,
        )
        references = []
        for name, candidate in (("raw", model), ("reseeded", reseed_snowball_head(model, controls))):
            metrics = eval_model(evaluator, candidate, prefix="eval")
            audit = evaluation_input_audit(evaluator, controls.special_token_ids)
            references.append(metrics["eval/loss"])
            write_evaluation_result(str(tmp_path / f"{name}.json"), metrics, audit)
        gate = InitialEvalMatchConfig(
            data,
            str(tmp_path / "reseeded.json"),
            8,
            1,
            controls.special_token_ids,
            pre_reseed_reference_path=str(tmp_path / "raw.json"),
        )
        config = TrainLmConfig(
            data=data,
            model=model_config,
            snowball_sft=controls,
            initial_eval_match=gate,
            hf_save_path=str(tmp_path / "hf"),
        )
        loaded = _load_lm_model_from_configured_source(
            config=config,
            converter=None,
            Vocab=vocab,
            model_key=key,
            adapter_key=jax.random.key(0),
            parameter_axis_mapping=trainer.parameter_axis_mapping,
            trainer=trainer,
        )
        receipt = check_initial_evaluation(
            loaded, gate, trainer, gate.reference_path, StoragePath(str(tmp_path)), "initial-eval"
        )
    assert abs(references[1] - references[0]) > 0.001
    assert receipt["raw_match"]["passed"] and receipt["passed"]
    assert receipt["raw_loss"] == pytest.approx(references[0], abs=1e-5)
    assert receipt["post_reseed_loss"] == pytest.approx(references[1], abs=1e-5)
    assert receipt["intentional_reseed_delta"] == pytest.approx(references[1] - references[0], abs=1e-5)
    assert (
        json.loads((tmp_path / "initial-eval-match.json").read_text())["intentional_reseed_delta"]
        == receipt["intentional_reseed_delta"]
    )


def test_reference_match_survives_batch_changes_and_rejects_changed_tokens(tmp_path):
    with use_test_mesh(tensor_parallelism=1) as mesh:
        pos = hax.Axis("position", 8)
        examples = [
            LmExample.causal(
                hax.named(jnp.arange(i, i + 8, dtype=jnp.int32), pos),
                segment_ids=hax.named(jnp.asarray([0, 0, 0, 0, 1, 1, 1, 1]), pos),
            )
            for i in range(4)
        ]
        model = ToyLmHeadModel.init(hax.Axis("vocab", 32), ToyLmConfig(), key=jax.random.key(0))
        results = []
        for batch_size in (1, 2):
            batch = hax.Axis("batch", batch_size)
            evaluator = TaggedEvaluator(
                EvalBatch=batch,
                tagged_eval_sets=[
                    (ListAsyncDataset(examples[:2]), ["heldout/a"]),
                    (ListAsyncDataset(examples[2:]), ["heldout/b"]),
                ],
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
        path = tmp_path / "reference.json"
        write_evaluation_result(str(path), results[0]["metrics"], results[0]["input_audit"])
        match = initial_eval_match(
            json.loads(path.read_text()), results[1]["metrics"], results[1]["input_audit"], 1e-5
        )
        assert match["passed"]
        assert match["input_audit"]["special_token_counts"] == {"7": 4}
        changed_audit = dict(results[1]["input_audit"], tokens_sha256="different-corpus")
        with pytest.raises(ValueError, match="inputs do not match"):
            initial_eval_match(results[0], results[1]["metrics"], changed_audit, 1e-5)
        with pytest.raises(ValueError, match="loss mismatch"):
            initial_eval_match(
                results[0], {"eval/loss": results[0]["metrics"]["eval/loss"] + 0.1}, results[1]["input_audit"], 1e-5
            )


def _audit_on_cpu_rank(coordinator):
    jax.distributed.initialize(
        coordinator_address=coordinator,
        num_processes=2,
        process_id=int(os.environ[IRIS_MULTIGPU_PROCESS_INDEX_ENV]),
        local_device_ids=[0],
    )
    pos = hax.Axis("position", 8)
    examples = [
        LmExample.causal(
            hax.named(jnp.arange(i, i + 8, dtype=jnp.int32), pos),
            segment_ids=hax.named(jnp.asarray([0, 0, 0, 0, 1, 1, 1, 1]), pos),
        )
        for i in range(4)
    ]
    results = []
    for shape, batch_size in (((2, 1), 2), ((1, 2), 1)):
        mesh = Mesh(np.asarray(jax.devices()).reshape(shape), ("data", "context"))
        batch = hax.Axis("batch", batch_size)
        evaluator = TaggedEvaluator(
            EvalBatch=batch,
            tagged_eval_sets=[(ListAsyncDataset(examples), ["heldout"])],
            loss_fn=partial(lm_eval_loss, EvalBatch=batch, mp=None),
            device_mesh=mesh,
            axis_mapping={"batch": "data", "position": "context"},
        )
        results.append(evaluation_input_audit(evaluator, (7,)))
    assert results[0] == results[1]
    assert results[0]["positions"] == 32
    assert results[0]["scored_tokens"] == 28
    assert results[0]["special_token_counts"] == {"7": 4}
    jax.distributed.shutdown()


@pytest.mark.multidevice
def test_input_audit_matches_between_two_cpu_ranks_with_data_or_context_sharding(tmp_path):
    entrypoint = Entrypoint.from_callable(_audit_on_cpu_rank, f"127.0.0.1:{find_free_port()}")
    for name, contents in entrypoint.workdir_files.items():
        (tmp_path / name).write_bytes(contents)
    env = {
        key: value
        for key, value in os.environ.items()
        if key.lower() not in ("http_proxy", "https_proxy", "all_proxy")
    }
    env.update(
        JAX_PLATFORMS="cpu",
        XLA_FLAGS="--xla_force_host_platform_device_count=1",
        IRIS_PYTHON=sys.executable,
        IRIS_WORKDIR=str(tmp_path),
    )
    command = MultiGpuHook(nproc=2).wrap(entrypoint.command)
    with subprocess.Popen([sys.executable, *command[1:]], env=env, start_new_session=True) as process:
        try:
            assert process.wait(timeout=45) == 0
        finally:
            terminate_process_group(process)
