# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from types import SimpleNamespace

import jax
import jmp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.store.cache import write_levanter_cache
from levanter.testing.cpu_devices import run_on_cpu_devices
from levanter.tracker import current_tracker
from levanter.tracker.json_logger import JsonLoggerConfig
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from experiments.grug.moe_hero_ep import model, train


@pytest.mark.timeout(180)
def test_dropless_multilength_evaluation_preserves_short_corpora_and_training_mesh():
    script = """
        import io
        import json
        import logging
        import tempfile
        from types import SimpleNamespace

        import jax
        import jax.numpy as jnp
        import jmp
        import numpy as np
        from jax.sharding import AxisType, Mesh
        from levanter.callbacks import ProgressEvent
        from levanter.data.text.datasets import DatasetComponent, LmDataConfig
        from levanter.data.text.formats import TextLmDatasetFormat
        from levanter.store.cache import write_levanter_cache
        from levanter.tracker import current_tracker
        from levanter.tracker.json_logger import JsonLoggerConfig
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel

        from experiments.grug.moe_hero_ep import model, train

        mesh = Mesh(
            np.asarray(jax.devices()).reshape(1, 1, 1, 4, 1),
            ("replica_dcn", "data", "context", "expert", "model"),
            axis_types=(AxisType.Explicit,) * 5,
        )
        config = model.GrugModelConfig(
            vocab_size=128, hidden_dim=32, intermediate_dim=16, shared_expert_intermediate_dim=16,
            num_shared_experts=1, num_experts=4, num_experts_per_token=2, num_layers=2, num_heads=4,
            num_kv_heads=2, local_kv_heads=2, global_kv_heads=2, head_dim=8, max_seq_len=4,
            sliding_window=4, global_every=2, capacity_factor=4.0, initializer_std=0.1,
            attention_implementation="reference", moe_implementation="fixed_all_to_all",
        )
        stream = io.StringIO()
        tracker = JsonLoggerConfig(logger_name="test.multilength.dropless").init(run_id=None)
        logger = logging.getLogger("test.multilength.dropless")
        logger.addHandler(logging.StreamHandler(stream))
        events = []
        with tempfile.TemporaryDirectory() as directory, jax.set_mesh(mesh):
            tokenizer = Tokenizer(WordLevel({"<unk>": 0, **{str(i): i for i in range(1, 128)}}, unk_token="<unk>"))
            tokenizer.save(directory + "/tokenizer.json")
            cache_path = directory + "/tiny"
            write_levanter_cache(
                [{"input_ids": np.array([1, 2, 3], dtype=np.int32)}], cache_path + "/validation", metadata={}
            )
            data = LmDataConfig(
                components={"paloma/tiny": DatasetComponent(
                    cache_dir=cache_path, format=TextLmDatasetFormat()
                )},
                tokenizer=directory,
            )
            transformer = model.Transformer.init(config, key=jax.random.key(0))
            tokens = jnp.asarray(np.tile([1, 2, 3, 4], (4, 1)).astype(np.int32))
            logits = jax.jit(lambda params, ids: params.logits(ids))
            before = np.asarray(logits(transformer, tokens))
            hooks = train.build_eval_hooks(
                data_config=data, train_seq_len=4, mesh=mesh,
                eval_cfg=train.GrugEvalConfig(
                    eval_batch_size=4, seq_lens=(2, 4, 8, 16, 4), compute_bpb=False,
                    eval_current=False, eval_ema=False, dropless_eval=True,
                    dropless_eval_moe_implementation="scatter",
                ),
                mp=jmp.get_policy("float32"), eval_ema=False,
            )
            step = SimpleNamespace(step=10, model=transformer, emit_event=events.append)
            with current_tracker(tracker):
                for hook in hooks:
                    hook(step)
                    hook(step, force=True)
            after = np.asarray(logits(transformer, tokens))

        records = [json.loads(line) for line in stream.getvalue().splitlines()]
        logs = [record for record in records if record["event"] == "log"]
        assert len(logs) == 4
        assert all(record["step"] == 10 for record in logs)
        training_length_logs = [record for record in logs if "eval_dropless/loss" in record["metrics"]]
        assert len(training_length_logs) == 1
        assert "eval_dropless/seq_len_4/loss" in training_length_logs[0]["metrics"]
        metrics = {key: value for record in logs for key, value in record["metrics"].items()}
        for length in (2, 4, 8, 16):
            prefix = f"eval_dropless/seq_len_{length}"
            assert np.isfinite(metrics[prefix + "/loss"])
            assert metrics[prefix + "/paloma/tiny/loss"] > 0
        np.testing.assert_allclose(
            [metrics[f"eval_dropless/seq_len_{length}/loss"] for length in (8, 16)],
            metrics["eval_dropless/seq_len_4/loss"], rtol=1e-5, atol=1e-5,
        )
        assert metrics["eval_dropless/loss"] == metrics["eval_dropless/seq_len_4/loss"]
        np.testing.assert_array_equal(after, before)
        assert events == [ProgressEvent.EVALUATION_STARTED, ProgressEvent.EVALUATION_FINISHED] * 4
    """
    run_on_cpu_devices(script, device_count=4)


def test_multilength_evaluation_logs_current_and_ema_metrics_separately(tmp_path, caplog):
    tokenizer = Tokenizer(WordLevel({"<unk>": 0, **{str(i): i for i in range(1, 32)}}, unk_token="<unk>"))
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    cache_path = str(tmp_path / "tiny")
    write_levanter_cache([{"input_ids": np.array([1, 2, 3], dtype=np.int32)}], cache_path + "/validation", metadata={})
    data = LmDataConfig(
        components={"paloma/tiny": DatasetComponent(cache_dir=cache_path, format=TextLmDatasetFormat())},
        tokenizer=str(tmp_path),
    )
    mesh = Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1, 1),
        ("replica_dcn", "data", "context", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 5,
    )
    config = model.GrugModelConfig(
        vocab_size=32,
        hidden_dim=16,
        intermediate_dim=8,
        shared_expert_intermediate_dim=8,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        local_kv_heads=1,
        global_kv_heads=1,
        head_dim=8,
        max_seq_len=4,
        sliding_window=4,
        global_every=2,
        initializer_std=0.1,
        attention_implementation="reference",
        moe_implementation="scatter",
    )
    logger_name = "test.multilength.ema"
    tracker = JsonLoggerConfig(logger_name=logger_name).init(run_id=None)
    caplog.set_level(logging.INFO, logger=logger_name)
    with jax.set_mesh(mesh):
        transformer = model.Transformer.init(config, key=jax.random.key(0))
        ema = model.Transformer.init(config, key=jax.random.key(1))
        hooks = train.build_eval_hooks(
            data_config=data,
            train_seq_len=4,
            mesh=mesh,
            eval_cfg=train.GrugEvalConfig(eval_batch_size=4, seq_lens=(4, 8), compute_bpb=False),
            mp=jmp.get_policy("float32"),
            eval_ema=True,
        )
        step = SimpleNamespace(step=2, model=transformer, eval_model=ema, emit_event=lambda _event: None)
        with current_tracker(tracker):
            for hook in hooks:
                hook(step)
                hook(step, force=True)

    records = [json.loads(record.message) for record in caplog.records if record.name == logger_name]
    logs = [record for record in records if record["event"] == "log"]
    assert len(logs) == 4
    assert sum("eval/loss" in record["metrics"] for record in logs) == 1
    assert sum("eval/ema/loss" in record["metrics"] for record in logs) == 1
    metrics = {key: value for record in logs for key, value in record["metrics"].items()}
    for length in (4, 8):
        assert np.isfinite(metrics[f"eval/seq_len_{length}/loss"])
        assert np.isfinite(metrics[f"eval/seq_len_{length}/ema/loss"])
        assert metrics[f"eval/seq_len_{length}/loss"] != metrics[f"eval/seq_len_{length}/ema/loss"]
    assert metrics["eval/loss"] == metrics["eval/seq_len_4/loss"]
    assert metrics["eval/ema/loss"] == metrics["eval/seq_len_4/ema/loss"]
