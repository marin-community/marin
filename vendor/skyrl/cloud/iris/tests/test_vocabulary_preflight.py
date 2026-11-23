"""The CPU launch gate exercises the same vocabulary evidence as the driver."""

import json
from pathlib import Path

import fsspec
import numpy as np
from omegaconf import OmegaConf
import pytest
from safetensors.numpy import save
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from cloud.iris import hf_model_cache, task_runtime
from cloud.iris.tests.launch_config_fixtures import raw_launch_config
from experiments.post_training.long_loop.validate_launch import main, validate_policy_vocabulary


def write_tokenizer(path: Path, *, reversed_ids: bool = False) -> None:
    vocabulary = {"[UNK]": 0, "hello": 2 if reversed_ids else 1, "world": 1 if reversed_ids else 2}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]").save_pretrained(path)


@pytest.mark.parametrize("case", ["model_only", "embedded_mismatch", "missing_policy_backend"])
def test_cpu_preflight_loads_fast_metadata_and_rejects_missing_or_mismatched_vocab(tmp_path, monkeypatch, case):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "llama", "vocab_size": 3}))
    (model / "model.safetensors").write_bytes(save({"weight": np.arange(4, dtype=np.float32)}))
    tokenizer = tmp_path / "tokenizer"
    write_tokenizer(tokenizer)
    if case == "embedded_mismatch":
        write_tokenizer(model, reversed_ids=True)
    if case == "missing_policy_backend":
        (tokenizer / "tokenizer.json").unlink()
    monkeypatch.setattr(task_runtime.tempfile, "gettempdir", lambda: str(tmp_path / "node"))
    launch = OmegaConf.create(
        {
            "inputs": {
                "model": {
                    "uri": model.as_uri(),
                    "identity": "model@v1",
                    "tokenizer_uri": str(tokenizer),
                    "tokenizer_revision": "a" * 40,
                }
            },
            "skyrl": {
                "trainer": {"strategy": "megatron", "disable_fast_tokenizer": False, "policy": {"model": {}}},
                "generator": {"chat_template": None},
            },
        }
    )
    if case != "model_only":
        with pytest.raises(ValueError, match="differs" if case == "embedded_mismatch" else "backend tokenizer"):
            validate_policy_vocabulary(launch)
        return
    receipt = validate_policy_vocabulary(launch)
    assert receipt["tokenizer_size"] == receipt["checkpoint_vocab_size"] == 3
    assert receipt["model_only_binding"]["tokenizer_source"] == str(tokenizer)
    assert (Path(receipt["checkpoint_metadata"]) / "tokenizer.json").is_file()
    assert not (model / "tokenizer.json").exists()


def test_native_preflight_keeps_metadata_logs_out_of_json_receipt(tmp_path, monkeypatch, capsys):
    model = tmp_path / "model"
    model.mkdir()
    config_path = model / "config.json"
    config_path.write_text('{"model_type":"llama","vocab_size":3}')
    (model / "model.safetensors").write_bytes(save({"weight": np.arange(4, dtype=np.float32)}))
    tokenizer = tmp_path / "tokenizer"
    write_tokenizer(tokenizer)
    monkeypatch.setattr(task_runtime.tempfile, "gettempdir", lambda: str(tmp_path / "node"))
    monkeypatch.setenv("WANDB_API_KEY", "offline-fixture")
    document = raw_launch_config()
    document["runtime"]["entrypoint"] = "skyrl_train.entrypoints.taskcompendium"
    document["skyrl"]["entrypoint"] = "taskcompendium"
    document["skyrl"]["terminal_bench"] = {"harbor": {"n_concurrent_trials": 1}}
    document["inputs"]["model"].update(
        uri="s3://test/policy", identity="model@v1", tokenizer_uri=str(tokenizer), tokenizer_revision="a" * 40
    )
    monkeypatch.setattr(hf_model_cache, "fs_and_path", lambda _uri: (fsspec.filesystem("file"), str(model)))
    source = tmp_path / "launch.yaml"
    OmegaConf.save(OmegaConf.create(document), source)
    main(source, tmp_path / "compiled.yaml", config_path, check_vocabulary=True)
    captured = capsys.readouterr()
    receipt = json.loads(captured.out)
    assert receipt["vocabulary_check"]["tokenizer_size"] == 3
    assert "[task-runtime]" in captured.err
