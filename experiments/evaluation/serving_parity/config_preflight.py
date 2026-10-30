# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check the actual isolated TPU configuration before any model is served."""

import argparse
import json
from pathlib import Path

from vllm.engine.arg_utils import EngineArgs
from vllm.plugins import load_general_plugins
from vllm.transformers_utils.configs.grugmoe import grug_moe_layer_types


def check_config(settings: dict) -> dict:
    load_general_plugins()
    probe = settings["probe"]
    engine_args = EngineArgs(
        model=settings["weights"],
        trust_remote_code=True,
        tokenizer=probe["model"],
        tokenizer_revision=probe["revision"],
        load_format="runai_streamer",
        dtype="bfloat16",
        max_model_len=probe["max_model_len"],
        tensor_parallel_size=settings["tensor_parallel_size"],
        max_num_batched_tokens=probe["max_num_batched_tokens"],
        max_num_seqs=probe["max_num_seqs"],
    )
    config = engine_args.create_engine_config()
    actual = getattr(config.model_config.hf_text_config, "layer_types", None)
    expected = grug_moe_layer_types(probe["num_hidden_layers"], probe["full_attention_interval"])
    report = dict(
        status="PASS" if actual == expected and config.cache_config.sliding_window is None else "FAIL",
        layer_types=actual,
        expected=expected,
        cache_config_sliding_window=config.cache_config.sliding_window,
        override_applied=False,
    )
    Path(settings["preflight_path"]).write_text(json.dumps(report))
    if report["status"] != "PASS":
        raise RuntimeError("Shipped TPU configuration failed the layer schedule preflight")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(check_config(json.loads(args.config.read_text()))))


if __name__ == "__main__":
    main()
