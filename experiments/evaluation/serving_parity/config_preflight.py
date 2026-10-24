# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run inside the selected TPU uv environment, without allocating TPU devices."""

import argparse
import json
from pathlib import Path

from vllm.engine.arg_utils import EngineArgs
from vllm.plugins import load_general_plugins
from vllm.transformers_utils.configs.grugmoe import grug_moe_layer_types


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    load_general_plugins()
    engine_args = EngineArgs(
        model=args.model_config,
        trust_remote_code=True,
        tokenizer="open-athena/Grug-67B-A2B-Antidoom-RLVR1-Step12-2026.10.02",
        tokenizer_revision="50fe2f488aee7730bbccc721024b35229075602e",
        load_format="runai_streamer",
        dtype="bfloat16",
        max_model_len=8192,
        tensor_parallel_size=8,
        max_num_batched_tokens=512,
        max_num_seqs=16,
    )
    config = engine_args.create_engine_config()
    actual = getattr(config.model_config.hf_text_config, "layer_types", None)
    expected = grug_moe_layer_types(26, 4)
    passed = actual == expected and config.cache_config.sliding_window is None
    report = dict(
        status="PASS" if passed else "KNOWN_FAIL",
        hf_config_class=type(config.model_config.hf_config).__module__
        + "."
        + type(config.model_config.hf_config).__name__,
        layer_types=actual,
        expected=expected,
        cache_config_sliding_window=config.cache_config.sliding_window,
        override_applied=False,
        issue="https://github.com/marin-community/tpu-inference/pull/34",
    )
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
