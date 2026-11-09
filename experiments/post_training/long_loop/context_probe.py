# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only ModelConfig proof in the pinned SkyRL/vLLM runtime."""

import hashlib
import json
from pathlib import Path

import vllm
from vllm.config import ModelConfig

from experiments.post_training.long_loop.serving import serving_release


def main():
    root = Path(__file__).with_name("context-fixtures")
    if serving_release().source_commit[:12] not in vllm.__version__:
        raise ValueError(f"Wrong vLLM runtime: {vllm.__version__}")
    cases = [
        ("old", {"max_position_embeddings": 262144}, False),
        ("fixed", {"max_position_embeddings": 262144, "max_seq_len": 262144}, True),
        ("other-alias", {"max_position_embeddings": 262144, "max_seq_len": 262144, "n_positions": 32768}, False),
    ]
    results = []
    for name, overrides, expected in cases:
        try:
            config = ModelConfig(
                model=str(root), tokenizer=str(root), dtype="bfloat16", max_model_len=65536, hf_overrides=overrides
            )
            result = {"name": name, "accepted": True, "max_model_len": config.max_model_len}
        except ValueError as exc:
            if "max_model_len" not in str(exc):
                raise
            result = {"name": name, "accepted": False, "error": str(exc)}
        if result["accepted"] != expected:
            raise AssertionError(result)
        result.update(
            config_digest=hashlib.sha256(
                json.dumps(json.loads((root / "config.json").read_text()), sort_keys=True).encode()
            ).hexdigest(),
            overrides=overrides,
            requested_window=65536,
        )
        results.append(result)
    evidence = {
        "vllm_commit": serving_release().source_commit,
        "vllm": vllm.__version__,
        "config_sha256": hashlib.sha256((root / "config.json").read_bytes()).hexdigest(),
        "cases": results,
    }
    Path("/tmp/context-proof.json").write_text(json.dumps(evidence, indent=2))
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
