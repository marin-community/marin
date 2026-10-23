# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Job-private, idempotent static-window trace; no attention arithmetic changes."""
import hashlib
import importlib.util
import sys
from pathlib import Path
from vllm.transformers_utils.configs.grugmoe import grug_moe_layer_types
expected_layers = ['sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'sliding_attention', 'sliding_attention', 'full_attention', 'sliding_attention', 'full_attention']
assert grug_moe_layer_types(26, 4) == expected_layers
print("RSI236_LAYER_TYPES", expected_layers, flush=True)
root = Path(importlib.util.find_spec("tpu_inference").origin).parent
target = root / "layers/vllm/backends/flash_attn.py"
source = target.read_text()
current = hashlib.sha256(target.read_bytes()).hexdigest()
if current == '8bba8f9f71acdc618fdda20614b5c49a0f6949a176ad929db9a24b5c38a828bd':
    assert source.count('                # With update_kv_cache=False the kernel returns the cache\n') == 1
    target.write_text(source.replace('                # With update_kv_cache=False the kernel returns the cache\n', '                # Task236 static trace only; no output or cache changes.\n                print("RSI236_LAYER_STATIC", layer.layer_name,\n                      "sliding_window=" + str(self.sliding_window), flush=True)\n                # With update_kv_cache=False the kernel returns the cache\n'))
elif current != 'd836c8ddd1fab5f702476ad2a69be19e247dd210e6d383322faf94067e23930c':
    raise RuntimeError("Unexpected fork source; refusing static trace patch")
print("RSI236_LAYER_PATCH", hashlib.sha256(target.read_bytes()).hexdigest(), flush=True)
if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main
    sys.argv[0] = "vllm"
    main()
