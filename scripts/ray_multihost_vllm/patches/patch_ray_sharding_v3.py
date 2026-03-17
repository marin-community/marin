# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fix weight sharding for Ray multi-host: hardcode TP partition specs.

nnx.get_named_sharding(params, mesh) fails under Ray, leaving all specs None.
This patch adds a fallback that assigns correct TP partition specs based on
weight name patterns, matching the model definitions in llama3.py/gpt_oss.py.

Weight -> Shape -> PartitionSpec mapping (for 2D mesh with axes ('data', 'model')):
  embed_tokens:   (vocab, hidden)         -> P('model', None)     -- shard vocab
  lm_head:        (hidden, vocab)         -> P(None, 'model')     -- shard vocab (transposed)
  q_proj:         (hidden, heads, head_d) -> P(None, 'model', None) -- shard heads
  k_proj:         (hidden, kv_h, head_d)  -> P(None, 'model', None)
  v_proj:         (hidden, kv_h, head_d)  -> P(None, 'model', None)
  o_proj:         (heads, head_d, hidden) -> P('model', None, None)
  gate_proj:      (hidden, inter)         -> P(None, 'model')     -- shard intermediate
  up_proj:        (hidden, inter)         -> P(None, 'model')
  down_proj:      (inter, hidden)         -> P('model', None)
  layernorm/bias: (hidden,)               -> P()                  -- replicate
"""

import os
import re

BASE = "/workspace/tpu_inference/tpu_inference"
PATH = os.path.join(BASE, "models/jax/utils/weight_utils.py")

with open(PATH) as f:
    code = f.read()

# Find the weight spec extraction block and replace with hardcoded fallback
# Match the original code pattern (before any of our patches modified it)
patterns_to_try = [
    # Pattern 1: original unpatched code
    """    # Update the model weight
    spec = model_weight.sharding.spec if isinstance(
        model_weight.sharding, NamedSharding) else model_weight.sharding
    model_weight.value = shard(hf_weight, spec)""",
    # Pattern 2: our v3 debug patch
    """    # Update the model weight - read sharding from underlying JAX array""",
    # Pattern 3: our v2 patch
    """    # Update the model weight
    # Try nnx.Param sharding metadata first, then fall back to the""",
]

# The replacement code with hardcoded TP sharding
replacement = """    # Update the model weight
    # When nnx.get_named_sharding fails (Ray multi-host), hardcode TP partition
    # specs based on weight name patterns from model definitions (llama3.py).
    spec = None
    _mw_sharding = getattr(model_weight, 'sharding', None)
    _val_sharding = getattr(model_weight.value, 'sharding', None) if hasattr(model_weight, 'value') else None
    if isinstance(_mw_sharding, NamedSharding):
        spec = _mw_sharding.spec
    elif isinstance(_val_sharding, NamedSharding):
        spec = _val_sharding.spec
    else:
        # Hardcoded TP sharding fallback based on weight name
        ndim = len(hf_weight.shape)
        if any(k in hf_key for k in ('gate_proj', 'up_proj')):
            spec = P(None, 'model') if ndim == 2 else P()
        elif 'down_proj' in hf_key:
            spec = P('model', None) if ndim == 2 else P()
        elif 'q_proj' in hf_key or 'k_proj' in hf_key or 'v_proj' in hf_key:
            spec = P(None, 'model', None) if ndim == 3 else (P(None, 'model') if ndim == 2 else P())
        elif 'o_proj' in hf_key:
            spec = P('model', None, None) if ndim == 3 else (P('model', None) if ndim == 2 else P())
        elif 'embed_tokens' in hf_key:
            spec = P('model', None) if ndim == 2 else P()
        elif 'lm_head' in hf_key:
            spec = P(None, 'model') if ndim == 2 else P()
        elif 'norm' in hf_key or 'bias' in hf_key:
            spec = P(*([None] * ndim)) if ndim > 0 else P()
        else:
            # Unknown weight: replicate (safe default)
            spec = P(*([None] * ndim)) if ndim > 0 else P()
        logger.info(f"TP sharding fallback | {hf_key} | spec={spec} | shape={hf_weight.shape}")
    model_weight.value = shard(hf_weight, spec)"""

# Find which pattern is in the code and replace everything up to model_weight.value = shard


# Find the "# Update the model weight" section and replace it entirely
# Match from "# Update the model weight" to "model_weight.value = shard(hf_weight, spec)"
pattern = r"    # Update the model weight.*?model_weight\.value = shard\(hf_weight, spec\)"
match = re.search(pattern, code, re.DOTALL)

if match:
    code = code[: match.start()] + replacement + code[match.end() :]
    with open(PATH, "w") as f:
        f.write(code)
    print("PATCHED weight_utils.py: hardcoded TP sharding fallback for Ray multi-host")
else:
    print("SKIP: could not find weight update block")
