#!/usr/bin/env python
# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3.5 HF parity test.

Compares Levanter Qwen3.5 forward pass against HuggingFace reference implementation.
This test generates reference logits from HF in a subprocess with transformers>=5.x,
then compares against Levanter's output.

Run standalone:  python tests/test_qwen35_hf_parity.py
Run via pytest:  pytest tests/test_qwen35_hf_parity.py -v --timeout=600
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import haliax as hax
from haliax import Axis

from levanter.layers.attention import AttentionMask
from levanter.models.qwen35 import Qwen35LMHeadModel

jax.config.update("jax_default_matmul_precision", "float32")

MODEL_ID = "Qwen/Qwen3.5-0.8B"
TEST_TOKENS = [1, 2, 3, 4, 5, 6, 7, 8]

# Subprocess script that generates HF reference logits
HF_REFERENCE_SCRIPT = """
import sys, json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoConfig

model_id = sys.argv[1]
tokens = json.loads(sys.argv[2])
output_path = sys.argv[3]

config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype=torch.float32
)
model.eval()

with torch.no_grad():
    input_ids = torch.tensor([tokens])
    logits = model(input_ids).logits[0].numpy()

np.save(output_path, logits)
print(f"Saved logits shape={logits.shape} to {output_path}")
"""


def _generate_hf_reference(model_id: str, tokens: list) -> np.ndarray:
    """Run HF model in a subprocess and return logits."""
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = Path(tmpdir) / "hf_ref.py"
        script_path.write_text(HF_REFERENCE_SCRIPT)
        output_path = Path(tmpdir) / "logits.npy"

        result = subprocess.run(
            [sys.executable, str(script_path), model_id, json.dumps(tokens), str(output_path)],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            raise RuntimeError(f"HF reference script failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

        return np.load(str(output_path))


def _can_load_qwen35_hf():
    """Check if the current transformers can load Qwen3.5."""
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from transformers import AutoConfig; AutoConfig.from_pretrained('Qwen/Qwen3.5-0.8B', trust_remote_code=True)",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return result.returncode == 0
    except Exception:
        return False


@pytest.mark.slow
def test_qwen35_hf_parity():
    """Compare Levanter Qwen3.5-0.8B logits against HuggingFace reference."""
    if not _can_load_qwen35_hf():
        pytest.skip("transformers version cannot load Qwen3.5 (need >=5.x with qwen3_5 support)")

    # Generate HF reference logits
    print("Generating HF reference logits...")
    hf_logits = _generate_hf_reference(MODEL_ID, TEST_TOKENS)
    print(f"HF logits: shape={hf_logits.shape}, range=[{hf_logits.min():.2f}, {hf_logits.max():.2f}]")

    # Load in Levanter
    print("Loading Levanter model...")
    lev_model = Qwen35LMHeadModel.load_from_hf_checkpoint(MODEL_ID)

    Batch = Axis("batch", 1)
    Pos = Axis("position", len(TEST_TOKENS))
    input_ids = hax.named(jnp.array([TEST_TOKENS], dtype=jnp.int32), (Batch, Pos))
    causal_mask = AttentionMask.causal()
    lev_logits = np.array(lev_model(input_ids, attn_mask=causal_mask, key=None).array[0])
    print(f"Lev logits: shape={lev_logits.shape}, range=[{lev_logits.min():.2f}, {lev_logits.max():.2f}]")

    # Compare
    diff = np.abs(hf_logits - lev_logits)
    print(f"Max diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")
    print(f"First 5 logits (pos 0) - HF:  {hf_logits[0, :5]}")
    print(f"First 5 logits (pos 0) - Lev: {lev_logits[0, :5]}")

    # Allow some tolerance for float32 accumulation differences between JAX and PyTorch
    np.testing.assert_allclose(lev_logits, hf_logits, rtol=1e-3, atol=5e-3)
    print("PARITY CHECK PASSED!")


if __name__ == "__main__":
    test_qwen35_hf_parity()
