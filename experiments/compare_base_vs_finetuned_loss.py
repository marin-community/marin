# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Compare base model loss vs fine-tuned model loss on OpenThoughts3 data.
This establishes a baseline to understand if ~1.5 loss is expected.

Run with: uv run python experiments/compare_base_vs_finetuned_loss.py
"""
import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
FINETUNED_PATH = "/tmp/finetuned_checkpoint"  # Already downloaded from previous script
TOKENIZED_DATA = "gs://marin-us-central2/tokenized/openthoughts3_qwen2_5_7b_instruct_tokenizer-0905ba/train"
NUM_SAMPLES = 5
MAX_SEQ_LEN = 16384  # Full sequence length to match debug script

print("=" * 80)
print("BASE vs FINE-TUNED LOSS COMPARISON")
print("=" * 80)

# Load tokenized data from GCS using Levanter's JaggedArrayStore
print(f"\n*** Loading tokenized data from: {TOKENIZED_DATA}")
from levanter.store.jagged_array import JaggedArrayStore

input_ids_store = JaggedArrayStore.open(f"{TOKENIZED_DATA}/input_ids", mode='r', dtype=np.int32)
assistant_masks_store = JaggedArrayStore.open(f"{TOKENIZED_DATA}/assistant_masks", mode='r', dtype=np.int32)

print(f"    Total samples in cache: {len(input_ids_store)}")

# Sample some examples
samples = []
for i in range(NUM_SAMPLES):
    # Get sample i - JaggedArrayStore supports __getitem__ for sync access
    input_ids = input_ids_store[i]
    assistant_mask = assistant_masks_store[i]
    original_len = len(input_ids)

    # Truncate to MAX_SEQ_LEN for faster computation
    if len(input_ids) > MAX_SEQ_LEN:
        input_ids = input_ids[:MAX_SEQ_LEN]
        assistant_mask = assistant_mask[:MAX_SEQ_LEN]

    samples.append({
        'input_ids': input_ids,
        'assistant_mask': assistant_mask,
        'original_len': original_len
    })
    print(f"    Sample {i}: {len(input_ids)} tokens (orig {original_len}), {sum(assistant_mask > 0)} with loss")

def compute_loss(model, input_ids, assistant_mask, device):
    """Compute weighted cross-entropy loss matching Levanter's approach."""
    input_ids_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_ids_tensor)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Shift for next-token prediction
    shift_logits = logits[0, :-1, :].float()  # (seq_len-1, vocab_size)
    shift_labels = input_ids_tensor[0, 1:]     # (seq_len-1,)
    shift_mask = torch.tensor(assistant_mask[:-1], dtype=torch.float32, device=device)  # (seq_len-1,)

    # Don't compute loss on last token (causal mask)
    shift_mask[-1] = 0.0

    # Compute per-token cross entropy
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    per_token_loss = loss_fn(shift_logits, shift_labels)  # (seq_len-1,)

    # Weighted mean (only over tokens with mask > 0)
    weighted_loss = per_token_loss * shift_mask
    total_weight = shift_mask.sum()

    if total_weight > 0:
        mean_loss = weighted_loss.sum() / total_weight
    else:
        mean_loss = torch.tensor(0.0)

    return mean_loss.item(), total_weight.item()

# Use CPU to avoid GPU memory issues
device = torch.device("cpu")
print(f"\n*** Using device: {device}")

# Load and evaluate BASE model
print(f"\n*** Loading BASE model: {BASE_MODEL}")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32)
base_model.eval()

print("\n*** Computing BASE model losses:")
base_losses = []
for i, sample in enumerate(samples):
    loss, num_tokens = compute_loss(base_model, sample['input_ids'], sample['assistant_mask'], device)
    base_losses.append(loss)
    print(f"    Sample {i}: loss = {loss:.4f} ({int(num_tokens)} tokens with loss)")

base_mean = np.mean(base_losses)
print(f"\n    BASE MODEL MEAN LOSS: {base_mean:.4f}")

# Free memory
del base_model
torch.cuda.empty_cache() if torch.cuda.is_available() else None
import gc
gc.collect()

# Load and evaluate FINE-TUNED model
print(f"\n*** Loading FINE-TUNED model from: {FINETUNED_PATH}")
if not os.path.exists(FINETUNED_PATH):
    print("    ERROR: Fine-tuned model not found. Run check_weights.py first to download it.")
    exit(1)

ft_model = AutoModelForCausalLM.from_pretrained(FINETUNED_PATH, torch_dtype=torch.float32)
ft_model.eval()

print("\n*** Computing FINE-TUNED model losses:")
ft_losses = []
for i, sample in enumerate(samples):
    loss, num_tokens = compute_loss(ft_model, sample['input_ids'], sample['assistant_mask'], device)
    ft_losses.append(loss)
    print(f"    Sample {i}: loss = {loss:.4f} ({int(num_tokens)} tokens with loss)")

ft_mean = np.mean(ft_losses)
print(f"\n    FINE-TUNED MODEL MEAN LOSS: {ft_mean:.4f}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Base model mean loss:       {base_mean:.4f}")
print(f"Fine-tuned model mean loss: {ft_mean:.4f}")
print(f"Improvement:                {base_mean - ft_mean:.4f} ({(base_mean - ft_mean) / base_mean * 100:.1f}%)")
print()
if ft_mean < base_mean:
    print("*** Fine-tuning DID reduce loss compared to base model.")
else:
    print("*** WARNING: Fine-tuned model has HIGHER loss than base model!")
print()
print("If fine-tuned loss ~1.5 and base loss ~2.5-3.0, then training worked correctly.")
print("The ~0.84 loss from WandB was likely a smoothed/EMA value, not instantaneous loss.")
print("=" * 80)
