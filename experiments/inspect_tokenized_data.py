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
Simple script to inspect tokenized data samples from a Levanter cache.

Run with:
    uv run python experiments/inspect_tokenized_data.py

This script loads samples directly from the tokenized cache and prints:
- Token IDs
- Loss weights (which tokens have loss computed)
- Decoded text
- Where loss weight transitions from 0 to 1 (if properly masked)

Output is saved to logs.txt
"""
import sys
import numpy as np
from transformers import AutoTokenizer

# Configuration
TOKENIZED_DATA_PATH = "gs://marin-us-central2/tokenized/openthoughts3_qwen2_5_7b_instruct_tokenizer-0905ba"
TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"
NUM_SAMPLES = 1  # Number of samples to inspect
LOG_FILE = "logs.txt"


class TeeOutput:
    """Write to both stdout and a file."""
    def __init__(self, filename):
        self.file = open(filename, 'w', encoding='utf-8')
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()


def print_sample(input_ids: list, assistant_mask: list, tokenizer, sample_idx: int):
    """Print detailed info about a tokenized sample."""
    print("\n" + "=" * 100)
    print(f"*** SAMPLE {sample_idx}")
    print("=" * 100)

    tokens = input_ids
    loss_weights = assistant_mask

    print(f"\nSequence length: {len(tokens)}")

    # Find transition point
    transition_idx = None
    for i, w in enumerate(loss_weights):
        if w > 0:
            transition_idx = i
            break

    num_with_loss = sum(1 for w in loss_weights if w > 0)
    print(f"Tokens with loss_weight > 0: {num_with_loss} / {len(loss_weights)}")
    print(f"First token with loss_weight > 0: index {transition_idx}")

    if transition_idx == 0:
        print("\n*** WARNING: ALL tokens have loss computed! No prompt masking! ***")
    elif transition_idx is None:
        print("\n*** WARNING: NO tokens have loss computed! ***")

    # Decode full text
    decoded = tokenizer.decode(tokens)
    print(f"\n*** FULL DECODED TEXT")
    print(decoded)

    # Show ALL tokens with loss mask
    print(f"\n*** TOKEN-BY-TOKEN (all {len(tokens)} tokens)")
    for i in range(len(tokens)):
        tok_id = tokens[i]
        tok_text = tokenizer.decode([tok_id])
        loss_marker = "LOSS" if loss_weights[i] > 0 else "----"
        transition_marker = " <-- TRANSITION" if i == transition_idx else ""
        print(f"[{i:5d}] {loss_marker} | {tok_id:6d} | {repr(tok_text)}{transition_marker}")


def main():
    # Set up logging to both stdout and file
    tee = TeeOutput(LOG_FILE)
    sys.stdout = tee

    try:
        print("=" * 100)
        print("TOKENIZED DATA INSPECTOR")
        print("=" * 100)
        print(f"Cache path: {TOKENIZED_DATA_PATH}")
        print(f"Tokenizer: {TOKENIZER_NAME}")
        print(f"Samples to inspect: {NUM_SAMPLES}")
        print(f"Log file: {LOG_FILE}")

        # Load tokenizer
        print("\nLoading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

        # Use Zarr directly to read the arrays
        print("\nLoading data from Zarr arrays...")
        import zarr
        import json
        import fsspec

        cache_path = f"{TOKENIZED_DATA_PATH}/train"
        print(f"Cache path: {cache_path}")

        # First, let's read the shard ledger to understand the data layout
        fs = fsspec.filesystem("gs")
        ledger_path = f"{cache_path.replace('gs://', '')}/shard_ledger.json"
        with fs.open(ledger_path, 'r') as f:
            ledger = json.load(f)
        print(f"Shard ledger: {ledger}")
        total_rows = ledger['total_num_rows']
        print(f"Total rows: {total_rows}")

        # Open Zarr arrays for input_ids and assistant_masks
        input_ids_path = f"{cache_path}/input_ids/data"
        assistant_masks_path = f"{cache_path}/assistant_masks/data"

        print(f"\nOpening input_ids at: {input_ids_path}")
        input_ids_store = zarr.open(input_ids_path, mode='r')
        print(f"input_ids shape: {input_ids_store.shape}, dtype: {input_ids_store.dtype}")

        print(f"Opening assistant_masks at: {assistant_masks_path}")
        assistant_masks_store = zarr.open(assistant_masks_path, mode='r')
        print(f"assistant_masks shape: {assistant_masks_store.shape}, dtype: {assistant_masks_store.dtype}")

        # The data is stored as a flat array - we need to know the sequence length
        # Let's read the first part's ledger to get sequence info
        part_ledger_path = f"{cache_path.replace('gs://', '')}/part-00000/shard_ledger.json"
        with fs.open(part_ledger_path, 'r') as f:
            part_ledger = json.load(f)
        print(f"\nPart-00000 ledger: {part_ledger}")

        # Assuming fixed sequence length, we need to figure it out
        # Let's look at a chunk of data and find EOS tokens to determine boundaries
        # For now, let's assume a reasonable max_seq_len
        MAX_SEQ_LEN = 16384  # This is the training seq len from the experiment

        # Read first chunk to find document boundaries
        IM_START = 151644  # <|im_start|> token ID for Qwen
        IM_END = 151645    # <|im_end|> token ID
        SYSTEM = 8948      # 'system' token ID

        print(f"\n--- Finding document boundaries ---")
        # Read a larger chunk to find multiple documents
        chunk = input_ids_store[0:200000].tolist()
        mask_chunk = assistant_masks_store[0:200000].tolist()

        # Find positions where <|im_start|>system pattern appears
        doc_starts = []
        for i in range(len(chunk) - 2):
            if chunk[i] == IM_START and chunk[i+1] == SYSTEM:
                # This is a system message start - indicates new document
                doc_starts.append(i)
                if len(doc_starts) >= NUM_SAMPLES + 1:
                    break

        print(f"Found {len(doc_starts)} document starts: {doc_starts[:10]}")

        if len(doc_starts) < 2:
            print("Not enough document boundaries found. Showing fixed-size samples...")
            # Fall back to fixed-size chunks
            for sample_idx in range(NUM_SAMPLES):
                start = sample_idx * 15000
                end = start + 15000
                input_ids = chunk[start:end]
                assistant_mask = mask_chunk[start:end]
                print_sample(input_ids, assistant_mask, tokenizer, sample_idx)
        else:
            print(f"\n--- Reading {NUM_SAMPLES} complete documents ---")
            for sample_idx in range(min(NUM_SAMPLES, len(doc_starts) - 1)):
                start = doc_starts[sample_idx]
                end = doc_starts[sample_idx + 1] if sample_idx + 1 < len(doc_starts) else min(start + 30000, len(chunk))

                input_ids = chunk[start:end]
                assistant_mask = mask_chunk[start:end]

                print_sample(input_ids, assistant_mask, tokenizer, sample_idx)

        print(f"\n\nOutput saved to: {LOG_FILE}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Restore stdout and close log file
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()
