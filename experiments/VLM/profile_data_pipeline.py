#!/usr/bin/env python3
"""
Simple profiler to measure data pipeline batch loading time and TPU transfer time.

Usage:
    python experiments/VLM/profile_data_pipeline.py --num_batches 10
"""

import argparse
import asyncio
import logging
import time
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
from levanter.data.image import ConversationDatasetSourceConfig, ImageMixtureDatasetConfig
# Note: processor is loaded via data_config.the_processor (cached)

# Enable logging to see what's happening
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Profile VLM data pipeline batch loading time")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches to profile")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size (default: 16 for v5p-16)")
    parser.add_argument(
        "--train_urls",
        type=str,
        default="gs://marin-vlm/stage1_sharded/*.parquet",
        help="Training data URLs",
    )
    args = parser.parse_args()

    print(f"=== VLM Data Pipeline Profiler ===")
    print(f"Batch size: {args.batch_size}")
    print(f"Num batches: {args.num_batches}")
    print(f"Data URLs: {args.train_urls}")
    print()

    # Data configuration (same as demo_vlm_train.py)
    data_source = ConversationDatasetSourceConfig(
        train_urls=[args.train_urls],
        messages_key="messages",
        images_key="images",
    )

    data_config = ImageMixtureDatasetConfig(
        cache_dir="cache/vlm_profile",
        processor="llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        configs={"train": data_source},
        train_weights={"train": 1.0},
        use_cache=False,  # Streaming mode
        max_length=8192,
    )

    # Pre-load processor (this is cached by the_processor property)
    print("Loading processor...")
    t0 = time.time()
    processor = data_config.the_processor  # Use cached processor
    print(f"Processor loaded in {time.time() - t0:.2f}s: {type(processor).__name__}")
    print()

    # Build the dataset - should be fast now (step-based mode, no data counting)
    print("Building streaming dataset (step-based mode)...")
    t0 = time.time()

    # JAX initialization can be slow on first use
    print("  Creating JAX PRNGKey...")
    t_jax = time.time()
    data_key = jax.random.PRNGKey(42)
    print(f"  JAX PRNGKey created in {time.time() - t_jax:.2f}s")

    # Build dataset
    print("  Calling train_set()...")
    t_train = time.time()
    dataset = data_config.train_set(key=data_key)
    print(f"  train_set() completed in {time.time() - t_train:.2f}s")

    print(f"Dataset built in {time.time() - t0:.2f}s (total)")
    print(f"Dataset type: {type(dataset).__name__}")
    print()

    # Profile batch loading
    batch_times: List[float] = []
    transfer_times: List[float] = []
    batch_sizes_bytes: List[int] = []
    batch_size = args.batch_size

    def get_batch_size_bytes(batch) -> int:
        """Calculate total size of batch data in bytes."""
        total = 0
        if isinstance(batch, dict):
            for key, value in batch.items():
                total += get_item_size_bytes(value)
        elif isinstance(batch, (list, tuple)):
            for item in batch:
                total += get_batch_size_bytes(item)
        else:
            total += get_item_size_bytes(batch)
        return total

    def get_item_size_bytes(value) -> int:
        """Calculate size of a single item in bytes."""
        if isinstance(value, np.ndarray):
            return value.nbytes
        elif isinstance(value, jnp.ndarray):
            return value.nbytes
        elif isinstance(value, dict):
            return sum(get_item_size_bytes(v) for v in value.values())
        elif isinstance(value, (list, tuple)) and len(value) > 0:
            if isinstance(value[0], (np.ndarray, jnp.ndarray)):
                return sum(v.nbytes for v in value)
            else:
                return sum(get_item_size_bytes(v) for v in value)
        return 0

    def print_value_info(key, value):
        """Print info about a value in the batch."""
        if isinstance(value, (np.ndarray, jnp.ndarray)):
            size_mb = value.nbytes / (1024 * 1024)
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}, size={size_mb:.2f} MB")
        elif isinstance(value, (list, tuple)) and len(value) > 0:
            if isinstance(value[0], (np.ndarray, jnp.ndarray)):
                total_size = sum(v.nbytes for v in value) / (1024 * 1024)
                print(f"  {key}: list of {len(value)} arrays, first shape={value[0].shape}, total={total_size:.2f} MB")
            else:
                print(f"  {key}: {type(value).__name__} of length {len(value)}")
        else:
            print(f"  {key}: {type(value).__name__}")

    def transfer_to_tpu(batch, device):
        """Transfer batch data to TPU device."""
        if isinstance(batch, dict):
            result = {}
            for key, value in batch.items():
                result[key] = transfer_to_tpu(value, device)
            return result
        elif isinstance(batch, list):
            return [transfer_to_tpu(item, device) for item in batch]
        elif isinstance(batch, tuple):
            return tuple(transfer_to_tpu(item, device) for item in batch)
        elif isinstance(batch, (np.ndarray, jnp.ndarray)):
            return jax.device_put(batch, device)
        else:
            return batch

    # Get TPU device
    devices = jax.devices()
    print(f"Available devices: {devices}")
    tpu_device = devices[0]  # Use first device
    print(f"Using device: {tpu_device}")
    print()

    print(f"Profiling {args.num_batches} batches (batch_size={batch_size})...")
    print("-" * 85)
    print(f"{'Batch':>5} | {'Load Time':>10} | {'Transfer Time':>13} | {'Total':>10} | {'Size':>9} | Indices")
    print("-" * 85)

    first_batch_info_printed = False

    async def profile_batches():
        nonlocal first_batch_info_printed
        for i in range(args.num_batches):
            # Generate indices for this batch
            start_idx = i * batch_size
            indices = list(range(start_idx, start_idx + batch_size))

            # Time the batch fetch (CPU)
            start_load = time.time()
            batch = await dataset.get_batch(indices)
            load_elapsed = time.time() - start_load

            # Print first batch structure info
            if not first_batch_info_printed:
                print("\n=== First Batch Structure ===")
                print(f"  Batch type: {type(batch).__name__}")
                if isinstance(batch, dict):
                    for key, value in batch.items():
                        print_value_info(key, value)
                elif isinstance(batch, (list, tuple)):
                    print(f"  Batch is a list of {len(batch)} items")
                    if len(batch) > 0:
                        first_item = batch[0]
                        print(f"  First item type: {type(first_item).__name__}")
                        if isinstance(first_item, dict):
                            for key, value in first_item.items():
                                print_value_info(f"  [0].{key}", value)
                        elif isinstance(first_item, (np.ndarray, jnp.ndarray)):
                            size_mb = first_item.nbytes / (1024 * 1024)
                            print(f"    [0]: shape={first_item.shape}, dtype={first_item.dtype}, size={size_mb:.2f} MB")
                print("-" * 85)
                first_batch_info_printed = True

            # Time the TPU transfer
            start_transfer = time.time()
            # Transfer batch to TPU
            batch_on_tpu = transfer_to_tpu(batch, tpu_device)
            # Block until transfer completes
            jax.block_until_ready(batch_on_tpu)
            transfer_elapsed = time.time() - start_transfer

            # Calculate batch size
            size_bytes = get_batch_size_bytes(batch)

            batch_times.append(load_elapsed)
            transfer_times.append(transfer_elapsed)
            batch_sizes_bytes.append(size_bytes)

            total = load_elapsed + transfer_elapsed
            size_mb = size_bytes / (1024 * 1024)
            print(f"{i:5d} | {load_elapsed:10.2f}s | {transfer_elapsed:13.3f}s | {total:10.2f}s | {size_mb:7.1f} MB | {start_idx}-{start_idx + batch_size - 1}")

    # Run the async profiling
    asyncio.run(profile_batches())

    # Print summary
    print("-" * 85)
    print()
    print("=== Summary ===")
    print(f"Total batches: {len(batch_times)}")
    print(f"Batch size: {batch_size}")
    print()

    if batch_times:
        # Load time stats
        avg_load = sum(batch_times) / len(batch_times)
        min_load = min(batch_times)
        max_load = max(batch_times)
        sorted_load = sorted(batch_times)
        p95_idx = int(len(sorted_load) * 0.95)
        p95_load = sorted_load[min(p95_idx, len(sorted_load) - 1)]

        # Transfer time stats
        avg_transfer = sum(transfer_times) / len(transfer_times)
        min_transfer = min(transfer_times)
        max_transfer = max(transfer_times)
        sorted_transfer = sorted(transfer_times)
        p95_transfer = sorted_transfer[min(p95_idx, len(sorted_transfer) - 1)]

        # Total time stats
        total_times = [l + t for l, t in zip(batch_times, transfer_times)]
        avg_total = sum(total_times) / len(total_times)
        min_total = min(total_times)
        max_total = max(total_times)
        sorted_total = sorted(total_times)
        p95_total = sorted_total[min(p95_idx, len(sorted_total) - 1)]

        print(f"Data Loading Time (CPU processing):")
        print(f"  Average: {avg_load:.3f}s")
        print(f"  Min:     {min_load:.3f}s")
        print(f"  Max:     {max_load:.3f}s")
        print(f"  P95:     {p95_load:.3f}s")
        print()

        print(f"TPU Transfer Time:")
        print(f"  Average: {avg_transfer:.3f}s")
        print(f"  Min:     {min_transfer:.3f}s")
        print(f"  Max:     {max_transfer:.3f}s")
        print(f"  P95:     {p95_transfer:.3f}s")
        print()

        # Batch size and bandwidth stats
        avg_size_bytes = sum(batch_sizes_bytes) / len(batch_sizes_bytes)
        avg_size_mb = avg_size_bytes / (1024 * 1024)
        avg_bandwidth_mbps = avg_size_mb / avg_transfer if avg_transfer > 0 else 0
        avg_bandwidth_gbps = avg_bandwidth_mbps * 8 / 1024  # Convert MB/s to Gbps

        print(f"Data Size per Batch:")
        print(f"  Average: {avg_size_mb:.1f} MB")
        print()

        print(f"TPU Transfer Bandwidth:")
        print(f"  Average: {avg_bandwidth_mbps:.1f} MB/s ({avg_bandwidth_gbps:.2f} Gbps)")
        print()

        print(f"Total Time (Load + Transfer):")
        print(f"  Average: {avg_total:.3f}s")
        print(f"  Min:     {min_total:.3f}s")
        print(f"  Max:     {max_total:.3f}s")
        print(f"  P95:     {p95_total:.3f}s")
        print()

        print(f"Throughput: {1/avg_total:.2f} batches/sec")
        print(f"            {batch_size/avg_total:.2f} examples/sec")
        print()

        # Breakdown
        load_pct = (avg_load / avg_total) * 100
        transfer_pct = (avg_transfer / avg_total) * 100
        print(f"Time Breakdown:")
        print(f"  Data Loading:  {load_pct:.1f}%")
        print(f"  TPU Transfer:  {transfer_pct:.1f}%")

        # Calculate if this would be a bottleneck
        # Assuming ~1s per training step on TPU, we need < 1s batch load time
        if avg_total > 1.0:
            print()
            print(f"WARNING: Total batch time ({avg_total:.2f}s) > 1s may be a bottleneck!")
            print(f"         Consider enabling disk caching or parallel image loading.")


if __name__ == "__main__":
    main()
