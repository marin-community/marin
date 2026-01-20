#!/usr/bin/env python3
"""
Convert LLaVA-OneVision-1.5-Mid-Training-85M dataset to Levanter image conversation format.

Uses streaming to avoid large local storage, with TWO-LAYER shuffling for cross-subset mixing:
1. HF built-in shuffle: Shuffles shard (parquet file) order AND maintains a streaming buffer
2. Write-time shuffle: Additional shuffle before writing each output shard

This ensures data from different subsets (coyo, datacomp1b, imagenet, etc.) are mixed together.

Example:
    # Basic conversion
    python convert_llava_onevision_to_levanter.py \
        --output-gcs gs://your-bucket/llava_onevision_levanter/ \
        --buffer-size 100000 \
        --rows-per-shard 10000 \
        --seed 42

    # Test with small subset
    python convert_llava_onevision_to_levanter.py \
        --output-gcs gs://your-bucket/test_output/ \
        --buffer-size 1000 \
        --rows-per-shard 500 \
        --max-rows 5000 \
        --seed 42

    # Write to local directory for testing
    python convert_llava_onevision_to_levanter.py \
        --output-local /tmp/llava_test/ \
        --buffer-size 1000 \
        --rows-per-shard 500 \
        --max-rows 5000
"""

import argparse
import gzip
import io
import json
import os
import random
import sys
import time
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional, Set, Tuple

# Set up temporary directory before importing datasets
if 'TMPDIR' not in os.environ:
    for temp_dir in ['/tmp', '/var/tmp', '/usr/tmp', str(Path.home() / '.tmp')]:
        if os.path.exists(temp_dir) and os.access(temp_dir, os.W_OK):
            os.environ['TMPDIR'] = temp_dir
            break
    else:
        temp_dir = Path.cwd() / '.tmp'
        temp_dir.mkdir(exist_ok=True)
        os.environ['TMPDIR'] = str(temp_dir)

from datasets import Dataset, load_dataset
import shutil
import glob as glob_module

try:
    from huggingface_hub import HfApi, hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

try:
    import pandas as pd
    import pyarrow.parquet as pq
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class AsyncShardWriter:
    """Background thread pool for writing shards concurrently."""

    def __init__(self, output_path: str, fs=None, max_workers: int = 4):
        self.output_path = output_path
        self.fs = fs
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []
        self.lock = threading.Lock()
        self.total_written = 0
        self.shards_completed = 0

    def submit_shard(self, rows: List[Dict[str, Any]], shard_idx: int) -> None:
        """Submit a shard for async writing."""
        future = self.executor.submit(self._write_shard, rows, shard_idx)
        with self.lock:
            self.futures.append((future, shard_idx, len(rows)))

    def _write_shard(self, rows: List[Dict[str, Any]], shard_idx: int) -> int:
        """Write a single shard (runs in thread pool)."""
        if not rows:
            return 0

        shard_name = f"train-{shard_idx:05d}.parquet"

        # Create dataset from list
        dataset = Dataset.from_list(rows)

        if self.fs is not None:
            # Write to GCS
            full_path = f"{self.output_path.rstrip('/')}/{shard_name}"
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
                tmp_path = tmp.name

            try:
                dataset.to_parquet(tmp_path)
                self.fs.put(tmp_path, full_path)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        else:
            # Write to local filesystem
            output_dir = Path(self.output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            shard_path = output_dir / shard_name
            dataset.to_parquet(str(shard_path))

        return len(rows)

    def wait_for_completed(self) -> List[Tuple[int, int]]:
        """Check for completed futures and return (shard_idx, rows_written) for each."""
        completed = []
        with self.lock:
            remaining = []
            for future, shard_idx, num_rows in self.futures:
                if future.done():
                    try:
                        written = future.result()
                        completed.append((shard_idx, written))
                        self.total_written += written
                        self.shards_completed += 1
                    except Exception as e:
                        print(f"Error writing shard {shard_idx}: {e}", file=sys.stderr)
                else:
                    remaining.append((future, shard_idx, num_rows))
            self.futures = remaining
        return completed

    def wait_all(self) -> int:
        """Wait for all pending writes to complete. Returns total rows written."""
        with self.lock:
            futures_to_wait = list(self.futures)
            self.futures = []

        for future, shard_idx, num_rows in futures_to_wait:
            try:
                written = future.result()
                self.total_written += written
                self.shards_completed += 1
                print(f"  Shard {shard_idx} written ({written} rows)")
            except Exception as e:
                print(f"Error writing shard {shard_idx}: {e}", file=sys.stderr)

        return self.total_written

    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


class ProgressTracker:
    """Background thread to show status during HF buffer filling phase."""

    def __init__(self, shuffle_buffer_size: int, update_interval: int = 30):
        self.shuffle_buffer_size = shuffle_buffer_size
        self.update_interval = update_interval
        self.first_output_received = False
        self.start_time = None
        self.stop_event = threading.Event()
        self.thread = None

    def start(self):
        """Start the progress tracking thread."""
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        """Stop the progress tracking thread."""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)

    def mark_first_output(self):
        """Mark that the first output has been received (buffer is active)."""
        self.first_output_received = True

    def _run(self):
        """Background thread that prints status updates."""
        update_count = 0
        while not self.stop_event.is_set():
            self.stop_event.wait(self.update_interval)

            if self.stop_event.is_set() or self.first_output_received:
                break

            update_count += 1
            elapsed = time.time() - self.start_time
            timestamp = datetime.now().strftime("%H:%M:%S")

            # Print status message
            print(f"[{timestamp}] Still waiting for HF buffer to fill... "
                  f"(elapsed: {elapsed/60:.1f} min)")

            # Provide helpful tips periodically
            if update_count == 2:  # After ~1 minute
                print(f"           Tip: Network timeouts are normal - HF auto-retries downloads")
            elif update_count == 4:  # After ~2 minutes
                print(f"           Tip: Use --hf-verbose to see HuggingFace download logs")
            elif update_count == 6:  # After ~3 minutes
                print(f"           Tip: Buffer size is {self.shuffle_buffer_size:,}. "
                      f"Use smaller --shuffle-buffer-size for faster initial output")
            elif update_count % 4 == 0:  # Every ~2 minutes after that
                print(f"           Note: Large buffer = better shuffle but slower start. "
                      f"Script IS working.")

# Prompts for long captions (> 200 characters)
LONG_PROMPTS = [
    "Describe this image in detail.",
    "Provide a comprehensive description of what you see in this image.",
    "What is shown in this image? Please be thorough in your description.",
    "Analyze and describe all the elements visible in this image.",
    "Give a detailed account of everything you observe in this picture.",
    "Explain what this image depicts, including all relevant details.",
    "Describe the contents of this image as completely as possible.",
    "What can you see in this image? Provide an extensive description.",
    "Look at this image carefully and describe what you see in detail.",
    "Provide a rich, detailed description of this image.",
]

# Prompts for short captions (<= 200 characters)
SHORT_PROMPTS = [
    "Describe this image.",
    "What is in this image?",
    "What do you see?",
    "Describe what you see.",
    "What is shown here?",
    "Caption this image.",
    "What does this image show?",
    "Briefly describe this image.",
    "What's in the picture?",
    "Describe the image.",
]

CAPTION_LENGTH_THRESHOLD = 200


def select_prompt(caption: str, item_id: str) -> str:
    """
    Select a prompt based on caption length.
    Uses item_id hash as seed for reproducibility.
    """
    # Use hash of item_id for reproducible random selection
    seed = hash(item_id) % (2**32)
    rng = random.Random(seed)

    if len(caption) > CAPTION_LENGTH_THRESHOLD:
        return rng.choice(LONG_PROMPTS)
    else:
        return rng.choice(SHORT_PROMPTS)


def extract_source(item_id: str) -> str:
    """
    Extract source name from item id.

    Different sources have different ID formats:
    - coyo: "coyo/part_02/coyo700m_01_id_xxx.jpg" (has '/')
    - obelics: "recs_obelics_07_data_xxx.jpg" (NO '/', contains 'obelics')
    - sa1b/sam1b: "sam1b/part_01/recs_batch_xxx.jpg" (prefix is 'sam1b')
    - laioncn: "laioncn/..." (has '/')
    - datacomp1b: "datacomp1b/..." (has '/')
    - zero250m, imagenet, mint: various formats

    Returns the normalized source name.
    """
    if not item_id:
        return 'unknown'

    # Method 1: Check for '/' separator (coyo, laioncn, sam1b, datacomp1b, etc.)
    if '/' in item_id:
        prefix = item_id.split('/')[0]
        # Normalize some names
        if prefix == 'sam1b':
            return 'sa1b'
        return prefix

    # Method 2: Check for known patterns in the ID string (for IDs without '/')
    item_id_lower = item_id.lower()

    # Known sources to search for in the ID string
    # Order matters: more specific names should come first
    known_sources = [
        'datacomp1b', 'datacomp',  # datacomp variants
        'obelics',
        'zero250m',
        'imagenet',
        'laioncn',
        'sam1b', 'sa1b',  # SA-1B variants
        'coyo',
        'mint',
    ]

    for source in known_sources:
        if source in item_id_lower:
            # Normalize sam1b -> sa1b
            if source == 'sam1b':
                return 'sa1b'
            return source

    return 'unknown'


def get_checkpoint_path(output_path: str) -> str:
    """Get the checkpoint file path for the given output path."""
    return f"{output_path.rstrip('/')}/checkpoint.json.gz"


def load_checkpoint(checkpoint_path: str, fs=None) -> Tuple[Set[str], int, Counter, int, int]:
    """
    Load checkpoint from file.

    Returns:
        processed_ids: Set of already processed item ids
        shard_idx: Current shard index to continue from
        source_counts: Source distribution counter
        total_processed: Total rows processed so far
        total_written: Total rows written so far
    """
    processed_ids: Set[str] = set()
    shard_idx = 0
    source_counts: Counter = Counter()
    total_processed = 0
    total_written = 0

    try:
        if fs is not None:
            # GCS
            if fs.exists(checkpoint_path):
                with fs.open(checkpoint_path, 'rb') as f:
                    with gzip.open(f, 'rt', encoding='utf-8') as gz:
                        data = json.load(gz)
        else:
            # Local
            path = Path(checkpoint_path)
            if path.exists():
                with gzip.open(path, 'rt', encoding='utf-8') as gz:
                    data = json.load(gz)
            else:
                return processed_ids, shard_idx, source_counts, total_processed, total_written

        # Parse checkpoint data
        processed_ids = set(data.get('processed_ids', []))
        shard_idx = data.get('shard_idx', 0)
        source_counts = Counter(data.get('source_counts', {}))
        total_processed = data.get('total_processed', 0)
        total_written = data.get('total_written', 0)

        print(f"Loaded checkpoint: {len(processed_ids):,} processed ids, shard_idx={shard_idx}")

    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")

    return processed_ids, shard_idx, source_counts, total_processed, total_written


def save_checkpoint(
    checkpoint_path: str,
    processed_ids: Set[str],
    shard_idx: int,
    source_counts: Counter,
    total_processed: int,
    total_written: int,
    fs=None
):
    """Save checkpoint to file (gzip compressed JSON)."""
    data = {
        'processed_ids': list(processed_ids),
        'shard_idx': shard_idx,
        'source_counts': dict(source_counts),
        'total_processed': total_processed,
        'total_written': total_written,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    try:
        if fs is not None:
            # GCS - write to temp file then upload
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.json.gz', delete=False) as tmp:
                tmp_path = tmp.name

            with gzip.open(tmp_path, 'wt', encoding='utf-8') as gz:
                json.dump(data, gz)

            fs.put(tmp_path, checkpoint_path)
            os.remove(tmp_path)
        else:
            # Local
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(checkpoint_path, 'wt', encoding='utf-8') as gz:
                json.dump(data, gz)

    except Exception as e:
        print(f"Warning: Could not save checkpoint: {e}", file=sys.stderr)


def convert_to_levanter(item: Dict[str, Any], include_source: bool = True) -> Dict[str, Any]:
    """
    Convert a single item from LLaVA-OneVision format to Levanter format.

    Input format:
        {
            "id": str,
            "image": PIL.Image or dict with bytes,
            "caption": str,
            "split": str
        }

    Output format:
        {
            "messages": [
                {"role": "user", "content": [{"type": "image", "text": None}, {"type": "text", "text": prompt}]},
                {"role": "assistant", "content": [{"type": "text", "text": caption}]}
            ],
            "images": [{"bytes": image_bytes}],
            "source": str  # Optional: source dataset name (coyo, datacomp1b, etc.)
        }
    """
    item_id = item.get('id', '')
    caption = item.get('caption', '')
    image = item.get('image')

    # Get image bytes
    if hasattr(image, 'tobytes'):
        # PIL Image - convert to bytes
        img_byte_arr = io.BytesIO()
        # Save as PNG to preserve quality
        image.save(img_byte_arr, format='PNG')
        image_bytes = img_byte_arr.getvalue()
    elif isinstance(image, dict) and 'bytes' in image:
        image_bytes = image['bytes']
    elif isinstance(image, bytes):
        image_bytes = image
    else:
        raise ValueError(f"Unknown image format: {type(image)}")

    # Select prompt based on caption length
    prompt = select_prompt(caption, item_id)

    # Build Levanter format
    result = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": caption}
                ]
            }
        ],
        "images": [{"bytes": image_bytes}]
    }

    # Add source field for shuffle quality analysis
    if include_source:
        result["source"] = extract_source(item_id)

    return result


def write_shard(
    rows: List[Dict[str, Any]],
    shard_idx: int,
    output_path: str,
    fs=None
) -> int:
    """
    Write a shard to local filesystem or GCS.

    Returns number of rows written.
    """
    if not rows:
        return 0

    shard_name = f"train-{shard_idx:05d}.parquet"

    # Create dataset from list
    dataset = Dataset.from_list(rows)

    if fs is not None:
        # Write to GCS
        full_path = f"{output_path.rstrip('/')}/{shard_name}"
        # Write to a temporary local file first, then upload
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            dataset.to_parquet(tmp_path)
            # Upload to GCS
            fs.put(tmp_path, full_path)
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        print(f"  Written shard {shard_idx}: {full_path} ({len(rows)} rows)")
    else:
        # Write to local filesystem
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        shard_path = output_dir / shard_name
        dataset.to_parquet(str(shard_path))
        print(f"  Written shard {shard_idx}: {shard_path} ({len(rows)} rows)")

    return len(rows)


# ============================================================================
# Download-First Mode: Parallel batch downloading and processing
# ============================================================================

def list_dataset_parquet_files(repo_id: str = "mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M") -> List[str]:
    """List all parquet files in the dataset repository."""
    if not HF_HUB_AVAILABLE:
        raise RuntimeError("huggingface_hub not installed. Install with: pip install huggingface_hub")

    api = HfApi()
    files = api.list_repo_files(repo_id, repo_type="dataset")
    parquet_files = [f for f in files if f.endswith('.parquet')]
    print(f"Found {len(parquet_files)} parquet files in dataset")
    return parquet_files


def get_download_first_checkpoint_path(output_path: str) -> str:
    """Get the checkpoint file path for download-first mode."""
    return f"{output_path.rstrip('/')}/checkpoint_download_first.json.gz"


def load_download_first_checkpoint(
    checkpoint_path: str,
    fs=None
) -> Tuple[List[str], int, int, Counter, int, int]:
    """
    Load checkpoint for download-first mode.

    Returns:
        shuffled_files: The shuffled file list (to maintain same order on resume)
        batch_idx: Current batch index to resume from
        shard_idx: Current shard index to continue from
        source_counts: Source distribution counter
        total_processed: Total rows processed so far
        total_written: Total rows written so far
    """
    shuffled_files: List[str] = []
    batch_idx = 0
    shard_idx = 0
    source_counts: Counter = Counter()
    total_processed = 0
    total_written = 0

    try:
        data = None
        if fs is not None:
            # GCS
            if fs.exists(checkpoint_path):
                with fs.open(checkpoint_path, 'rb') as f:
                    with gzip.open(f, 'rt', encoding='utf-8') as gz:
                        data = json.load(gz)
        else:
            # Local
            path = Path(checkpoint_path)
            if path.exists():
                with gzip.open(path, 'rt', encoding='utf-8') as gz:
                    data = json.load(gz)

        if data:
            shuffled_files = data.get('shuffled_files', [])
            batch_idx = data.get('batch_idx', 0)
            shard_idx = data.get('shard_idx', 0)
            source_counts = Counter(data.get('source_counts', {}))
            total_processed = data.get('total_processed', 0)
            total_written = data.get('total_written', 0)

            print(f"Loaded download-first checkpoint:")
            print(f"  Batch to resume: {batch_idx}")
            print(f"  Shard index: {shard_idx}")
            print(f"  Total processed: {total_processed:,}")
            print(f"  Total written: {total_written:,}")

    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")

    return shuffled_files, batch_idx, shard_idx, source_counts, total_processed, total_written


def save_download_first_checkpoint(
    checkpoint_path: str,
    shuffled_files: List[str],
    batch_idx: int,
    shard_idx: int,
    source_counts: Counter,
    total_processed: int,
    total_written: int,
    fs=None
):
    """Save checkpoint for download-first mode (gzip compressed JSON)."""
    data = {
        'shuffled_files': shuffled_files,
        'batch_idx': batch_idx,
        'shard_idx': shard_idx,
        'source_counts': dict(source_counts),
        'total_processed': total_processed,
        'total_written': total_written,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    try:
        if fs is not None:
            # GCS - write to temp file then upload
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.json.gz', delete=False) as tmp:
                tmp_path = tmp.name

            with gzip.open(tmp_path, 'wt', encoding='utf-8') as gz:
                json.dump(data, gz)

            fs.put(tmp_path, checkpoint_path)
            os.remove(tmp_path)
        else:
            # Local
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(checkpoint_path, 'wt', encoding='utf-8') as gz:
                json.dump(data, gz)

    except Exception as e:
        print(f"Warning: Could not save checkpoint: {e}", file=sys.stderr)


def download_parquet_batch(
    files: List[str],
    download_dir: str,
    repo_id: str = "mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M",
    max_workers: int = 16
) -> List[str]:
    """
    Download a batch of parquet files in parallel.

    Returns list of local file paths.
    """
    if not HF_HUB_AVAILABLE:
        raise RuntimeError("huggingface_hub not installed")

    # Enable high-performance mode
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    local_paths = []
    download_errors = []

    def download_file(filename: str) -> Optional[str]:
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=download_dir,
                local_dir_use_symlinks=False
            )
            return local_path
        except Exception as e:
            print(f"Error downloading {filename}: {e}", file=sys.stderr)
            return None

    print(f"Downloading {len(files)} files with {max_workers} workers...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, f): f for f in files}

        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result:
                local_paths.append(result)
            else:
                download_errors.append(futures[future])

            # Progress update every 10 files
            if (i + 1) % 10 == 0 or i + 1 == len(files):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  Downloaded {i + 1}/{len(files)} files ({rate:.1f} files/sec)")

    if download_errors:
        print(f"Warning: Failed to download {len(download_errors)} files")

    return local_paths


def process_dataset_download_first(
    output_path: str,
    download_dir: str,
    buffer_size: int = 100000,
    rows_per_shard: int = 10000,
    batch_size: int = 500,
    max_rows: Optional[int] = None,
    seed: int = 42,
    use_gcs: bool = False,
    num_workers: int = 4,
    download_workers: int = 16,
    resume: bool = False
):
    """
    Process dataset using download-first mode with batch processing.

    Three-layer shuffle:
    1. File-level shuffle: Shuffle parquet file order before batching
    2. Batch-level shuffle: Shuffle data within each batch
    3. Write-time shuffle: Shuffle buffer before writing shard

    Supports resume from checkpoint with --resume flag.
    """
    if not PANDAS_AVAILABLE:
        raise RuntimeError("pandas/pyarrow not installed. Install with: pip install pandas pyarrow")

    # Setup filesystem
    fs = None
    if use_gcs:
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            print(f"Using GCS filesystem, output: {output_path}")
        except ImportError:
            print("gcsfs not installed. Install with: pip install gcsfs")
            sys.exit(1)
    else:
        print(f"Using local filesystem, output: {output_path}")

    # Checkpoint path
    checkpoint_path = get_download_first_checkpoint_path(output_path)

    # Initialize state (may be overwritten by checkpoint)
    all_files: List[str] = []
    start_batch_idx = 0
    shard_idx = 0
    source_counts: Counter = Counter()
    total_processed = 0
    total_written = 0

    # Load checkpoint if resuming
    if resume:
        print(f"Checking for checkpoint at: {checkpoint_path}")
        (
            checkpoint_files,
            start_batch_idx,
            shard_idx,
            source_counts,
            total_processed,
            total_written
        ) = load_download_first_checkpoint(checkpoint_path, fs)

        if checkpoint_files:
            all_files = checkpoint_files
            print(f"Resuming from batch {start_batch_idx} with {len(all_files)} files")
        else:
            print("No valid checkpoint found, starting fresh")
            resume = False

    # If not resuming (or no valid checkpoint), list and shuffle files
    if not all_files:
        # Set random seed
        random.seed(seed)

        # List all parquet files
        print("Listing parquet files from HuggingFace...")
        all_files = list_dataset_parquet_files()

        # LAYER 1: File-level shuffle
        print(f"Shuffling {len(all_files)} files (seed={seed})...")
        random.shuffle(all_files)

    # Create download directory
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    # Initialize async shard writer
    shard_writer = AsyncShardWriter(output_path, fs, max_workers=num_workers)

    # Processing state
    start_time = time.time()

    print()
    print("=" * 60)
    print("Download-First Mode with Proportional Sampling")
    print("=" * 60)
    print(f"  Total files: {len(all_files)}")
    print(f"  Batch size: {batch_size} files")
    print(f"  Download workers: {download_workers}")
    print(f"  Write workers: {num_workers}")
    print(f"  Rows per shard: {rows_per_shard}")
    print(f"  Max rows: {max_rows or 'all'}")
    print(f"  Resume mode: {resume}")
    if resume:
        print(f"  Resuming from batch: {start_batch_idx}")
        print(f"  Already processed: {total_processed:,}")
        print(f"  Already written: {total_written:,}")
    print("=" * 60)
    print()

    # Process in batches
    num_batches = (len(all_files) + batch_size - 1) // batch_size

    try:
        for batch_idx in range(start_batch_idx, num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(all_files))
            batch_files = all_files[batch_start:batch_end]

            print()
            print(f"{'=' * 60}")
            print(f"Batch {batch_idx + 1}/{num_batches}: files {batch_start}-{batch_end - 1}")
            print(f"{'=' * 60}")

            # Download this batch
            local_paths = download_parquet_batch(
                batch_files,
                download_dir,
                max_workers=download_workers
            )

            if not local_paths:
                print(f"Warning: No files downloaded in batch {batch_idx + 1}")
                continue

            # Shuffle local_paths to ensure random file processing order
            # (download order depends on network speed, not our shuffled file order)
            random.shuffle(local_paths)

            # ========== PASS 1: Count rows in each file ==========
            print(f"Pass 1: Counting rows in {len(local_paths)} parquet files...")
            file_row_counts = {}
            total_available_rows = 0

            for local_path in local_paths:
                try:
                    parquet_file = pq.ParquetFile(local_path)
                    num_rows = parquet_file.metadata.num_rows
                    file_row_counts[local_path] = num_rows
                    total_available_rows += num_rows
                except Exception as e:
                    print(f"  Error reading metadata from {Path(local_path).name}: {e}", file=sys.stderr)
                    file_row_counts[local_path] = 0

            print(f"  Total available rows in batch: {total_available_rows:,}")

            # ========== Calculate number of shards for this batch ==========
            # Each batch uses ALL data and produces ceil(T / rows_per_shard) shards
            # For each shard, sample proportionally: rows_per_shard / T * a_i rows from file i
            reached_max_rows = False

            # Check if we've already reached max_rows limit
            if max_rows is not None and total_processed >= max_rows:
                reached_max_rows = True
                print(f"  Already reached max_rows limit: {max_rows}")
            else:
                # Calculate full batch shards (use all data)
                full_batch_shards = (total_available_rows + rows_per_shard - 1) // rows_per_shard

                # If max_rows is set, limit the number of shards we produce
                if max_rows is not None:
                    remaining_rows = max_rows - total_processed
                    # Use floor division to ensure we don't exceed max_rows
                    max_shards_allowed = remaining_rows // rows_per_shard
                    if max_shards_allowed <= 0:
                        reached_max_rows = True
                        print(f"  Remaining rows ({remaining_rows}) < rows_per_shard ({rows_per_shard}), stopping")
                    else:
                        num_shards_in_batch = min(full_batch_shards, max_shards_allowed)
                        if num_shards_in_batch < full_batch_shards:
                            print(f"  Limited by max_rows: {num_shards_in_batch} shards (would be {full_batch_shards} for full batch)")
                else:
                    num_shards_in_batch = full_batch_shards

            if not reached_max_rows:
                print(f"  Will produce {num_shards_in_batch} shards from this batch")

                # Initialize shard buckets - each will hold data for one output shard
                shard_buckets = [[] for _ in range(num_shards_in_batch)]

                # ========== PASS 2: Load each file, shuffle, and distribute to shard buckets ==========
                print(f"Pass 2: Loading, shuffling, and distributing data to {num_shards_in_batch} shards...")

                for file_idx, local_path in enumerate(local_paths):
                    row_count = file_row_counts.get(local_path, 0)
                    if row_count == 0:
                        continue

                    try:
                        df = pd.read_parquet(local_path)

                        # LAYER 2: Shuffle within each file
                        df = df.sample(frac=1, random_state=seed + file_idx).reset_index(drop=True)

                        # Distribute rows to shards proportionally
                        # Each shard s gets rows [s * row_count / num_shards, (s+1) * row_count / num_shards)
                        for shard_s in range(num_shards_in_batch):
                            start_row = int(shard_s * row_count / num_shards_in_batch)
                            end_row = int((shard_s + 1) * row_count / num_shards_in_batch)

                            for idx in range(start_row, end_row):
                                row = df.iloc[idx]
                                item = row.to_dict()
                                item_id = item.get('id', '')

                                # Track source
                                source = extract_source(item_id)
                                source_counts[source] += 1

                                # Convert to Levanter format
                                try:
                                    converted = convert_to_levanter(item)
                                    shard_buckets[shard_s].append(converted)
                                except Exception as e:
                                    print(f"Error converting item {item_id}: {e}", file=sys.stderr)

                        # Free memory after processing this file
                        del df

                    except Exception as e:
                        print(f"Error reading {local_path}: {e}", file=sys.stderr)
                        continue

                # ========== Write all shards from this batch ==========
                batch_rows_processed = sum(len(bucket) for bucket in shard_buckets)
                print(f"Loaded {batch_rows_processed} items, writing {len(shard_buckets)} shards...")

                for shard_s, bucket in enumerate(shard_buckets):
                    if bucket:
                        # LAYER 3: Final shuffle within each shard
                        random.shuffle(bucket)
                        shard_writer.submit_shard(bucket, shard_idx)
                        shard_idx += 1
                        total_processed += len(bucket)

                # Check completed shards
                completed = shard_writer.wait_for_completed()
                for completed_idx, written in completed:
                    total_written += written

                # Free shard buckets memory
                del shard_buckets

            # Clean up downloaded files to free space
            print(f"Cleaning up batch {batch_idx + 1} files...")
            for local_path in local_paths:
                try:
                    if os.path.exists(local_path):
                        os.remove(local_path)
                except Exception:
                    pass

            # Also clean up any empty directories
            try:
                for root, dirs, files in os.walk(download_dir, topdown=False):
                    for d in dirs:
                        dir_path = os.path.join(root, d)
                        if os.path.isdir(dir_path) and not os.listdir(dir_path):
                            os.rmdir(dir_path)
            except Exception:
                pass

            # Progress update
            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            print(f"Progress: {total_processed:,} processed, {total_written:,} written, "
                  f"{rate:.1f} items/sec, {elapsed/60:.1f} min elapsed")

            # Save checkpoint after each batch (save next batch index)
            save_download_first_checkpoint(
                checkpoint_path, all_files, batch_idx + 1, shard_idx,
                source_counts, total_processed, total_written, fs
            )

            # Check max rows - break after this batch if limit reached
            if max_rows is not None and total_processed >= max_rows:
                print(f"Reached max_rows limit: {max_rows} (processed: {total_processed})")
                break

        interrupted = False
    except KeyboardInterrupt:
        interrupted = True
        print("\nInterrupted! Saving checkpoint...")
        # Save checkpoint with current batch index (will resume this batch)
        save_download_first_checkpoint(
            checkpoint_path, all_files, batch_idx, shard_idx,
            source_counts, total_processed, total_written, fs
        )
        print(f"Checkpoint saved. Run with --resume to continue from batch {batch_idx}.")

    # Wait for pending writes
    print("Waiting for pending writes...")
    pending_written = shard_writer.wait_all()
    total_written += pending_written

    shard_writer.shutdown()

    # Clean up download directory
    print(f"Cleaning up download directory: {download_dir}")
    try:
        shutil.rmtree(download_dir, ignore_errors=True)
    except Exception:
        pass

    # Summary
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("Processing complete!")
    print(f"  Output: {output_path}")
    print(f"  Total shards: {shard_idx}")
    print(f"  Total rows processed: {total_processed:,}")
    print(f"  Total rows written: {total_written:,}")
    print(f"  Time elapsed: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
    print(f"  Average rate: {total_processed/elapsed:.1f} items/sec")
    print("=" * 60)

    # Source distribution
    print()
    print("=" * 60)
    print("Source Distribution:")
    print("-" * 60)
    total_sources = sum(source_counts.values())
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        percentage = count / total_sources * 100 if total_sources > 0 else 0
        print(f"  {source}: {count:,} ({percentage:.2f}%)")
    print("=" * 60)

    # Save stats
    stats = {
        "mode": "download-first",
        "total_rows_processed": total_processed,
        "total_rows_written": total_written,
        "total_shards": shard_idx,
        "source_distribution": dict(source_counts),
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "rows_per_shard": rows_per_shard,
        "seed": seed,
        "elapsed_minutes": elapsed / 60
    }

    stats_path = f"{output_path.rstrip('/')}/stats.json"
    if use_gcs and fs is not None:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(stats, tmp, indent=2)
            tmp_path = tmp.name
        try:
            fs.put(tmp_path, stats_path)
            print(f"\nStatistics saved to: {stats_path}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to: {stats_path}")


def process_dataset(
    output_path: str,
    buffer_size: int = 100000,
    rows_per_shard: int = 10000,
    shuffle_buffer_size: int = 500000,
    max_rows: Optional[int] = None,
    seed: int = 42,
    use_gcs: bool = True,
    resume: bool = False,
    hf_verbose: bool = False,
    no_hf_shuffle: bool = False,
    num_workers: int = 4,
    prefetch: int = 1000
):
    """
    Main processing function.

    Args:
        output_path: GCS path (gs://...) or local directory path
        buffer_size: Number of rows to keep in memory buffer for write-time shuffling
        rows_per_shard: Number of rows per output shard
        shuffle_buffer_size: Buffer size for HF's built-in streaming shuffle (cross-subset mixing)
        max_rows: Maximum number of rows to process (None for all)
        seed: Random seed for shuffling
        use_gcs: Whether to write to GCS or local filesystem
        resume: Whether to resume from checkpoint (lossless resume)
    """
    # Set random seed
    random.seed(seed)

    # Setup filesystem
    fs = None
    if use_gcs:
        try:
            import gcsfs
            fs = gcsfs.GCSFileSystem()
            print(f"Using GCS filesystem, output: {output_path}")
        except ImportError:
            print("gcsfs not installed. Install with: pip install gcsfs")
            sys.exit(1)
    else:
        print(f"Using local filesystem, output: {output_path}")

    # Configure HF download timeout
    # This helps with slower/unreliable network connections
    hf_timeout = os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT', '120')
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = hf_timeout
    print(f"HF download timeout: {hf_timeout}s")

    # Enable HF verbose logging if requested
    if hf_verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
        # Enable HF datasets logging
        from datasets.utils.logging import set_verbosity_info, enable_progress_bar
        set_verbosity_info()
        enable_progress_bar()
        print("HF verbose logging enabled - you'll see download progress from HF library")

    # Load dataset in streaming mode
    print("Loading LLaVA-OneVision dataset in streaming mode...")
    dataset = load_dataset(
        "mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M",
        streaming=True,
        split="train"
    )

    # Apply HF's built-in shuffle for cross-subset mixing (unless disabled)
    if no_hf_shuffle:
        print("HF shuffle DISABLED - data will be processed in original order")
        print("Note: Write-time buffer shuffle still applies for local mixing")
    else:
        # This shuffles shard order AND maintains a streaming buffer
        print(f"Applying HF shuffle with buffer_size={shuffle_buffer_size}...")
        print(f"Note: Shuffle requires downloading data to fill internal reservoir.")
        print(f"      Network timeouts are normal - HF will auto-retry.")
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer_size)

    # Note: HF IterableDataset doesn't support prefetch directly
    # Parallel speedup comes from async shard writing instead

    # Checkpoint path
    checkpoint_path = get_checkpoint_path(output_path)

    # Load checkpoint if resuming
    processed_ids: Set[str] = set()
    shard_idx = 0
    source_counts: Counter = Counter()
    total_processed = 0
    total_written = 0

    if resume:
        print(f"Checking for checkpoint at: {checkpoint_path}")
        processed_ids, shard_idx, source_counts, total_processed, total_written = \
            load_checkpoint(checkpoint_path, fs)
        if processed_ids:
            print(f"Resuming: will skip {len(processed_ids):,} already processed items")

    # Processing state
    buffer: List[Tuple[str, Dict[str, Any]]] = []  # (item_id, converted_data)
    skipped_count = 0
    skipped_already_processed = 0
    start_time = time.time()
    last_log_time = start_time

    # Initialize async shard writer
    shard_writer = AsyncShardWriter(output_path, fs, max_workers=num_workers)
    print(f"Initialized async shard writer with {num_workers} workers")

    print(f"Starting processing...")
    print(f"  Write-time buffer size: {buffer_size}")
    print(f"  HF shuffle buffer size: {shuffle_buffer_size}")
    print(f"  Rows per shard: {rows_per_shard}")
    print(f"  Max rows: {max_rows or 'all'}")
    print(f"  Resume mode: {resume}")
    print(f"  Parallel writers: {num_workers}")
    print()

    # Track newly processed ids for this session (for checkpoint)
    new_processed_ids: Set[str] = set()

    # Flag to track if we've shown the "buffer filling" message
    first_item_time = None

    print()
    print("=" * 60)
    print(f"HF Shuffle Buffer Info:")
    print(f"  - Buffer size: {shuffle_buffer_size:,} items")
    print(f"  - HF needs to download data to fill its internal reservoir")
    print(f"  - Network timeouts are normal during this phase - HF auto-retries")
    print(f"  - Progress updates will appear every 30 seconds")
    print(f"  - Dataset total: ~91M rows across multiple parquet shards")
    print("=" * 60)
    print()

    # Start progress tracker
    progress_tracker = ProgressTracker(shuffle_buffer_size, update_interval=30)
    progress_tracker.start()

    try:
        for idx, item in enumerate(dataset):
            # Log when first item arrives (buffer has started outputting)
            if idx == 0:
                first_item_time = time.time()
                buffer_fill_time = first_item_time - start_time
                progress_tracker.mark_first_output()
                progress_tracker.stop()
                print()
                print("=" * 60)
                print(f"FIRST ITEM RECEIVED after {buffer_fill_time:.1f}s!")
                print(f"HF shuffle buffer is now active. Data processing begins...")
                print("=" * 60)
                print()

            # Track items for initial phase (first 10 items for debugging)
            if idx < 10:
                item_id = item.get('id', 'unknown')
                source = extract_source(item_id)
                print(f"  Item {idx}: source={source}, id={item_id[:50]}...")

            if idx == 10:
                print(f"  ... (detailed logging disabled, see periodic progress updates)")
                print()
            # Check max rows (based on newly processed, not total)
            new_count = total_processed - (len(processed_ids) - len(new_processed_ids)) + len(new_processed_ids)
            if max_rows is not None and new_count >= max_rows:
                print(f"Reached max_rows limit: {max_rows}")
                break

            try:
                # Get item id
                item_id = item.get('id', '')

                # Skip if already processed (lossless resume)
                if item_id in processed_ids:
                    skipped_already_processed += 1
                    if skipped_already_processed % 100000 == 0:
                        print(f"Skipped {skipped_already_processed:,} already processed items...")
                    continue

                # Track source distribution
                source = extract_source(item_id)
                source_counts[source] += 1

                # Convert to Levanter format
                converted = convert_to_levanter(item)
                buffer.append((item_id, converted))
                new_processed_ids.add(item_id)
                total_processed += 1

                # Log progress periodically (more frequent at start, then every 30s)
                current_time = time.time()
                log_interval = 10 if total_processed < 10000 else 30  # Log every 10s at start
                if current_time - last_log_time >= log_interval:
                    elapsed = current_time - start_time
                    rate = len(new_processed_ids) / elapsed if elapsed > 0 else 0
                    print(f"Progress: {total_processed:,} total, {len(new_processed_ids):,} new, "
                          f"{total_written:,} written, {len(buffer):,} in buffer, "
                          f"{skipped_already_processed:,} resumed, {skipped_count:,} errors, "
                          f"{rate:.1f} rows/sec")
                    last_log_time = current_time

                # Check if buffer is full
                if len(buffer) >= buffer_size:
                    # Shuffle buffer
                    random.shuffle(buffer)

                    # Write shards until buffer is below threshold (ASYNC)
                    while len(buffer) >= rows_per_shard:
                        shard_items = buffer[:rows_per_shard]
                        buffer = buffer[rows_per_shard:]

                        # Extract just the converted data for writing
                        shard_data = [item[1] for item in shard_items]
                        shard_ids = [item[0] for item in shard_items]

                        # Submit shard for async writing
                        shard_writer.submit_shard(shard_data, shard_idx)
                        shard_idx += 1

                        # Update processed_ids (shard is being written)
                        processed_ids.update(shard_ids)

                    # Check for completed shards and update counts
                    completed = shard_writer.wait_for_completed()
                    for completed_idx, written in completed:
                        total_written += written
                        print(f"  Shard {completed_idx} written ({written} rows)")

                    # Save checkpoint periodically
                    if completed:
                        save_checkpoint(
                            checkpoint_path, processed_ids, shard_idx,
                            source_counts, total_processed, total_written, fs
                        )

            except Exception as e:
                print(f"Error processing row {idx}: {e}", file=sys.stderr)
                skipped_count += 1
                continue

    except KeyboardInterrupt:
        print(f"\nInterrupted! Waiting for pending writes to complete...")
        progress_tracker.stop()
        # Wait for all pending shard writes
        total_written += shard_writer.wait_all()
        shard_writer.shutdown()
        # Save checkpoint with current progress
        processed_ids.update(new_processed_ids)
        save_checkpoint(
            checkpoint_path, processed_ids, shard_idx,
            source_counts, total_processed, total_written, fs
        )
        print(f"Checkpoint saved. Run with --resume to continue.")
        return
    finally:
        # Ensure progress tracker is stopped
        progress_tracker.stop()

    # Wait for any pending writes from the main loop
    print("Waiting for pending shard writes to complete...")
    pending_written = shard_writer.wait_all()
    total_written += pending_written

    # Process remaining buffer
    if buffer:
        print(f"Processing remaining {len(buffer)} rows in buffer...")
        random.shuffle(buffer)

        while len(buffer) >= rows_per_shard:
            shard_items = buffer[:rows_per_shard]
            buffer = buffer[rows_per_shard:]

            shard_data = [item[1] for item in shard_items]
            shard_ids = [item[0] for item in shard_items]

            # Submit shard for async writing
            shard_writer.submit_shard(shard_data, shard_idx)
            shard_idx += 1
            processed_ids.update(shard_ids)

        # Write final partial shard if any data remains
        if buffer:
            shard_data = [item[1] for item in buffer]
            shard_ids = [item[0] for item in buffer]

            shard_writer.submit_shard(shard_data, shard_idx)
            shard_idx += 1
            processed_ids.update(shard_ids)

        # Wait for all remaining writes
        remaining_written = shard_writer.wait_all()
        total_written += remaining_written

    # Final checkpoint and cleanup
    save_checkpoint(
        checkpoint_path, processed_ids, shard_idx,
        source_counts, total_processed, total_written, fs
    )
    shard_writer.shutdown()

    # Summary
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("Processing complete!")
    print(f"  Output: {output_path}")
    print(f"  Total shards: {shard_idx}")
    print(f"  Total rows written: {total_written:,}")
    print(f"  Rows skipped: {skipped_count:,}")
    print(f"  Time elapsed: {elapsed/3600:.2f} hours")
    print(f"  Average rate: {total_written/elapsed:.1f} rows/sec")
    print("=" * 60)

    # Source distribution statistics
    print()
    print("=" * 60)
    print("Source Distribution:")
    print("-" * 60)
    total_sources = sum(source_counts.values())
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        percentage = count / total_sources * 100 if total_sources > 0 else 0
        print(f"  {source}: {count:,} ({percentage:.2f}%)")
    print("=" * 60)

    # Save statistics to JSON
    stats = {
        "total_rows_processed": total_processed,
        "total_rows_written": total_written,
        "rows_skipped": skipped_count,
        "total_shards": shard_idx,
        "source_distribution": dict(source_counts),
        "shuffle_buffer_size": shuffle_buffer_size,
        "buffer_size": buffer_size,
        "rows_per_shard": rows_per_shard,
        "seed": seed,
        "elapsed_hours": elapsed / 3600
    }

    # Write stats file
    if use_gcs and fs is not None:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(stats, tmp, indent=2)
            tmp_path = tmp.name
        try:
            stats_path = f"{output_path.rstrip('/')}/stats.json"
            fs.put(tmp_path, stats_path)
            print(f"\nStatistics saved to: {stats_path}")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        stats_path = Path(output_path) / "stats.json"
        Path(output_path).mkdir(parents=True, exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nStatistics saved to: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LLaVA-OneVision dataset to Levanter format with streaming and shuffling."
    )
    parser.add_argument(
        "--output-gcs",
        type=str,
        default=None,
        help="GCS output path (e.g., gs://bucket/path/)"
    )
    parser.add_argument(
        "--output-local",
        type=str,
        default=None,
        help="Local output directory path"
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=100000,
        help="Buffer size for write-time shuffling (default: 100000)"
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=500000,
        help="Buffer size for HF's streaming shuffle, for cross-subset mixing (default: 500000)"
    )
    parser.add_argument(
        "--rows-per-shard",
        type=int,
        default=1000,
        help="Rows per output shard (default: 1000)"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum rows to process (default: all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling (default: 42)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (lossless resume). Skips already processed items."
    )
    parser.add_argument(
        "--hf-verbose",
        action="store_true",
        help="Enable HuggingFace verbose logging to see download progress"
    )
    parser.add_argument(
        "--hf-timeout",
        type=int,
        default=120,
        help="HuggingFace download timeout in seconds (default: 120)"
    )
    parser.add_argument(
        "--no-hf-shuffle",
        action="store_true",
        help="Disable HF streaming shuffle (much faster, but less cross-subset mixing)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for writing shards (default: 4)"
    )
    parser.add_argument(
        "--prefetch",
        type=int,
        default=1000,
        help="(Deprecated - not used) HF IterableDataset doesn't support prefetch"
    )

    # Download-first mode arguments
    parser.add_argument(
        "--download-first",
        action="store_true",
        help="Use download-first mode: parallel download parquet files, then process locally (RECOMMENDED for speed)"
    )
    parser.add_argument(
        "--download-dir",
        type=str,
        default="/dev/shm/hf_cache",
        help="Directory for temporary parquet downloads (default: /dev/shm/hf_cache)"
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=16,
        help="Number of parallel download workers (default: 16)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=40,
        help="Number of parquet files per batch in download-first mode (default: 40, ~144GB per batch with 3.6GB/file avg)"
    )

    args = parser.parse_args()

    # Validate output
    if args.output_gcs is None and args.output_local is None:
        print("Error: Must specify either --output-gcs or --output-local", file=sys.stderr)
        sys.exit(1)

    if args.output_gcs is not None and args.output_local is not None:
        print("Error: Cannot specify both --output-gcs and --output-local", file=sys.stderr)
        sys.exit(1)

    use_gcs = args.output_gcs is not None
    output_path = args.output_gcs if use_gcs else args.output_local

    # Validate buffer size
    if args.buffer_size < args.rows_per_shard:
        print(f"Warning: buffer_size ({args.buffer_size}) < rows_per_shard ({args.rows_per_shard}). "
              f"Setting buffer_size = rows_per_shard * 2")
        args.buffer_size = args.rows_per_shard * 2

    # Set HF timeout environment variable early
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = str(args.hf_timeout)

    print("=" * 60)
    print("LLaVA-OneVision to Levanter Converter")
    print("=" * 60)
    print(f"Mode: {'DOWNLOAD-FIRST (parallel)' if args.download_first else 'STREAMING'}")
    print(f"Output: {output_path}")
    print(f"Use GCS: {use_gcs}")
    print(f"Buffer size (write-time): {args.buffer_size}")
    print(f"Rows per shard: {args.rows_per_shard}")
    print(f"Max rows: {args.max_rows or 'all'}")
    print(f"Seed: {args.seed}")

    if args.download_first:
        print(f"Download dir: {args.download_dir}")
        print(f"Download workers: {args.download_workers}")
        print(f"Batch size: {args.batch_size} files")
        print(f"Write workers: {args.num_workers}")
        print(f"Resume mode: {args.resume}")
    else:
        print(f"Shuffle buffer size (HF): {args.shuffle_buffer_size}")
        print(f"Resume mode: {args.resume}")
        print(f"HF verbose: {args.hf_verbose}")
        print(f"HF timeout: {args.hf_timeout}s")
        print(f"HF shuffle: {'DISABLED' if args.no_hf_shuffle else 'enabled'}")
        print(f"Parallel writers: {args.num_workers}")
    print("=" * 60)
    print()

    if args.download_first:
        # Use download-first mode (RECOMMENDED)
        process_dataset_download_first(
            output_path=output_path,
            download_dir=args.download_dir,
            buffer_size=args.buffer_size,
            rows_per_shard=args.rows_per_shard,
            batch_size=args.batch_size,
            max_rows=args.max_rows,
            seed=args.seed,
            use_gcs=use_gcs,
            num_workers=args.num_workers,
            download_workers=args.download_workers,
            resume=args.resume
        )
    else:
        # Use streaming mode (original)
        process_dataset(
            output_path=output_path,
            buffer_size=args.buffer_size,
            rows_per_shard=args.rows_per_shard,
            shuffle_buffer_size=args.shuffle_buffer_size,
            max_rows=args.max_rows,
            seed=args.seed,
            use_gcs=use_gcs,
            resume=args.resume,
            hf_verbose=args.hf_verbose,
            no_hf_shuffle=args.no_hf_shuffle,
            num_workers=args.num_workers,
            prefetch=args.prefetch
        )


if __name__ == "__main__":
    main()
