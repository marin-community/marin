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
import gc
import gzip
import io
import json
import os
import random
import subprocess
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
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback: simple iterator that does nothing
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from huggingface_hub import HfApi, hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class AsyncGCSUploader:
    """
    Background thread pool for uploading files to GCS.

    This class ONLY handles GCS uploads (I/O bound), NOT parquet conversion (CPU bound).
    Parquet conversion should be done serially before calling this uploader.
    """

    def __init__(self, fs, max_workers: int = 8):
        self.fs = fs
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.futures = []
        self.lock = threading.Lock()
        self.total_uploaded = 0
        self.files_completed = 0

    def submit_upload(self, local_path: str, remote_path: str, num_rows: int = 0) -> None:
        """Submit a file for async upload to GCS."""
        future = self.executor.submit(self._upload_file, local_path, remote_path)
        with self.lock:
            self.futures.append((future, local_path, remote_path, num_rows))

    def _upload_file(self, local_path: str, remote_path: str) -> bool:
        """Upload a single file to GCS using gcloud storage (faster than gcsfs)."""
        try:
            # Use gcloud storage cp (recommended by Google, 2-3x faster than gsutil)
            result = subprocess.run(
                ["gcloud", "storage", "cp", "--quiet", local_path, remote_path],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            if result.returncode == 0:
                return True
            else:
                # gcloud failed, try fallback to gcsfs
                print(f"gcloud upload failed ({result.returncode}), falling back to gcsfs: {result.stderr.strip()}", file=sys.stderr)
                self.fs.put(local_path, remote_path)
                return True
        except subprocess.TimeoutExpired:
            print(f"Upload timeout for {local_path}, falling back to gcsfs", file=sys.stderr)
            try:
                self.fs.put(local_path, remote_path)
                return True
            except Exception as e:
                print(f"Fallback upload also failed: {e}", file=sys.stderr)
                return False
        except FileNotFoundError:
            # gcloud not installed, use gcsfs
            try:
                self.fs.put(local_path, remote_path)
                return True
            except Exception as e:
                print(f"Error uploading {local_path} to {remote_path}: {e}", file=sys.stderr)
                return False
        except Exception as e:
            print(f"Error uploading {local_path} to {remote_path}: {e}", file=sys.stderr)
            return False

    def wait_for_completed(self, delete_local: bool = True) -> List[Tuple[str, int]]:
        """
        Check for completed uploads and return list of (remote_path, num_rows).
        Optionally delete local files after successful upload.
        """
        completed = []
        with self.lock:
            remaining = []
            for future, local_path, remote_path, num_rows in self.futures:
                if future.done():
                    try:
                        success = future.result()
                        if success:
                            completed.append((remote_path, num_rows))
                            self.total_uploaded += num_rows
                            self.files_completed += 1
                            # Delete local file after successful upload
                            if delete_local and os.path.exists(local_path):
                                os.remove(local_path)
                    except Exception as e:
                        print(f"Error in upload future for {local_path}: {e}", file=sys.stderr)
                else:
                    remaining.append((future, local_path, remote_path, num_rows))
            self.futures = remaining
        return completed

    def wait_all(self, delete_local: bool = True) -> int:
        """Wait for all pending uploads to complete. Returns total rows uploaded."""
        with self.lock:
            futures_to_wait = list(self.futures)
            self.futures = []

        for future, local_path, remote_path, num_rows in futures_to_wait:
            try:
                success = future.result()
                if success:
                    self.total_uploaded += num_rows
                    self.files_completed += 1
                    print(f"  Uploaded {Path(local_path).name} ({num_rows} rows)")
                    # Delete local file after successful upload
                    if delete_local and os.path.exists(local_path):
                        os.remove(local_path)
                else:
                    print(f"  Failed to upload {local_path}", file=sys.stderr)
            except Exception as e:
                print(f"Error uploading {local_path}: {e}", file=sys.stderr)

        return self.total_uploaded

    def pending_count(self) -> int:
        """Return the number of pending uploads."""
        with self.lock:
            return len(self.futures)

    def shutdown(self):
        """Shutdown the executor."""
        self.executor.shutdown(wait=True)


def write_shard_to_local(
    rows: List[Dict[str, Any]],
    shard_idx: int,
    local_dir: str
) -> Tuple[str, int]:
    """
    Write a shard to local filesystem (SERIAL, not async).

    Uses Dataset.from_list which properly handles complex nested structures
    (like images with bytes). Includes explicit memory cleanup.

    Args:
        rows: List of row dictionaries
        shard_idx: Shard index for naming
        local_dir: Local directory to write to

    Returns:
        Tuple of (local_path, num_rows)
    """
    if not rows:
        return None, 0

    shard_name = f"train-{shard_idx:05d}.parquet"
    local_path = str(Path(local_dir) / shard_name)

    # Ensure directory exists
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    num_rows = len(rows)

    # Use Dataset.from_list (handles complex nested structures like images)
    dataset = Dataset.from_list(rows)
    dataset.to_parquet(local_path)

    # Explicit memory cleanup
    del dataset
    gc.collect()

    return local_path, num_rows


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
        # Explicitly close and delete BytesIO to free memory
        img_byte_arr.close()
        del img_byte_arr
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

def list_dataset_parquet_files_with_sizes(repo_ids: List[str]) -> List[Tuple[str, str, int]]:
    """
    List all parquet files from multiple dataset repositories with their sizes.

    Args:
        repo_ids: List of HuggingFace repo IDs to fetch files from

    Returns:
        List of (repo_id, filename, size_in_bytes) tuples
    """
    if not HF_HUB_AVAILABLE:
        raise RuntimeError("huggingface_hub not installed. Install with: pip install huggingface_hub")

    api = HfApi()
    all_files_with_sizes = []

    for repo_id in repo_ids:
        print(f"Fetching file list and sizes from {repo_id}...")
        repo_files = []
        for item in api.list_repo_tree(repo_id, repo_type="dataset", recursive=True):
            if item.path.endswith('.parquet'):
                # item.size is in bytes
                repo_files.append((repo_id, item.path, item.size or 0))

        repo_size_gb = sum(size for _, _, size in repo_files) / (1024**3)
        print(f"  Found {len(repo_files)} parquet files from {repo_id}, size: {repo_size_gb:.1f} GB")
        all_files_with_sizes.extend(repo_files)

    total_size_gb = sum(size for _, _, size in all_files_with_sizes) / (1024**3)
    print(f"Total: {len(all_files_with_sizes)} parquet files from {len(repo_ids)} repo(s), total size: {total_size_gb:.1f} GB")
    return all_files_with_sizes


def create_size_limited_batches(
    files_with_sizes: List[Tuple[str, str, int]],
    max_batch_bytes: int
) -> Tuple[List[List[Tuple[str, str]]], List[int]]:
    """
    Create batches of files where each batch doesn't exceed max_batch_bytes.

    Args:
        files_with_sizes: List of (repo_id, filename, size_in_bytes) tuples
        max_batch_bytes: Maximum total size per batch in bytes

    Returns:
        Tuple of (batches, batch_sizes) where:
        - batches: List of batches, each batch is a list of (repo_id, filename) tuples
        - batch_sizes: List of total sizes in bytes for each batch
    """
    batches = []
    batch_sizes = []
    current_batch = []
    current_size = 0

    for repo_id, filename, size in files_with_sizes:
        # If adding this file would exceed limit, start new batch
        if current_size + size > max_batch_bytes and current_batch:
            batches.append(current_batch)
            batch_sizes.append(current_size)
            current_batch = []
            current_size = 0

        current_batch.append((repo_id, filename))
        current_size += size

    # Don't forget the last batch
    if current_batch:
        batches.append(current_batch)
        batch_sizes.append(current_size)

    return batches, batch_sizes


def get_download_first_checkpoint_path(output_path: str, local_checkpoint_dir: Optional[str] = None) -> str:
    """
    Get the checkpoint file path for download-first mode.

    Tries hash-based filename first, falls back to non-hash version for backward compatibility.

    Args:
        output_path: The output path (GCS or local)
        local_checkpoint_dir: If specified, save checkpoint locally instead of with output.
                             This is useful when output is on GCS to avoid slow checkpoint I/O.

    Returns:
        Local file path for the checkpoint
    """
    if local_checkpoint_dir:
        # Use local directory for checkpoint
        Path(local_checkpoint_dir).mkdir(parents=True, exist_ok=True)
        # Create a unique checkpoint name based on output path hash to avoid conflicts
        import hashlib
        path_hash = hashlib.md5(output_path.encode()).hexdigest()[:8]

        # 优先使用带 hash 的文件名
        hash_path = Path(local_checkpoint_dir) / f"checkpoint_download_first_{path_hash}.json.gz"
        # 回退到无 hash 的文件名（向后兼容）
        legacy_path = Path(local_checkpoint_dir) / "checkpoint_download_first.json.gz"

        # 如果带 hash 的存在，用它；否则如果旧版本存在，用旧版本；否则返回新格式路径
        if hash_path.exists():
            return str(hash_path)
        elif legacy_path.exists():
            print(f"Note: Using legacy checkpoint file (no hash): {legacy_path}")
            return str(legacy_path)
        else:
            # 新 checkpoint 用带 hash 的格式
            return str(hash_path)
    else:
        # Save with output (original behavior for local output)
        return f"{output_path.rstrip('/')}/checkpoint_download_first.json.gz"


def load_download_first_checkpoint(
    checkpoint_path: str
) -> Tuple[List[List[Tuple[str, str]]], List[int], int, int, Counter, int, int]:
    """
    Load checkpoint for download-first mode from LOCAL filesystem.

    Returns:
        batches: Pre-computed batches (list of (repo_id, filename) tuple lists) to maintain same batching on resume
        batch_sizes: Size in bytes for each batch
        batch_idx: Current batch index to resume from
        shard_idx: Current shard index to continue from
        source_counts: Source distribution counter
        total_processed: Total rows processed so far
        total_written: Total rows written so far
    """
    batches: List[List[Tuple[str, str]]] = []
    batch_sizes: List[int] = []
    batch_idx = 0
    shard_idx = 0
    source_counts: Counter = Counter()
    total_processed = 0
    total_written = 0

    try:
        path = Path(checkpoint_path)
        if path.exists():
            with gzip.open(path, 'rt', encoding='utf-8') as gz:
                data = json.load(gz)

            # Support both old format (shuffled_files) and new format (batches)
            if 'batches' in data:
                # Convert lists back to tuples (JSON doesn't preserve tuple type)
                raw_batches = data.get('batches', [])
                batches = [[tuple(item) for item in batch] for batch in raw_batches]
                batch_sizes = data.get('batch_sizes', [0] * len(batches))  # Default to 0 if not present
            elif 'shuffled_files' in data:
                # Old format - convert to single batch (will need re-batching)
                print("Warning: Old checkpoint format detected, will re-batch files")
                batches = []  # Signal that re-batching is needed
            batch_idx = data.get('batch_idx', 0)
            shard_idx = data.get('shard_idx', 0)
            source_counts = Counter(data.get('source_counts', {}))
            total_processed = data.get('total_processed', 0)
            total_written = data.get('total_written', 0)

            print(f"Loaded download-first checkpoint from: {checkpoint_path}")
            print(f"  Number of batches: {len(batches)}")
            print(f"  Batch to resume: {batch_idx}")
            print(f"  Shard index: {shard_idx}")
            print(f"  Total processed: {total_processed:,}")
            print(f"  Total written: {total_written:,}")

    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")

    return batches, batch_sizes, batch_idx, shard_idx, source_counts, total_processed, total_written


def save_download_first_checkpoint(
    checkpoint_path: str,
    batches: List[List[Tuple[str, str]]],
    batch_sizes: List[int],
    batch_idx: int,
    shard_idx: int,
    source_counts: Counter,
    total_processed: int,
    total_written: int,
    gcs_checkpoint_path: Optional[str] = None,
    fs=None
):
    """Save checkpoint for download-first mode to LOCAL filesystem and optionally to GCS (gzip compressed JSON)."""
    data = {
        'batches': batches,
        'batch_sizes': batch_sizes,
        'batch_idx': batch_idx,
        'shard_idx': shard_idx,
        'source_counts': dict(source_counts),
        'total_processed': total_processed,
        'total_written': total_written,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    # Save to local filesystem
    try:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(checkpoint_path, 'wt', encoding='utf-8') as gz:
            json.dump(data, gz)
        print(f"  Checkpoint saved locally: {checkpoint_path} (batch {batch_idx})")
    except Exception as e:
        print(f"ERROR: Could not save checkpoint to {checkpoint_path}: {e}", file=sys.stderr)

    # Also save to GCS if path provided
    if gcs_checkpoint_path and fs is not None:
        try:
            fs.put(checkpoint_path, gcs_checkpoint_path)
            print(f"  Checkpoint saved to GCS: {gcs_checkpoint_path}")
        except Exception as e:
            print(f"WARNING: Could not save checkpoint to GCS {gcs_checkpoint_path}: {e}", file=sys.stderr)


def download_parquet_batch(
    files: List[Tuple[str, str]],
    download_dir: str,
    max_workers: int = 16
) -> List[str]:
    """
    Download a batch of parquet files in parallel.

    Args:
        files: List of (repo_id, filename) tuples
        download_dir: Directory to download files to
        max_workers: Number of parallel download workers

    Returns list of local file paths.
    """
    if not HF_HUB_AVAILABLE:
        raise RuntimeError("huggingface_hub not installed")

    # Enable high-performance mode
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    local_paths = []
    download_errors = []

    def download_file(repo_id: str, filename: str) -> Optional[str]:
        try:
            # Create a repo-specific subdirectory to avoid filename conflicts
            repo_download_dir = os.path.join(download_dir, repo_id.replace("/", "_"))
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=repo_download_dir,
                local_dir_use_symlinks=False
            )
            return local_path
        except Exception as e:
            print(f"Error downloading {repo_id}/{filename}: {e}", file=sys.stderr)
            return None

    print(f"Downloading {len(files)} files with {max_workers} workers...")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, repo_id, filename): (repo_id, filename) for repo_id, filename in files}

        # Use tqdm for progress bar
        pbar = tqdm(
            as_completed(futures),
            total=len(files),
            desc="  Downloading",
            unit="file"
        )

        for future in pbar:
            result = future.result()
            if result:
                local_paths.append(result)
                pbar.set_postfix(ok=len(local_paths), err=len(download_errors))
            else:
                download_errors.append(futures[future])
                pbar.set_postfix(ok=len(local_paths), err=len(download_errors))

        pbar.close()

    elapsed = time.time() - start_time
    rate = len(files) / elapsed if elapsed > 0 else 0
    print(f"  Download complete: {len(local_paths)} files in {elapsed:.1f}s ({rate:.1f} files/sec)")

    if download_errors:
        print(f"  Warning: Failed to download {len(download_errors)} files")

    return local_paths


def process_dataset_download_first(
    output_path: str,
    download_dir: str,
    repo_ids: List[str],
    buffer_size: int = 100000,
    rows_per_shard: int = 10000,
    max_batch_gb: float = 150.0,
    max_rows: Optional[int] = None,
    seed: int = 42,
    use_gcs: bool = False,
    upload_workers: int = 8,
    download_workers: int = 16,
    read_workers: int = 8,
    write_workers: int = 8,
    resume: bool = False,
    checkpoint_dir: Optional[str] = None,
    local_shard_dir: Optional[str] = None,
    max_local_shards: int = 20
):
    """
    Process dataset using download-first mode with batch processing.

    Three-layer shuffle:
    1. File-level shuffle: Shuffle parquet file order before batching
    2. Batch-level shuffle: Shuffle data within each batch
    3. Write-time shuffle: Shuffle buffer before writing shard

    Supports resume from checkpoint with --resume flag.
    Batches are limited by size (max_batch_gb) to avoid disk space issues.
    Checkpoint is saved locally (not on GCS) for faster I/O.

    Parquet conversion (CPU-intensive) is done serially.
    GCS uploads (I/O-intensive) are done in parallel.
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

    # Checkpoint path (always local for faster I/O)
    # Default to download_dir if no checkpoint_dir specified
    effective_checkpoint_dir = checkpoint_dir or download_dir
    checkpoint_path = get_download_first_checkpoint_path(output_path, effective_checkpoint_dir)
    print(f"Checkpoint will be saved locally at: {checkpoint_path}")

    # GCS checkpoint path (same directory as output data)
    gcs_checkpoint_path = None
    if use_gcs:
        gcs_checkpoint_path = f"{output_path.rstrip('/')}/checkpoint_download_first.json.gz"
        print(f"Checkpoint will also be saved to GCS: {gcs_checkpoint_path}")

    # Initialize state (may be overwritten by checkpoint)
    batches: List[List[str]] = []
    batch_sizes: List[int] = []
    start_batch_idx = 0
    shard_idx = 0
    source_counts: Counter = Counter()
    total_processed = 0
    total_written = 0

    # Load checkpoint if resuming (always from local filesystem)
    if resume:
        print(f"Checking for checkpoint at: {checkpoint_path}")
        
        # If local checkpoint doesn't exist but GCS checkpoint does, download it first
        local_ckpt_exists = Path(checkpoint_path).exists()
        if not local_ckpt_exists and use_gcs and gcs_checkpoint_path and fs is not None:
            print(f"Local checkpoint not found, checking GCS: {gcs_checkpoint_path}")
            try:
                if fs.exists(gcs_checkpoint_path):
                    print(f"Found checkpoint on GCS, downloading to local...")
                    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                    fs.get(gcs_checkpoint_path, checkpoint_path)
                    print(f"Downloaded checkpoint from GCS to: {checkpoint_path}")
                else:
                    print(f"No checkpoint found on GCS either.")
            except Exception as e:
                print(f"Warning: Could not download checkpoint from GCS: {e}")
        
        (
            checkpoint_batches,
            checkpoint_batch_sizes,
            start_batch_idx,
            shard_idx,
            source_counts,
            total_processed,
            total_written
        ) = load_download_first_checkpoint(checkpoint_path)

        if checkpoint_batches:
            batches = checkpoint_batches
            batch_sizes = checkpoint_batch_sizes
            print(f"Resuming from batch {start_batch_idx} with {len(batches)} total batches")
        else:
            print("No valid checkpoint found, starting fresh")
            resume = False

    # If not resuming (or no valid checkpoint), list files and create size-limited batches
    if not batches:
        # Set random seed
        random.seed(seed)

        # List all parquet files with sizes from all repos
        print(f"Listing parquet files from {len(repo_ids)} repo(s)...")
        files_with_sizes = list_dataset_parquet_files_with_sizes(repo_ids)

        # LAYER 1: File-level shuffle
        print(f"Shuffling {len(files_with_sizes)} files (seed={seed})...")
        random.shuffle(files_with_sizes)

        # Create size-limited batches
        max_batch_bytes = int(max_batch_gb * 1024 * 1024 * 1024)
        batches, batch_sizes = create_size_limited_batches(files_with_sizes, max_batch_bytes)
        print(f"Created {len(batches)} batches with max {max_batch_gb:.1f} GB each")

    # Create download directory
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    # Create local shard directory for temporary parquet files
    effective_local_shard_dir = local_shard_dir or f"{download_dir}/shards"
    Path(effective_local_shard_dir).mkdir(parents=True, exist_ok=True)

    # Initialize GCS uploader (only for GCS mode)
    gcs_uploader = None
    if use_gcs and fs is not None:
        gcs_uploader = AsyncGCSUploader(fs, max_workers=upload_workers)

    # Processing state
    start_time = time.time()

    # Calculate total files across all batches
    total_files = sum(len(batch) for batch in batches)

    print()
    print("=" * 60)
    print("Download-First Mode with Proportional Sampling")
    print("=" * 60)
    print(f"  Total files: {total_files}")
    print(f"  Total batches: {len(batches)}")
    print(f"  Max batch size: {max_batch_gb:.1f} GB")
    print(f"  Download workers: {download_workers}")
    print(f"  Read workers: {read_workers}")
    print(f"  Write workers: {write_workers}")
    print(f"  Upload workers: {upload_workers}")
    print(f"  Local shard dir: {effective_local_shard_dir}")
    print(f"  Max local shards: {max_local_shards}")
    print(f"  Rows per shard: {rows_per_shard}")
    print(f"  Max rows: {max_rows or 'all'}")
    print(f"  Resume mode: {resume}")
    if resume:
        print(f"  Resuming from batch: {start_batch_idx}")
        print(f"  Already processed: {total_processed:,}")
        print(f"  Already written: {total_written:,}")
    print("=" * 60)
    print()

    # Process batches
    num_batches = len(batches)

    try:
        for batch_idx in range(start_batch_idx, num_batches):
            # Check if we should stop BEFORE downloading
            if max_rows is not None:
                remaining_rows = max_rows - total_processed
                if remaining_rows < rows_per_shard:
                    print(f"\nStopping: remaining rows ({remaining_rows}) < rows_per_shard ({rows_per_shard})")
                    print(f"Total processed: {total_processed:,}")
                    break

            batch_files = batches[batch_idx]
            batch_size_gb = batch_sizes[batch_idx] / (1024**3) if batch_idx < len(batch_sizes) else 0

            print()
            print(f"{'=' * 60}")
            print(f"Batch {batch_idx + 1}/{num_batches}: {len(batch_files)} files, {batch_size_gb:.1f} GB")
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
                # Calculate sample ratio - if we're limiting shards, we need to sample fewer rows
                sample_ratio = num_shards_in_batch / full_batch_shards
                expected_rows = int(total_available_rows * sample_ratio)
                print(f"  Will produce {num_shards_in_batch} shards from this batch")
                print(f"  Sample ratio: {sample_ratio:.4f} (sampling ~{expected_rows:,} of {total_available_rows:,} rows)")

                # ========== PASS 2: Batch Shard Processing (I/O Efficient) ==========
                # Process N output shards at a time:
                #   - Keep N shard buffers in memory
                #   - Read each source file ONCE and distribute to all N shards
                #   - Shuffle and write all N shards
                # This reduces file reads from (num_shards × num_files) to (num_files × num_shard_batches)

                # Number of shards to process in parallel (memory vs I/O tradeoff)
                shards_per_round = min(max_local_shards, num_shards_in_batch)  # Use max_local_shards as batch size
                num_rounds = (num_shards_in_batch + shards_per_round - 1) // shards_per_round

                print(f"Pass 2: Processing {num_shards_in_batch} shards in {num_rounds} rounds ({shards_per_round} shards/round)...")
                print(f"  File reads reduced from {num_shards_in_batch * len(local_paths)} to {num_rounds * len(local_paths)}")

                batch_rows_processed = 0
                local_shard_files = []
                shards_written_in_batch = 0

                # Calculate how many rows each file contributes per shard
                # a_i = file_row_count / num_shards (fractional)
                samples_per_file = {}
                for local_path, row_count in file_row_counts.items():
                    if row_count > 0:
                        samples_per_file[local_path] = row_count / num_shards_in_batch

                # Pre-shuffle row indices for each file (for better shuffle quality)
                print(f"  Pre-shuffling row indices for {len(local_paths)} files...")
                file_shuffled_indices = {}
                for local_path in local_paths:
                    n_rows = file_row_counts.get(local_path, 0)
                    if n_rows > 0:
                        indices = list(range(n_rows))
                        random.shuffle(indices)
                        file_shuffled_indices[local_path] = indices

                # Track pending uploads for pipeline parallelism
                # GCS upload runs in parallel with next round's file reading
                pending_upload_files = []

                # Process shards in rounds
                for round_idx in range(num_rounds):
                    round_start_shard = round_idx * shards_per_round
                    round_end_shard = min((round_idx + 1) * shards_per_round, num_shards_in_batch)
                    num_shards_in_round = round_end_shard - round_start_shard

                    print(f"  Round {round_idx + 1}/{num_rounds}: shards {round_start_shard}-{round_end_shard - 1}")

                    # Create buffers for all shards in this round
                    shard_buffers = [[] for _ in range(num_shards_in_round)]

                    # ===== Parallel file reading and distribution =====
                    def read_and_distribute_file(args):
                        """Read one file and return data grouped by shard (thread-safe)."""
                        local_path, row_count, a_i, shuffled_indices, round_start, n_shards, n_shards_total = args

                        # Results: list of converted items for each shard
                        shard_results = [[] for _ in range(n_shards)]
                        local_source_counts = Counter()
                        rows_processed = 0

                        try:
                            df = pd.read_parquet(local_path)

                            for shard_offset in range(n_shards):
                                shard_k = round_start + shard_offset
                                is_last_shard = (shard_k == n_shards_total - 1)

                                start_idx = int(shard_k * a_i)
                                end_idx = int((shard_k + 1) * a_i)

                                if is_last_shard:
                                    end_idx = row_count

                                if start_idx >= end_idx or start_idx >= row_count:
                                    continue

                                indices_to_read = shuffled_indices[start_idx:min(end_idx, len(shuffled_indices))]

                                for idx in indices_to_read:
                                    if idx >= len(df):
                                        continue

                                    row = df.iloc[idx]
                                    item = row.to_dict()
                                    item_id = item.get('id', '')

                                    source = extract_source(item_id)
                                    local_source_counts[source] += 1

                                    try:
                                        converted = convert_to_levanter(item)
                                        shard_results[shard_offset].append(converted)
                                        rows_processed += 1
                                    except Exception as e:
                                        print(f"Error converting item {item_id}: {e}", file=sys.stderr)

                                    del row, item

                            del df

                        except Exception as e:
                            print(f"Error reading {local_path}: {e}", file=sys.stderr)

                        return shard_results, local_source_counts, rows_processed

                    # Prepare tasks for parallel execution
                    read_tasks = []
                    for local_path in local_paths:
                        row_count = file_row_counts.get(local_path, 0)
                        if row_count == 0:
                            continue
                        a_i = samples_per_file.get(local_path, 0)
                        if a_i == 0:
                            continue
                        shuffled_indices = file_shuffled_indices.get(local_path, [])
                        if not shuffled_indices:
                            continue
                        read_tasks.append((
                            local_path, row_count, a_i, shuffled_indices,
                            round_start_shard, num_shards_in_round, num_shards_in_batch
                        ))

                    # Parallel read using ThreadPoolExecutor
                    num_file_readers = min(read_workers, len(read_tasks))
                    print(f"    Reading {len(read_tasks)} files in parallel ({num_file_readers} workers)...")

                    with ThreadPoolExecutor(max_workers=num_file_readers) as read_executor:
                        futures = [read_executor.submit(read_and_distribute_file, task) for task in read_tasks]

                        file_pbar = tqdm(
                            as_completed(futures),
                            total=len(futures),
                            desc=f"    Reading files",
                            unit="file",
                            leave=False
                        )

                        for future in file_pbar:
                            try:
                                shard_results, local_counts, rows_processed = future.result()
                                # Merge results into shard_buffers
                                for shard_offset, results in enumerate(shard_results):
                                    shard_buffers[shard_offset].extend(results)
                                source_counts.update(local_counts)
                                batch_rows_processed += rows_processed
                                file_pbar.set_postfix(rows=batch_rows_processed)
                            except Exception as e:
                                print(f"Error in parallel file read: {e}", file=sys.stderr)

                        file_pbar.close()

                    # ===== Shuffle all shard buffers (can run in parallel with upload) =====
                    print(f"    Shuffling {num_shards_in_round} shard buffers...")
                    for shard_buffer in shard_buffers:
                        if shard_buffer:
                            random.shuffle(shard_buffer)

                    # ===== Wait for previous round's upload before writing to local cache =====
                    # This prevents local disk from filling up
                    if gcs_uploader and pending_upload_files:
                        print(f"    Waiting for previous upload to complete...")
                        gcs_uploader.wait_all(delete_local=True)
                        uploaded_rows = sum(nrows for _, _, nrows in pending_upload_files)
                        total_written += uploaded_rows
                        print(f"    Upload complete: {len(pending_upload_files)} shards, {uploaded_rows:,} rows uploaded to GCS")
                        pending_upload_files = []

                    # ===== Parallel write shards (already shuffled) =====
                    def write_shard_only(args):
                        """Write a single shard to local disk (already shuffled)."""
                        shard_buffer, shard_idx_to_write, local_dir = args
                        if not shard_buffer:
                            return None, 0
                        # Write to local (no shuffle needed, already done)
                        local_path, num_rows = write_shard_to_local(
                            shard_buffer, shard_idx_to_write, local_dir
                        )
                        return local_path, num_rows

                    # Prepare write tasks
                    write_tasks = []
                    for shard_offset in range(num_shards_in_round):
                        shard_buffer = shard_buffers[shard_offset]
                        current_shard_idx = shard_idx + shard_offset
                        if shard_buffer:
                            # Make a copy of the buffer for thread safety
                            write_tasks.append((
                                list(shard_buffer),  # Copy to avoid race conditions
                                current_shard_idx,
                                effective_local_shard_dir
                            ))
                            shard_buffer.clear()  # Clear original immediately

                    # Parallel write using ThreadPoolExecutor
                    num_write_workers = min(write_workers, len(write_tasks))
                    print(f"    Writing {len(write_tasks)} shards in parallel ({num_write_workers} workers)...")

                    write_results = []
                    with ThreadPoolExecutor(max_workers=num_write_workers) as write_executor:
                        futures = {write_executor.submit(write_shard_only, task): task[1]
                                   for task in write_tasks}

                        write_pbar = tqdm(
                            as_completed(futures),
                            total=len(futures),
                            desc=f"    Writing shards",
                            unit="shard",
                            leave=False
                        )

                        for future in write_pbar:
                            shard_idx_written = futures[future]
                            try:
                                local_path, num_rows = future.result()
                                if local_path and num_rows > 0:
                                    write_results.append((local_path, shard_idx_written, num_rows))
                                    write_pbar.set_postfix(shard=shard_idx_written, rows=num_rows)
                            except Exception as e:
                                print(f"Error writing shard {shard_idx_written}: {e}", file=sys.stderr)

                        write_pbar.close()

                    # Process write results
                    for local_path, written_shard_idx, num_rows in write_results:
                        total_processed += num_rows
                        shards_written_in_batch += 1
                        shard_name = f"train-{written_shard_idx:05d}.parquet"
                        remote_path = f"{output_path.rstrip('/')}/{shard_name}"
                        local_shard_files.append((local_path, remote_path, num_rows))

                    # Update shard_idx for next round
                    shard_idx += num_shards_in_round

                    # Clear all shard buffers for this round
                    del shard_buffers
                    gc.collect()

                    # ===== Start async upload (don't wait - runs in parallel with next round) =====
                    if gcs_uploader and local_shard_files:
                        print(f"    Starting async upload of {len(local_shard_files)} shards...")
                        for lpath, rpath, nrows in local_shard_files:
                            gcs_uploader.submit_upload(lpath, rpath, nrows)
                        # Store for tracking, don't wait - will be checked in next round
                        pending_upload_files = local_shard_files
                        local_shard_files = []
                        print(f"    Round {round_idx + 1} complete: {shards_written_in_batch} shards, {total_processed:,} rows (upload in background)")

                # Clear pre-shuffled indices to free memory
                del file_shuffled_indices
                gc.collect()

                # Wait for any pending uploads from the last round
                print(f"Processed {batch_rows_processed} items, wrote {shards_written_in_batch} shards locally")
                if gcs_uploader and pending_upload_files:
                    print(f"Waiting for final upload to complete ({len(pending_upload_files)} shards)...")
                    gcs_uploader.wait_all(delete_local=True)
                    uploaded_rows = sum(nrows for _, _, nrows in pending_upload_files)
                    total_written += uploaded_rows
                    print(f"Final upload complete: {len(pending_upload_files)} shards, {uploaded_rows:,} rows uploaded to GCS")
                    pending_upload_files = []

                # Upload any remaining local shards (edge case)
                if gcs_uploader and local_shard_files:
                    print(f"Uploading {len(local_shard_files)} remaining shards to GCS...")
                    for lpath, rpath, nrows in local_shard_files:
                        gcs_uploader.submit_upload(lpath, rpath, nrows)
                    gcs_uploader.wait_all(delete_local=True)
                    uploaded_rows = sum(nrows for _, _, nrows in local_shard_files)
                    total_written += uploaded_rows
                    print(f"Upload complete: {len(local_shard_files)} shards, {uploaded_rows:,} rows uploaded to GCS")
                    local_shard_files = []
                elif not gcs_uploader:
                    # Local mode - files are already in the right place, just count them
                    total_written += sum(nrows for _, _, nrows in local_shard_files)
                    # Move files to final output directory if different from local_shard_dir
                    if effective_local_shard_dir != output_path:
                        for lpath, _, _ in local_shard_files:
                            dest_path = Path(output_path) / Path(lpath).name
                            Path(output_path).mkdir(parents=True, exist_ok=True)
                            shutil.move(lpath, dest_path)
                    local_shard_files = []
                print(f"Batch {batch_idx + 1} complete: wrote {shards_written_in_batch} shards")

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

            # Force garbage collection after batch cleanup
            gc.collect()

            # Progress update
            elapsed = time.time() - start_time
            rate = total_processed / elapsed if elapsed > 0 else 0
            print(f"Progress: {total_processed:,} processed, {total_written:,} written, "
                  f"{rate:.1f} items/sec, {elapsed/60:.1f} min elapsed")

            # Save checkpoint after each batch (save next batch index)
            save_download_first_checkpoint(
                checkpoint_path, batches, batch_sizes, batch_idx + 1, shard_idx,
                source_counts, total_processed, total_written,
                gcs_checkpoint_path=gcs_checkpoint_path, fs=fs
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
            checkpoint_path, batches, batch_sizes, batch_idx, shard_idx,
            source_counts, total_processed, total_written,
            gcs_checkpoint_path=gcs_checkpoint_path, fs=fs
        )
        print(f"Checkpoint saved. Run with --resume to continue from batch {batch_idx}.")

    # Wait for any pending GCS uploads
    if gcs_uploader:
        print("Waiting for any pending GCS uploads...")
        gcs_uploader.wait_all(delete_local=True)
        gcs_uploader.shutdown()

    # Clean up download directory and local shard directory
    print(f"Cleaning up download directory: {download_dir}")
    try:
        shutil.rmtree(download_dir, ignore_errors=True)
    except Exception:
        pass

    if effective_local_shard_dir != output_path:
        print(f"Cleaning up local shard directory: {effective_local_shard_dir}")
        try:
            shutil.rmtree(effective_local_shard_dir, ignore_errors=True)
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
    repo_ids: List[str],
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
        repo_ids: List of HuggingFace repo IDs to process
        buffer_size: Number of rows to keep in memory buffer for write-time shuffling
        rows_per_shard: Number of rows per output shard
        shuffle_buffer_size: Buffer size for HF's built-in streaming shuffle (cross-subset mixing)
        max_rows: Maximum number of rows to process (None for all)
        seed: Random seed for shuffling
        use_gcs: Whether to write to GCS or local filesystem
        resume: Whether to resume from checkpoint (lossless resume)
    """
    from datasets import interleave_datasets

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

    # Load datasets in streaming mode from all repos
    print(f"Loading datasets from {len(repo_ids)} repo(s) in streaming mode...")
    datasets_list = []
    for repo_id in repo_ids:
        print(f"  Loading {repo_id}...")
        ds = load_dataset(repo_id, streaming=True, split="train")
        datasets_list.append(ds)

    # Interleave datasets if multiple repos
    if len(datasets_list) == 1:
        dataset = datasets_list[0]
    else:
        print(f"Interleaving {len(datasets_list)} datasets...")
        dataset = interleave_datasets(datasets_list)

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
        help="(Deprecated, use --upload-workers) Number of parallel workers"
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
        "--upload-workers",
        type=int,
        default=8,
        help="Number of parallel GCS upload workers (default: 8). Parquet conversion is done serially."
    )
    parser.add_argument(
        "--max-batch-gb",
        type=float,
        default=150.0,
        help="Maximum size per batch in GB for download-first mode (default: 150 GB). Batches are created based on actual file sizes to avoid disk space issues."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for saving checkpoint locally (default: same as --download-dir). Checkpoint is always saved locally for faster I/O, even when output is on GCS."
    )
    parser.add_argument(
        "--local-shard-dir",
        type=str,
        default=None,
        help="Directory for temporary local parquet shards before GCS upload (default: {download-dir}/shards)"
    )
    parser.add_argument(
        "--max-local-shards",
        type=int,
        default=20,
        help="Maximum number of local shards before triggering GCS upload (default: 20). Lower values use less disk space."
    )
    parser.add_argument(
        "--read-workers",
        type=int,
        default=8,
        help="Number of parallel workers for reading parquet files during shard processing (default: 8). Higher values speed up processing but use more memory."
    )
    parser.add_argument(
        "--write-workers",
        type=int,
        default=8,
        help="Number of parallel workers for writing shards to local disk (default: 8). Higher values speed up writing but use more memory."
    )
    parser.add_argument(
        "--repos",
        type=str,
        nargs="+",
        default=["mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M"],
        help="HuggingFace repo IDs to process (space-separated). Data from all repos will be merged and shuffled."
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
    print("HuggingFace Dataset to Levanter Converter")
    print("=" * 60)
    print(f"Mode: {'DOWNLOAD-FIRST (parallel)' if args.download_first else 'STREAMING'}")
    print(f"Repos: {len(args.repos)} repo(s)")
    for repo in args.repos:
        print(f"  - {repo}")
    print(f"Output: {output_path}")
    print(f"Use GCS: {use_gcs}")
    print(f"Buffer size (write-time): {args.buffer_size}")
    print(f"Rows per shard: {args.rows_per_shard}")
    print(f"Max rows: {args.max_rows or 'all'}")
    print(f"Seed: {args.seed}")

    if args.download_first:
        print(f"Download dir: {args.download_dir}")
        print(f"Checkpoint dir: {args.checkpoint_dir or args.download_dir} (local)")
        print(f"Local shard dir: {args.local_shard_dir or args.download_dir + '/shards'}")
        print(f"Download workers: {args.download_workers}")
        print(f"Read workers: {args.read_workers}")
        print(f"Write workers: {args.write_workers}")
        print(f"Upload workers: {args.upload_workers}")
        print(f"Max batch size: {args.max_batch_gb} GB")
        print(f"Max local shards: {args.max_local_shards}")
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
            repo_ids=args.repos,
            buffer_size=args.buffer_size,
            rows_per_shard=args.rows_per_shard,
            max_batch_gb=args.max_batch_gb,
            max_rows=args.max_rows,
            seed=args.seed,
            use_gcs=use_gcs,
            upload_workers=args.upload_workers,
            download_workers=args.download_workers,
            read_workers=args.read_workers,
            write_workers=args.write_workers,
            resume=args.resume,
            checkpoint_dir=args.checkpoint_dir,
            local_shard_dir=args.local_shard_dir,
            max_local_shards=args.max_local_shards
        )
    else:
        # Use streaming mode (original)
        process_dataset(
            output_path=output_path,
            repo_ids=args.repos,
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
