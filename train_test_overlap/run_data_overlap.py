import gc
import json
import logging
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import time

import draccus
import fsspec
import psutil
import pyarrow.parquet as pq
import ray

from marin.core.runtime import cached_or_construct_output
from marin.utils import fsspec_exists, fsspec_glob, fsspec_isdir, fsspec_mkdirs, rebase_file_path
from train_test_overlap.compute_data_overlap_metrics import (
    compute_all_data_overlap,
    create_ngram_index,
    load_light_scenarios_from_jsonl,
)
from train_test_overlap.compute_metrics_from_ngrams import get_metrics
from train_test_overlap.data_overlap_spec import (
    DataOverlapStats,
    EntryOverlapNgrams,
)
from train_test_overlap.metrics import aggregate_metrics
from train_test_overlap.utils import asdict_without_nones

logger = logging.getLogger(__name__)


@dataclass
class DataOverlapPipelineConfig:
    input_data: str
    scenario_data: str
    output_path: str
    N: list[int] = field(default_factory=lambda: [5, 9, 13])


@cached_or_construct_output(success_suffix="SUCCESS")
def copy_metrics_out(input_file_path: str, output_file_path: str):
    """
    Idempotent copy of all metric files under input_file_path into output_file_path/metrics
    """
    dest_base = os.path.join(output_file_path, "metrics")
    for local_file in fsspec_glob(f"{input_file_path}/**/*"):
        dest = rebase_file_path(input_file_path, local_file, dest_base)
        fsspec_mkdirs(os.path.dirname(dest))
        with fsspec.open(local_file, "rb") as src, fsspec.open(dest, "wb") as dst:
            dst.write(src.read())


def create_compressed_file_iterator(
    input_patterns: list[str],
    base_path: str,
) -> Iterator[tuple[str, str]]:
    """
    Create a streaming iterator over compressed files.
    Only holds one file path and one line in memory at a time.

    Args:
        input_patterns: List of glob patterns to match
        base_path: Base path to search from

    Yields:
        Tuple of (document_text, source_info) where source_info contains file info for logging
    """
    start_time = time()
    processed_files = 0
    failed_files = 0

    @contextmanager
    def managed_file_open(file_path: str):
        """Context manager to ensure proper file handle cleanup"""
        f = None
        try:
            f = fsspec.open(file_path, "rt", compression="infer").open()
            yield f
        finally:
            if f is not None:
                f.close()
                del f  # Explicitly delete the file handle

    # Determine list of files to process: single file or all matching patterns
    try:
        is_file = fsspec_exists(base_path) and not fsspec_isdir(base_path)
    except Exception:
        is_file = False
    if is_file:
        file_list = [base_path]
    else:
        file_list: list[str] = []
        for pattern in input_patterns:
            remote_pattern = os.path.join(base_path.rstrip("/"), pattern)
            files = fsspec_glob(remote_pattern)
            file_list.extend(files)
    # Process each file one at a time
    for file_path in file_list:
        if not file_path.strip():
            continue

        # Skip unsupported file types (only JSONL or Parquet)
        if not file_path.lower().endswith((".jsonl", ".jsonl.gz", ".jsonl.zst", ".json.gz", ".json.zst", ".parquet")):
            failed_files += 1
            logger.error(f"Skipping unsupported file type: {file_path}")
            continue

        file_start = time()
        print(f"Starting file: {file_path}", flush=True)

        # Parquet files: read via pyarrow
        if file_path.lower().endswith(".parquet"):
            try:
                with fsspec.open(file_path, "rb") as f:
                    parquet_file = pq.ParquetFile(f)
                    row_num = 0
                    for batch in parquet_file.iter_batches():
                        for record in batch.to_pylist():
                            row_num += 1
                            text = record.get("text")
                            source_info = f"file={file_path}, row={row_num}"
                            if text is None:
                                logger.warning(f"Missing 'text' field in parquet row: {source_info}")
                                continue
                            yield text, source_info
                processed_files += 1
                gc.collect()
            except Exception as e:
                failed_files += 1
                logger.error(f"Error processing parquet file {file_path}: {e!s}")
            continue

        try:
            with managed_file_open(file_path) as f:
                # Process one line at a time
                for line_num, line in enumerate(f, 1):
                    try:
                        document = json.loads(line)
                        source_info = f"file={file_path}, line={line_num}"

                        if "text" not in document:
                            logger.warning(f"Missing 'text' field in document: {source_info}")
                            continue

                        yield document["text"], source_info

                        # Explicitly clear memory
                        del line
                        del document

                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error in {file_path}:{line_num}: {e!s}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing line in {file_path}:{line_num}: {e!s}")
                        continue

            processed_files += 1
            elapsed = time() - start_time
            rate = processed_files / (elapsed / 60) if elapsed > 0 else 0
            file_time = time() - file_start
            print(
                f"Completed file in {file_time:.1f}s. Total: {processed_files} files ({rate:.1f} files/min)", flush=True
            )

            gc.collect()  # Force garbage collection after each file

        except Exception as e:
            failed_files += 1
            logger.error(f"Failed to process file {file_path}: {e!s}")
            continue

    total_time = time() - start_time
    print(f"Completed processing {processed_files} files in {total_time:.1f}s. Failed: {failed_files}", flush=True)


def _print_mem(label: str, include_children: bool = True):
    """Helper function to print memory usage information for debugging purposes.

    Args:
        label: String label to identify the memory snapshot
        include_children: Whether to include memory usage of child processes
    """
    proc = psutil.Process(os.getpid())
    info = proc.memory_info()
    uss = getattr(proc.memory_full_info(), "uss", None)

    private = (info.rss - info.shared) / 2**20  # MiB
    shared = info.shared / 2**20
    txt = f"[MEM] {label}: private={private:,.1f} MiB  shared={shared:,.1f}"
    if uss is not None:
        txt += f"  uss={uss/2**20:,.1f}"
    if include_children:
        children_rss = sum(c.memory_info().rss for c in proc.children(recursive=True)) / 2**20
        txt += f"  children_rss={children_rss:,.1f}"
    print(txt, flush=True)


@ray.remote(memory=1024 * 1024 * 1024 * 16, num_cpus=16)
def run_data_overlap(config: DataOverlapPipelineConfig) -> str:
    # Idempotence: skip if already completed
    print(f"starting run_data_overlap for {config.input_data}", flush=True)
    success_file = config.output_path.rstrip("/") + ".SUCCESS"
    if fsspec_exists(success_file):
        logger.info(f"Skipping run_data_overlap for {config.input_data}, success exists at {success_file}")
        return f"Skipped {config.input_data}"

    # Create a temporary directory under /tmp that will be cleaned up automatically
    with tempfile.TemporaryDirectory(dir="/dev/shm") as tmpdir:
        stats_prefix = os.path.join(tmpdir, "stats")

        # Copy and merge scenario JSONL files to local tmpdir
        local_scenarios_path = os.path.join(tmpdir, "scenario.jsonl")
        scenario_root = config.scenario_data.rstrip("/")
        remote_pattern = os.path.join(scenario_root, "**", "*.jsonl*")
        scenario_files = fsspec_glob(remote_pattern)
        if not scenario_files:
            # fallback: treat scenario_data as a single file
            scenario_files = [config.scenario_data]
        scenario_files = sorted(scenario_files)
        with open(local_scenarios_path, "wb") as out_f:
            for remote_file in scenario_files:
                with fsspec.open(remote_file, "rb", compression="infer") as in_f:
                    out_f.write(in_f.read())
        scenarios = load_light_scenarios_from_jsonl(local_scenarios_path)

        # track ngram counts with a stats_key which is unique for each test instance
        stats_key_counts: defaultdict = defaultdict(int)

        # and choice of n
        # TODO: this is the only time we need scenarios,
        # refactor to get rid of this.
        # create ngram_index for each N requested for test set
        ngram_index = create_ngram_index(scenarios, config.N, stats_key_counts)

        stats_key_to_input_ids = defaultdict(set)
        stats_key_to_reference_ids = defaultdict(set)
        entry_ngram_counts = defaultdict(lambda: defaultdict(int))

        # Define patterns for supported file types
        input_patterns = [
            "**/*.jsonl.gz",
            "**/*.jsonl.zst",
            "**/*.jsonl.gs",
            "**/*.json.gz",
            "**/*.json.zst",
            "**/*.jsonl",
            "**/*.parquet",
        ]

        # Create streaming iterator for compressed files
        document_iterator = create_compressed_file_iterator(
            input_patterns=input_patterns,
            base_path=config.input_data,
        )

        # Process documents using the iterator
        compute_all_data_overlap(
            document_iterator=document_iterator,
            ngram_index=ngram_index,
            stats_key_to_input_ids=stats_key_to_input_ids,
            stats_key_to_reference_ids=stats_key_to_reference_ids,
            entry_overlap_key_to_ngram_counts=entry_ngram_counts,
            output_ngrams=True,
        )

        # 2) Write raw ngrams
        ngrams_out = stats_prefix + "_ngrams"
        with open(ngrams_out, "w") as f:
            for entry_key, counts in entry_ngram_counts.items():
                # the entry overlap key tracks the dataset name, instance / example id
                # whether it's an input or reference, a dataoverlap key which tracks
                # the dataset and the length of the ngrams we're looking at.
                # the counts here are the number of times each ngram appears in the training data
                # for a given test instance.
                eon = EntryOverlapNgrams(entry_data_overlap_key=entry_key, overlapping_ngram_counts=list(counts.items()))
                f.write(json.dumps(asdict_without_nones(eon)) + "\n")

        # 3) Write overlap stats to track which instances from test data
        # have either overlapping inputs or reference answers
        stats_out = stats_prefix
        with open(stats_out, "w") as f:
            for sk, count in stats_key_counts.items():
                dos = DataOverlapStats(
                    data_overlap_stats_key=sk,
                    instance_ids_with_overlapping_input=sorted(stats_key_to_input_ids[sk]),
                    instance_ids_with_overlapping_reference=sorted(stats_key_to_reference_ids[sk]),
                    num_instances=count,
                )
                f.write(json.dumps(asdict_without_nones(dos)) + "\n")

        # 4) Compute metrics from ngrams
        metric_dirs = []
        for n in config.N:
            met_dir = os.path.join(tmpdir, f"metrics_{n}")
            get_metrics(
                ngrams_path=ngrams_out,
                scenario_path=local_scenarios_path,
                out_path=met_dir,
                filter_path="",
                N=n,
            )
            metric_dirs.append(met_dir)

        # 5) Aggregate metrics (include original GCS input_data path)
        agg_dirs = []
        for met_dir in metric_dirs:
            agg_dir = os.path.join(tmpdir, f"aggregate_{os.path.basename(met_dir)}")
            aggregate_metrics(path=met_dir, out_path=agg_dir, metrics_input_path=config.input_data)
            agg_dirs.append(agg_dir)

        # 6) Copy each aggregated metrics file into its own folder under the output path
        for agg_file in agg_dirs:
            # agg_file is a JSONL file, e.g. /tmp/.../aggregate_metrics_5
            agg_name = os.path.basename(agg_file)
            remote_dir = os.path.join(config.output_path, agg_name)
            # Destination is <remote_dir>/<filename>
            dest_file = os.path.join(remote_dir, agg_name)
            fsspec_mkdirs(os.path.dirname(dest_file))
            with fsspec.open(agg_file, "rb") as src, fsspec.open(dest_file, "wb") as dst:
                dst.write(src.read())

        # 7) Copy raw ngrams file to raw_ngrams directory in output path
        raw_ngrams_dir = os.path.join(config.output_path, "raw_ngrams")
        fsspec_mkdirs(raw_ngrams_dir)
        raw_ngrams_dest = os.path.join(raw_ngrams_dir, "raw_ngrams.jsonl")
        with fsspec.open(ngrams_out, "rb") as src, fsspec.open(raw_ngrams_dest, "wb") as dst:
            dst.write(src.read())

        # 8) Copy stats file to stats directory in output path
        stats_dir = os.path.join(config.output_path, "stats")
        fsspec_mkdirs(stats_dir)
        stats_dest = os.path.join(stats_dir, "overlap_stats.jsonl")
        with fsspec.open(stats_out, "rb") as src, fsspec.open(stats_dest, "wb") as dst:
            dst.write(src.read())

        # 9) Write a mapping file to track instance IDs to their metrics for easier analysis
        print(f"Writing instance mapping to {config.output_path}", flush=True)
        instance_mapping_dir = os.path.join(str(config.output_path), "instance_mapping")
        fsspec_mkdirs(instance_mapping_dir)
        instance_mapping_dest = os.path.join(instance_mapping_dir, "instance_mapping.json")

        # Create mapping of instance IDs to their stats_keys
        instance_id_mapping = {}
        for sk, _count in stats_key_counts.items():
            scenario_name = sk.light_scenario_key.scenario_spec.class_name.split(".")[-1]
            subject = sk.light_scenario_key.scenario_spec.args.get("subject", "unknown")
            n_value = sk.overlap_protocol_spec.n
            key_name = f"{scenario_name}_{subject}_n{n_value}"

            # Add input overlapping instances
            for instance_id in stats_key_to_input_ids[sk]:
                if instance_id not in instance_id_mapping:
                    instance_id_mapping[instance_id] = {}
                if "input_overlaps" not in instance_id_mapping[instance_id]:
                    instance_id_mapping[instance_id]["input_overlaps"] = []
                instance_id_mapping[instance_id]["input_overlaps"].append(key_name)

            # Add reference overlapping instances
            for instance_id in stats_key_to_reference_ids[sk]:
                if instance_id not in instance_id_mapping:
                    instance_id_mapping[instance_id] = {}
                if "reference_overlaps" not in instance_id_mapping[instance_id]:
                    instance_id_mapping[instance_id]["reference_overlaps"] = []
                instance_id_mapping[instance_id]["reference_overlaps"].append(key_name)

        with fsspec.open(instance_mapping_dest, "w") as f:
            json.dump(instance_id_mapping, f, indent=2)

    # Temporary directory is automatically cleaned up here
    # Write success marker
    fsspec_mkdirs(os.path.dirname(success_file))
    with fsspec.open(success_file, "w") as _f:
        _f.write("")

    return "Train test overlap pipeline completed!"


@draccus.wrap()
def main(config: DataOverlapPipelineConfig):
    ray.get(run_data_overlap.options().remote(config))
