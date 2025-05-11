import os
import json
import tempfile
from collections import defaultdict
import logging
import ray
import fsspec

from train_test_overlap.run_data_overlap import DataOverlapPipelineConfig, SCRATCH_DIR, create_compressed_file_iterator
from train_test_overlap.compute_data_overlap_metrics import (
    compute_all_data_overlap,
    create_ngram_index,
    load_light_scenarios_from_jsonl,
)
from train_test_overlap.compute_metrics_from_ngrams import get_metrics
from train_test_overlap.data_overlap_spec import DataOverlapStats, EntryOverlapNgrams
from train_test_overlap.metrics import aggregate_metrics
from train_test_overlap.utils import asdict_without_nones
from marin.utils import fsspec_glob, fsspec_mkdirs

logger = logging.getLogger(__name__)

# 4 GiB per actor, 1 CPU each
@ray.remote(memory=1024 * 1024 * 1024 * 4, num_cpus=1)
class OverlapWorker:
    def __init__(self, scenario_data: str, N: list[int]):
        # Create a per-actor scratch dir
        self.tmpdir = tempfile.mkdtemp(dir=SCRATCH_DIR)
        # Copy and merge scenario JSONLs to local file
        local_scenarios_path = os.path.join(self.tmpdir, "scenario.jsonl")
        scenario_root = scenario_data.rstrip("/")
        remote_pattern = os.path.join(scenario_root, "**", "*.jsonl*")
        scenario_files = fsspec_glob(remote_pattern)
        if not scenario_files:
            scenario_files = [scenario_data]
        scenario_files = sorted(scenario_files)
        with open(local_scenarios_path, "wb") as out_f:
            for remote_file in scenario_files:
                with fsspec.open(remote_file, "rb", compression="infer") as in_f:
                    out_f.write(in_f.read())
        # Load and index scenarios
        scenarios = load_light_scenarios_from_jsonl(local_scenarios_path)
        stats_key_counts = defaultdict(int)
        self.ngram_index = create_ngram_index(scenarios, N, stats_key_counts)
        # Keep template counts for writing later
        self.stats_key_counts_template = stats_key_counts
        self.local_scenarios_path = local_scenarios_path
        self.N = N

    def run(self, input_data: str, output_path: str) -> str:
        # Fresh state per shard
        stats_key_counts = dict(self.stats_key_counts_template)
        stats_key_to_input_ids = defaultdict(set)
        stats_key_to_reference_ids = defaultdict(set)
        entry_ngram_counts = defaultdict(lambda: defaultdict(int))

        # Stream and compute overlap
        input_patterns = [
            "**/*.jsonl.gz",
            "**/*.jsonl.zst",
            "**/*.jsonl.gs",
            "**/*.json.gz",
            "**/*.json.zst",
            "**/*.jsonl",
        ]
        document_iterator = create_compressed_file_iterator(
            input_patterns=input_patterns, base_path=input_data
        )
        compute_all_data_overlap(
            document_iterator=document_iterator,
            ngram_index=self.ngram_index,
            stats_key_to_input_ids=stats_key_to_input_ids,
            stats_key_to_reference_ids=stats_key_to_reference_ids,
            entry_overlap_key_to_ngram_counts=entry_ngram_counts,
            output_ngrams=True,
        )

        # 1) Write raw ngrams
        stats_prefix = os.path.join(self.tmpdir, "stats")
        ngrams_out = stats_prefix + "_ngrams"
        with open(ngrams_out, "w") as f:
            for entry_key, counts in entry_ngram_counts.items():
                eon = EntryOverlapNgrams(
                    entry_data_overlap_key=entry_key,
                    overlapping_ngram_counts=list(counts.items()),
                )
                f.write(json.dumps(asdict_without_nones(eon)) + "\n")

        # 2) Write overlap stats
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

        # 3) Compute metrics per N
        metric_dirs = []
        for n in self.N:
            met_dir = os.path.join(self.tmpdir, f"metrics_{n}")
            get_metrics(
                ngrams_path=ngrams_out,
                scenario_path=self.local_scenarios_path,
                out_path=met_dir,
                filter_path="",
                N=n,
            )
            metric_dirs.append(met_dir)

        # 4) Aggregate metrics
        agg_dirs = []
        for met_dir in metric_dirs:
            agg_dir = os.path.join(self.tmpdir, f"aggregate_{os.path.basename(met_dir)}")
            aggregate_metrics(path=met_dir, out_path=agg_dir)
            agg_dirs.append(agg_dir)

        # 5) Copy aggregated files to GCS
        for agg_file in agg_dirs:
            agg_name = os.path.basename(agg_file)
            remote_dir = os.path.join(output_path, agg_name)
            dest_file = os.path.join(remote_dir, agg_name)
            fsspec_mkdirs(os.path.dirname(dest_file))
            with fsspec.open(agg_file, "rb") as src, fsspec.open(dest_file, "wb") as dst:
                dst.write(src.read())

        # 6) Copy raw ngrams file
        raw_ngrams_dir = os.path.join(output_path, "raw_ngrams")
        fsspec_mkdirs(raw_ngrams_dir)
        raw_ngrams_dest = os.path.join(raw_ngrams_dir, "raw_ngrams.jsonl")
        with fsspec.open(ngrams_out, "rb") as src, fsspec.open(raw_ngrams_dest, "wb") as dst:
            dst.write(src.read())

        # 7) Copy stats file
        stats_dir = os.path.join(output_path, "stats")
        fsspec_mkdirs(stats_dir)
        stats_dest = os.path.join(stats_dir, "overlap_stats.jsonl")
        with fsspec.open(stats_out, "rb") as src, fsspec.open(stats_dest, "wb") as dst:
            dst.write(src.read())

        # 8) Write instance mapping
        instance_mapping_dir = os.path.join(output_path, "instance_mapping")
        fsspec_mkdirs(instance_mapping_dir)
        instance_mapping_dest = os.path.join(instance_mapping_dir, "instance_mapping.json")
        instance_id_mapping = {}
        for sk in stats_key_counts:
            scenario_name = sk.light_scenario_key.scenario_spec.class_name.split(".")[-1]
            subject = sk.light_scenario_key.scenario_spec.args.get("subject", "unknown")
            n_value = sk.overlap_protocol_spec.n
            key_name = f"{scenario_name}_{subject}_n{n_value}"
            for instance_id in stats_key_to_input_ids[sk]:
                mapping = instance_id_mapping.setdefault(instance_id, {})
                mapping.setdefault("input_overlaps", []).append(key_name)
            for instance_id in stats_key_to_reference_ids[sk]:
                mapping = instance_id_mapping.setdefault(instance_id, {})
                mapping.setdefault("reference_overlaps", []).append(key_name)
        with fsspec.open(instance_mapping_dest, "w") as f:
            json.dump(instance_id_mapping, f, indent=2)

        return "Train test overlap pipeline completed!"


# Pool of actors, round-robin dispatch
_worker_pool = []
_next_worker_idx = 0

@ray.remote(num_cpus=0)
def run_data_overlap(config: DataOverlapPipelineConfig) -> str:
    global _worker_pool, _next_worker_idx
    if not _worker_pool:
        # initialize one actor per process slot
        for _ in range(config.processes):
            _worker_pool.append(OverlapWorker.remote(config.scenario_data, config.N))
    actor = _worker_pool[_next_worker_idx]
    _next_worker_idx = (_next_worker_idx + 1) % len(_worker_pool)
    # Delegate to actor and wait
    return ray.get(actor.run.remote(config.input_data, config.output_path)) 