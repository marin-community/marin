import logging
import os
from dataclasses import dataclass, field

import ray

from experiments.pretraining_datasets import dclm_baseline  # InputName for DCLM download step
from marin.core.runtime import simple_backpressure
from marin.execution.executor import ExecutorStep, executor_main, get_executor_step, this_output_path
from marin.processing.classification.dedupe import DedupeConfig, NGramConfig, dedupe
from marin.utils import fsspec_glob

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ShardedDedupeConfig:
    # Base directory containing DCLM shards
    base_input_dir: str
    # MMLU test JSONL directory to dedupe against
    mmlu_test_dir: str
    # Base output path under which each shard's results will be written
    output_base: str
    # Attribute name for marking overlaps
    attribute_name: str = "mmlu_overlap"
    # False positive rate for bloom filter
    false_positive_rate: float = 0.00001
    # NGram settings
    ngram: NGramConfig = field(
        default_factory=lambda: NGramConfig(
            ngram_length=[10, 15],
            overlap_threshold=1e-6,
            stride=0,
        )
    )
    # Number of processes per dedupe task
    processes: int = 16
    # Whether to decontaminate (remove overlaps)
    decontaminate: bool = True
    # Max concurrent tasks in flight
    max_in_flight: int = 256


@ray.remote
def run_all_dedupe(cfg: ShardedDedupeConfig) -> str:
    """
    Launch one Dolma-dedupe task per shard, up to cfg.max_in_flight in parallel.
    """
    # Patterns matching compressed JSONL or Parquet shard files
    input_patterns = [
        "**/*.jsonl.gz",
        "**/*.jsonl.zst",
        "**/*.jsonl",
        "**/*.json.gz",
        "**/*.json.zst",
        "**/*.parquet",
    ]
    all_files = set()
    for patt in input_patterns:
        pattern = os.path.join(cfg.base_input_dir.rstrip("/"), patt)
        all_files.update(fsspec_glob(pattern))
    shards = sorted(all_files)
    if not shards:
        raise FileNotFoundError(f"No DCLM shard files found under {cfg.base_input_dir}")
    print(f"Found {len(shards)} shards", flush=True)

    def make_task(shard_path: str) -> DedupeConfig:
        # Preserve the original directory structure under base_input_dir
        base_dir = cfg.base_input_dir.rstrip("/")
        # compute path of shard relative to base_input_dir
        if shard_path.startswith(base_dir + "/"):
            relative_path = shard_path[len(base_dir) + 1 :]
        else:
            relative_path = os.path.basename(shard_path)
        # strip known file extensions from final component
        for suffix in [".jsonl.gz", ".jsonl.zst", ".jsonl", ".json.gz", ".json.zst", ".parquet"]:
            if relative_path.endswith(suffix):
                relative_path = relative_path[: -len(suffix)]
                break
        # use nested folders matching the original layout
        out_base = os.path.join(cfg.output_base, relative_path)
        return DedupeConfig(
            input_path=cfg.mmlu_test_dir,
            output_path=out_base,
            attribute_name=cfg.attribute_name,
            false_positive_rate=cfg.false_positive_rate,
            ngram=cfg.ngram,
            processes=cfg.processes,
            decontaminate=cfg.decontaminate,
            decontaminate_path=shard_path,
        )

    # Generator of arguments for each Ray task
    # for idx, path in enumerate(shards):
    #     print(f"Processing shard {path}", flush=True)
    #     if idx == 100:
    #         os._exit(0)
    task_gen = ((make_task(path),) for path in shards)

    # Launch tasks with backpressure
    for ref in simple_backpressure(
        dedupe,
        task_gen,
        max_in_flight=cfg.max_in_flight,
        fetch_local=True,
    ):
        ray.get(ref)

    return "Sharded dedupe pipeline completed!"


# Determine GCS prefix for Marin (must be set via environment)
prefix = os.environ.get("MARIN_PREFIX")
if not prefix:
    raise ValueError("MARIN_PREFIX environment variable must be set to your GCS prefix")

# Resolve the full directory where DCLM shards live:
producing_step = get_executor_step(dclm_baseline)
dataset_id = producing_step.config.hf_dataset_id  # e.g. "mlfoundations/dclm-baseline-1.0"
revision = producing_step.config.revision  # e.g. "a3b142c"
namespace, repo = dataset_id.split("/", 1)
base_input = os.path.join(
    prefix.rstrip("/"),
    producing_step.override_output_path,  # "raw/dclm"
    revision,
    "huggingface.co",
    "datasets",
    namespace,
    repo,
    "resolve",
    revision,
)
logger.info(f"Looking for DCLM shards in {base_input}")

# MMLU test set JSONL directory
mmlu_dir = "gs://marin-us-central2/decontamination/mmlu-9fbdd5/cais/"

# Configure sharded dedupe runner
config = ShardedDedupeConfig(
    base_input_dir=base_input,
    mmlu_test_dir=mmlu_dir,
    output_base=this_output_path(),
    max_in_flight=256,
)
# Single ExecutorStep to kick off all per-shard dedupe tasks
dedupe_sharded_step = ExecutorStep(
    name="train_test_overlap/dolma/dclm_dedupe_sharded",
    fn=run_all_dedupe,
    config=config,
)

if __name__ == "__main__":
    executor_main(
        steps=[dedupe_sharded_step],
        description="Run sharded Dolma dedupe on DCLM shards against MMLU test set",
    )
