import logging
import os

from experiments.train_test_overlap.format.convert_finemath3plus_parquet2jsonl import finemath3plus_to_jsonl
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.processing.classification.dedupe import DedupeConfig, NGramConfig, dedupe
from marin.utils import fsspec_glob

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
# Determine GCS prefix for Marin (must be set via environment)
prefix = os.environ.get("MARIN_PREFIX")
if not prefix:
    raise ValueError("MARIN_PREFIX environment variable must be set to your GCS prefix")

# Directory where Finemath-3plus JSONL shards live
finemath_dir = os.path.join(prefix, finemath3plus_to_jsonl.override_output_path)
logger.info(f"Looking for Finemath-3plus shards in {finemath_dir}")

# Glob for all compressed JSONL shard files
patterns = ["**/*.jsonl.gz", "**/*.jsonl.zst", "**/*.jsonl", "**/*.json.gz", "**/*.json.zst"]
all_files = set()
for patt in patterns:
    glob_pat = os.path.join(finemath_dir.rstrip("/"), patt)
    matches = fsspec_glob(glob_pat)
    all_files.update(matches)
shard_paths = sorted(all_files)
if not shard_paths:
    raise FileNotFoundError(f"No Finemath-3plus shard files found under {finemath_dir}")

# MMLU test set JSONL directory (already converted and stored)
mmlu_test_dir = "gs://marin-us-central2/decontamination/mmlu-9fbdd5/cais/"

# Build steps: first the conversion, then one dedupe step per shard
steps: list[ExecutorStep] = []
for shard_path in shard_paths:
    # derive a unique name from the shard file
    shard_basename = os.path.basename(shard_path).replace(".jsonl.zst", "")
    step_name = f"train_test_overlap/dolma/finemath3plus_dedupe/{shard_basename}"
    cfg = DedupeConfig(
        input_path=mmlu_test_dir,
        output_path=this_output_path(),
        attribute_name="mmlu_overlap",
        false_positive_rate=0.0001,
        ngram=NGramConfig(
            ngram_length=[10, 15],
            overlap_threshold=1e-6,
            stride=0,
        ),
        processes=8,
        decontaminate=True,
        decontaminate_path=shard_path,
    )
    steps.append(ExecutorStep(name=step_name, fn=dedupe, config=cfg))


if __name__ == "__main__":
    executor_main(
        steps=steps,
        description="Run per-shard Dolma dedupe for each Finemath-3plus shard against MMLU test set",
    )
