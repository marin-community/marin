#!/usr/bin/env python3
"""Review HuggingFace datasets for pretraining quality using Claude.

Full pipeline:
  1. Discover datasets from a HF collection (or read existing queue)
  2. Fetch metadata (schema, README, size) for each dataset
  3. Sample 100 random rows per dataset
  4. Score samples in batches of 10 via `claude -p --model sonnet`
  5. Merge batch scores into a final review via `claude -p --model opus`

All steps are incremental — cached results are skipped on re-run.
Designed to run from a terminal via `claude -p` (not inside Claude Code).

Usage:
    # First run: discover collection, summarize, and review
    uv run scripts/review_datasets.py --collection https://huggingface.co/collections/WillHeld/pretraining-data

    # Re-run: picks up where it left off (skips completed work)
    uv run scripts/review_datasets.py

    # Review a single dataset
    uv run scripts/review_datasets.py --dataset "common-pile/pubmed_filtered"
"""

import argparse
import json
import logging
import re
import subprocess
import time
from pathlib import Path

import requests
from huggingface_hub import HfApi

from marin.tools import get_dataset_size, sample_dataset
from marin.tools.get_hf_dataset_schema import hf_auth_headers

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

NUM_SAMPLES = 100
BATCH_SIZE = 10
SEED = 42
MAX_CHARS = 3000

HF_INFO_URL = "https://datasets-server.huggingface.co/info"


def safe_name(dataset: str) -> str:
    return dataset.replace("/", "__")


# ---------------------------------------------------------------------------
# Step 1: Discover datasets from a HF collection
# ---------------------------------------------------------------------------


def discover_collection(collection_url: str) -> list[str]:
    """Return dataset IDs from a HuggingFace collection URL."""
    match = re.search(r"collections/(.+)", collection_url.rstrip("/"))
    slug = match.group(1) if match else collection_url
    api = HfApi()
    collection = api.get_collection(slug)
    return [item.item_id for item in collection.items if item.item_type == "dataset"]


# ---------------------------------------------------------------------------
# Step 2: Fetch metadata (schema, README, size) per dataset
# ---------------------------------------------------------------------------


def fetch_info(dataset_name: str) -> dict | None:
    for attempt in range(5):
        try:
            resp = requests.get(
                HF_INFO_URL,
                params={"dataset": dataset_name},
                headers=hf_auth_headers(),
                timeout=60,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = float(resp.headers.get("Retry-After", 2**attempt))
                time.sleep(max(retry_after, 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException:
            if attempt < 4:
                time.sleep(2**attempt)
                continue
            return None
    return None


def summarize_dataset(dataset_name: str) -> dict:
    summary = {"dataset": dataset_name}

    # README
    try:
        api = HfApi()
        path = api.hf_hub_download(dataset_name, "README.md", repo_type="dataset")
        with open(path) as f:
            summary["readme"] = f.read()
    except Exception:
        pass

    # Info (features, configs, splits)
    info = fetch_info(dataset_name)
    if info:
        dataset_info = info.get("dataset_info", {})
        configs = list(dataset_info.keys())
        summary["configs"] = configs

        # Per-config splits and row counts
        configs_info = {}
        for cfg_name in configs:
            cfg_data = dataset_info[cfg_name]
            cfg_splits = cfg_data.get("splits", {})
            configs_info[cfg_name] = {
                s_name: {"num_examples": s_data.get("num_examples")}
                for s_name, s_data in cfg_splits.items()
            }
        summary["configs_info"] = configs_info

        if configs:
            first_config = configs[0]
            summary["config_used"] = first_config
            summary["splits"] = list(dataset_info[first_config].get("splits", {}).keys())

            # Parse features per config, collect text candidates across all
            all_features = {}
            text_candidates = []
            for cfg_name in configs:
                features = dataset_info[cfg_name].get("features", {})
                parsed = {}
                for name, feat in features.items():
                    if isinstance(feat, dict):
                        dtype = feat.get("dtype", feat.get("_type", "unknown"))
                        parsed[name] = str(dtype)
                        if (dtype == "string" or "text" in name.lower()) and name not in text_candidates:
                            text_candidates.append(name)
                    elif isinstance(feat, list):
                        parsed[name] = "list"
                    else:
                        parsed[name] = str(type(feat).__name__)
                all_features[cfg_name] = parsed

            # If all configs share the same schema, flatten to a single dict
            unique_schemas = {json.dumps(v, sort_keys=True) for v in all_features.values()}
            if len(unique_schemas) == 1:
                summary["features"] = all_features[first_config]
            else:
                summary["features_per_config"] = all_features
                summary["features"] = all_features[first_config]
            summary["text_field_candidates"] = text_candidates
    else:
        summary["info_error"] = "Failed to fetch from /info endpoint"

    # Size — fetch for all configs
    configs_to_query = summary.get("configs", [])
    if not configs_to_query:
        configs_to_query = [summary.get("config_used")]
    summary["size"] = {}
    for cfg in configs_to_query:
        try:
            size_info = get_dataset_size(dataset_name, cfg)
            summary["partial"] = summary.get("partial", False) or size_info.get("partial", False)
            for split_name, split_meta in size_info.get("splits", {}).items():
                key = f"{cfg}/{split_name}" if len(configs_to_query) > 1 else split_name
                summary["size"][key] = {
                    "num_rows": split_meta.get("num_rows"),
                    "num_bytes_parquet_files": split_meta.get("num_bytes_parquet_files"),
                }
        except Exception as e:
            summary.setdefault("size_errors", {})[cfg] = str(e)

    # Total file bytes
    try:
        api = HfApi()
        hub_info = api.dataset_info(dataset_name, files_metadata=True)
        total_bytes = sum(s.size for s in hub_info.siblings if s.size)
        summary["total_file_bytes"] = total_bytes
    except Exception:
        pass

    return summary


def ensure_summaries(datasets: list[str], output_dir: Path) -> None:
    """Fetch and cache metadata for all datasets."""
    for i, ds in enumerate(datasets, 1):
        sn = safe_name(ds)
        out_path = output_dir / f"{sn}.json"

        if out_path.exists():
            log.info(f"[{i}/{len(datasets)}] {ds} — summary cached")
        else:
            log.info(f"[{i}/{len(datasets)}] {ds} — fetching summary...")
            try:
                summary = summarize_dataset(ds)
            except Exception as e:
                log.error(f"  FAILED: {e}")
                summary = {"dataset": ds, "error": str(e)}
            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            time.sleep(0.5)


# ---------------------------------------------------------------------------
# Step 3: Build review queue (auto-detect text field per dataset)
# ---------------------------------------------------------------------------


COMMON_TEXT_FIELDS = ("text", "content", "document", "body", "code")


def detect_text_fields(
    dataset_name: str, config: str = "default", split: str = "train"
) -> list[str] | None:
    """Detect which columns contain text content relevant for pretraining.

    Tries common field names first, falling back to an LLM call for
    datasets with non-obvious schemas.
    """
    try:
        result = sample_dataset(dataset_name, config=config, split=split, n=1, seed=0)
    except Exception as e:
        log.warning(f"  detect_text_fields: could not fetch sample row: {e}")
        return None
    samples = result.get("samples", [])
    if not samples:
        return None

    row = samples[0]

    # Fast path: check common field names
    matches = [f for f in COMMON_TEXT_FIELDS if f in row]
    if matches:
        return matches

    # Fallback: ask the model
    schema_lines = []
    preview_lines = []
    for col, val in row.items():
        dtype = type(val).__name__
        schema_lines.append(f"  {col}: {dtype}")
        preview_lines.append(f"  {col}: {str(val)[:200]}")

    prompt_parts = [
        f"Dataset: {dataset_name}",
        "Columns and types:\n" + "\n".join(schema_lines),
        "Sample row (truncated):\n" + "\n".join(preview_lines),
        "\nWhich columns contain the primary text content useful for LLM pretraining?"
        "\nReturn ONLY a JSON list of column names, e.g. [\"text\"] or [\"instruction\", \"response\"]."
        "\nPick only string columns that carry meaningful natural language or code.",
    ]

    try:
        response = run_claude("\n\n".join(prompt_parts), "", model="sonnet")
        fields = parse_json_response(response)
        if isinstance(fields, list) and all(isinstance(f, str) for f in fields):
            valid = [f for f in fields if f in row]
            return valid if valid else None
        return None
    except Exception as e:
        log.warning(f"  detect_text_fields failed: {e}")
        return None


def detect_configs_and_splits(
    dataset_name: str, configs_info: dict[str, dict]
) -> list[dict]:
    """Pick which (config, split) pairs to sample for a quality review.

    configs_info maps config name -> {split_name: {"num_examples": N}, ...}.
    Returns a list of {"config": ..., "split": ...} dicts.
    """
    if not configs_info:
        return [{"config": "default", "split": "train"}]

    # Single config with standard splits — just pick train (or the only split).
    if len(configs_info) == 1:
        cfg = next(iter(configs_info))
        splits = list(configs_info[cfg].keys())
        if "train" in splits:
            return [{"config": cfg, "split": "train"}]
        if len(splits) == 1:
            return [{"config": cfg, "split": splits[0]}]

    # Few configs each with a train split (e.g. "default" + "v2") — take train from each.
    all_have_train = all("train" in splits for splits in configs_info.values())
    if len(configs_info) <= 3 and all_have_train:
        return [{"config": cfg, "split": "train"} for cfg in configs_info]

    # Non-trivial layout — ask the model.
    desc_lines = []
    for cfg, splits in configs_info.items():
        split_parts = []
        for s_name, s_meta in splits.items():
            n = s_meta.get("num_examples", "?")
            split_parts.append(f"{s_name} ({n} rows)")
        desc_lines.append(f"  {cfg}: {', '.join(split_parts)}")

    prompt_parts = [
        f"Dataset: {dataset_name}",
        f"Configs and splits:\n" + "\n".join(desc_lines),
        "\nWe want to review this dataset's quality for LLM pretraining."
        "\nPick the (config, split) pairs we should sample from to get a representative view."
        "\nPrefer training splits. If configs represent subsets (e.g. languages), pick a diverse handful (up to 5)."
        "\nIf configs are just versions, pick the latest/best one."
        '\nReturn ONLY a JSON list, e.g. [{"config": "en-fr", "split": "train"}]',
    ]

    try:
        response = run_claude("\n\n".join(prompt_parts), "", model="sonnet")
        pairs = parse_json_response(response)
        if isinstance(pairs, list) and pairs:
            valid = [
                p for p in pairs
                if isinstance(p, dict)
                and p.get("config") in configs_info
                and p.get("split") in configs_info[p["config"]]
            ]
            if valid:
                return valid
    except Exception as e:
        log.warning(f"  detect_configs_and_splits failed: {e}")

    # Fallback: train split from first config.
    cfg = next(iter(configs_info))
    splits = list(configs_info[cfg].keys())
    split = "train" if "train" in splits else splits[0]
    return [{"config": cfg, "split": split}]


def make_entry(dataset_name: str, output_dir: Path) -> dict | None:
    """Build a review entry (config_splits, text_fields) for a single dataset."""
    summary_path = output_dir / f"{safe_name(dataset_name)}.json"
    if not summary_path.exists():
        return None

    summary = json.loads(summary_path.read_text())
    if summary.get("error"):
        return None

    candidates = summary.get("text_field_candidates", [])

    # Build configs_info from summary, falling back to old single-config format.
    configs_info = summary.get("configs_info")
    if not configs_info:
        config = summary.get("config_used", "default")
        splits = summary.get("splits", ["train"])
        configs_info = {config: {s: {} for s in splits}}

    config_splits = detect_configs_and_splits(dataset_name, configs_info)

    # Detect text fields once per unique schema across selected configs.
    # Group configs by their feature set so we only call detect_text_fields
    # once per distinct schema.
    all_features = summary.get("features_per_config", {})
    if not all_features:
        # All configs share one schema — use it for everything
        feats = summary.get("features", {})
        all_features = {cs["config"]: feats for cs in config_splits}

    schema_to_configs: dict[str, list[str]] = {}
    for cs in config_splits:
        cfg = cs["config"]
        schema_key = json.dumps(all_features.get(cfg, {}), sort_keys=True)
        schema_to_configs.setdefault(schema_key, []).append(cfg)

    text_fields_per_config: dict[str, list[str]] = {}
    for schema_key, cfgs in schema_to_configs.items():
        # Pick the first config/split pair with this schema to probe
        representative = next(cs for cs in config_splits if cs["config"] in cfgs)
        detected = detect_text_fields(dataset_name, representative["config"], representative["split"])
        fields = detected if detected else candidates
        for cfg in cfgs:
            text_fields_per_config[cfg] = fields

    if not any(text_fields_per_config.values()):
        return None

    return {
        "dataset": dataset_name,
        "config_splits": config_splits,
        "text_fields_per_config": text_fields_per_config,
    }


# ---------------------------------------------------------------------------
# Step 4 & 5: Sample, batch review, merge
# ---------------------------------------------------------------------------


def prefetch(entry: dict, output_dir: Path) -> Path | None:
    ds = entry["dataset"]
    samples_path = output_dir / f"{safe_name(ds)}_samples.json"

    if samples_path.exists():
        data = json.loads(samples_path.read_text())
        if data.get("error"):
            log.info(f"  Previous fetch failed, retrying...")
            samples_path.unlink()
        else:
            return samples_path

    config_splits = entry.get("config_splits")
    if not config_splits:
        config_splits = [{"config": entry.get("config"), "split": entry.get("split", "train")}]

    samples_per_pair = max(1, NUM_SAMPLES // len(config_splits))
    log.info(
        f"  Fetching {NUM_SAMPLES} samples across {len(config_splits)} config/split pair(s) (seed={SEED})..."
    )

    all_samples = []
    try:
        for cs in config_splits:
            result = sample_dataset(
                ds,
                config=cs["config"],
                split=cs["split"],
                n=samples_per_pair,
                seed=SEED,
            )
            for row in result.get("samples", []):
                row["_source_config"] = cs["config"]
                row["_source_split"] = cs["split"]
            all_samples.extend(result.get("samples", []))

        output = {
            "dataset": ds,
            "config_splits": config_splits,
            "samples": all_samples,
        }
        with open(samples_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        return samples_path
    except Exception as e:
        log.error(f"  Prefetch failed: {e}")
        with open(samples_path, "w") as f:
            json.dump({"dataset": ds, "error": str(e)}, f, indent=2)
        return None


def run_claude(prompt: str, stdin_text: str, model: str = "sonnet", timeout: int = 120, retries: int = 2) -> str:
    for attempt in range(retries):
        try:
            result = subprocess.run(
                ["claude", "-p", "--model", model, "--no-session-persistence", prompt],
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            if attempt < retries - 1:
                log.warning(f"  claude -p timed out after {timeout}s, retrying...")
                continue
            raise RuntimeError(f"claude -p timed out after {timeout}s ({retries} attempts)")
        if result.returncode != 0:
            raise RuntimeError(f"claude -p failed: {result.stderr[:500]}")
        return result.stdout


def parse_json_response(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def _format_sample(row: dict, text_fields: list[str], max_chars: int = MAX_CHARS) -> str:
    """Format a raw row dict for review, showing each text field labeled and truncated."""
    parts = []
    for field in text_fields:
        val = str(row.get(field, ""))[:max_chars]
        if len(text_fields) == 1:
            parts.append(val)
        else:
            parts.append(f"{field}: {val}")
    return "\n".join(parts)


def review_batch(
    dataset: str,
    samples_path: Path,
    batch_idx: int,
    output_dir: Path,
    text_fields_per_config: dict[str, list[str]],
) -> Path | None:
    sn = safe_name(dataset)
    batch_path = output_dir / f"{sn}_review_batch_{batch_idx}.json"
    if batch_path.exists():
        return batch_path

    start = batch_idx * BATCH_SIZE
    end = start + BATCH_SIZE

    samples_data = json.loads(samples_path.read_text())
    batch_samples = samples_data["samples"][start:end]

    # Pick a representative text_fields list for the header
    all_fields = sorted({f for fields in text_fields_per_config.values() for f in fields})
    formatted = f"Dataset: {dataset}\nText fields: {', '.join(all_fields)}\n\n"
    for i, row in enumerate(batch_samples):
        config = row.get("_source_config", "default")
        fields = text_fields_per_config.get(config, all_fields)
        formatted += f"===== SAMPLE {start + i} =====\n{_format_sample(row, fields)}\n\n"

    prompt = f"""You are reviewing samples from `{dataset}` to decide whether it's worth \
including in a 20T-token LLM pretraining pipeline and filtering with a quality classifier.

Messy but natural text (forums, comments, chat) is fine for pretraining. \
Only penalize noise that a model would learn nothing useful from.

Review the batch of {len(batch_samples)} samples (indices {start}-{end - 1}) as a whole. \
Do NOT score individual samples.

Respond with ONLY valid JSON, no markdown fences:
{{
  "batch": {batch_idx},
  "usability": 0,
  "usability_rationale": "Why this score — what would/wouldn't survive filtering",
  "content_types": "What kinds of text observed (prose, code, dialogue, tables, etc.)",
  "noise_profile": "Dominant noise if any — systematic (filterable) or random?",
  "provenance_signal": "Human-written, synthetic/LLM-generated, or template-based? Look for: repetitive structure, unnaturally uniform formatting, or other telltale signs.",
  "notes": "Anything else notable"
}}

Usability scale (can a classifier extract good pretraining data from this?):
  5 = Almost all samples useful as-is or with minimal filtering
  4 = Most samples have real content; some noise a classifier can catch
  3 = Mixed — meaningful content exists but significant filtering needed
  2 = Mostly noise/boilerplate/junk, but some signal might be extractable
  1 = Not useful — garbled, empty, or fundamentally non-textual"""

    try:
        response = run_claude(prompt, formatted, model="sonnet")
        batch_path.write_text(response)
        return batch_path
    except Exception as e:
        log.error(f"  Batch {batch_idx} failed: {e}")
        return None


def merge_reviews(dataset: str, output_dir: Path) -> Path | None:
    sn = safe_name(dataset)
    review_path = output_dir / f"{sn}_review.json"
    if review_path.exists():
        return review_path

    batch_files = sorted(output_dir.glob(f"{sn}_review_batch_*.json"))
    if not batch_files:
        log.error(f"  No batch reviews found for {dataset}")
        return None

    # Read dataset summary for full metadata context
    summary_path = output_dir / f"{sn}.json"
    summary_text = ""
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        meta_parts = [f"Dataset: {dataset}"]

        # Configs, splits, and row counts
        configs_info = summary.get("configs_info", {})
        if configs_info:
            lines = []
            for cfg, splits in configs_info.items():
                split_parts = []
                for s_name, s_meta in splits.items():
                    n = s_meta.get("num_examples", "?")
                    split_parts.append(f"{s_name} ({n} rows)")
                lines.append(f"  {cfg}: {', '.join(split_parts)}")
            meta_parts.append("Configs and splits:\n" + "\n".join(lines))

        # Size info
        size_info = summary.get("size", {})
        if size_info:
            size_lines = []
            for split_name, split_meta in size_info.items():
                rows = split_meta.get("num_rows", "?")
                nbytes = split_meta.get("num_bytes_parquet_files")
                size_str = f"{rows} rows"
                if nbytes:
                    approx_tokens = nbytes // 4
                    size_str += f", ~{approx_tokens:,} tokens (estimated)"
                size_lines.append(f"  {split_name}: {size_str}")
            meta_parts.append("Size:\n" + "\n".join(size_lines))

        total_bytes = summary.get("total_file_bytes")
        if total_bytes:
            meta_parts.append(f"Total file size: {total_bytes:,} bytes")

        # Features
        features_per_config = summary.get("features_per_config")
        if features_per_config:
            feat_lines = []
            for cfg, feats in features_per_config.items():
                feat_lines.append(f"  [{cfg}]")
                feat_lines.extend(f"    {name}: {dtype}" for name, dtype in feats.items())
            meta_parts.append("Features (vary by config):\n" + "\n".join(feat_lines))
        else:
            features = summary.get("features", {})
            if features:
                feat_lines = [f"  {name}: {dtype}" for name, dtype in features.items()]
                meta_parts.append("Features:\n" + "\n".join(feat_lines))

        # README
        readme = summary.get("readme")
        if readme:
            meta_parts.append(f"README (first 2000 chars):\n{readme[:2000]}")

        summary_text = "\n\n".join(meta_parts) + "\n\n"

    all_batches = [bf.read_text() for bf in batch_files]
    batches_text = "\n\n---\n\n".join(all_batches)

    prompt = f"""You are assessing `{dataset}` for inclusion in a 20T-token LLM pretraining \
pipeline. Below are batch-level reviews of randomly sampled data, the dataset README, and \
dataset metadata.

The batch reviews are based on random samples — trust them over README claims about quality. \
Use the README for context about provenance, intended use, and coverage.

Respond with ONLY valid JSON, no markdown fences:
{{
  "dataset": "{dataset}",
  "priority": 0,
  "content_summary": "What this dataset contains — domain, register, text types",
  "usability": 0,
  "estimated_usable_fraction": "<25% | 25-50% | 50-75% | >75%",
  "provenance": "human | synthetic | template | mixed — with brief evidence",
  "noise_profile": "What needs filtering, systematic vs random",
  "language": "Primary language(s)",
  "scale": "Size context",
  "concerns": ["List: toxicity, licensing, etc. Empty list if none."],
  "strengths": ["What makes this valuable for pretraining"],
  "summary": "2-3 sentence overall assessment and justification for priority score"
}}

Priority rubric — use these anchors for self-consistency:

9-10: Large-scale, high usability (4-5), valuable domain.
      Filter immediately.
7-8:  Either large and good quality, or moderate scale with a high-value niche domain.
5-6:  Moderate yield. Usable content exists but lower yield or less unique domain.
3-4:  Low yield. Small dataset in a common domain, or large dataset with poor usability
      (<25% likely usable). Filter only if you need to fill gaps.
1-2:  Not worth the compute. Fundamentally unusable, or so small and common-domain that
      even perfect content wouldn't matter at 20T scale and doesn't fill a niche."""

    stdin_text = summary_text + "Batch review results:\n" + batches_text

    try:
        response = run_claude(prompt, stdin_text, model="opus", timeout=180)
        parsed = parse_json_response(response)
        with open(review_path, "w") as f:
            json.dump(parsed, f, indent=2)
        return review_path
    except Exception as e:
        log.error(f"  Merge failed: {e}")
        return None


def review_one(entry: dict, output_dir: Path) -> bool:
    ds = entry["dataset"]
    sn = safe_name(ds)

    review_path = output_dir / f"{sn}_review.json"
    if review_path.exists():
        log.info(f"  Already reviewed, skipping")
        return True

    # Prefetch samples
    samples_path = prefetch(entry, output_dir)
    if not samples_path:
        return False

    samples_data = json.loads(samples_path.read_text())
    num_samples = len(samples_data.get("samples", []))
    num_batches = (num_samples + BATCH_SIZE - 1) // BATCH_SIZE

    # Batch reviews
    text_fields_per_config = entry.get("text_fields_per_config")
    if not text_fields_per_config:
        # Backwards compat with old-format entries
        fields = entry.get("text_fields", entry.get("text_field", ["text"]))
        if isinstance(fields, str):
            fields = [fields]
        text_fields_per_config = {"default": fields}
    log.info(f"  Reviewing {num_samples} samples in {num_batches} batches...")
    failed = 0
    for batch_idx in range(num_batches):
        batch_path = review_batch(ds, samples_path, batch_idx, output_dir, text_fields_per_config)
        if batch_path:
            log.info(f"    Batch {batch_idx + 1}/{num_batches} done")
        else:
            failed += 1
        time.sleep(1)

    if failed == num_batches:
        log.error(f"  All batches failed")
        return False

    # Merge
    log.info(f"  Merging batch reviews...")
    merged = merge_reviews(ds, output_dir)
    if merged:
        log.info(f"  Review written to {merged}")
        return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Review HuggingFace datasets for pretraining quality.")
    parser.add_argument("--collection", help="HuggingFace collection URL (discovers datasets)")
    parser.add_argument("--dataset", help="Review a single dataset")
    parser.add_argument("--output_dir", default="data/dataset_reviews", help="Output directory")
    args = parser.parse_args()

    if not args.collection and not args.dataset:
        parser.error("Provide --collection <url> or --dataset <name>")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset:
        datasets = [args.dataset]
    else:
        log.info(f"Discovering datasets from collection...")
        datasets = discover_collection(args.collection)
        log.info(f"Found {len(datasets)} datasets")

    already_reviewed = [ds for ds in datasets if (output_dir / f"{safe_name(ds)}_review.json").exists()]
    remaining = [ds for ds in datasets if ds not in set(already_reviewed)]
    if already_reviewed:
        log.info(f"{len(already_reviewed)}/{len(datasets)} datasets already reviewed, {len(remaining)} remaining")
    datasets = remaining

    log.info(f"\nFetching summaries...")
    ensure_summaries(datasets, output_dir)

    results = {"reviewed": 0, "skipped": len(already_reviewed), "failed": 0}
    for i, ds in enumerate(datasets, 1):
        log.info(f"[{i}/{len(datasets)}] {ds} — detecting text fields...")
        entry = make_entry(ds, output_dir)
        if not entry:
            log.warning(f"  No text fields detected, skipping")
            results["failed"] += 1
            continue

        if review_one(entry, output_dir):
            results["reviewed"] += 1
        else:
            results["failed"] += 1

    log.info(f"\nDone. Reviewed: {results['reviewed']}, Skipped: {results['skipped']}, Failed: {results['failed']}")


if __name__ == "__main__":
    main()
