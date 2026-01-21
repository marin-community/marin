#!/usr/bin/env python3
"""
Analyze shuffle quality of converted LLaVA-OneVision dataset.

This script reads parquet shards from GCS and analyzes how well the data
from different sources (coyo, datacomp1b, imagenet, etc.) is shuffled
across shards.

Usage:
    python analyze_shuffle_quality.py gs://your-bucket/llava_onevision_levanter/

    # Analyze only first N shards
    python analyze_shuffle_quality.py gs://your-bucket/path/ --max-shards 100

    # Save report to file
    python analyze_shuffle_quality.py gs://your-bucket/path/ --output report.txt
"""

import argparse
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import gcsfs
    import pandas as pd
    import pyarrow.parquet as pq
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install gcsfs pandas pyarrow")
    sys.exit(1)


def list_parquet_shards(gcs_path: str, fs) -> List[str]:
    """List all parquet shard files in GCS path, sorted by name."""
    # Remove gs:// prefix for gcsfs
    path = gcs_path.replace("gs://", "")
    path = path.rstrip("/")

    try:
        files = fs.ls(path)
    except Exception as e:
        print(f"Error listing files in {gcs_path}: {e}")
        return []

    # Filter for parquet files and sort
    parquet_files = [f for f in files if f.endswith(".parquet")]
    parquet_files.sort()

    return parquet_files


def analyze_shard(shard_path: str, fs) -> Tuple[Counter, int]:
    """
    Analyze source distribution in a single shard.

    Returns:
        Tuple of (source_counts, total_rows)
    """
    try:
        # Read only the 'source' column if it exists
        with fs.open(shard_path, 'rb') as f:
            table = pq.read_table(f, columns=['source'] if 'source' in pq.read_schema(f).names else None)
            df = table.to_pandas()

        if 'source' in df.columns:
            source_counts = Counter(df['source'].tolist())
        else:
            # If no source column, count as 'unknown'
            source_counts = Counter({'unknown': len(df)})

        return source_counts, len(df)
    except Exception as e:
        print(f"  Warning: Error reading {shard_path}: {e}", file=sys.stderr)
        return Counter(), 0


def calculate_shuffle_metrics(
    shard_stats: Dict[str, Counter],
    global_stats: Counter,
    total_rows: int
) -> Dict:
    """
    Calculate shuffle quality metrics using variance ratio method.

    Good shuffle means each shard has similar source distribution to global distribution.
    We compare observed variance to expected variance under perfect random shuffle.
    """
    if not shard_stats or total_rows == 0:
        return {}

    # Get all sources
    all_sources = list(global_stats.keys())

    # Calculate global proportions
    global_proportions = {
        source: count / total_rows
        for source, count in global_stats.items()
    }

    # Calculate average shard size
    avg_shard_size = total_rows / len(shard_stats)

    # Calculate per-shard proportions
    shard_proportions = {}
    for shard_name, counts in shard_stats.items():
        shard_total = sum(counts.values())
        if shard_total > 0:
            shard_proportions[shard_name] = {
                source: counts.get(source, 0) / shard_total
                for source in all_sources
            }

    # Calculate variance for each source across shards
    source_variances = {}
    for source in all_sources:
        proportions = [
            props.get(source, 0)
            for props in shard_proportions.values()
        ]
        if proportions:
            source_variances[source] = np.var(proportions)

    # Calculate expected variance under perfect random shuffle (binomial distribution)
    # expected_var = p * (1-p) / n
    expected_variances = {}
    for source in all_sources:
        p = global_proportions[source]
        expected_variances[source] = p * (1 - p) / avg_shard_size

    # Calculate variance ratios (observed / expected)
    variance_ratios = {}
    for source in all_sources:
        expected_var = expected_variances[source]
        observed_var = source_variances.get(source, 0)
        if expected_var > 0:
            variance_ratios[source] = observed_var / expected_var
        else:
            variance_ratios[source] = 0.0

    # Calculate weighted average ratio (weighted by global proportion)
    total_weight = sum(global_proportions[s] for s in variance_ratios if variance_ratios[s] > 0)
    if total_weight > 0:
        weighted_ratio = sum(
            variance_ratios[s] * global_proportions[s]
            for s in variance_ratios
        ) / total_weight
    else:
        weighted_ratio = 1.0

    # Calculate max deviation from global mean (kept for reference)
    max_deviations = {}
    for source in all_sources:
        global_prop = global_proportions[source]
        deviations = [
            abs(props.get(source, 0) - global_prop)
            for props in shard_proportions.values()
        ]
        if deviations:
            max_deviations[source] = max(deviations)

    # Overall metrics
    avg_variance = np.mean(list(source_variances.values())) if source_variances else 0
    max_deviation = max(max_deviations.values()) if max_deviations else 0

    # Shuffle quality score (0-100, higher is better)
    # Based on variance ratio: ratio=1 means perfect random shuffle
    # ratio=1 -> 100, ratio=10 -> 50, ratio=100 -> 0
    if weighted_ratio <= 1:
        # Better than random (stratified sampling) - cap at 100
        quality_score = 100.0
    else:
        quality_score = max(0, 100 * (1 - np.log10(weighted_ratio) / 2))

    return {
        'global_proportions': global_proportions,
        'shard_proportions': shard_proportions,
        'source_variances': source_variances,
        'expected_variances': expected_variances,
        'variance_ratios': variance_ratios,
        'weighted_ratio': weighted_ratio,
        'avg_shard_size': avg_shard_size,
        'max_deviations': max_deviations,
        'avg_variance': avg_variance,
        'max_deviation': max_deviation,
        'quality_score': quality_score
    }


def generate_report(
    gcs_path: str,
    shard_stats: Dict[str, Counter],
    global_stats: Counter,
    total_rows: int,
    metrics: Dict,
    output_file: Optional[str] = None
) -> str:
    """Generate the shuffle quality report."""
    lines = []

    lines.append("=" * 70)
    lines.append("SHUFFLE QUALITY REPORT")
    lines.append("=" * 70)
    lines.append(f"Dataset: {gcs_path}")
    lines.append(f"Total shards analyzed: {len(shard_stats)}")
    lines.append(f"Total rows: {total_rows:,}")
    lines.append("")

    # Global distribution
    lines.append("-" * 70)
    lines.append("GLOBAL DISTRIBUTION (across all shards)")
    lines.append("-" * 70)

    sorted_sources = sorted(global_stats.items(), key=lambda x: -x[1])
    for source, count in sorted_sources:
        pct = 100 * count / total_rows if total_rows > 0 else 0
        lines.append(f"  {source:25s}: {pct:6.2f}% ({count:,} samples)")
    lines.append("")

    # Per-shard distribution (summary)
    lines.append("-" * 70)
    lines.append("PER-SHARD DISTRIBUTION (all shards)")
    lines.append("-" * 70)

    shard_names = sorted(shard_stats.keys())
    top_sources = [s for s, _ in sorted_sources[:5]]  # Top 5 sources

    # Header
    header = f"  {'Shard':20s}"
    for source in top_sources:
        header += f" {source[:8]:>8s}"
    header += "  (rows)"
    lines.append(header)

    for shard_name in shard_names:
        counts = shard_stats[shard_name]
        shard_total = sum(counts.values())

        # Extract shard number from path
        shard_display = shard_name.split("/")[-1].replace(".parquet", "")

        row = f"  {shard_display:20s}"
        for source in top_sources:
            pct = 100 * counts.get(source, 0) / shard_total if shard_total > 0 else 0
            row += f" {pct:7.1f}%"
        row += f"  ({shard_total:,})"
        lines.append(row)

    lines.append("")

    # Shuffle quality metrics
    lines.append("-" * 70)
    lines.append("SHUFFLE QUALITY METRICS (Variance Ratio Method)")
    lines.append("-" * 70)

    if metrics:
        lines.append(f"  Average shard size: {metrics['avg_shard_size']:,.0f} rows")
        lines.append(f"  Weighted variance ratio: {metrics['weighted_ratio']:.2f} (1.0 = perfect random shuffle)")
        lines.append(f"  Shuffle quality score: {metrics['quality_score']:.1f}/100")

        # Interpretation
        score = metrics['quality_score']
        if score >= 90:
            interpretation = "EXCELLENT - Close to perfect random shuffle"
        elif score >= 70:
            interpretation = "GOOD - Shuffle quality is good"
        elif score >= 50:
            interpretation = "FAIR - Some clustering detected"
        else:
            interpretation = "POOR - Significant clustering, consider re-shuffling"

        lines.append(f"  Interpretation: {interpretation}")
        lines.append("")

        # Per-source variance ratios
        lines.append("  Per-source variance ratio (observed/expected, 1.0 = ideal):")
        sorted_ratios = sorted(metrics['variance_ratios'].items(), key=lambda x: -x[1])
        for source, ratio in sorted_ratios[:10]:
            obs_var = metrics['source_variances'].get(source, 0)
            exp_var = metrics['expected_variances'].get(source, 0)
            lines.append(f"    {source:20s}: {ratio:6.2f}x  (obs={obs_var:.6f}, exp={exp_var:.6f})")
    else:
        lines.append("  (No metrics available)")

    lines.append("")
    lines.append("=" * 70)

    report = "\n".join(lines)

    # Output
    print(report)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_file}")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Analyze shuffle quality of converted LLaVA-OneVision dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_shuffle_quality.py gs://your-bucket/llava_onevision_levanter/
    python analyze_shuffle_quality.py gs://your-bucket/path/ --max-shards 100
    python analyze_shuffle_quality.py gs://your-bucket/path/ --output report.txt
        """
    )
    parser.add_argument(
        "--gcs_path",
        type=str,default='gs://marin-vlm/stage2_sharded/',
        help="GCS path to the converted dataset (e.g., gs://bucket/path/)"
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=100,
        help="Maximum number of shards to analyze (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for the report (default: print to stdout only)"
    )

    args = parser.parse_args()

    # Initialize GCS filesystem
    print("Initializing GCS filesystem...")
    fs = gcsfs.GCSFileSystem()

    # List shards
    print(f"Listing parquet files in {args.gcs_path}...")
    shard_paths = list_parquet_shards(args.gcs_path, fs)

    if not shard_paths:
        print("No parquet files found!")
        sys.exit(1)

    print(f"Found {len(shard_paths)} parquet files")

    # Limit shards if requested
    if args.max_shards and args.max_shards < len(shard_paths):
        shard_paths = shard_paths[:args.max_shards]
        print(f"Analyzing first {args.max_shards} shards...")

    # Analyze each shard
    shard_stats = {}
    global_stats = Counter()
    total_rows = 0

    print("Analyzing shards...")
    for i, shard_path in enumerate(shard_paths):
        if (i + 1) % 100 == 0 or i == 0 or i == len(shard_paths) - 1:
            print(f"  Processing shard {i + 1}/{len(shard_paths)}...")

        counts, rows = analyze_shard(shard_path, fs)

        if rows > 0:
            shard_stats[shard_path] = counts
            global_stats.update(counts)
            total_rows += rows

    print(f"Analyzed {len(shard_stats)} shards with {total_rows:,} total rows")
    print()

    # Calculate metrics
    metrics = calculate_shuffle_metrics(shard_stats, global_stats, total_rows)

    # Generate report
    generate_report(
        args.gcs_path,
        shard_stats,
        global_stats,
        total_rows,
        metrics,
        args.output
    )


if __name__ == "__main__":
    main()
