from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np

from marin.utils import fsspec_exists, fsspec_glob, fsspec_mkdirs, fsspec_size

logger = logging.getLogger(__name__)


@dataclass
class PlotSlidingLogitsConfig:
    """Configuration for creating plots from sliding logits results."""
    
    # Input/Output paths - these will be resolved by the executor framework
    input_path: str  # Path to sliding logits results directory (InputName)
    original_text_path: str  # Path to original text file
    output_path: str  # This step's output directory (OutputName)
    
    # Plot configuration
    plot_title: str = "Sliding Logits: Maximum per-character probability"
    colormap: str = "Blues"
    figsize: tuple[int, int] = (14, 2)
    dpi: int = 300
    
    # Optional: save combined arrays for further analysis
    save_combined_arrays: bool = True
    
    # Optional: compute and save extraction rate statistics
    compute_extraction_stats: bool = True


def create_sliding_logits_plot(cfg: PlotSlidingLogitsConfig) -> Dict[str, Any]:
    """
    Read sliding logits results and create character-level heatmap.
    
    This function:
    1. Discovers all char_max_part_*.npy files in the input directory
    2. Loads and combines them using element-wise maximum
    3. Creates a single-row heatmap showing max probability per character
    4. Optionally computes extraction rate statistics
    5. Saves plot and arrays to output directory
    
    Returns:
        Dictionary with paths and summary statistics
    """
    
    print(f"Creating sliding logits plot from {cfg.input_path}", flush=True)
    logger.info("Creating sliding logits plot from %s", cfg.input_path)
    
    # Ensure output directory exists
    logger.info("Creating output directory: %s", cfg.output_path)
    print(f"Creating output directory: {cfg.output_path}", flush=True)
    fsspec_mkdirs(cfg.output_path)
    
    # Step 1: Discover char_max shard files using marin's fsspec utilities
    logger.info("Discovering char_max shard files...")
    print(f"Discovering char_max shard files...", flush=True)
    
    # First try the multi-core pattern (char_max_part_*.npy)
    search_pattern = os.path.join(cfg.input_path, "char_max_part_*.npy")
    logger.info("Searching for files matching: %s", search_pattern)
    print(f"Searching for files matching: {search_pattern}", flush=True)
    char_max_files = fsspec_glob(search_pattern)
    print(f"Found {len(char_max_files)} char_max shard files", flush=True)
    # If no multi-core files found, try the tensor parallel patterns
    if not char_max_files:
        # Try different tensor parallel patterns
        tp_patterns = [
            os.path.join(cfg.input_path, "char_max_tp.npy"),  # Actual pattern from gsutil ls
            os.path.join(cfg.input_path, "char_max.npy"),     # Original expected pattern
        ]
        print(f"No multi-core files found, searching for tensor parallel file: {tp_patterns}", flush=True)
        for tp_pattern in tp_patterns:
            print(f"Searching for tensor parallel file: {tp_pattern}", flush=True)
            logger.info("No multi-core files found, searching for tensor parallel file: %s", tp_pattern)
            if fsspec_exists(tp_pattern):
                char_max_files = [tp_pattern]
                print(f"Found tensor parallel char_max file: {tp_pattern}", flush=True)
                logger.info("Found tensor parallel char_max file: %s", tp_pattern)
                break
        else:
            raise FileNotFoundError(
                f"No char_max files found in {cfg.input_path}. "
                f"Searched for 'char_max_part_*.npy' (multi-core), 'char_max_tp.npy', and 'char_max.npy' (tensor parallel)"
            )
    else:
        logger.info("Found multi-core char_max files")
    
    char_max_files.sort()  # Ensure consistent ordering
    
    logger.info("Found %d char_max shard files", len(char_max_files))
    print(f"Found {len(char_max_files)} char_max shard files", flush=True)
    for i, file_path in enumerate(char_max_files):
        try:
            file_size = fsspec_size(file_path)
            print(f"  Shard {i}: {file_path} (size: {file_size} bytes)", flush=True)
            logger.info("  Shard %d: %s (size: %d bytes)", i, file_path, file_size)
        except Exception as e:
            logger.warning("  Shard %d: %s (could not get size: %s)", i, file_path, e)
    
    # Step 2: Load original text to get expected length
    print(f"Loading original text from {cfg.original_text_path}", flush=True)
    logger.info("Loading original text from %s", cfg.original_text_path)
    print(f"Loading original text from {cfg.original_text_path}", flush=True)
    # Use marin's fsspec utilities
    import fsspec
    with fsspec.open(cfg.original_text_path, "r") as f:
        original_text = f.read()
    expected_length = len(original_text)
    logger.info("Original text length: %d characters", expected_length)
    print(f"Original text length: {expected_length} characters", flush=True)
    # Step 3: Load and combine character-level data
    logger.info("Loading and combining char_max arrays...")
    print(f"Loading and combining char_max arrays...", flush=True)
    combined_char_max = None

    for i, file_path in enumerate(char_max_files):
        logger.info("Loading shard %d/%d: %s", i + 1, len(char_max_files), file_path)
        print(f"Loading shard {i + 1}/{len(char_max_files)}: {file_path}", flush=True)
        # Load each file carefully to avoid OOM
        try:
            with fsspec.open(file_path, "rb") as f:
                part_array = np.load(f)
                logger.info("  Loaded array shape: %s, dtype: %s, memory: %.2f MB", 
                           part_array.shape, part_array.dtype, part_array.nbytes / 1024 / 1024)
                print(f"  Loaded array shape: {part_array.shape}, dtype: {part_array.dtype}, memory: {part_array.nbytes / 1024 / 1024:.2f} MB", flush=True)
                
                if combined_char_max is None:
                    combined_char_max = part_array.copy()
                    logger.info("  Initialized combined array with shape: %s", combined_char_max.shape)
                    print(f"  Initialized combined array with shape: {combined_char_max.shape}", flush=True)
                else:
                    # Verify shapes match
                    if part_array.shape != combined_char_max.shape:
                        raise ValueError(
                            f"Shape mismatch: shard {i} has shape {part_array.shape}, "
                            f"but expected {combined_char_max.shape}"
                        )
                    
                    # Element-wise maximum across cores
                    combined_char_max = np.maximum(combined_char_max, part_array)
                    logger.info("  Combined with existing array using element-wise maximum")
                    print(f"  Combined with existing array using element-wise maximum", flush=True)

                # Free memory from this shard
                del part_array
                
        except Exception as e:
            logger.error("Error loading shard %d (%s): %s", i, file_path, e)
            raise
    
    if combined_char_max is None:
        raise RuntimeError("No char_max arrays were successfully loaded")
    
    # Validate combined array
    actual_length = len(combined_char_max)
    logger.info("Combined char_max array: shape=%s, dtype=%s, memory=%.2f MB", 
                combined_char_max.shape, combined_char_max.dtype, combined_char_max.nbytes / 1024 / 1024)
    
    if actual_length != expected_length:
        logger.warning(
            "Length mismatch: combined array has %d elements, but original text has %d characters",
            actual_length, expected_length
        )
    
    # Compute basic statistics
    max_prob = float(combined_char_max.max())
    mean_prob = float(combined_char_max.mean())
    min_prob = float(combined_char_max.min())
    nonzero_count = int(np.count_nonzero(combined_char_max))
    
    logger.info("Combined array statistics:")
    logger.info("  Max probability: %.6f", max_prob)
    logger.info("  Mean probability: %.6f", mean_prob)
    logger.info("  Min probability: %.6f", min_prob)
    logger.info("  Non-zero positions: %d/%d (%.1f%%)", 
                nonzero_count, actual_length, 100 * nonzero_count / actual_length)
    print(f"Combined array statistics: Max probability: {max_prob}, Mean probability: {mean_prob}, Min probability: {min_prob}, Non-zero positions: {nonzero_count}/{actual_length} ({100 * nonzero_count / actual_length:.1f}%)", flush=True)
    # Step 4: Create visualization
    logger.info("Creating heatmap visualization...")
    
    try:
        fig, ax = plt.subplots(figsize=cfg.figsize)
        
        # Create single-row heatmap
        print(f"Creating heatmap visualization...", flush=True)
        im = ax.imshow(
            combined_char_max[np.newaxis, :],  # shape (1, text_len)
            cmap=cfg.colormap,
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )
        
        ax.set_title(cfg.plot_title)
        ax.set_xlabel("Character position")
        ax.set_yticks([])  # Hide y-axis (only one row)
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
        cbar.set_label("Max. suffix probability")
        
        plt.tight_layout()
        
        # Save plot using fsspec
        print(f"Saving plot to {cfg.output_path}/sliding_logits_plot.png", flush=True)
        plot_path = os.path.join(cfg.output_path, "sliding_logits_plot.png")
        logger.info("Saving plot to %s", plot_path)
        with fsspec.open(plot_path, "wb") as f:
            plt.savefig(f, dpi=cfg.dpi, bbox_inches='tight')
        
        plt.close()
        print(f"Plot saved successfully", flush=True)
        logger.info("Plot saved successfully")
        
    except Exception as e:
        logger.error("Error creating or saving plot: %s", e)
        raise
    
    # Step 5: Save combined arrays if requested
    results = {
        "plot_path": plot_path,
        "text_length": actual_length,
        "max_probability": max_prob,
        "mean_probability": mean_prob,
        "min_probability": min_prob,
        "nonzero_positions": nonzero_count,
        "coverage_percent": 100 * nonzero_count / actual_length,
    }
    
    if cfg.save_combined_arrays:
        # Save combined char_max array
        array_path = os.path.join(cfg.output_path, "char_max_combined.npy")
        logger.info("Saving combined char_max array to %s", array_path)
        
        try:
            with fsspec.open(array_path, "wb") as f:
                np.save(f, combined_char_max)
            results["char_max_array_path"] = array_path
            logger.info("Combined array saved successfully")
        except Exception as e:
            logger.error("Error saving combined array: %s", e)
            raise
    
    # Step 6: Compute extraction statistics if requested
    if cfg.compute_extraction_stats:
        logger.info("Computing extraction rate statistics...")
        
        try:
            # Look for the main sliding_logits files (not just char_max)
            # Try both patterns: multi-core and tensor parallel
            logits_pattern_multicore = os.path.join(cfg.input_path, "sliding_logits_*_part*.np*")
            logits_pattern_tp = os.path.join(cfg.input_path, "sliding_logits_tp_part*.np*")  # Actual pattern from gsutil ls
            logits_pattern_tp_alt = os.path.join(cfg.input_path, "sliding_logits_part*.np*")  # Alternative pattern
            
            logits_files = fsspec_glob(logits_pattern_multicore)
            if not logits_files:
                logger.info("No multi-core logits files found, trying tensor parallel patterns")
                logits_files = fsspec_glob(logits_pattern_tp)
                if not logits_files:
                    logits_files = fsspec_glob(logits_pattern_tp_alt)
            
            if logits_files:
                logger.info("Found %d logits shard files for extraction statistics", len(logits_files))
                
                # Collect all pz values
                pz_values = []
                for file_path in logits_files:
                    try:
                        logger.info("  Loading pz from %s", file_path)
                        with fsspec.open(file_path, "rb") as f:
                            if file_path.endswith('.npy'):
                                # Uncompressed format - load as dict
                                data = np.load(f, allow_pickle=True).item()
                            else:
                                # Compressed format
                                data = np.load(f)
                            
                            if 'pz' in data:
                                pz_batch = data['pz']
                                pz_values.extend(pz_batch.tolist())
                                logger.info("    Loaded %d pz values", len(pz_batch))
                            else:
                                logger.warning("    No 'pz' key found")
                                
                    except Exception as e:
                        logger.warning("    Error loading pz: %s", e)
                        continue
                
                if pz_values:
                    logger.info("Computing extraction rates for %d pz values", len(pz_values))
                    
                    # Compute basic extraction statistics
                    pz_array = np.array(pz_values)
                    
                    # Compute some key extraction rates
                    # For p=0.5, 0.9, 0.99, compute mean n needed
                    p_thresholds = [0.5, 0.9, 0.99]
                    extraction_stats = {}
                    
                    for p in p_thresholds:
                        # n such that 1 - (1-pz)^n >= p
                        # n >= log(1-p) / log(1-pz)
                        valid_pz = pz_array[(pz_array > 0) & (pz_array < 1)]
                        if len(valid_pz) > 0:
                            log1m_p = np.log(1 - p)
                            log1m_pz = np.log(1 - valid_pz)
                            n_values = np.ceil(log1m_p / log1m_pz)
                            n_values = n_values[np.isfinite(n_values)]
                            
                            if len(n_values) > 0:
                                extraction_stats[f"mean_n_for_p_{p}"] = float(np.mean(n_values))
                                extraction_stats[f"median_n_for_p_{p}"] = float(np.median(n_values))
                                extraction_stats[f"max_n_for_p_{p}"] = float(np.max(n_values))
                    
                    extraction_stats["num_pz_values"] = len(pz_values)
                    extraction_stats["mean_pz"] = float(np.mean(pz_array))
                    extraction_stats["median_pz"] = float(np.median(pz_array))
                    extraction_stats["max_pz"] = float(np.max(pz_array))
                    
                    results["extraction_stats"] = extraction_stats
                    
                    logger.info("Extraction statistics computed:")
                    for key, value in extraction_stats.items():
                        logger.info("  %s: %s", key, value)
                    
                    # Save extraction stats
                    if cfg.save_combined_arrays:
                        stats_path = os.path.join(cfg.output_path, "extraction_stats.npy")
                        logger.info("Saving extraction statistics to %s", stats_path)
                        
                        with fsspec.open(stats_path, "wb") as f:
                            np.save(f, extraction_stats)
                        results["extraction_stats_path"] = stats_path
                        
                else:
                    logger.warning("No pz values found in shard files")
            else:
                logger.warning("No logits shard files found for extraction statistics")
                
        except Exception as e:
            logger.error("Error computing extraction statistics: %s", e)
            # Don't raise - extraction stats are optional
    
    logger.info("Sliding logits plot creation completed successfully")
    return results


if __name__ == "__main__":
    import draccus

    @draccus.wrap()
    def main(cfg: PlotSlidingLogitsConfig):  # pragma: no cover
        results = create_sliding_logits_plot(cfg)
        print("Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")

    main() 