# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utilities for benchmarking RL training pipeline performance.

This module provides tools to compute and summarize key performance metrics:
- Communication overhead per training step
- vLLM inference throughput (tokens/second)
- Levanter training throughput (tokens/second)
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Summary of benchmark metrics for RL training pipeline."""
    
    # Training metrics
    avg_step_duration_sec: float = 0.0
    avg_training_tokens_per_sec: float = 0.0
    total_training_steps: int = 0
    
    # Inference metrics
    avg_inference_tokens_per_sec: float = 0.0
    total_inference_tokens: int = 0
    total_inference_requests: int = 0
    
    # Weight transfer metrics
    avg_weight_transfer_duration_sec: float = 0.0
    total_bytes_transferred: int = 0
    total_weight_transfers: int = 0


def compute_benchmark_summary(wandb_run) -> BenchmarkMetrics:
    """
    Compute benchmark summary from wandb run history.
    
    Args:
        wandb_run: wandb.Run object with logged metrics
        
    Returns:
        BenchmarkMetrics with computed averages
    """
    try:
        import pandas as pd
        
        history = wandb_run.history()
        
        metrics = BenchmarkMetrics()
        
        # Training metrics
        if 'train.step_duration_sec' in history.columns:
            train_data = history['train.step_duration_sec'].dropna()
            if len(train_data) > 0:
                metrics.avg_step_duration_sec = train_data.mean()
                metrics.total_training_steps = len(train_data)
        
        if 'train.tokens_per_second' in history.columns:
            tps_data = history['train.tokens_per_second'].dropna()
            if len(tps_data) > 0:
                metrics.avg_training_tokens_per_sec = tps_data.mean()
        
        # Inference metrics
        if 'inference.vllm.avg_tokens_per_second' in history.columns:
            inf_data = history['inference.vllm.avg_tokens_per_second'].dropna()
            if len(inf_data) > 0:
                metrics.avg_inference_tokens_per_sec = inf_data.iloc[-1]  # Latest value
        
        if 'inference.vllm.total_tokens_generated' in history.columns:
            tok_data = history['inference.vllm.total_tokens_generated'].dropna()
            if len(tok_data) > 0:
                metrics.total_inference_tokens = int(tok_data.iloc[-1])
        
        if 'inference.vllm.total_requests' in history.columns:
            req_data = history['inference.vllm.total_requests'].dropna()
            if len(req_data) > 0:
                metrics.total_inference_requests = int(req_data.iloc[-1])
        
        # Weight transfer metrics
        if 'train.weight_transfer_duration_sec' in history.columns:
            wt_data = history['train.weight_transfer_duration_sec'].dropna()
            if len(wt_data) > 0:
                metrics.avg_weight_transfer_duration_sec = wt_data.mean()
        
        if 'train.weight_transfer.total_bytes_transferred' in history.columns:
            bytes_data = history['train.weight_transfer.total_bytes_transferred'].dropna()
            if len(bytes_data) > 0:
                metrics.total_bytes_transferred = int(bytes_data.iloc[-1])
        
        if 'train.weight_transfer.total_transfers' in history.columns:
            transfers_data = history['train.weight_transfer.total_transfers'].dropna()
            if len(transfers_data) > 0:
                metrics.total_weight_transfers = int(transfers_data.iloc[-1])
        
        return metrics
        
    except ImportError:
        logger.warning("pandas not available, cannot compute benchmark summary")
        return BenchmarkMetrics()
    except Exception as e:
        logger.error(f"Failed to compute benchmark summary: {e}")
        return BenchmarkMetrics()


def log_benchmark_summary(wandb_run, step: int = None):
    """
    Log benchmark summary to wandb as a summary metric.
    
    Args:
        wandb_run: wandb.Run object
        step: Optional step number for logging
    """
    metrics = compute_benchmark_summary(wandb_run)
    
    summary_dict = {
        "benchmark/avg_step_duration_sec": metrics.avg_step_duration_sec,
        "benchmark/avg_training_tokens_per_sec": metrics.avg_training_tokens_per_sec,
        "benchmark/avg_inference_tokens_per_sec": metrics.avg_inference_tokens_per_sec,
        "benchmark/total_training_steps": metrics.total_training_steps,
        "benchmark/total_inference_tokens": metrics.total_inference_tokens,
        "benchmark/avg_weight_transfer_duration_sec": metrics.avg_weight_transfer_duration_sec,
        "benchmark/total_bytes_transferred_mb": metrics.total_bytes_transferred / (1024 * 1024),
    }
    
    # Log as summary metrics
    for key, value in summary_dict.items():
        wandb_run.summary[key] = value
    
    # Optionally log as regular metrics
    if step is not None:
        wandb_run.log(summary_dict, step=step)
    
    logger.info("Benchmark Summary:")
    logger.info(f"  Weight transfer time: {metrics.avg_weight_transfer_duration_sec:.3f}s per step")
    logger.info(f"  Training throughput: {metrics.avg_training_tokens_per_sec:.1f} tokens/sec")
    logger.info(f"  Inference throughput: {metrics.avg_inference_tokens_per_sec:.1f} tokens/sec")
    
    return metrics


def print_benchmark_report(metrics: BenchmarkMetrics):
    """
    Print a formatted benchmark report.
    
    Args:
        metrics: BenchmarkMetrics to report
    """
    print("\n" + "=" * 70)
    print("BENCHMARK REPORT")
    print("=" * 70)
    print("\nTraining Performance:")
    print(f"  Average step duration:        {metrics.avg_step_duration_sec:.3f} seconds")
    print(f"  Training throughput:          {metrics.avg_training_tokens_per_sec:.1f} tokens/sec")
    print(f"  Total training steps:         {metrics.total_training_steps}")
    
    print("\nvLLM Inference Performance:")
    print(f"  Inference throughput:         {metrics.avg_inference_tokens_per_sec:.1f} tokens/sec")
    print(f"  Total tokens generated:       {metrics.total_inference_tokens:,}")
    print(f"  Total inference requests:     {metrics.total_inference_requests}")
    
    print("\nWeight Transfer Performance:")
    print(f"  Average transfer duration:    {metrics.avg_weight_transfer_duration_sec:.3f} seconds")
    print(f"  Total bytes transferred:      {metrics.total_bytes_transferred / (1024**2):.1f} MB")
    print(f"  Total transfers:              {metrics.total_weight_transfers}")
    if metrics.total_weight_transfers > 0:
        avg_mb_per_transfer = (metrics.total_bytes_transferred / (1024**2)) / metrics.total_weight_transfers
        print(f"  Average MB per transfer:      {avg_mb_per_transfer:.1f} MB")
        if metrics.avg_weight_transfer_duration_sec > 0:
            throughput = avg_mb_per_transfer / metrics.avg_weight_transfer_duration_sec
            print(f"  Transfer throughput:          {throughput:.1f} MB/sec")
    
    print("=" * 70 + "\n")
