#!/usr/bin/env python3
"""
Debug script for VLM batched inference empty string issue.
Run: python debug_vlm_batch.py

This script helps diagnose why ~37% of samples have empty generation results
in batched VLM inference.
"""

import logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def minimal_repro_test():
    """
    Minimal test case to reproduce the empty string issue.
    Simulates the batching logic in vlm_eval_harness.py.
    """
    logger.info("\n" + "="*80)
    logger.info("MINIMAL REPRODUCTION TEST")
    logger.info("="*80)

    # Simulate 8 requests, some are None (no images)
    vlm_requests = ["req0", "req1", None, "req3", None, "req5", "req6", "req7"]
    valid_requests = [r for r in vlm_requests if r is not None]
    valid_indices = [i for i, r in enumerate(vlm_requests) if r is not None]

    logger.info(f"vlm_requests: {vlm_requests}")
    logger.info(f"valid_requests: {valid_requests} (count: {len(valid_requests)})")
    logger.info(f"valid_indices: {valid_indices}")

    # Simulate engine.generate() result
    # Normal case: result.tokens should have 6 elements (one per valid request)
    class MockResultComplete:
        tokens = [[1, 2, 3], [4, 5, 6], [7, 8], [9], [10, 11], [12, 13, 14]]

    logger.info("\n--- Case 1: Complete results (6/6) ---")
    result = MockResultComplete()
    batch_results = [""] * len(vlm_requests)

    for i, idx in enumerate(valid_indices):
        logger.debug(f"  i={i}, idx={idx}, condition: {i} < {len(result.tokens)}")
        if result.tokens and i < len(result.tokens):
            batch_results[idx] = f"text_{i}"
        else:
            logger.error(f"  ✗ EMPTY at batch_results[{idx}]")

    empty_count = sum(1 for r in batch_results if r == "")
    logger.info(f"batch_results: {batch_results}")
    logger.info(f"Empty count: {empty_count}")

    # Simulate problem case: result.tokens only has 4 elements (2 requests lost)
    logger.info("\n--- Case 2: Incomplete results (4/6) - BUG REPRODUCTION ---")
    class MockResultIncomplete:
        tokens = [[1, 2], [3, 4], [5, 6], [7, 8]]  # Only 4 results!

    result2 = MockResultIncomplete()
    batch_results2 = [""] * len(vlm_requests)

    for i, idx in enumerate(valid_indices):
        logger.debug(f"  i={i}, idx={idx}, condition: {i} < {len(result2.tokens)}")
        if result2.tokens and i < len(result2.tokens):
            batch_results2[idx] = f"text_{i}"
        else:
            logger.error(f"  ✗ EMPTY at batch_results2[{idx}]! (i={i} >= len={len(result2.tokens)})")

    empty_count2 = sum(1 for r in batch_results2 if r == "")
    logger.info(f"batch_results2: {batch_results2}")
    logger.info(f"Empty count: {empty_count2} (Expected some text, got empty!)")

    # Simulate problem case 2: result.tokens[i] is empty list
    logger.info("\n--- Case 3: Empty token lists in result ---")
    class MockResultWithEmpty:
        tokens = [[1, 2, 3], [], [7, 8], [], [10, 11], []]  # 3 empty lists!

    result3 = MockResultWithEmpty()
    batch_results3 = [""] * len(vlm_requests)

    for i, idx in enumerate(valid_indices):
        if result3.tokens and i < len(result3.tokens):
            generated_tokens = result3.tokens[i]
            if not generated_tokens:
                logger.warning(f"  ⚠ Empty token list at result.tokens[{i}]")
            # Simulate tokenizer.decode([]) → ""
            batch_results3[idx] = "text" if generated_tokens else ""
        else:
            logger.error(f"  ✗ Index out of range")

    empty_count3 = sum(1 for r in batch_results3 if r == "")
    logger.info(f"batch_results3: {batch_results3}")
    logger.info(f"Empty count: {empty_count3}")

    logger.info("\n" + "="*80)
    logger.info("CONCLUSION:")
    logger.info("  - Case 2 shows: When engine returns fewer tokens than valid_requests, later requests get empty")
    logger.info("  - Case 3 shows: When individual token lists are empty, decode returns empty string")
    logger.info("  - Need to check WHY engine returns incomplete/empty tokens")
    logger.info("="*80)


def check_inference_engine_results():
    """
    More detailed test using actual InferenceEngine logic simulation.
    """
    logger.info("\n" + "="*80)
    logger.info("INFERENCE ENGINE RESULT COLLECTION SIMULATION")
    logger.info("="*80)

    # Simulate InferenceEngine result collection
    from dataclasses import dataclass, field
    from typing import Dict, List

    @dataclass
    class DecodeResult:
        id: int
        choice: int
        token_list: List[int] = field(default_factory=list)
        done: bool = False

    @dataclass
    class Request:
        request_id: int
        n_generations: int = 1

    # Simulate self.results dict
    results: Dict[int, Dict[int, DecodeResult]] = {}

    # Simulate 8 requests
    requests = [Request(request_id=i) for i in range(8)]

    # Simulate _extract_outputs behavior
    # Assume only some requests' outputs are correctly mapped
    mapped_outputs = {0: [101, 102], 1: [103], 3: [104, 105, 106], 5: [107]}

    logger.info("Simulating _extract_outputs behavior:")
    for rid, tokens in mapped_outputs.items():
        results.setdefault(rid, {}).setdefault(0, DecodeResult(id=rid, choice=0)).token_list.extend(tokens)
        logger.info(f"  Request {rid}: Added tokens {tokens}")

    logger.info(f"\nUnmapped requests: {[i for i in range(8) if i not in mapped_outputs]}")

    # Simulate generate() result collection (Lines 1158-1183)
    outputs_list = []
    for r in requests:
        rid = int(r.request_id)
        kid_map = results.get(rid, {})
        for k in range(int(r.n_generations)):
            dr = kid_map.get(k)
            if dr is None:
                logger.warning(f"  Request {rid}: No DecodeResult found, creating empty")
                kid_map[k] = DecodeResult(id=rid, choice=k, token_list=[])
                dr = kid_map[k]
            outputs_list.append(dr.token_list)

    logger.info(f"\nFinal outputs_list:")
    for i, tokens in enumerate(outputs_list):
        status = "✓" if tokens else "✗ EMPTY"
        logger.info(f"  Request {i}: {status} - {tokens}")

    empty_count = sum(1 for t in outputs_list if not t)
    logger.info(f"\nEmpty results: {empty_count}/{len(outputs_list)} ({100*empty_count/len(outputs_list):.1f}%)")


def analyze_result_file(filepath: str):
    """
    Analyze a VLM evaluation result file to understand the pattern of empty generations.
    """
    import json

    logger.info("\n" + "="*80)
    logger.info(f"ANALYZING RESULT FILE: {filepath}")
    logger.info("="*80)

    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load file: {e}")
        return

    # Get the first task's results
    task_name = list(data.keys())[0]
    samples = data[task_name]

    logger.info(f"Task: {task_name}")
    logger.info(f"Total samples: {len(samples)}")

    # Analyze empty vs non-empty
    empty_indices = []
    non_empty_indices = []

    for i, sample in enumerate(samples):
        gen = sample.get("generation", "")
        if gen == "":
            empty_indices.append(i)
        else:
            non_empty_indices.append(i)

    logger.info(f"\nEmpty generations: {len(empty_indices)} ({100*len(empty_indices)/len(samples):.1f}%)")
    logger.info(f"Non-empty generations: {len(non_empty_indices)} ({100*len(non_empty_indices)/len(samples):.1f}%)")

    # Check batch patterns
    batch_size = 8
    logger.info(f"\nBatch analysis (batch_size={batch_size}):")

    for batch_start in range(0, min(40, len(samples)), batch_size):
        batch_end = min(batch_start + batch_size, len(samples))
        batch_samples = samples[batch_start:batch_end]

        batch_status = []
        for i, sample in enumerate(batch_samples):
            gen = sample.get("generation", "")
            num_images = sample.get("num_images", 0)
            status = "✓" if gen else "✗"
            batch_status.append(f"{status}({num_images}img)")

        empty_in_batch = sum(1 for s in batch_samples if s.get("generation", "") == "")
        logger.info(f"  Batch [{batch_start}:{batch_end}]: {' '.join(batch_status)} | Empty: {empty_in_batch}/{len(batch_samples)}")

    # Check if empty correlates with num_images
    logger.info("\nCorrelation with num_images:")
    from collections import defaultdict
    empty_by_images = defaultdict(int)
    total_by_images = defaultdict(int)

    for sample in samples:
        num_images = sample.get("num_images", 0)
        total_by_images[num_images] += 1
        if sample.get("generation", "") == "":
            empty_by_images[num_images] += 1

    for num_images in sorted(total_by_images.keys()):
        empty = empty_by_images[num_images]
        total = total_by_images[num_images]
        logger.info(f"  {num_images} images: {empty}/{total} empty ({100*empty/total:.1f}%)")


if __name__ == "__main__":
    import sys

    # Run basic tests
    minimal_repro_test()
    print("\n" + "="*80 + "\n")
    check_inference_engine_results()

    # If a result file is provided, analyze it
    if len(sys.argv) > 1:
        print("\n" + "="*80 + "\n")
        analyze_result_file(sys.argv[1])
    else:
        print("\n" + "="*80)
        print("TIP: Run with a result file to analyze:")
        print("  python debug_vlm_batch.py vlm_eval_results/samples_mmmu_val_20260130_055548.json")
        print("="*80)
