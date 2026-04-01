# Autodoc Variation Experiment — Round 3

## Setup
- **Coding agent**: haiku (weaker model)
- **Reviewer**: sonnet
- **Task**: Write a fuzzy dedup script (docs show exact dedup examples only)
- **Docs**: Organized by package/folder, not by topic

## Results

| # | Variation | Context | Gen Cost | Score | Acc |
|---|-----------|---------|----------|-------|-----|
| 1 | Full package doc + semantic overview + exact dedup example | 5,292 | $0.0451 | 5/10 | 50% |
| 2 | Compressed package doc (signatures + 1-liners, no example) | 2,725 | $0.0249 | 6/10 | 60% |
| 3 | Conceptual/semantic doc + calling convention + exact example | 3,507 | $0.0220 | 10/10 | 100% |
| 4 | Hierarchical: package overview + separate module docs | 3,708 | $0.0190 | 6/10 | 60% |
| 5 | Minimal: just signatures in code blocks, no prose | 2,563 | $0.0277 | 6/10 | 60% |

## Analysis

### Best perfect score
**V3** (Conceptual/semantic doc + calling convention + exact example): $0.0220 at 3,507 chars

### Best cost-accuracy tradeoff (≥80%)
**V3** (Conceptual/semantic doc + calling convention + exact example): $0.0220 at 100%

### Key takeaways

1. **Conceptual explanation + calling convention > raw signatures.** V3 (semantic doc
   explaining *how* dedup works, plus a generic calling convention block) scored 100%.
   V5 (just signatures) and V2 (signatures + 1-liners) both scored 60%. The weaker
   model needs the *why* to know which parameters matter.

2. **The "include defaults explicitly" problem.** The consistent failure mode across
   V1/V2/V4/V5 was that haiku didn't pass the MinHash defaults explicitly — it assumed
   the function would use its own defaults. V3 succeeded because its "Calling Convention"
   section showed parameters with `=` values inline, making it obvious those are the
   values to use.

3. **More context ≠ better.** V1 had the most context (5,292 chars) and scored *worst*
   (50%). V3 at 3,507 chars was the only perfect score. The semantic explanation was
   more signal-dense.

4. **Hierarchical docs don't help a weaker model.** V4 spread information across three
   doc sections (package + exact module + fuzzy module). Haiku extracted the basics but
   missed the parameter defaults that were in the fuzzy module doc.

5. **Examples of a *different* feature still help.** V3 included an exact dedup example
   (different from the task). This didn't directly solve the problem but showed the
   calling *pattern* — keyword args, argparse structure — which haiku followed.

6. **This test is honest.** Unlike Round 2, the example here shows exact dedup while the
   task asks for fuzzy dedup. The agent had to generalize from the pattern, not copy-paste.

### Per-criterion pass rates

| Criterion | Pass Rate |
|-----------|-----------|
| IMPORT_PATH | 5/5 |
| FUNCTION_NAME | 5/5 |
| NUM_PERMS | 1/5 |
| NUM_BANDS | 1/5 |
| NGRAM_SIZE | 1/5 |
| SEED | 1/5 |
| MAX_PARALLELISM | 5/5 |
| KEYWORD_ONLY | 5/5 |
| RUNNABLE | 5/5 |
| NO_HALLUCINATION | 4/5 |

## Scripts

### V1: Full package doc + semantic overview + exact dedup example
*Context: 5,292 | Cost: $0.0451 | Score: 5/10*

```python
#!/usr/bin/env python3
import argparse
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document


def main():
    parser = argparse.ArgumentParser(
        description="Perform fuzzy document deduplication on Parquet files"
    )
    parser.add_argument(
        "input_glob",
        help="Input glob pattern (e.g., gs://bucket/data/*.parquet)"
    )
    parser.add_argument(
        "output_path",
        help="Output path for vortex attribute files"
    )
    parser.add_argument(
        "--max-parallelism",
        type=int,
        default=16,
        help="Number of Zephyr workers (default: 16)"
    )

    args = parser.parse_args()

    result = dedup_fuzzy_document(
        input_paths=args.input_glob,
        output_path=args.output_path,
        max_parallelism=args.max_parallelism,
    )

    print("Deduplication complete!")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
```

**Review:** NUM_PERMS/NUM_BANDS/NGRAM_SIZE/SEED all score 0: the script never passes fuzzy_minhash_num_perms=286, fuzzy_minhash_num_bands=26, fuzzy_minhash_ngram_size=5, or fuzzy_minhash_seed=42 to the function — it silently relies on whatever the library defaults happen to be, which is incorrect per spec. MAX_PARALLELISM scores 1 because the value is forwarded to the function call, but note the script gives it an argparse default of 16, contradicting the ground truth that it is required with no default. NO_HALLUCINATION scores 0 for two reasons: (1) the return value is iterated as a dict via result.items(), which is an invented assumption about the return type; (2) the parameter name 'input_paths' is unverified against the actual API signature.

- ✓ IMPORT_PATH
- ✓ FUNCTION_NAME
- ✗ NUM_PERMS
- ✗ NUM_BANDS
- ✗ NGRAM_SIZE
- ✗ SEED
- ✓ MAX_PARALLELISM
- ✓ KEYWORD_ONLY
- ✓ RUNNABLE
- ✗ NO_HALLUCINATION

### V2: Compressed package doc (signatures + 1-liners, no example)
*Context: 2,725 | Cost: $0.0249 | Score: 6/10*

```python
import argparse
import glob
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document


def main():
    parser = argparse.ArgumentParser(
        description="Perform fuzzy document deduplication on Parquet files"
    )
    parser.add_argument("input_pattern", help="Glob pattern for input Parquet files")
    parser.add_argument("output_path", help="Output path for deduplicated data")
    parser.add_argument(
        "--max-parallelism",
        type=int,
        default=4,
        help="Number of Zephyr workers (default: 4)",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Text field name in input data (default: text)",
    )

    args = parser.parse_args()

    # Expand glob pattern
    input_paths = glob.glob(args.input_pattern)
    if not input_paths:
        raise ValueError(f"No files found matching pattern: {args.input_pattern}")

    # Perform fuzzy dedup with default MinHash parameters
    result = dedup_fuzzy_document(
        input_paths=input_paths,
        output_path=args.output_path,
        text_field=args.text_field,
        max_parallelism=args.max_parallelism,
    )

    print(f"Deduplication completed. Result: {result}")


if __name__ == "__main__":
    main()
```

**Review:** Import path and function name are correct. text_field is a real parameter (default='text'). max_parallelism is provided but the script gives it a default of 4 — the real signature has no default (required). The four minhash parameters (num_perms=286, num_bands=26, ngram_size=5, seed=42) are never explicitly passed; the script relies on the function's own defaults, so criteria 3-6 score 0.

- ✓ IMPORT_PATH
- ✓ FUNCTION_NAME
- ✗ NUM_PERMS
- ✗ NUM_BANDS
- ✗ NGRAM_SIZE
- ✗ SEED
- ✓ MAX_PARALLELISM
- ✓ KEYWORD_ONLY
- ✓ RUNNABLE
- ✓ NO_HALLUCINATION

### V3: Conceptual/semantic doc + calling convention + exact example
*Context: 3,507 | Cost: $0.0220 | Score: 10/10*

```python
#!/usr/bin/env python3
"""Fuzzy document deduplication script."""

import argparse
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document


def main():
    parser = argparse.ArgumentParser(
        description="Perform fuzzy document deduplication on Parquet files."
    )
    parser.add_argument(
        "input_pattern",
        help="Input glob pattern (e.g., 'gs://bucket/data/*.parquet')",
    )
    parser.add_argument(
        "output_path",
        help="Output path for vortex attribute files",
    )
    parser.add_argument(
        "--max-parallelism",
        type=int,
        default=64,
        help="Number of Zephyr workers (default: 64)",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Column name containing text (default: 'text')",
    )
    parser.add_argument(
        "--num-perms",
        type=int,
        default=286,
        help="MinHash permutations (default: 286)",
    )
    parser.add_argument(
        "--num-bands",
        type=int,
        default=26,
        help="LSH bands (default: 26)",
    )
    parser.add_argument(
        "--ngram-size",
        type=int,
        default=5,
        help="Character n-gram size (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    result = dedup_fuzzy_document(
        input_paths=args.input_pattern,
        output_path=args.output_path,
        text_field=args.text_field,
        max_parallelism=args.max_parallelism,
        fuzzy_minhash_num_perms=args.num_perms,
        fuzzy_minhash_num_bands=args.num_bands,
        fuzzy_minhash_ngram_size=args.ngram_size,
        fuzzy_minhash_seed=args.seed,
    )

    print(f"Deduplication complete: {result}")


if __name__ == "__main__":
    main()
```

**Review:** All criteria pass. Import path, function name, and all default values (num_perms=286, num_bands=26, ngram_size=5, seed=42) match the source exactly. max_parallelism is provided to the function call (the function signature requires it with no default). All args are passed as keyword-only arguments (the function signature enforces this with bare `*`). The script has `if __name__ == '__main__'` and argparse. The one minor discrepancy is that argparse gives max_parallelism a default of 64, meaning it is optional at the CLI level, whereas the real function signature declares it required (no default). However, the criterion asks only whether max_parallelism is *provided* to the function, which it is.

- ✓ IMPORT_PATH
- ✓ FUNCTION_NAME
- ✓ NUM_PERMS
- ✓ NUM_BANDS
- ✓ NGRAM_SIZE
- ✓ SEED
- ✓ MAX_PARALLELISM
- ✓ KEYWORD_ONLY
- ✓ RUNNABLE
- ✓ NO_HALLUCINATION

### V4: Hierarchical: package overview + separate module docs
*Context: 3,708 | Cost: $0.0190 | Score: 6/10*

```python
import argparse
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document


def main():
    parser = argparse.ArgumentParser(
        description="Perform fuzzy document deduplication on Parquet files."
    )
    parser.add_argument(
        "input_pattern",
        type=str,
        help="Input glob pattern (e.g., 'gs://bucket/data/*.parquet')"
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Output path for deduplication results"
    )
    parser.add_argument(
        "--max-parallelism",
        type=int,
        default=64,
        help="Number of Zephyr workers (default: 64)"
    )

    args = parser.parse_args()

    result = dedup_fuzzy_document(
        input_paths=args.input_pattern,
        output_path=args.output_path,
        max_parallelism=args.max_parallelism,
    )

    print(f"Deduplication complete. Result: {result}")


if __name__ == "__main__":
    main()
```

**Review:** Import path, function name, keyword-only calling convention, max_parallelism presence, argparse entrypoint, and API shape are all correct. The four minhash parameters (num_perms=286, num_bands=26, ngram_size=5, seed=42) are never passed to the call site — the script relies silently on function defaults rather than setting them explicitly, so criteria 3-6 fail. Additionally, the real signature has max_parallelism as a required positional-keyword arg (no default), but the script supplies a default of 64 in argparse, which means the function will always receive a value — a minor semantic mismatch but max_parallelism is still provided.

- ✓ IMPORT_PATH
- ✓ FUNCTION_NAME
- ✗ NUM_PERMS
- ✗ NUM_BANDS
- ✗ NGRAM_SIZE
- ✗ SEED
- ✓ MAX_PARALLELISM
- ✓ KEYWORD_ONLY
- ✓ RUNNABLE
- ✓ NO_HALLUCINATION

### V5: Minimal: just signatures in code blocks, no prose
*Context: 2,563 | Cost: $0.0277 | Score: 6/10*

```python
import argparse
import glob
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document


def main():
    parser = argparse.ArgumentParser(
        description="Perform fuzzy document deduplication on Parquet files"
    )
    parser.add_argument("input_glob", help="Glob pattern for input files")
    parser.add_argument("output_path", help="Output directory path")
    parser.add_argument(
        "--max-parallelism",
        type=int,
        default=4,
        help="Maximum parallelism for workers (default: 4)",
    )

    args = parser.parse_args()

    # Expand glob pattern
    input_files = sorted(glob.glob(args.input_glob, recursive=True))
    if not input_files:
        print(f"No files found matching pattern: {args.input_glob}")
        return

    print(f"Found {len(input_files)} files to deduplicate")

    # Run fuzzy deduplication with default MinHash parameters
    result = dedup_fuzzy_document(
        input_paths=input_files,
        output_path=args.output_path,
        max_parallelism=args.max_parallelism,
    )

    print(f"Deduplication completed: {result}")


if __name__ == "__main__":
    main()
```

**Review:** Import path and function name are correct. max_parallelism is passed (always, via argparse default=4, satisfying the required-with-no-default contract at the function level). All args are keyword-only matching the `*` signature. The script is runnable with argparse and __main__ guard. However, the four minhash parameters (num_perms=286, num_bands=26, ngram_size=5, seed=42) are never explicitly passed — the script relies on function defaults rather than setting them, so criteria 3–6 score 0. No invented APIs detected.

- ✓ IMPORT_PATH
- ✓ FUNCTION_NAME
- ✗ NUM_PERMS
- ✗ NUM_BANDS
- ✗ NGRAM_SIZE
- ✗ SEED
- ✓ MAX_PARALLELISM
- ✓ KEYWORD_ONLY
- ✓ RUNNABLE
- ✓ NO_HALLUCINATION

## Cross-Round Summary

| Round | Best Variation | Context | Cost | Accuracy | Model | Task |
|-------|---------------|---------|------|----------|-------|------|
| R1 | V7: Curated snippet | 2,470 | $0.035 | 100% | sonnet | Q&A |
| R2 | V4: Overview + descriptions + example* | 3,800 | $0.051 | 100% | sonnet | Code (cheating) |
| R3 | V3: Conceptual + calling convention + example | 3,507 | $0.022 | 100% | haiku | Code (honest) |

*R2 was unfair — the example directly solved the task.

### Consistent findings across all rounds

1. **Sweet spot is 2.5–4K chars.** All winning variations land here regardless of model or task.
2. **Conceptual explanations beat raw signatures.** Models need to understand *what* to do, not just *what exists*.
3. **Examples of calling patterns are high-signal**, even when they show a different feature than the task.
4. **More context hurts.** 5K+ chars consistently underperforms 3-4K chars.
5. **No context is the most expensive option** — models hallucinate more and output more tokens.

### Implications for doc generation

The winning doc structure is:
1. Ecosystem overview table (~1.8K, shared across all topics)
2. Conceptual explanation of the feature area (~0.5-1K)
3. Calling convention block showing parameter names, types, and defaults in code-like format (~0.5-1K)
4. Import paths as real Python imports (~0.2K)
5. Short example of a *related* feature showing the pattern (~0.5K)

Total per-topic: ~2-4K chars. This maps cleanly to the three-tier design:
- **Tier 0** (MAP.md): ecosystem overview, always loaded
- **Tier 1** (topics/): conceptual + calling convention + imports
- **Tier 2** (examples/): runnable examples showing patterns
