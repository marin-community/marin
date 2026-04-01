# Autodoc Variation Experiment — Round 2

## Objective
Minimize agent token cost while producing a correct, runnable dedup script.
Agent starts from a top-level Marin ecosystem overview and must navigate to the right APIs.

## Task
> Write a Python script that fuzzy-deduplicates Parquet files using Marin's built-in pipeline.
> Script must include proper imports, argparse, correct function name and defaults.

## Scoring Criteria
| Criterion | Description |
|-----------|-------------|
| IMPORT_PATH | Correct import path for `dedup_fuzzy_document` |
| FUNCTION_NAME | Uses `dedup_fuzzy_document` (not invented names) |
| NUM_PERMS | Default 286 (not 128) |
| NUM_BANDS | Default 26 |
| NGRAM_SIZE | Default 5 |
| SEED | Default 42 |
| MAX_PARALLELISM | Required arg provided |
| KEYWORD_ONLY | Keyword arguments used |
| RUNNABLE | Has `__main__` block + argparse |
| NO_HALLUCINATION | No invented APIs |

## Results

| # | Variation | Context | Gen Cost | Score | Accuracy |
|---|-----------|---------|----------|-------|----------|
| 1 | Overview + full signatures | 4,968 chars | $0.0563 | 9/10 | 90% |
| 2 | Overview + 1-sentence descriptions | 3,085 chars | $0.0444 | 5/10 | 50% |
| 3 | Overview + grep-friendly index | 3,671 chars | $0.0475 | 6/10 | 60% |
| 4 | Overview + 1-sentence + example | 3,934 chars | $0.0507 | 10/10 | 100% |
| 5 | Overview + full signatures + example | 5,817 chars | $0.0473 | 9/10 | 90% |
| 6 | Overview only (no module docs) | 1,779 chars | $0.1176 | 6/10 | 60% |
| 7 | Overview + generated marin.processing doc | 21,193 chars | $0.0556 | 8/10 | 80% |
| 8 | Overview + generated 3-module docs | 32,342 chars | $0.0891 | 8/10 | 80% |
| 9 | Overview + 1-sentence + grep index | 4,977 chars | $0.2147 | 6/10 | 60% |
| 10 | No context baseline | 0 chars | $0.2187 | 9/10 | 90% |

## Analysis

### Best cost-accuracy tradeoff
**V5** (Overview + full signatures + example): $0.0473 at 90% accuracy (5,817 chars)

### Cheapest perfect score
**V4** (Overview + 1-sentence + example): $0.0507 (3,934 chars)

### Per-criterion pass rates

| Criterion | Pass Rate |
|-----------|-----------|
| IMPORT_PATH | 8/10 (80%) |
| FUNCTION_NAME | 10/10 (100%) |
| NUM_PERMS | 6/10 (60%) |
| NUM_BANDS | 6/10 (60%) |
| NGRAM_SIZE | 6/10 (60%) |
| SEED | 6/10 (60%) |
| MAX_PARALLELISM | 5/10 (50%) |
| KEYWORD_ONLY | 10/10 (100%) |
| RUNNABLE | 10/10 (100%) |
| NO_HALLUCINATION | 9/10 (90%) |

## Detailed Results

### V1: Overview + full signatures
*Context: 4,968 chars | Gen cost: $0.0563 | Score: 9/10*

```python
import argparse

from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuzzy-deduplicate a set of Parquet files.")
    parser.add_argument("input_glob", help="Glob pattern for input Parquet files (e.g. gs://bucket/data/*.parquet)")
    parser.add_argument("output_path", help="Output path for deduplicated results")
    parser.add_argument("--text-field", default="text", help="Name of the text column (default: text)")
    parser.add_argument("--max-parallelism", type=int, default=64, help="Maximum parallelism (default: 64)")
    parser.add_argument("--num-perms", type=int, default=286, help="Number of MinHash permutations (default: 286)")
    parser.add_argument("--num-bands", type=int, default=26, help="Number of LSH bands (default: 26)")
    parser.add_argument("--ngram-size", type=int, default=5, help="N-gram size for MinHash (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for MinHash (default: 42)")
    args = parser.parse_args()

    result = dedup_fuzzy_document(
        input_paths=args.input_glob,
        output_path=args.output_path,
        text_field=args.text_field,
        filetypes=["parquet"],
        fuzzy_minhash_num_perms=args.num_perms,
        fuzzy_minhash_num_bands=args.num_bands,
        fuzzy_minhash_ngram_size=args.ngram_size,
        fuzzy_minhash_seed=args.seed,
        max_parallelism=args.max_parallelism,
    )
    print(result)


if __name__ == "__main__":
    main()
```

**Review notes:** max_parallelism is declared as an optional argparse argument with default=64 rather than as a required argument, violating the ground-truth contract that it has no default.

**Scores:**
- ✓ IMPORT_PATH
- ✓ FUNCTION_NAME
- ✓ NUM_PERMS
- ✓ NUM_BANDS
- ✓ NGRAM_SIZE
- ✓ SEED
- ✗ MAX_PARALLELISM
- ✓ KEYWORD_ONLY
- ✓ RUNNABLE
- ✓ NO_HALLUCINATION

### V2: Overview + 1-sentence descriptions
*Context: 3,085 chars | Gen cost: $0.0444 | Score: 5/10*

```python
import argparse

from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document


def main(input_glob: str, output_path: str) -> None:
    dedup_fuzzy_document(
        input_glob=input_glob,
        output_path=output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fuzzy-deduplicate Parquet files using MinHash/LSH.")
    parser.add_argument("input_glob", help='Input glob pattern, e.g. "gs://bucket/data/*.parquet"')
    parser.add_argument("output_path", help="Output path for deduplicated results")
    args = parser.parse_args()
    main(args.input_glob, args.output_path)
```

**Review notes:** Import path and function name are correct, but the script omits all five required/configured parameters (fuzzy_minhash_num_perms, fuzzy_minhash_num_bands, fuzzy_minhash_ngram_size, fuzzy_minhash_seed, and the required max_parallelism), so the call would fail at runtime.

**Scores:**
- ✓ IMPORT_PATH
- ✓ FUNCTION_NAME
- ✗ NUM_PERMS
- ✗ NUM_BANDS
- ✗ NGRAM_SIZE
- ✗ SEED
- ✗ MAX_PARALLELISM
- ✓ KEYWORD_ONLY
- ✓ RUNNABLE
- ✓ NO_HALLUCINATION

### V3: Overview + grep-friendly index
*Context: 3,671 chars | Gen cost: $0.0475 | Score: 6/10*

```python
import argparse
import glob

from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuzzy-deduplicate a set of Parquet files.")
    parser.add_argument("input_glob", help="Glob pattern for input Parquet files (e.g. gs://bucket/data/*.parquet)")
    parser.add_argument("output_path", help="Output path for deduplicated Parquet files")
    parser.add_argument("--max-parallelism", type=int, default=8, help="Maximum parallelism for the dedup job")
    args = parser.parse_args()

    input_paths = glob.glob(args.input_glob)
    if not input_paths:
        raise ValueError(f"No files matched glob pattern: {args.input_glob}")

    result = dedup_fuzzy_document(
        input_paths=input_paths,
        output_path=args.output_path,
        max_parallelism=args.max_parallelism,
    )
    print(result)


if __name__ == "__main__":
    main()
```

**Review notes:** Import path, function name, keyword-only calling convention, max_parallelism, and CLI structure are all correct, but the script never explicitly passes fuzzy_minhash_num_perms (286), fuzzy_minhash_num_bands (26), fuzzy_minhash_ngram_size (5), or fuzzy_minhash_seed (42), relying entirely on undocumented defaults.

**Scores:**
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

### V4: Overview + 1-sentence + example
*Context: 3,934 chars | Gen cost: $0.0507 | Score: 10/10*

```python
import argparse

from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuzzy-deduplicate Parquet files using MinHash/LSH.")
    parser.add_argument("input_glob", help="Input glob pattern (e.g. gs://bucket/data/*.parquet)")
    parser.add_argument("output_path", help="Output path for deduplicated files")
    args = parser.parse_args()

    result = dedup_fuzzy_document(
        input_paths=[args.input_glob],
        output_path=args.output_path,
        text_field="text",
        fuzzy_minhash_num_perms=286,
        fuzzy_minhash_num_bands=26,
        fuzzy_minhash_ngram_size=5,
        fuzzy_minhash_seed=42,
        max_parallelism=64,
    )

    print(result)


if __name__ == "__main__":
    main()
```

**Review notes:** The script is fully correct: it uses the exact import path, correct function name, all default parameter values match ground truth, max_parallelism is supplied, all arguments are keyword-only, and no non-existent APIs are invoked.

**Scores:**
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

### V5: Overview + full signatures + example
*Context: 5,817 chars | Gen cost: $0.0473 | Score: 9/10*

```python
import argparse

from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuzzy-deduplicate a set of Parquet files.")
    parser.add_argument("input_glob", help="Input glob pattern (e.g. gs://bucket/data/*.parquet)")
    parser.add_argument("output_path", help="Output path for deduplicated results")
    parser.add_argument("--text-field", default="text", help="Name of the text column (default: text)")
    parser.add_argument("--max-parallelism", type=int, default=64, help="Max worker parallelism (default: 64)")
    parser.add_argument("--num-perms", type=int, default=286, help="MinHash number of permutations (default: 286)")
    parser.add_argument("--num-bands", type=int, default=26, help="MinHash LSH number of bands (default: 26)")
    parser.add_argument("--ngram-size", type=int, default=5, help="MinHash n-gram size (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="MinHash random seed (default: 42)")
    args = parser.parse_args()

    if args.num_perms % args.num_bands != 0:
        parser.error(f"--num-perms ({args.num_perms}) must be divisible by --num-bands ({args.num_bands})")

    result = dedup_fuzzy_document(
        input_paths=args.input_glob,
        output_path=args.output_path,
        text_field=args.text_field,
        fuzzy_minhash_num_perms=args.num_perms,
        fuzzy_minhash_num_bands=args.num_bands,
        fuzzy_minhash_ngram_size=args.ngram_size,
        fuzzy_minhash_seed=args.seed,
        max_parallelism=args.max_parallelism,
    )

    print("Deduplication complete.")
    print("Result counters:", result)


if __name__ == "__main__":
    main()
```

**Review notes:** MAX_PARALLELISM fails because the script defines --max-parallelism with default=64, making it optional, whereas the ground truth specifies it is a required argument with no default.

**Scores:**
- ✓ IMPORT_PATH
- ✓ FUNCTION_NAME
- ✓ NUM_PERMS
- ✓ NUM_BANDS
- ✓ NGRAM_SIZE
- ✓ SEED
- ✗ MAX_PARALLELISM
- ✓ KEYWORD_ONLY
- ✓ RUNNABLE
- ✓ NO_HALLUCINATION

### V6: Overview only (no module docs)
*Context: 1,779 chars | Gen cost: $0.1176 | Score: 6/10*

```python
import argparse

from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuzzy-deduplicate a set of Parquet files using MinHash/LSH.")
    parser.add_argument("input_glob", help='Input glob pattern, e.g. "gs://bucket/data/*.parquet"')
    parser.add_argument("output_path", help="Output path for deduplicated files")
    parser.add_argument(
        "--max-parallelism",
        type=int,
        default=64,
        help="Maximum number of parallel workers (default: 64)",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Name of the text column in the Parquet files (default: text)",
    )
    args = parser.parse_args()

    result = dedup_fuzzy_document(
        input_paths=args.input_glob,
        output_path=args.output_path,
        text_field=args.text_field,
        max_parallelism=args.max_parallelism,
    )

    print(result)


if __name__ == "__main__":
    main()
```

**Review notes:** Import path, function name, keyword-argument style, max_parallelism, and script structure are correct, but the four required MinHash parameters (num_perms=286, num_bands=26, ngram_size=5, seed=42) are never passed to dedup_fuzzy_document, leaving them at whatever the function's own defaults are.

**Scores:**
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

### V7: Overview + generated marin.processing doc
*Context: 21,193 chars | Gen cost: $0.0556 | Score: 8/10*

```python
import argparse
import glob

from marin.processing.dedup_commons import DedupMode
from marin.processing.fuzzy import dedup_fuzzy_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuzzy-deduplicate a set of Parquet files.")
    parser.add_argument("input_glob", help="Glob pattern for input Parquet files (e.g. gs://bucket/data/*.parquet)")
    parser.add_argument("output_path", help="Output path for deduplicated files")
    args = parser.parse_args()

    input_paths = sorted(glob.glob(args.input_glob))
    if not input_paths:
        raise ValueError(f"No files matched glob pattern: {args.input_glob}")

    result = dedup_fuzzy_document(
        input_paths=input_paths,
        output_path=args.output_path,
        text_field="text",
        fuzzy_minhash_num_perms=286,
        fuzzy_minhash_num_bands=26,
        fuzzy_minhash_ngram_size=5,
        fuzzy_minhash_seed=42,
        max_parallelism=64,
    )

    print(result)


if __name__ == "__main__":
    main()
```

**Review notes:** The script uses the wrong import path (`marin.processing.fuzzy` instead of `marin.processing.classification.deduplication.fuzzy`) and also hallucinates an unused import (`from marin.processing.dedup_commons import DedupMode`) that does not correspond to any real module in the ground truth; all numeric hyperparameters and calling conventions are otherwise correct.

**Scores:**
- ✗ IMPORT_PATH
- ✓ FUNCTION_NAME
- ✓ NUM_PERMS
- ✓ NUM_BANDS
- ✓ NGRAM_SIZE
- ✓ SEED
- ✓ MAX_PARALLELISM
- ✓ KEYWORD_ONLY
- ✓ RUNNABLE
- ✗ NO_HALLUCINATION

### V8: Overview + generated 3-module docs
*Context: 32,342 chars | Gen cost: $0.0891 | Score: 8/10*

```python
import argparse
import fsspec

from marin.processing.dedup.fuzzy import dedup_fuzzy_document


def glob_input_paths(pattern: str) -> list[str]:
    protocol = pattern.split("://")[0] if "://" in pattern else "file"
    fs = fsspec.filesystem(protocol)
    paths = fs.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    if "://" in pattern:
        return [f"{protocol}://{p}" for p in paths]
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuzzy-deduplicate a set of Parquet files using MinHash LSH."
    )
    parser.add_argument(
        "input_pattern",
        help='Glob pattern for input Parquet files (e.g. "gs://bucket/data/*.parquet")',
    )
    parser.add_argument(
        "output_path",
        help="Output path for deduplicated Parquet files",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Name of the text column in the Parquet files (default: text)",
    )
    parser.add_argument(
        "--num-perms",
        type=int,
        default=286,
        help="Number of MinHash permutations; must be divisible by --num-bands (default: 286)",
    )
    parser.add_argument(
        "--num-bands",
        type=int,
        default=26,
        help="Number of LSH bands; must divide --num-perms evenly (default: 26)",
    )
    parser.add_argument(
        "--ngram-size",
        type=int,
        default=5,
        help="N-gram size for MinHash computation (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for MinHash (default: 42)",
    )
    parser.add_argument(
        "--max-parallelism",
        type=int,
        default=64,
        help="Maximum number of parallel workers (default: 64)",
    )
    args = parser.parse_args()

    if args.num_perms % args.num_bands != 0:
        parser.error(
            f"--num-perms ({args.num_perms}) must be divisible by --num-bands ({args.num_bands})"
        )

    input_paths = glob_input_paths(args.input_pattern)
    print(f"Found {len(input_paths)} input file(s) matching '{args.input_pattern}'")

    result = dedup_fuzzy_document(
        input_paths=input_paths,
        output_path=args.output_path,
        text_field=args.text_field,
        fuzzy_minhash_num_perms=args.num_perms,
        fuzzy_minhash_num_bands=args.num_bands,
        fuzzy_minhash_ngram_size=args.ngram_size,
        fuzzy_minhash_seed=args.seed,
        max_parallelism=args.max_parallelism,
    )
    print("Deduplication complete:", result)


if __name__ == "__main__":
    main()
```

**Review notes:** Import path is wrong (`marin.processing.dedup.fuzzy` instead of `marin.processing.classification.deduplication.fuzzy`), and `max_parallelism` is given a default of 64 rather than being a required argument.

**Scores:**
- ✗ IMPORT_PATH
- ✓ FUNCTION_NAME
- ✓ NUM_PERMS
- ✓ NUM_BANDS
- ✓ NGRAM_SIZE
- ✓ SEED
- ✗ MAX_PARALLELISM
- ✓ KEYWORD_ONLY
- ✓ RUNNABLE
- ✓ NO_HALLUCINATION

### V9: Overview + 1-sentence + grep index
*Context: 4,977 chars | Gen cost: $0.2147 | Score: 6/10*

```python
import argparse
import glob

from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuzzy-deduplicate a set of Parquet files.")
    parser.add_argument("input_glob", help="Glob pattern for input Parquet files (e.g. gs://bucket/data/*.parquet)")
    parser.add_argument("output_path", help="Output path for deduplicated Parquet files")
    args = parser.parse_args()

    input_paths = glob.glob(args.input_glob)
    if not input_paths:
        raise ValueError(f"No files matched glob pattern: {args.input_glob}")

    result = dedup_fuzzy_document(
        input_paths=input_paths,
        output_path=args.output_path,
        max_parallelism=8,
    )

    print(result)


if __name__ == "__main__":
    main()
```

The session's write permissions are restricted to `/Users/power/code/marin-prime/.claude/worktrees/quirky-bouman` and cannot reach `/tmp/autodoc-r2/variation_09/dedup_script.py`. Run this in your terminal to write the file:

```bash
mkdir -p /tmp/autodoc-r2/variation_09 && cat > /tmp/autodoc-r2/variation_09/dedup_script.py << 'EOF'
import argparse
import glob

from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuzzy-deduplicate a set of Parquet files.")
    parser.add_argument("input_glob", help="Glob pattern for input Parquet files (e.g. gs://bucket/data/*.parquet)")
    parser.add_argument("output_path", help="Output path for deduplicated Parquet files")
    args = parser.parse_args()

    input_paths = glob.glob(args.input_glob)
    if not input_paths:
        raise ValueError(f"No files matched glob pattern: {args.input_glob}")

    result = dedup_fuzzy_document(
        input_paths=input_paths,
        output_path=args.output_path,
        max_parallelism=8,
    )

    print(result)


if __name__ == "__main__":
    main()
EOF
```

**Review notes:** Import path and function name are correct and max_parallelism is supplied, but the script omits explicit values for fuzzy_minhash_num_perms (286), fuzzy_minhash_num_bands (26), fuzzy_minhash_ngram_size (5), and fuzzy_minhash_seed (42), relying silently on defaults.

**Scores:**
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

### V10: No context baseline
*Context: 0 chars | Gen cost: $0.2187 | Score: 9/10*

```python
import argparse

from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuzzy-deduplicate a set of Parquet files using MinHash LSH."
    )
    parser.add_argument(
        "input_glob",
        help='Glob pattern for input Parquet files (e.g. "gs://bucket/data/*.parquet")',
    )
    parser.add_argument(
        "output_path",
        help="Output path where deduplicated results will be written",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Name of the field containing document text (default: text)",
    )
    parser.add_argument(
        "--max-parallelism",
        type=int,
        default=64,
        help="Maximum number of parallel workers (default: 64)",
    )
    parser.add_argument(
        "--num-perms",
        type=int,
        default=286,
        help="Number of MinHash permutations (default: 286)",
    )
    parser.add_argument(
        "--num-bands",
        type=int,
        default=26,
        help="Number of LSH bands; must divide evenly into --num-perms (default: 26)",
    )
    parser.add_argument(
        "--ngram-size",
        type=int,
        default=5,
        help="N-gram size used when computing MinHash signatures (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for MinHash (default: 42)",
    )
    args = parser.parse_args()

    if args.num_perms % args.num_bands != 0:
        parser.error(
            f"--num-perms ({args.num_perms}) must be divisible by --num-bands ({args.num_bands})"
        )

    result = dedup_fuzzy_document(
        input_paths=args.input_glob,
        output_path=args.output_path,
        text_field=args.text_field,
        filetypes=["parquet"],
        fuzzy_minhash_num_perms=args.num_perms,
        fuzzy_minhash_num_bands=args.num_bands,
        fuzzy_minhash_ngram_size=args.ngram_size,
        fuzzy_minhash_seed=args.seed,
        max_parallelism=args.max_parallelism,
    )

    total = result.get("dedup/fuzzy/document/total", "N/A")
    dups = result.get("dedup/fuzzy/document/dups", "N/A")
    unique = result.get("dedup/fuzzy/document/unique", "N/A")
    print(f"Deduplication complete: total={total}, duplicates={dups}, unique={unique}")
    print(f"Output written to: {args.output_path}")


if __name__ == "__main__":
    main()
```

**Review notes:** All parameters are correct except MAX_PARALLELISM, which is declared with `default=64` making it optional, whereas the ground truth specifies it is a required argument with no default.

**Scores:**
- ✓ IMPORT_PATH
- ✓ FUNCTION_NAME
- ✓ NUM_PERMS
- ✓ NUM_BANDS
- ✓ NGRAM_SIZE
- ✓ SEED
- ✗ MAX_PARALLELISM
- ✓ KEYWORD_ONLY
- ✓ RUNNABLE
- ✓ NO_HALLUCINATION

