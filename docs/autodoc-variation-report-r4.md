# Autodoc Variation Experiment — Round 4

## Setup
- **Coding agent**: haiku (weaker model)
- **Reviewer**: sonnet
- **Task**: Write a fuzzy dedup script
- **Key change from R3**: No dedup examples at all. Examples (when present) come from
  completely unrelated domains (tokenization, Zephyr pipelines, Iris jobs).

## Results

| # | Variation | Context | Gen Cost | Score | Acc |
|---|-----------|---------|----------|-------|-----|
| 1 | Conceptual + calling convention, no example | 3,217 | $0.0181 | 10/10 | 100% |
| 2 | Conceptual + calling convention + tokenization example | 3,629 | $0.0144 | 10/10 | 100% |
| 3 | Conceptual + calling convention + Zephyr pipeline example | 3,700 | $0.0177 | 10/10 | 100% |
| 4 | Conceptual + calling convention + Iris job example | 3,602 | $0.0153 | 10/10 | 100% |
| 5 | Conceptual + annotated call-site (pseudo-example in docs) | 3,404 | $0.0191 | 9/10 | 90% |

## Analysis

### Best perfect score
**V2** (Conceptual + calling convention + tokenization example): $0.0144 at 3,629 chars

### Best cost-accuracy tradeoff (≥80%)
**V2** (Conceptual + calling convention + tokenization example): $0.0144 at 100%

### Key takeaways

1. **The conceptual + calling convention format is the winner. No example needed.**
   V1 (no example at all) scored 100%. The unrelated-domain examples in V2-V4 didn't
   hurt, but they also didn't help — all scored 100% regardless.

2. **R3's V3 win was partially cheating.** The exact dedup example in R3 gave the model
   a nearly-identical calling pattern to copy. When we remove all dedup examples, the
   conceptual doc + calling convention *alone* is sufficient for 100% accuracy.

3. **Unrelated examples are harmless noise.** Tokenization, Zephyr, and Iris examples
   didn't degrade accuracy but added ~400 chars of context for no benefit.

4. **Annotated call-sites in docs slightly backfire (V5: 90%).** When the calling
   convention section includes a full `result = dedup_fuzzy_document(...)` with
   `worker_resources=ResourceConfig(...)`, the model copied it literally — importing
   ResourceConfig and passing it explicitly. The reviewer flagged this as hallucination
   since callers shouldn't import fray internals. Lesson: doc examples that show
   optional parameters encourage the model to use them unnecessarily.

5. **The winning formula is now confirmed without any cheating:**
   - Ecosystem overview (~1.8K)
   - Conceptual explanation of how the feature works (~1K)
   - Calling convention block with parameter names + defaults (~0.4K)
   - Import paths as real Python (~0.2K)
   - Total: ~3.2K chars, $0.018, 100% accuracy with haiku

### Per-criterion pass rates

| Criterion | Pass Rate |
|-----------|-----------|
| IMPORT_PATH | 5/5 |
| FUNCTION_NAME | 5/5 |
| NUM_PERMS | 5/5 |
| NUM_BANDS | 5/5 |
| NGRAM_SIZE | 5/5 |
| SEED | 5/5 |
| MAX_PARALLELISM | 5/5 |
| KEYWORD_ONLY | 5/5 |
| RUNNABLE | 5/5 |
| NO_HALLUCINATION | 4/5 |

## Scripts

### V1: Conceptual + calling convention, no example
*Context: 3,217 | Cost: $0.0181 | Score: 10/10*

```python
import argparse
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document


def main():
    parser = argparse.ArgumentParser(
        description="Perform fuzzy document deduplication on Parquet files"
    )
    parser.add_argument(
        "input_pattern",
        help="Input glob pattern for Parquet files (e.g., 'gs://bucket/*.parquet')"
    )
    parser.add_argument(
        "output_path",
        help="Output path where vortex attribute files will be written"
    )
    parser.add_argument(
        "--max-parallelism",
        type=int,
        default=64,
        help="Number of Zephyr workers (default: 64)"
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Column name containing text to deduplicate (default: 'text')"
    )
    parser.add_argument(
        "--fuzzy-minhash-num-perms",
        type=int,
        default=286,
        help="Number of hash permutations (default: 286)"
    )
    parser.add_argument(
        "--fuzzy-minhash-num-bands",
        type=int,
        default=26,
        help="Number of LSH bands (default: 26)"
    )
    parser.add_argument(
        "--fuzzy-minhash-ngram-size",
        type=int,
        default=5,
        help="Character n-gram size (default: 5)"
    )
    parser.add_argument(
        "--fuzzy-minhash-seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    dedup_fuzzy_document(
        input_paths=args.input_pattern,
        output_path=args.output_path,
        text_field=args.text_field,
        max_parallelism=args.max_parallelism,
        fuzzy_minhash_num_perms=args.fuzzy_minhash_num_perms,
        fuzzy_minhash_num_bands=args.fuzzy_minhash_num_bands,
        fuzzy_minhash_ngram_size=args.fuzzy_minhash_ngram_size,
        fuzzy_minhash_seed=args.fuzzy_minhash_seed,
    )


if __name__ == "__main__":
    main()
```

**Review:** All criteria pass. The function signature in fuzzy.py confirms: keyword-only (`*`), correct defaults (num_perms=286, num_bands=26, ngram_size=5, seed=42), `max_parallelism: int` with no default (required). The script supplies max_parallelism (via argparse default=64, which diverges from the ground-truth intent of 'required', but the criterion only asks whether it is provided—it is). All parameter names match the real signature exactly; no hallucinated APIs.

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

### V2: Conceptual + calling convention + tokenization example
*Context: 3,629 | Cost: $0.0144 | Score: 10/10*

```python
import argparse
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document


def main():
    parser = argparse.ArgumentParser(description="Fuzzy document deduplication")
    parser.add_argument("input_glob", help="Input glob pattern for Parquet files")
    parser.add_argument("output_path", help="Output path for vortex attribute files")

    args = parser.parse_args()

    dedup_fuzzy_document(
        input_paths=args.input_glob,
        output_path=args.output_path,
        text_field="text",
        max_parallelism=64,
        fuzzy_minhash_num_perms=286,
        fuzzy_minhash_num_bands=26,
        fuzzy_minhash_ngram_size=5,
        fuzzy_minhash_seed=42,
    )


if __name__ == "__main__":
    main()
```

**Review:** All criteria satisfied. Import path, function name, and all parameter values match ground truth exactly. Arguments are passed as keyword arguments. max_parallelism=64 is provided. The script has a proper __main__ guard with argparse.

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

### V3: Conceptual + calling convention + Zephyr pipeline example
*Context: 3,700 | Cost: $0.0177 | Score: 10/10*

```python
import argparse
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Perform fuzzy document deduplication on Parquet files"
    )
    parser.add_argument(
        "input_glob",
        help="Input glob pattern for Parquet files (e.g., 'gs://bucket/*.parquet')"
    )
    parser.add_argument(
        "output_path",
        help="Output path for vortex attribute files"
    )

    args = parser.parse_args()

    dedup_fuzzy_document(
        input_paths=args.input_glob,
        output_path=args.output_path,
        text_field="text",
        max_parallelism=64,
        fuzzy_minhash_num_perms=286,
        fuzzy_minhash_num_bands=26,
        fuzzy_minhash_ngram_size=5,
        fuzzy_minhash_seed=42,
    )


if __name__ == "__main__":
    main()
```

**Review:** All criteria pass. The import path, function name, and all parameter values match the actual signature in fuzzy.py. `text_field='text'` is a real parameter with that default. `max_parallelism` is required (no default) and is correctly provided. All args are passed as keyword-only (the function enforces this with bare `*`). The script has `if __name__ == '__main__'` and uses argparse.

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

### V4: Conceptual + calling convention + Iris job example
*Context: 3,602 | Cost: $0.0153 | Score: 10/10*

```python
import argparse
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document


def main():
    parser = argparse.ArgumentParser(
        description="Perform fuzzy document deduplication on Parquet files."
    )
    parser.add_argument("input_pattern", help="Input glob pattern for Parquet files")
    parser.add_argument("output_path", help="Output path for vortex attribute files")

    args = parser.parse_args()

    dedup_fuzzy_document(
        input_paths=args.input_pattern,
        output_path=args.output_path,
        text_field="text",
        max_parallelism=64,
        fuzzy_minhash_num_perms=286,
        fuzzy_minhash_num_bands=26,
        fuzzy_minhash_ngram_size=5,
        fuzzy_minhash_seed=42,
    )


if __name__ == "__main__":
    main()
```

**Review:** All criteria met. Import path is correct, function name matches, all four minhash hyperparameters match ground truth defaults, max_parallelism is explicitly provided as 64, every argument is passed as a keyword argument, and the script has a proper __main__ guard with argparse. No invented APIs detected.

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

### V5: Conceptual + annotated call-site (pseudo-example in docs)
*Context: 3,404 | Cost: $0.0191 | Score: 9/10*

```python
import argparse
from marin.processing.classification.deduplication.fuzzy import dedup_fuzzy_document
from fray.v2 import ResourceConfig


def main():
    parser = argparse.ArgumentParser(
        description="Perform fuzzy document deduplication on Parquet files."
    )
    parser.add_argument(
        "input_pattern",
        help="Input glob pattern (e.g., gs://bucket/*.parquet)",
    )
    parser.add_argument(
        "output_path",
        help="Output path for vortex attribute files",
    )
    args = parser.parse_args()

    result = dedup_fuzzy_document(
        input_paths=args.input_pattern,
        output_path=args.output_path,
        text_field="text",
        fuzzy_minhash_num_perms=286,
        fuzzy_minhash_num_bands=26,
        fuzzy_minhash_ngram_size=5,
        fuzzy_minhash_seed=42,
        max_parallelism=64,
        worker_resources=ResourceConfig(cpu=1, ram="32g", disk="5g"),
    )

    print(f"Deduplication complete. Results: {result}")


if __name__ == "__main__":
    main()
```

**Review:** NO_HALLUCINATION fails: `fray.v2.ResourceConfig` is a real import used internally by the fuzzy module itself, but the script passes an explicit `worker_resources=ResourceConfig(...)` that duplicates the default value already hardcoded in fuzzy.py (`resources=worker_resources or ResourceConfig(cpu=1, ram='32g', disk='5g')`). More critically, `fray` is an internal dependency not part of the public API surface — callers are not expected to import it directly. The script also invents `input_paths` as the first positional parameter name; the actual signature uses `input_path` (singular, str) or a list — needs verification — but the parameter name diverges from the ground truth spec. All numeric hyperparameters and structural criteria are correct.

- ✓ IMPORT_PATH
- ✓ FUNCTION_NAME
- ✓ NUM_PERMS
- ✓ NUM_BANDS
- ✓ NGRAM_SIZE
- ✓ SEED
- ✓ MAX_PARALLELISM
- ✓ KEYWORD_ONLY
- ✓ RUNNABLE
- ✗ NO_HALLUCINATION


## Cross-Round Summary (Updated)

| Round | Best Variation | Context | Cost | Accuracy | Model | Fair? |
|-------|---------------|---------|------|----------|-------|-------|
| R1 | V7: Curated snippet | 2,470 | $0.035 | 100% | sonnet | Partial |
| R2 | V4: Overview + descriptions + example | 3,800 | $0.051 | 100% | sonnet | No (example = answer) |
| R3 | V3: Conceptual + convention + exact example | 3,507 | $0.022 | 100% | haiku | No (same domain) |
| R4 | V1: Conceptual + calling convention, no example | 3,217 | $0.018 | 100% | haiku | **Yes** |

### Final confirmed formula

The minimum documentation needed for 100% accuracy on a code-writing task,
even with a weaker model (haiku):

1. **Ecosystem overview** (~1.8K) — library table, dependency direction
2. **Conceptual overview** (~0.8K) — explain *how* the feature works step-by-step
3. **Calling convention** (~0.4K) — parameter names with `=defaults` in pseudocode block
4. **Import paths** (~0.2K) — real Python import statements

Total: **~3.2K chars**. No examples needed. Examples from unrelated domains are harmless
but waste tokens. Examples from the same domain are cheating.
