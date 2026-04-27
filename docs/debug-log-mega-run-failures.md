# Debugging log for mega-run-failures

Fix code-path failures in the mega perplexity-gap branch so the raw materialization finishes and score jobs can launch.

## Initial status

Mega launcher on codex/perplexity-gap-mega is running from scratch/perplexity_gap_mega_matrix.log. Earlier failures were trimmed or fixed, but we need the current remaining failures from the live log.

## Hypothesis 1

Excluding `bio_chem/refseq/refseq_viral_gff` at the dataset-map layer is insufficient because `refseq_viral_fasta` and `refseq_viral_gff` share the same raw materialization step.

## Changes to make

- Update `experiments/evals/perplexity_gap_mega_bundle.py` to use a mega-specific RefSeq FASTA-only step.
- Extend `tests/evals/test_perplexity_gap_mega_bundle.py` to assert FASTA is present and GFF is excluded.

## Results

Pending validation.
