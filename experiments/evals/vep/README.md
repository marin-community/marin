# Served-model variant effect prediction

This development experiment compares unconditioned DNA likelihood, a
healthy-individual prefix, and zero-/two-shot FASTA/VCF classification on the
same autosomal missense variants. It implements the protocol in
[Marin #11181](https://github.com/marin-community/marin/issues/11181).
The research results and rejected serving configurations remain in the linked
logbook. This harness does not register an official benchmark result.

## Prepare and verify inputs

Download only the development (`train`) files from these pinned datasets:

| Local filename | Dataset | Revision | Remote filename |
| --- | --- | --- | --- |
| `variants-train.parquet` | `marin-dna/evals_mendelian_traits` | `4aed58e50c5dea0b878a665007af2ef9e5108e9f` | `train.parquet` |
| `sequences-train.parquet` | `marin-dna/evals_mendelian_traits_harness_255` | `7b92f047f9a36f90e9ac47886afa2a99264ee35c` | `train.parquet` |

`prepare.py` checks the complete file hashes before selecting ten groups of
one pathogenic variant and nine matched common controls. It selects exactly
two separate chromosome-1 demonstrations; query variants use other autosomes.
The reserved validation groups exclude pilot overlaps. Do not inspect validation
scores until the scoring protocol is frozen, or use the held-out test split
for prompt selection.

```bash
uv run python -m experiments.evals.vep.prepare \
  --inputs INPUT_DIRECTORY --output RUN_DIRECTORY/panels.json \
  --seed 11181 --validation-groups 30
uv run python -m experiments.evals.vep.verify_reference \
  --panels RUN_DIRECTORY/panels.json \
  --output RUN_DIRECTORY/reference.jsonl
```

The first command checks labels, central alleles, 255-base windows and reverse
complements. The second makes bounded batches of public Ensembl GRCh38 requests
and requires every reference window to match. Retain both the raw reference
responses and the resulting `reference.summary.json`.

## Score through an existing service

The likelihood client requires `/tokenize`, `/v1/models`, and
`/v1/completions` with echoed token IDs and native prompt log probabilities.
It verifies contextual token boundaries, exact target decoding, finite scores,
and a consistent serving fingerprint. A/B candidates must each be one token.
It scores the original unrestricted distribution, excluding conditioning text
and EOS. This is a general-vocabulary scoring protocol, not numerical parity
with MarinDNA's four-base softmax.

Supply an authenticated `requests.Session` from your environment. Keep the
service address, model deployment identifier, credentials, and raw wire logs
outside public repositories. Authentication setup belongs to the caller.
The following function binds the harness to that already configured session:

```python
from pathlib import Path

from experiments.evals.vep.paced import PacedClient
from experiments.evals.vep.run import run_panel


def score_pilot(session, base_url: str, model: str, run_directory: Path) -> None:
    with (
        (run_directory / "wire.jsonl").open("x") as wire,
        (run_directory / "sentinels.jsonl").open("x") as sentinels,
        (run_directory / "scores.jsonl").open("x") as scores,
    ):
        client = PacedClient(
            session, base_url, model, wire,
            max_requests=10000, max_seconds=5400, guard_log=sentinels,
        )
        run_panel(
            client, run_directory / "panels.json",
            run_directory / "reference.summary.json", "pilot", 100, scores,
        )
```

Create the run directory with private permissions before opening these files.
Set the request/time budget explicitly for the approved service use. Each run
needs a fresh client and fresh output files. Record the source commit and model
checkpoint alongside the private run record.

The runner pins data-parallel rank 0 and issues one request at a time, leaving
at least one second between completion calls. The service must support the
`X-data-parallel-rank` header. Six pilot sentinels repeat every 48 targets and
at completion. Drift above 1e-6 log-probability units, an HTTP/protocol failure,
a completion over five seconds, or a recent five-call median exceeding both
1.5 seconds and three times the initial eight-call median stops the run.
There are no retries. Partial scores remain on disk without a completion
marker; aggregation rejects them.

Rank pinning does not isolate shared expert-parallel GPUs, and another user's
traffic may change effective batching. The checks establish bounded repeatability,
not absence of impact on other users. Prefix caching is not assumed: identical
reference/alternate left flanks cancel algebraically, but this baseline scores
complete inputs. Mutated suffixes cannot share downstream KV states. A cache
optimization requires separate score-parity and cache-metric validation.

## Recompute scores

```bash
uv run python -m experiments.evals.vep.aggregate \
  --panels RUN_DIRECTORY/panels.json --scores RUN_DIRECTORY/scores.jsonl \
  --output RUN_DIRECTORY/variant-scores.json
```

The aggregate verifies the full target matrix and every prompt hash. DNA scores
are `log P(reference) - log P(alternate)`. Classification scores are pathogenic
minus benign answer log probability. Each is averaged over forward and
reverse-complement strands. Reversed label mappings, mapping averages and
strand differences remain available for sensitivity analysis.

Ten match groups support exploratory analysis only. Use matched-group resampling
for uncertainty and freeze methods before running the reserved 30-group panel.
Preserve per-token wire evidence so alternate aggregates need no new inference.
