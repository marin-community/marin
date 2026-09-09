# Mixture GP

Fit completed two-phase mixtures and propose the next ones. The core has one
model and one search path:

- `hf.py` reads named Hugging Face Parquet records into a flat `Data` object.
- `objective.py` converts BPB metrics into a score and observation-noise variance.
- `model.py` fits a ridge exposure mean and a Matérn-5/2 GP over square-root weights.
- `search.py` samples proportional-centered Dirichlet mixtures, ranks by posterior
  mean, perturbs the best starts, and returns the best feasible unseen mixtures.
- `train.py` saves the fitted GP as a Marin artifact tied to its training data.
- `generate.py` loads that artifact and writes named phase weights.

The model fits one swarm. Its mean shares domain effects across phases, learns
an additive coefficient for each phase, and includes fixed exposure and phase
alignment penalties. Outcomes use median/MAD scaling and are clipped at eight
scaled units. The model and search meet at `predict(weights) -> (mean, variance)`;
changing the predictor does not require a search implementation change.

The objective normalizes metrics against proportional-reference runs and
replicate noise. Higher scores are better. It averages target and guardrail contributions separately.
Targets receive a linear improvement reward plus a regression hinge; guardrails
receive the hinge alone. Standardized metric values are clipped to ±10.
Edit the named task lists in `objective.py` to change the objective.

## Run

From the repository root:

```bash
uv run python -m experiments.datakit.mixprior.train \
  --revision 9cdd208e42f926f2b23c204085c9dc45f63c6c92 \
  --swarm h100-d512-from10pct-store-4d2e363d \
  --output mixture-gp

uv run python -m experiments.datakit.mixprior.generate \
  --model mixture-gp \
  --output candidates.parquet
```

Input data comes from [marin-community/grug-moe-mix-swarm](https://huggingface.co/datasets/marin-community/grug-moe-mix-swarm).
A component is a named data cell in that swarm. Use a full dataset commit for `--revision`. Add `--data /path/to/snapshot`
to read an existing snapshot whose root contains `registry/v1/swarms/`.
The loader reads `swarm.parquet`, `buckets.parquet`, and `observations.parquet`
for the named swarm. Phase weights are aligned by cell name. Language-group
means are recomputed from their constituent metrics.

Training writes `model.npz` and Marin's `.artifact.json` to `--output`.
The `GPArtifact` record identifies the HF dataset revision, swarm, observation
IDs, component ordering, and objective metrics and hinge tolerance. The NPZ
stores the fitted GP's prediction state, including its feature parameters,
outcome scaling, Cholesky factor, and observed weights. Reusing the training
output directory replaces these files.

Candidate generation loads the saved GP without downloading the training data
or refitting. Move the model directory as a unit; loading uses the directory
you pass. Programmatic callers can use `load_model(Path("mixture-gp"))`
from `train.py` to obtain the artifact metadata, GP, and observed weights.
The shared manual artifact API handles the record; it does not add step
fingerprints or launch provenance automatically.

Defaults: 65,536 initial draws, one returned candidate, seed 111, and at most
16 cumulative epochs per component: sum each phase’s token budget times its
component weight, then divide by the component’s available tokens. Change these with `--pool-size`,
`--batch-size`, `--seed`, and `--max-epochs`. Selected phases sum to one on the
1/49,152 training lattice. Observed mixtures and duplicate proposals are excluded.
A batch ranks candidates independently by posterior mean; it has no joint
exploration or diversity objective.

The output is one Parquet row per candidate with `swarm_id`, `hf_revision`,
`phase0_weights`, and `phase1_weights`. Each phase is a map from cell names to
weights. Reusing `--output` replaces that file.

Offline replays and the model-choice tradeoff live in
[mixprior_benchmarks](../mixprior_benchmarks/README.md#model-choice).
Run the core tests with `uv run pytest tests/datakit/mixprior`.
