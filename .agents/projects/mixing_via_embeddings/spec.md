# Spec — mixing_via_embeddings

Contracts for the experiment code and persisted artifacts. Scope: H1/H2a/H2b/H4 (retrodiction on
existing swarm runs) plus the featurization job they share. H3 (live validation) reuses swarm launch
scaffolding and adds no new contracts beyond the policy-proposal note at the end.

All new code lives under `experiments/datakit/mixture_features/` as self-contained scripts + a small
importable module. Nothing imports from the swarm branch; swarm run histories (W&B) and mixture
configs (CSVs from `calvin/swarm-olmo3-regmix-test`) are consumed as data.

## Basis identity

```python
@dataclass(frozen=True)
class MixtureBasis:
    """Identity of the frozen space a histogram is expressed in.

    Two histograms are comparable iff their MixtureBasis is equal. Identity is by content
    hash, not path: a re-uploaded centroid file with different bytes is a different basis.
    Every persisted artifact embeds this identity; every function that combines histograms
    raises BasisMismatchError on inequality rather than silently mixing spaces.
    """
    embedder: str                  # e.g. "luxical-one-v0" (192-dim, int8 quantized)
    tokenizer: str                 # tokenizer defining the token measure (must match the swarm runs')
    centroids_path: str            # GCS path to the K=5000 spherical k-means centroids
    centroids_sha256: str
    k: int                         # fine cluster count (5000)
    view_paths: Mapping[int, str]  # coarse-view lookup tables, e.g. {1000: ..., 40: ...}
    view_sha256: Mapping[int, str]
    quality_scorer: str | None     # e.g. "datakit-quality-v0-fasttext"; None = no quality axis
    quality_scorer_sha256: str | None
    rff_dim: int                   # cluster-free RFF map identity — part of basis equality so
    rff_seed: int                  # histograms built with different RFF maps can never be
    rff_bandwidth: float           # concatenated (median-heuristic value frozen at basis creation)
```

v0 binds to the existing datakit artifacts: Luxical-One embedder, `train_centroids_22d1e89d`
centroids, K=5000 with 1000/40 agglomerative views, the qsplit240 run tokenizer, RFF map (dim 2048,
seed 0, bandwidth = median heuristic on the codebook training sample, frozen at basis creation), and
(behind the audit gate, see H4) the datakit v0 fasttext quality scorer with the store cutoffs
`[0.2,0.4,0.6,0.8]`.
Sampling metadata (sample size, seed) identifies an *estimate*, and lives on `DomainHistogram`, not on
the basis.

## Domain histograms (`build_domain_histograms.py`)

Zephyr map-only job, one shard per swarm domain. Histograms are **token-weighted**: each sampled
document contributes mass equal to its count of training-eligible tokens under `basis.tokenizer`
(post truncation and loss-masking, using the same serialization the swarm loader applied — for SFT
domains, the rendered prompt+response string with only loss-bearing tokens counted).

```python
def build_domain_histogram(
    domain: str,
    documents: Iterable[Document],   # (text, serialization already applied by the per-domain loader shim)
    basis: MixtureBasis,
    sample_size: int = 100_000,
    seed: int = 0,
) -> DomainHistogram:
    """Embed a uniform document sample, assign to basis centroids (and quality buckets when
    basis.quality_scorer is set), and accumulate token-weighted cell mass.

    Samples exactly `sample_size` documents (seeded reservoir); raises InsufficientSampleError
    below `min_sample_size` (10_000). `frac` sums to 1 over occupied cells. Also accumulates
    the bucket-stats summary (below).
    """

@dataclass(frozen=True)
class DomainHistogram:
    domain: str
    basis: MixtureBasis
    sample_size: int          # documents sampled
    token_count: int          # eligible tokens over the sample (the histogram's total mass)
    seed: int
    counts: Mapping[tuple[int, int], int]   # (cluster_id, quality_bucket) -> eligible tokens; quality_bucket = -1 when no quality axis
    rff_mean: tuple[float, ...]             # cluster-free summary (dim 2048), accumulated in the same pass
    stats: BucketStats

@dataclass(frozen=True)
class BucketStats:
    """Per-bucket scalars that embeddings cannot see but that remain computable for any new
    bucket; used as features alongside histogram mass."""
    total_tokens_available: int       # corpus size, from source metadata (not the sample)
    mean_doc_tokens: float
    duplicate_frac: float | None      # within-sample near-dup rate (minhash); None if not computed
    loss_masked_frac: float           # fraction of serialized tokens excluded from loss (0 for web)
```

**Persisted shape** — parquet at
`gs://marin-eu-west4/user/rav/projects/mixing_via_embeddings/v0/domain_histograms/part-<domain>.parquet` with columns
`domain` (string), `cluster_id` (int32), `quality_bucket` (int8, -1 when absent), `token_count`
(int64), `frac` (float64). Sidecar `_meta.json` per prefix: the full `MixtureBasis` (with hashes),
per-domain `sample_size`/`seed`/`token_count`/`BucketStats`, `created_at`, git SHA of the producing
script. Coarse views are derived at load time via the view tables (`composition_matrix` below), never
persisted, so K=40/1000 histograms cannot drift from the K=5000 source.

## Swarm run loading (`swarm_runs.py`)

```python
@dataclass(frozen=True)
class SwarmRun:
    """One proxy run re-read from W&B."""
    run_id: str
    model_size: int                                   # parameters, e.g. 60_000_000 / 300_000_000
    phase_weights: tuple[Mapping[str, float], ...]    # per-phase mixture, keys = domain names
    phase_tokens: tuple[int, ...]                     # tokens trained in each phase
    domain_tokens: tuple[Mapping[str, float], ...]    # per-phase per-domain consumed tokens (weights x phase_tokens)
    is_vertex: bool                                   # sampled by the vertex-biased strategy
    vertex_domain: str | None                         # the dominant domain if is_vertex (weight >= 0.7 per weight_sampler.py)
    metrics: Mapping[str, float]                      # e.g. {"eval/uncheatable_eval/bpb": ...}

def load_qsplit240(expected_missing: Collection[str] = ()) -> list[SwarmRun]:
    """Load all qsplit240 histories. Runs with missing/incomplete required fields fail loudly:
    the loader raises listing the offending run_ids unless they appear in `expected_missing`."""
```

Exposure features are **derived**, not stored, and are split into two families (see below): per-domain
epochs/exposure are computable from `domain_tokens` + `BucketStats.total_tokens_available`.

## Featurization (`featurize.py`)

Pure functions; no I/O.

```python
class FeatureFamily(StrEnum):
    HIST_K40 = "hist_k40"                  # primary granularity
    HIST_K1000 = "hist_k1000"              # ablation, smoothness-regularized models only
    HIST_K5000 = "hist_k5000"              # ablation/stress only
    KME_MEAN = "kme_mean"                  # mass-weighted mean of dequantized centroid vectors (192 dims)
    RFF_MEAN = "rff_mean"                  # cluster-free arm: token-weighted mean of random Fourier
                                           # features of raw document embeddings. Map identity
                                           # (rff_dim/rff_seed/rff_bandwidth) lives on MixtureBasis, so
                                           # mismatched maps raise BasisMismatchError. Bypasses the
                                           # k-means codebook.
    QUALITY_MASS = "quality_mass"          # mass per quality bucket; locked until the scorer audit passes
    BUCKET_STATS = "bucket_stats"          # mixture-weighted BucketStats scalars
    EXPOSURE_GLOBAL = "exposure_global"    # transferable: model_size, phase_tokens, per-cell exposure
    EXPOSURE_BUCKET = "exposure_bucket"    # bucket-indexed per-domain epochs; DIAGNOSTIC-CEILING ONLY,
                                           # never valid in a transfer model (leaks bucket identity)

def composition_matrix(histograms: Sequence[DomainHistogram], k: int) -> tuple[np.ndarray, list[str]]:
    """Stack per-domain fracs into V of shape (cells, n_domains) at granularity k (coarsened from
    K=5000 via the basis views when k < 5000). Raises BasisMismatchError on basis disagreement.
    Column order = sorted(domain). Also returns rank/condition diagnostics via .diagnostics on the
    returned array wrapper (numerical rank, singular spectrum) — H1 consumes these."""

def mixture_histogram(weights: Mapping[str, float], v: np.ndarray, domain_order: list[str]) -> np.ndarray:
    """h = V @ w. Missing domains get weight 0; raises ValueError on unknown domain or if weights
    do not sum to 1 (atol 1e-6)."""

class PhaseHandling(StrEnum):
    PER_PHASE = "per_phase"   # per-phase features, concatenated in phase order (default)
    POOLED = "pooled"         # phase-token-weighted average mixture (from run.phase_tokens)

def run_features(
    run: SwarmRun,
    histograms: Sequence[DomainHistogram],
    families: Sequence[FeatureFamily],
    phases: PhaseHandling = PhaseHandling.PER_PHASE,
) -> np.ndarray:
    """Concatenate the requested families under the given phase handling.
    Deterministic ordering: families as given, cells in index order, phases in order."""
```

**Control featurizations** (mandatory wherever a semantic result is claimed):

```python
def shuffled_columns_v(v: np.ndarray, seed: int) -> np.ndarray
    """Permute the domain -> histogram-column mapping. Destroys semantic alignment, preserves
    marginal geometry."""

def matched_random_v(v: np.ndarray, seed: int) -> np.ndarray
    """Independent random permutation of cell indices within each column. Every column keeps its
    exact mass profile (non-negative, sums to 1, same entropy/sparsity — valid input for the
    Hellinger/JS predictors), but the shared cell coordinate system, and hence all cross-column
    content similarity, is destroyed. (An earlier draft matched rank/singular spectrum with a
    generic random matrix; that control leaves the simplex and would feed invalid histograms to
    distance-based predictors.)"""
```

## H2a — domain response from content (`domain_response.py`)

```python
@dataclass(frozen=True)
class DomainResponse:
    """Per-domain, per-phase marginal-value/saturation parameters extracted from the sweep with the
    incumbent DSP/GRP structure, with bootstrap uncertainty."""
    domain: str
    phase: int
    params: Mapping[str, float]
    params_se: Mapping[str, float]

def fit_domain_responses(runs: Sequence[SwarmRun], target_metric: str, model_size: int, seed: int = 0) -> list[DomainResponse]

def lodo_response_prediction(
    responses: Sequence[DomainResponse],
    histograms: Sequence[DomainHistogram],
    holdout_domain: str,
    featurization: Literal["semantic", "shuffled", "matched_random", "cluster_free"],
    seed: int = 0,
) -> Mapping[str, float]:
    """Predict holdout_domain's response params from its content features using the other 38 domains.
    "cluster_free" uses RFF_MEAN features (codebook-bypassing); the other three use clustered
    histograms. The H2a headline: predicted-vs-fit correlation (uncertainty-weighted) for semantic vs
    control featurizations across all 39 folds, reported for both arms. Gate: if neither the semantic
    clustered arm nor the cluster-free arm beats both controls with CI separation, H2b does not run.
    Kill rule (pre-committed, see design.md): a negative gate under one MixtureBasis is inconclusive;
    abandoning the premise requires the gate to fail under a second basis (different embedder) as
    well."""
```

## H2b — held-out-dose retrodiction (`retrodiction.py`)

The estimand is **held-out-dose extrapolation** (train where domain k's dose ≈ 0, test where it
dominates), *not* zero-shot new-domain prediction. Featurization is always the full, physically
correct `V·w` — no column dropping.

```python
class PhaseReducer(StrEnum):
    MAX = "max"                        # dose(run, k) = max over phases of w_k (default)
    TOKEN_WEIGHTED = "token_weighted"  # phase-token-weighted mean of w_k

@dataclass(frozen=True)
class DoseSplit:
    holdout_domain: str
    model_size: int                # splits are per-scale; pooling scales is a contract violation
    phase_reducer: PhaseReducer = PhaseReducer.MAX
    train_max_dose: float = 0.02   # absolute dose threshold for train membership (not a quantile)
    test_min_dose: float = 0.30    # test = runs above this, plus vertex runs with vertex_domain == k
    # The band between the thresholds is intentionally discarded; sensitivity over
    # train_max_dose in {0.01, 0.02, 0.05} is part of the standard report.

class PredictorKind(StrEnum):
    HIST_RIDGE_K40 = "hist_ridge_k40"      # ridge on K=40 cells (+EXPOSURE_GLOBAL), primary
    KERNEL_HELLINGER = "kernel_hellinger"  # dual kernel ridge on per-phase Hellinger distances, primary
    KME_RIDGE = "kme_ridge"                # ridge on 192-dim mean embeddings (the theory-backed object)
    RFF_RIDGE = "rff_ridge"                # ridge on RFF mean maps (≈ MMD-kernel regression), cluster-free primary
    WEIGHTS_LGBM = "weights_lgbm"          # LightGBM on raw weights (+EXPOSURE_GLOBAL): the incumbent
    HIST_LGBM_K5000 = "hist_lgbm_k5000"    # stress ablation only
    MEAN_BASELINE = "mean_baseline"        # predicts the train-set mean metric (null floor)
    NN_HIST = "nn_hist"                    # 1-NN in Hellinger distance (cheap non-parametric floor)

@dataclass(frozen=True)
class RetrodictionResult:
    split: DoseSplit
    n_train: int
    n_test: int
    spearman: Mapping[str, float]          # key = f"{predictor}:{featurization}" over {semantic, shuffled, matched_random}
    spearman_ci: Mapping[str, tuple[float, float]]   # paired bootstrap 95% CI vs HIST_RIDGE_K40:semantic
    content_novelty: float                 # distance of V_k to the convex cone of V_{-k} (Hellinger geometry)
    design_support: tuple[float, ...]      # per-test-run distance to the convex hull of train feature vectors
    derivability_r2: float                 # mean R^2 predicting train h from train w (linear) — expected ~1, reported for honesty
    predictions_path: str                  # parquet of per-run predictions for every predictor
```

Contracts: all predictors fit on identical train rows and scored on identical test rows;
hyperparameters chosen by nested CV *inside train only*; `target_metric` direction is bpb
(lower-is-better) and `spearman` is computed on (-metric) so higher = better everywhere;
`run_retrodiction(...)` raises ValueError if `n_test < 20` or `n_train < 60`.

```python
def enumerate_splits(runs: Sequence[SwarmRun], candidate_domains: Sequence[str]) -> pd.DataFrame:
    """Feasibility table (n_train/n_test per domain x scale x reducer x thresholds). MUST be
    generated and committed to the results prefix before any model fit; the pre-registered
    eligible-domain list is derived from it without reference to outcomes."""
```

Driver `run_retrodiction_suite.py` executes the pre-registered grid and writes one row per
`RetrodictionResult` to `gs://marin-eu-west4/user/rav/projects/mixing_via_embeddings/v0/retrodiction/results.parquet`
(flattened + `families`, `k`, `target_metric`, git SHA), plus the W&B report.

## Errors

- `BasisMismatchError(ValueError)` — combining histograms/features under unequal `MixtureBasis`
  (including tokenizer).
- `InsufficientSampleError(ValueError)` — domain yielded < 10k sampled documents.
- `ValueError` — unknown domain in weights; weights not a distribution; infeasible split
  (`n_test < 20` / `n_train < 60`); pooled `model_size` in one split.
- W&B loader fails loudly listing dropped run ids; `expected_missing` is the only silence mechanism.

## File map

| path | contents |
|---|---|
| `experiments/datakit/mixture_features/build_domain_histograms.py` | zephyr job: sample → embed → assign → token-weighted V columns + BucketStats |
| `experiments/datakit/mixture_features/featurize.py` | `MixtureBasis`, `DomainHistogram`, `FeatureFamily`, `composition_matrix`, `mixture_histogram`, `run_features`, control featurizations |
| `experiments/datakit/mixture_features/swarm_runs.py` | `SwarmRun`, `load_qsplit240` |
| `experiments/datakit/mixture_features/domain_response.py` | H2a: `DomainResponse`, `fit_domain_responses`, `lodo_response_prediction` |
| `experiments/datakit/mixture_features/retrodiction.py` | H2b: `DoseSplit`, `PredictorKind`, `RetrodictionResult`, `enumerate_splits`, predictor fits |
| `experiments/datakit/mixture_features/run_retrodiction_suite.py` | driver: pre-registered grid, results parquet, W&B report |
| `tests/datakit/mixture_features/` | featurization algebra, synthetic recovery (incl. must-fail cases), control tests |
| `gs://marin-eu-west4/user/rav/projects/mixing_via_embeddings/v0/` | `domain_histograms/`, `_meta.json`, `retrodiction/results.parquet`, `retrodiction/feasibility.parquet` |

## Out of scope

- Any change to production mixtures, the datakit store layout, or `domains.py`.
- Scale-transfer modeling (results are per-scale; 60M→300M anchoring stays as-is).
- Training or selecting a new embedder (hill-climb thread owns that; a basis bump re-runs
  `build_domain_histograms.py` under a new `MixtureBasis`).
- Online/in-run mixing methods (Aioli/R&B-style) and pointwise selection (MATES thread).
- Merging or modifying PR #2393.
- H3 launch configs. Note the category distinction H3 must respect: Olmix-style reuse and
  token-proportional are mixture *proposers*, not metric predictors — they are compared by realized
  metric / policy regret of proposed mixtures (specified after H2 results), never by `spearman` on
  arbitrary test runs.
