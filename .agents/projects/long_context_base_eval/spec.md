# Contract: paired MRCR likelihood evaluation

## Public dataset surface

File: `experiments/datasets/mrcr.py`

```python
from dataclasses import dataclass
from enum import StrEnum

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.processing.tokenize.tokenize import TokenizedCache


MRCR_CONTEXT_CAPS: tuple[int, ...] = (
    8_192,
    16_384,
    32_768,
    65_536,
    131_072,
    262_144,
)
MRCR_NEEDLE_COUNTS: tuple[int, ...] = (2, 4, 8)
MRCR_DISTANCE_BOUNDS: tuple[int, ...] = (32_768, 65_536, 131_072)
MRCR_BOOTSTRAP_SAMPLES: int = 10_000
MRCR_PREAMBLE_PREFIX = "Here are some examples of conversations succeeded by a follow-up question answered correctly:"


class MrcrCondition(StrEnum):
    FULL_CONTEXT = "full_context"
    QUERY_ONLY = "query_only"


class MrcrPromptVariant(StrEnum):
    TWO_SHOT = "two_shot"
    ONE_SHOT = "one_shot"
    TWO_SHOT_NO_PREFIX = "two_shot_no_prefix"


@dataclass(frozen=True)
class MrcrDatasetBundle:
    datasets: dict[str, ArtifactStep[TokenizedCache]]
    manifests: dict[str, ArtifactStep[Artifact]]
    stats: ArtifactStep[Artifact]


@dataclass(frozen=True)
class MrcrTransformConfig:
    input_path: str
    output_path: str
    tokenizer: str
    context_caps: tuple[int, ...] = MRCR_CONTEXT_CAPS
    distance_bounds: tuple[int, ...] = MRCR_DISTANCE_BOUNDS
    prompt_variants: tuple[MrcrPromptVariant, ...] = (
        MrcrPromptVariant.TWO_SHOT,
        MrcrPromptVariant.ONE_SHOT,
        MrcrPromptVariant.TWO_SHOT_NO_PREFIX,
    )


def transform_mrcr(config: MrcrTransformConfig) -> None:
    """Build paired, tokenizer-binned MRCR records.

    For each requested prompt variant, every accepted source row produces one
    FULL_CONTEXT and one QUERY_ONLY record in the same context bin. TWO_SHOT
    preserves both official worked examples; ONE_SHOT retains the first
    complete worked example and removes the second. TWO_SHOT_NO_PREFIX keeps
    both examples, replaces the target query's exact leading ``Prepend
    {random_string_to_prepend} to `` directive with ``Return ``, and does not
    append the nonce to the assistant prompt. All variants have identical
    scored response bodies. The primary and one-shot variants move the required
    random prefix into the masked prompt. No variant truncates a prompt or target.
    Rows whose complete two-shot full-context prompt and target exceed every
    configured cap are omitted and included in transform statistics. The
    canonical two-shot length assigns every prompt variant for a source row to
    the same cap. The transform uses ``desired_msg_index`` to locate the selected
    user request and its following assistant response, then records the number
    of tokens between the end of that response and the first scored answer-body
    token. Source IDs are lowercase SHA-256 digests of the dataset revision,
    parquet path, row
    index, prompt, answer, and needle count.

    Raises ValueError when caps are not strictly increasing, a row has an
    unsupported needle count, the prompt does not contain exactly two complete
    official worked examples after ``MRCR_PREAMBLE_PREFIX``, the last message is
    not a user query, ``desired_msg_index`` does not identify a user request
    followed by an assistant message whose content matches the answer body, the
    answer does not start with the declared random prefix, removing that prefix
    leaves an empty response body, the no-prefix query cannot be rewritten by
    the exact rule above, or paired target tokenization differs.
    """


def mrcr_datasets(
    *,
    tokenizer: str,
    context_caps: tuple[int, ...] = MRCR_CONTEXT_CAPS,
    needle_counts: tuple[int, ...] = MRCR_NEEDLE_COUNTS,
    distance_bounds: tuple[int, ...] = MRCR_DISTANCE_BOUNDS,
    prompt_variants: tuple[MrcrPromptVariant, ...] = (MrcrPromptVariant.TWO_SHOT,),
) -> MrcrDatasetBundle:
    """Return validation caches and transform statistics for paired MRCR.

    Dataset keys have the form ``{prompt_variant}/cap_{cap}/{needles}needle/``
    ``{distance_band}/{condition}``. Components contain one example per
    sequence, reject overlength records, and expose the hierarchical tags
    defined below. A manifest keyed by the same path without ``condition``
    lists source IDs in cache order. ``stats`` and ``manifests`` are explicit
    dependencies for launchers that validate complete evaluation. The OpenAI
    source revision is pinned to ``f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d``.
    """
```

The transform and cache builder both call the same `_mrcr_format().build_preprocessor(...)`. Binning uses the complete `input_ids` and assistant mask returned by that preprocessor, including BOS and EOS. Cache construction asserts that those IDs and masks are unchanged.

`transform_mrcr` writes `{output_path}/stats.json` with this shape:

```json
{
  "dataset_revision": "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d",
  "tokenizer": "marin-community/marin-tokenizer",
  "accepted": {
    "two_shot/cap_65536/4needle/distance_32769_65536": {
      "examples": 100,
      "scored_tokens": 12345,
      "canonical_full_length_tokens_min": 50000,
      "canonical_full_length_tokens_max": 65000,
      "variant_full_length_tokens_min": 50000,
      "variant_full_length_tokens_max": 65000,
      "evidence_distance_tokens_min": 40000,
      "evidence_distance_tokens_max": 64000
    }
  },
  "excluded_over_262144": {"total": 300, "2needle": 100, "4needle": 100, "8needle": 100},
  "max_query_only_tokens": {
    "two_shot": 4096,
    "one_shot": 3072,
    "two_shot_no_prefix": 4088
  }
}
```

`accepted` contains every nonempty prompt-variant/cap/needle/distance cell. Empty distance cells are omitted; a requested cap with no accepted examples fails dataset construction. `scored_tokens` includes response-body tokens only and is identical across conditions. Counts refer to source records and are computed before padding.

Each `{prompt_variant}/{cap}/{needles}/{distance_band}/manifest.jsonl` contains one ordered record with `source_id`, `canonical_full_length_tokens`, `variant_full_length_tokens`, `evidence_distance_tokens`, and `scored_tokens` for every source row. The full-context and query-only cache builders preserve this order. The evaluator joins per-sequence outputs to source IDs by this ordinal after verifying both cache lengths and assistant-mask sums. A source has the same canonical cap and evidence-distance band in both prompt variants.

## Transformed record

Each gzip JSONL row has this shape:

```json
{
  "messages": [
    {"role": "user", "content": "<rendered preamble/context/query>\nAssistant: <10-char-prefix>"},
    {"role": "assistant", "content": "<answer body without prefix>"}
  ],
  "source_id": "<stable source-row identifier>",
  "prompt_variant": "two_shot",
  "context_cap": 65536,
  "n_needles": 4,
  "canonical_full_length_tokens": 61234,
  "variant_full_length_tokens": 61234,
  "evidence_distance_tokens": 48000,
  "distance_band": "distance_32769_65536",
  "condition": "full_context"
}
```

Only the second transformed message contributes loss. The first transformed message includes the random prefix for `TWO_SHOT` and `ONE_SHOT`; `TWO_SHOT_NO_PREFIX` includes neither the prefix nor its directive. Provenance fields do not affect tokenization. `canonical_full_length_tokens` is always computed from the primary two-shot full-context record; `variant_full_length_tokens` describes the rendered variant.

To compute `evidence_distance_tokens`, render and tokenize the complete two-shot full-context record once. Use offset mappings from that same tokenization to identify the last token overlapping `messages[desired_msg_index + 1].content`; independently tokenized substrings are forbidden. The distance is the count of token positions strictly after that response token and before the first token with loss weight one. The transform validates that `messages[desired_msg_index]` is a user request and the following message is the matching assistant response.

The chat format is:

```python
ChatLmDatasetFormat(
    chat_template=(
        "{{ messages[0]['content'] }}"
        "{% generation %}{{ messages[1]['content'] }}{% endgeneration %}"
        "{{ eos_token }}"
    ),
    pack=False,
)
```

Both conditions use `pack=False`, so one source record occupies one sequence and overlength input raises. No component may silently left- or right-slice a record.

## Dataset keys and tags

For the two-shot variant at cap `65_536`, four needles, evidence distance 32K–65K, and full context, the dataset key is:

`two_shot/cap_65536/4needle/distance_32769_65536/full_context`

Its tags are:

- `mrcr/two_shot/full_context`
- `mrcr/two_shot/cap_65536/full_context`
- `mrcr/two_shot/4needle/full_context`
- `mrcr/two_shot/distance_32769_65536/full_context`
- `mrcr/two_shot/cap_65536/4needle/distance_32769_65536/full_context`

The paired `query_only` cache has the same key/tag structure with the condition replaced. Each pair must contain the same `source_id` set and scored-token count.

## Parameters-only evaluation surface

File: `experiments/grug/moe/evaluate.py` on the pinned CP branch

```python
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from levanter.data.text.datasets import LmDataConfig
from levanter.tracker import TrackerConfig

from experiments.datasets.mrcr import MrcrPromptVariant
from experiments.grug.moe.model import GrugModelConfig


@dataclass(frozen=True)
class GrugCheckpointEvalRuntimeConfig:
    mp: str
    tracker: TrackerConfig
    seed: int = 0
    eval_batch_size: int = 256
    replica_axis_size: int = 1
    data_axis_size: int = 256
    context_axis_size: int = 4
    expert_axis_size: int = 1
    model_axis_size: int = 1


@dataclass(frozen=True)
class GrugCheckpointEvalConfig:
    run_id: str
    checkpoint_path: str
    context_cap: int
    prompt_variant: MrcrPromptVariant
    qk_mult: float
    model: GrugModelConfig
    data: LmDataConfig
    dataset_stats_path: str
    dataset_manifest_paths: dict[str, str]
    runtime: GrugCheckpointEvalRuntimeConfig
    output_path: str
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 0


def evaluate_grug_checkpoint(config: GrugCheckpointEvalConfig) -> dict[str, float]:
    """Evaluate one Grug checkpoint without constructing training state.

    The runner builds the model and configured device mesh, restores only
    ``step`` and ``params`` from the checkpoint, evaluates every tagged
    validation component once with no batch cap, preserves a loss sum and token
    count for every source record, logs raw and derived metrics at the
    checkpoint's stored step, and atomically writes ``eval_metrics.jsonl`` and
    ``mrcr_example_losses.jsonl`` below ``output_path``.

    The runner does not build training datasets, optimizer state, EMA state, or
    a checkpointer. It raises FileNotFoundError for an unresolved checkpoint and
    ValueError when ``context_cap != model.max_seq_len``, the data contains a
    different cap or prompt variant, a paired MRCR cell is incomplete, tokenized
    cache row/mask counts disagree with the stats artifact, or the model differs
    from the canonical 67B config outside ``max_seq_len``, ``qk_mult``, and the
    attention implementation.

    Returns every raw and derived numeric metric written to the tracker.
    """


def dispatch_grug_checkpoint_eval(
    config: GrugCheckpointEvalConfig,
    *,
    resources: ResourceConfig,
    processes_per_task: int = 1,
) -> None:
    """Dispatch ``evaluate_grug_checkpoint`` through the June Grug Fray runner.

    ``resources`` and ``processes_per_task`` affect dispatch only and are not
    part of the on-device evaluation contract.
    """
```

The first implementation targets `origin/june_tpu_67b_a2b@db7ffddd339dd4db71fbb83ae2555abe3522c894`. The default resource is non-preemptible `v4-2048`; the mesh is `(replica=1,data=256,context=4,expert=1,model=1)` and eval batch 256 at every cap. Smaller resource shapes require a separate measured memory/parity result before entering the checked-in matrix.

The implementation may extract model/mesh/evaluator helpers from the branch's `train.py`, but their behavior must remain shared between training evaluation and checkpoint evaluation. It records the actual evaluator commit plus the CP base commit.

## Per-example and metric contract

Raw hierarchical metrics retain the existing `TaggedEvaluator` naming, but a specialized MRCR evaluator owns per-example identity. It visits each dataset component in manifest order, computes one loss-sum and token-count pair per batch row before aggregation, gathers those vectors to process zero, removes zero-weight padding rows, and joins the remaining ordinals to the manifest. Evaluation sets `max_eval_batches=None`; completion without error means every source record in the stats artifact was consumed. For every cell prefix:

`eval/mrcr/{prompt_variant}/cap_{cap}/{needles}needle/{distance_band}`

the runner logs:

- `{prefix}/full_context/loss`
- `{prefix}/full_context/bpb`
- `{prefix}/query_only/loss`
- `{prefix}/query_only/bpb`
- `{prefix}/scored_tokens`
- `{prefix}/examples`
- `{prefix}/micro_context_gain_nll`
- `{prefix}/micro_context_gain_nll_ci95_low`
- `{prefix}/micro_context_gain_nll_ci95_high`
- `{prefix}/micro_context_ppl_ratio`
- `{prefix}/macro_context_gain_nll`
- `{prefix}/macro_context_gain_nll_ci95_low`
- `{prefix}/macro_context_gain_nll_ci95_high`
- `{prefix}/macro_context_ppl_ratio`

Derived values are:

```text
micro_context_gain_nll = query_only/loss - full_context/loss
macro_context_gain_nll = mean_i(query_only_nll_i - full_context_nll_i)
context_ppl_ratio = exp(context_gain_nll)
```

For each source record, `condition_nll_i = condition_loss_sum_i / scored_tokens_i`. The transform establishes identical target IDs, masks, `source_id` sets, and scored-token counts for the pair. Before evaluation, the runner compares each condition's tokenized-cache row count and assistant-mask sum with the explicit stats and manifest artifacts. `examples` means source records, not padded sequences.

Confidence intervals use 10,000 paired bootstrap samples by default and the configured seed. A cell resamples its source IDs with replacement. A cap-level aggregate resamples within each `(n_needles, distance_band)` stratum to preserve the observed composition. Each resample recomputes micro and macro gain; the 2.5th and 97.5th percentiles form the reported interval. PPL-ratio interval endpoints are the exponentials of the corresponding gain endpoints. Each slice uses NumPy `PCG64` seeded by `(bootstrap_seed + int.from_bytes(sha256(canonical_metric_prefix.encode()).digest()[:8], "big")) % 2**64`; language-runtime hash functions are forbidden.

Aggregate metrics over all needle counts use:

`eval/mrcr/{prompt_variant}/cap_{cap}/{condition}/micro_loss`

and corresponding micro/macro gain, ratio, and interval keys under `eval/mrcr/{prompt_variant}/cap_{cap}`. Distance aggregates use `eval/mrcr/{prompt_variant}/cap_{cap}/{distance_band}`. A claim about use beyond a threshold requires a band whose lower evidence-distance bound exceeds that threshold, source count at least the preregistered minimum, and primary micro-gain lower 95% bound above the preregistered gain floor.

File: `{output_path}/mrcr_example_losses.jsonl`

Each source record produces one deterministically sorted paired record:

```json
{
  "source_id": "<sha256>",
  "prompt_variant": "two_shot",
  "context_cap": 65536,
  "n_needles": 4,
  "distance_band": "distance_32769_65536",
  "evidence_distance_tokens": 48000,
  "scored_tokens": 312,
  "full_context_loss_sum": 640.1,
  "query_only_loss_sum": 702.4,
  "full_context_nll": 2.0516,
  "query_only_nll": 2.2513,
  "context_gain_nll": 0.1997
}
```

The file is ordered by `(prompt_variant, n_needles, distance_band, source_id)`. Missing pairs, duplicate IDs, or unequal scored-token counts fail the invocation.

## Persisted evaluation record

File: `{output_path}/eval_metrics.jsonl`

Each invocation writes one JSON object atomically. A retry that produces byte-identical metrics and per-example files succeeds idempotently; conflicting content for the same `(checkpoint_path, context_cap, prompt_variant, qk_mult, model_config_sha256, evaluation_config_sha256)` fails:

```json
{
  "run_id": "mrcr-67b-step157000-qk175-cap262144",
  "checkpoint_path": "gs://.../checkpoints/step-157000",
  "checkpoint_step": 157000,
  "context_cap": 262144,
  "prompt_variant": "two_shot",
  "qk_mult": 1.75,
  "dataset_revision": "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d",
  "model_config_sha256": "<normalized-config-digest>",
  "evaluation_config_sha256": "<dataset-and-bootstrap-config-digest>",
  "cp_base_commit": "db7ffddd339dd4db71fbb83ae2555abe3522c894",
  "evaluator_commit": "<git-commit-containing-the-evaluator>",
  "metrics": {"eval/mrcr/two_shot/cap_262144/micro_context_gain_nll": 0.0}
}
```

`metrics` contains every raw and derived numeric key from the run. JSON keys are sorted for deterministic records. The normalized model fingerprint compares all canonical 67B static fields after removing the approved overrides `max_seq_len`, `qk_mult`, and attention implementation; any other mismatch fails before checkpoint restore.

## Launcher

File: `experiments/grug/moe/eval_mrcr_context.py` on the pinned CP branch

The launcher exposes one lazy `ArtifactStep` per `(checkpoint, context_cap, qk_mult, prompt_variant)` cell and passes the dataset bundle's stats and manifest artifacts as explicit dependencies. The default two-shot matrix is the Cartesian product of:

- `(step-156000-source, qk=1.57)`
- `(step-156000-source, qk=1.75)`
- `(qk157-step156250, qk=1.57)`
- `(qk157-step156500, qk=1.57)`
- `(qk157-step156750, qk=1.57)`
- `(qk157-step157000, qk=1.57)`
- `(qk175-step156250, qk=1.75)`
- `(qk175-step156500, qk=1.75)`
- `(qk175-step156750, qk=1.75)`
- `(qk175-step157000, qk=1.75)`

with caps 8,192; 16,384; 32,768; 65,536; 131,072; and 262,144. This yields 60 jobs when every checkpoint exists. Each checkpoint's cells can launch as soon as its path is available; a later missing checkpoint does not block earlier cells. The 8K/32K engineering smoke submits only the source@1.57 and qk175-step157000 packages before expensive cells.

After the two-shot smoke passes, one-shot and two-shot-no-prefix sensitivity matrices run at 8K and 32K for source@qk1.57, source@qk1.75, qk157-step157000, and qk175-step157000. Each variant adds eight jobs, for 16 sensitivity jobs total. Intermediate checkpoints and caps above 32K do not run under either sensitivity variant.

### Matrix summary surface

File: `experiments/grug/moe/eval_mrcr_context.py` on the pinned CP branch

```python
@dataclass(frozen=True)
class MrcrEvaluationKey:
    package_name: str
    prompt_variant: MrcrPromptVariant


@dataclass(frozen=True)
class MrcrEvaluationArtifact:
    package_name: str
    prompt_variant: MrcrPromptVariant
    context_cap: int
    qk_mult: float
    training_offset: int
    baseline_package_name: str | None
    example_losses_path: str


def summarize_mrcr_matrix(
    evaluations: tuple[MrcrEvaluationArtifact, ...],
    *,
    summary_stage: str,
    expected_evaluations: tuple[MrcrEvaluationKey, ...],
    output_path: str,
    claim_gain_floor: float,
    claim_min_examples: int,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 0,
) -> None:
    """Persist paired comparisons across checkpoint packages.

    Inputs cover one context cap and one or both prompt variants. Available
    evaluation keys must be a subset of ``expected_evaluations`` and have
    identical source IDs within every computed comparison. The summary emits
    every comparison whose required inputs are present plus explicit available,
    missing, and blocked-comparison lists. Duplicate or unexpected packages,
    mismatched IDs, invalid claim thresholds, or conflicting output fail;
    absent expected packages do not.
    """
```

For checkpoint package `P` and prompt variant `v`, let `G(P,v)` be either its micro or macro context gain. The summary computes both versions of these paired estimands:

- Adaptation at qk `q` and offset `t`: `A(q,t,v) = G(extension[q,t],v) - G(source[q],v)`.
- Deployable-package arm difference: `D(t,v) = G(extension[1.75,t],v) - G(extension[1.57,t],v)`.
- Source inference-qk difference: `Q(v) = G(source[1.75],v) - G(source[1.57],v)`.
- Difference-in-differences: `I(t,v) = A(1.75,t,v) - A(1.57,t,v)`.
- Shot sensitivity: `S(P) = G(P,one_shot) - G(P,two_shot)`.
- Prefix sensitivity: `R(P) = G(P,two_shot_no_prefix) - G(P,two_shot)`.

Each difference is right minus left; a positive `I` favors greater qk-1.75 adaptation, positive `S` favors one shot, and positive `R` favors removing the nonce. Bootstrap samples preserve source-ID pairing across every term in a formula.

The function writes `{output_path}/mrcr_matrix_comparisons.jsonl` and `{output_path}/summary.json`. Each comparison row names its kind, packages and baselines, left/right prompt variants, cap, needle/distance slice, micro and macro difference, paired-bootstrap 95% bounds, source count, and claim eligibility. `summary.json` records `summary_stage`, expected/available/missing evaluation keys, blocked comparisons, thresholds, bootstrap configuration, and a `complete` boolean.

The launcher exposes summary stages independently: `smoke`, `source_qk`, one `offset_{250,500,750,1000}` stage, `prompt_sensitivity`, and `complete`. At 8K/32K, `smoke` expects primary two-shot source@qk1.57 and final qk1.75. `source_qk` expects both primary source packages. An offset stage expects the two source packages and two matching extension packages. `prompt_sensitivity` expects all three variants for the two source and two final packages at 8K/32K. `complete` expects all ten primary two-shot packages at one cap. A stage can be regenerated when another checkpoint becomes available without invalidating earlier summaries.

## Tests

- `tests/test_mrcr_dataset.py`: paired prompt/target behavior, deterministic one-shot extraction, exact no-prefix query rewriting, official-preamble validation, prefix masking/removal, `desired_msg_index` request/response semantics, full-render offset-based distance calculation, canonical token binning, manifest order, pair equality, empty-cell failure, and per-needle overlength counts.
- `experiments/grug/moe/test_evaluate.py`: parameters-only restore, static-config validation, prompt-variant validation, complete-cell validation, per-example output, fixed-seed paired bootstrap, idempotent output, and derived-metric behavior using a small model and local checkpoint.
- `experiments/grug/moe/test_eval_mrcr_context.py`: adaptation, direct-arm, source-qk, difference-in-differences, shot/prefix sensitivity formulas, paired resampling, partial-stage completeness, and conflicting-summary behavior.
- TPU smoke: two checkpoints at 8K and 32K. This is an experiment artifact, not a pytest test.

## Out of scope

- Standard MRCR free-generation scoring.
- RULER generation or likelihood adapters.
- Post-training or chat-template selection.
- Porting or landing the June context-parallel branch on main.
- A numerical threshold for choosing qk or declaring the 262K extension successful.
