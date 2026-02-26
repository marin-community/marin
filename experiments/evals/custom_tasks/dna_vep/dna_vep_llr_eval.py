# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
LLR-based variant effect prediction (VEP) evaluation task for lm-eval-harness.

Computes log-likelihood ratios (LLR) for DNA sequence variants and evaluates
them against ground-truth labels using configurable metrics (AUPRC, AUROC,
Pearson, Spearman). Supports per-subset stratification via an optional
"subset" column in the dataset.

Each variant is scored by computing:
    LLR = log P(alt_completion | context) - log P(ref_completion | context)
    score = llr_transform(LLR)

where context is the shared left flanking sequence, and ref/alt completions
are the reference/alternate allele plus the right flanking sequence.

Metrics are reported as {subset}/{metric} (e.g. "all/auprc", "missense/auprc").
"""

from collections import defaultdict

import datasets
from lm_eval.api.instance import Instance
from lm_eval.api.task import Task
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

METRIC_REGISTRY: dict[str, dict] = {
    "auprc": {
        "fn": lambda items: average_precision_score([t for _, t in items], [s for s, _ in items]),
        "higher_is_better": True,
    },
    "auroc": {
        "fn": lambda items: roc_auc_score([t for _, t in items], [s for s, _ in items]),
        "higher_is_better": True,
    },
    "pearson": {
        "fn": lambda items: float(pearsonr([s for s, _ in items], [t for _, t in items]).statistic),
        "higher_is_better": True,
    },
    "spearman": {
        "fn": lambda items: float(spearmanr([s for s, _ in items], [t for _, t in items]).statistic),
        "higher_is_better": True,
    },
}


class SubsetAwareAggregation:
    """Aggregation callable that computes metrics both overall and per-subset.

    Returns the overall ("all") metric as the scalar for lm-eval.
    Stores all results (including per-subset) in a shared dict using
    the format "{subset}/{metric}" for wandb grouping.
    """

    def __init__(self, base_fn, results_store: dict, metric_name: str):
        self.base_fn = base_fn
        self.results_store = results_store
        self.metric_name = metric_name

    def __call__(self, items: list[tuple[float, float, str | None]]) -> float:
        by_subset: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for score, target, subset in items:
            by_subset["all"].append((score, target))
            if subset is not None:
                by_subset[subset].append((score, target))

        for subset, subset_items in by_subset.items():
            self.results_store[f"{subset}/{self.metric_name}"] = self.base_fn(subset_items)

        return self.results_store[f"all/{self.metric_name}"]


LLR_TRANSFORMS: dict[str, callable] = {
    "identity": lambda x: x,
    "negate": lambda x: -x,
    "abs": abs,
}


class DnaVepLlrEvalTask(Task):
    """General LLR eval task for variant effect prediction.

    Parameterized by dataset and metrics via YAML config.

    Dataset docs must have: context, ref_completion, alt_completion, target.
    Optional: subset (str) — metrics computed per distinct value + "all".

    YAML config fields:
        dataset_path: HuggingFace dataset path
        dataset_name: HuggingFace dataset config name (optional)
        test_split: dataset split to evaluate on
        metrics: list of metric names from METRIC_REGISTRY
        llr_transform: identity | negate | abs (default: identity)
    """

    VERSION = 0
    DATASET_PATH = None
    DATASET_NAME = None

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None, config=None):
        # Task.__init__ calls self.download() BEFORE setting self._config,
        # and wraps the config dict via TaskConfig({**config}) which doesn't
        # correctly populate fields (passes the dict as the first positional
        # arg). We extract everything we need from the raw config dict here.
        cfg = config or {}
        self._task_name = cfg.get("task")
        self.DATASET_PATH = cfg.get("dataset_path") or self.DATASET_PATH
        self.DATASET_NAME = cfg.get("dataset_name") or self.DATASET_NAME
        self._metrics = cfg.get("metrics") or ["auprc"]
        self._test_split = cfg.get("test_split") or "test"

        transform_name = cfg.get("llr_transform") or "identity"
        if transform_name not in LLR_TRANSFORMS:
            raise ValueError(f"Unknown llr_transform: {transform_name}. Must be one of {list(LLR_TRANSFORMS.keys())}")
        self._llr_transform = LLR_TRANSFORMS[transform_name]
        self._subset_results: dict[str, float] = {}

        super().__init__(
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
            config=config,
        )

    @property
    def task_name(self):
        """Required by lm-eval's get_subtask_list (only defined on ConfigurableTask by default)."""
        return self._task_name

    def download(self, data_dir=None, cache_dir=None, download_mode=None) -> None:
        self.dataset = datasets.load_dataset(
            path=self.DATASET_PATH,
            name=self.DATASET_NAME,
            data_dir=data_dir,
            cache_dir=cache_dir,
            download_mode=download_mode,
        )

    def has_training_docs(self) -> bool:
        return False

    def has_validation_docs(self) -> bool:
        return False

    def has_test_docs(self) -> bool:
        return True

    def test_docs(self):
        return self.dataset[self._test_split]

    def doc_to_text(self, doc) -> str:
        return doc["context"]

    def doc_to_target(self, doc):
        raise NotImplementedError(
            "DnaVepLlrEvalTask does not use doc_to_target. "
            "It overrides construct_requests and should be used with num_fewshot=0."
        )

    def construct_requests(self, doc, ctx, **kwargs):
        # Only pass fields that Instance accepts; drop chat template args
        # passed by build_all_requests.
        metadata = kwargs.get("metadata", (None, None, None))
        return [
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(ctx, doc["ref_completion"]),
                idx=0,
                metadata=metadata,
            ),
            Instance(
                request_type="loglikelihood",
                doc=doc,
                arguments=(ctx, doc["alt_completion"]),
                idx=1,
                metadata=metadata,
            ),
        ]

    def process_results(self, doc, results):
        log_prob_ref = results[0][0]
        log_prob_alt = results[1][0]
        llr = log_prob_alt - log_prob_ref
        score = self._llr_transform(llr)
        target = doc["target"]
        subset = doc.get("subset")
        return {metric: (score, target, subset) for metric in self._metrics}

    def aggregation(self):
        return {
            metric: SubsetAwareAggregation(
                base_fn=METRIC_REGISTRY[metric]["fn"],
                results_store=self._subset_results,
                metric_name=metric,
            )
            for metric in self._metrics
        }

    def higher_is_better(self):
        return {metric: METRIC_REGISTRY[metric]["higher_is_better"] for metric in self._metrics}
