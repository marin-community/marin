import enum
import json
import logging
import time
from pathlib import Path
from typing import Any

import dspy
import requests


class _EnumSafeEncoder(json.JSONEncoder):
    """JSON encoder that converts Enum values to their .value string."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, enum.Enum):
            return obj.value
        return super().default(obj)


from experiments.dspy.adapters.baml import BAMLAdapter
from experiments.dspy.adapters.toon import ToonAdapter
from experiments.dspy.programs.hover import HoVer, HoverLabel
from experiments.dspy.programs.hotpotqa import HotpotQA
from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

ADAPTER_MAP: dict[str, type[dspy.Adapter]] = {
    "baml": BAMLAdapter,
    "chat": dspy.ChatAdapter,
    "toon": ToonAdapter,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _hover_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Compare HoVer gold string label with predicted HoverLabel enum.

    HoVer gold label is 'SUPPORTED' or 'NOT_SUPPORTED' (string).
    HoVer program returns ``label_int`` (0 or 1) derived from the HoverLabel enum.
    """
    if pred is None:
        return 0.0
    gold_int = 1 if str(getattr(example, "label", "")).upper() == "SUPPORTED" else 0
    return float(gold_int == getattr(pred, "label_int", -1))


def _hotpotqa_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Exact-match between gold answer string and predicted answer string."""
    if pred is None:
        return 0.0
    gold = str(getattr(example, "answer", "")).lower().strip()
    predicted = str(getattr(pred, "answer", "") or "").lower().strip()
    return float(gold == predicted)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASK_MAP: dict[str, dict] = {
    "hover": {
        "program": HoVer,
        "metric":  _hover_metric,
    },
    "hotpotqa": {
        "program": HotpotQA,
        "metric":  _hotpotqa_metric,
    },
}


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_hover(split: str, max_examples: int | None) -> list[dspy.Example]:
    """Load HoVer examples from *vincentkoc/hover-parquet* on HuggingFace.

    Only **num_hops == 2** examples are kept — our ClaimVerification program
    performs exactly 2 retrieval hops, so 3- and 4-hop examples would be
    evaluated unfairly.

    The dataset has a single 'train' split on HF; we partition it:
        train : first 80 %
        dev   : next  10 %
        test  : last  10 %
    """
    from dspy.datasets.dataloader import DataLoader

    dl = DataLoader()
    full = dl.from_huggingface(
        dataset_name="vincentkoc/hover-parquet",
        split="train",
        trust_remote_code=True,
        fields=("claim", "label", "num_hops", "hpqa_id"),
        input_keys=("claim",),
    )

    hpqa_ids: set = set()
    full = [
        ex for ex in full
        if ex["num_hops"] == 2
        and ex["hpqa_id"] not in hpqa_ids
        and not hpqa_ids.add(ex["hpqa_id"])
    ]

    n = len(full)
    boundaries = {
        "train": (0,             int(0.8 * n)),
        "dev":   (int(0.8 * n), int(0.9 * n)),
        "test":  (int(0.9 * n), n),
    }
    if split not in boundaries:
        raise ValueError(f"Unknown split '{split}'. Choose from {list(boundaries)}")

    lo, hi = boundaries[split]
    examples = full[lo:hi]

    if max_examples is not None:
        examples = examples[:max_examples]

    logger.info(f"Loaded {len(examples)} HoVer examples (split={split}, num_hops=2)")
    return examples


def _load_hotpotqa(split: str, max_examples: int | None) -> list[dspy.Example]:
    """Load HotpotQA examples from HuggingFace.

    Uses the 'fullwiki' configuration.  HotpotQA has a 'train' and a
    'validation' split; we partition 'train' into train/dev/test (80/10/10).
    """
    from dspy.datasets.dataloader import DataLoader

    dl = DataLoader()
    full = dl.from_huggingface(
        dataset_name="hotpotqa/hotpot_qa",
        name="fullwiki",
        split="train",
        input_keys=("question",),
    )

    n = len(full)
    boundaries = {
        "train": (0,             int(0.8 * n)),
        "dev":   (int(0.8 * n), int(0.9 * n)),
        "test":  (int(0.9 * n), n),
    }
    if split not in boundaries:
        raise ValueError(f"Unknown split '{split}'. Choose from {list(boundaries)}")

    lo, hi = boundaries[split]
    examples = full[lo:hi]

    if max_examples is not None:
        examples = examples[:max_examples]

    logger.info(f"Loaded {len(examples)} HotpotQA examples (split={split})")
    return examples


# ---------------------------------------------------------------------------
# BM25S retriever
# ---------------------------------------------------------------------------

_WIKI_URL  = "https://huggingface.co/dspy/cache/resolve/main/wiki.abstracts.2017.tar.gz"
_WIKI_FILE = "wiki.abstracts.2017.jsonl"


def _build_bm25s_retriever():
    """Build an offline BM25S retriever over the 2017 Wikipedia abstracts corpus.

    Downloads wiki.abstracts.2017.tar.gz (~500 MB) on first run and builds
    a BM25S index over all ~5 million Wikipedia article abstracts.

    Returns a callable ``rm(query, k)`` that returns a ``{text: score}`` dict
    of the top-k passages and their BM25S relevance scores.
    """
    import bm25s
    import orjson
    from dspy.utils import download

    if not Path(_WIKI_FILE).exists():
        logger.info("Downloading Wikipedia abstracts corpus (~500 MB)...")
        download(_WIKI_URL)
        import subprocess
        subprocess.run(["tar", "-xzvf", "wiki.abstracts.2017.tar.gz"], check=True)

    logger.info("Loading Wikipedia abstracts corpus...")
    corpus: list[str] = []
    with open(_WIKI_FILE) as f:
        for line in f:
            doc = orjson.loads(line)
            corpus.append(f"{doc['title']} | {' '.join(doc['text'])}")

    logger.info(f"Building BM25S index over {len(corpus):,} Wikipedia passages...")
    import Stemmer
    stemmer = Stemmer.Stemmer("english")
    retriever = bm25s.BM25(k1=0.9, b=0.4)
    retriever.index(bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer))
    logger.info("BM25S index ready.")

    def rm(query: str, k: int = 3, **kwargs) -> dict[str, float]:
        tokens = bm25s.tokenize(query, stopwords="en", stemmer=stemmer, show_progress=False)
        results, scores = retriever.retrieve(tokens, k=min(k, len(corpus)), n_threads=1, show_progress=False)
        return {corpus[idx]: float(sc) for idx, sc in zip(results[0], scores[0])}

    return rm


# ---------------------------------------------------------------------------
# Format-error detection
# ---------------------------------------------------------------------------

def _detect_format_error(pred: dspy.Prediction, task_name: str) -> bool:
    """Return True if the prediction is missing expected output fields."""
    if pred is None:
        return True
    if task_name == "hover":
        return getattr(pred, "label", None) is None
    if task_name == "hotpotqa":
        return not getattr(pred, "answer", None)
    return False


# ---------------------------------------------------------------------------
# DspyEvaluator
# ---------------------------------------------------------------------------

class DspyEvaluator(Evaluator):
    """Evaluates a DSPy program on a structured task across different adapters.

    Measures accuracy and format-error rate, and saves per-example trajectories
    (including per-passage BM25S scores) to a JSONL file for downstream
    analysis, SFT data generation, and Bayesian optimisation.

    Args:
        model: Model configuration (name, path, engine kwargs).
        endpoint: URL of a running OpenAI-compatible inference server.
        output_path: Directory where results and trajectories are saved.
        adapter_name: DSPy adapter to use. One of 'baml', 'chat', 'toon'.
        task_name: Task to evaluate. One of 'hover', 'hotpotqa'.
        split: Dataset split. One of 'train', 'dev', 'test'.
        max_eval_instances: Cap on examples to evaluate (None = full split).
        wandb_tags: Optional W&B tags (reserved for future use).

    Example usage::

        evaluator = DspyEvaluator(
            model=ModelConfig(name="Qwen/Qwen2.5-7B-Instruct", path=None, engine_kwargs={}),
            endpoint="http://localhost:8000",
            output_path="outputs/hover_baml_test",
            adapter_name="baml",
            task_name="hover",
            split="test",
            max_eval_instances=100,
        )
        results = evaluator.evaluate()
        print(results)
    """

    def __init__(
        self,
        model: ModelConfig,
        output_path: str,
        adapter_name: str = "baml",
        task_name: str = "hover",
        split: str = "test",
        max_eval_instances: int | None = 1000,
        endpoint: str | None = None,
        api_key: str | None = None,
        wandb_tags: list[str] | None = None,
        **kwargs,
    ):
        super().__init__()

        if adapter_name not in ADAPTER_MAP:
            raise ValueError(
                f"Unknown adapter '{adapter_name}'. Choose from {list(ADAPTER_MAP)}"
            )
        if task_name not in TASK_MAP:
            raise ValueError(
                f"Unknown task '{task_name}'. Choose from {list(TASK_MAP)}"
            )

        self.model = model
        self.output_path = output_path
        self.adapter_name = adapter_name
        self.task_name = task_name
        self.split = split
        self.max_eval_instances = max_eval_instances
        self.api_key = api_key
        self.wandb_tags = wandb_tags

        self.endpoint = self._resolve_endpoint(endpoint)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_endpoint(self, endpoint: str | None) -> str | None:
        """Return endpoint as-is for external APIs, health-check for local servers."""
        if endpoint is None:
            return None  # use DSPy default (OpenAI)

        # External providers don't expose /health — skip check
        if not endpoint.startswith("http://localhost") and not endpoint.startswith("http://127.0.0.1"):
            return endpoint

        try:
            resp = requests.get(endpoint + "/health", timeout=5)
            if resp.status_code == 200:
                return endpoint
        except requests.exceptions.RequestException:
            pass
        logger.warning("Local endpoint unreachable — falling back to http://localhost:8000")
        return "http://localhost:8000"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, **kwargs) -> dict:
        """Run evaluation and return a summary dict.

        Writes two files to ``output_path``:

        * ``<task>_<adapter>_<split>_trajectories.jsonl``
          One JSON line per example::

              {
                  "sample_id": 0,
                  "input":  {...},
                  "output": {...},
                  "score":  1.0,
                  "parsing_error": false,
                  "evidence_with_scores": [
                      {"text": "...", "score": 15.8},
                      ...
                  ]
              }

        * ``<task>_<adapter>_<split>_results.json``
          Aggregate metrics (accuracy, format_error_rate, elapsed_seconds, …).
        """
        task_cfg = TASK_MAP[self.task_name]

        # Configure DSPy — temperature=0 and cache=False for reproducibility
        lm = dspy.LM(
            model=f"openai/{self.model.name}",
            base_url=self.endpoint,
            api_key=self.api_key or "dspy",
            temperature=0.0,
            cache=False,
        )
        adapter = ADAPTER_MAP[self.adapter_name]()

        # Load dataset
        if self.task_name == "hover":
            examples = _load_hover(self.split, self.max_eval_instances)
        else:
            examples = _load_hotpotqa(self.split, self.max_eval_instances)

        # BM25S retriever — fully offline over 2017 Wikipedia abstracts.
        rm = _build_bm25s_retriever()
        dspy.configure(lm=lm, adapter=adapter)

        program = task_cfg["program"](search=rm)
        metric  = task_cfg["metric"]

        trajectories: list[dict] = []
        total_score         = 0.0
        total_format_errors = 0
        start_time          = time.time()

        for i, example in enumerate(examples):
            traj: dict = {
                "sample_id":   i,
                "input":       example.toDict() if hasattr(example, "toDict") else vars(example),
                "output":      None,
                "score":       None,
                "parsing_error": False,
                "hop_traces":  [],
            }

            try:
                pred      = program(**example.inputs())
                has_error = _detect_format_error(pred, self.task_name)
                score     = float(metric(example, pred))

                traj["output"]        = pred.toDict() if hasattr(pred, "toDict") else str(pred)
                traj["score"]         = score
                traj["parsing_error"] = has_error

                # Save per-hop traces for SFT data generation
                traj["hop_traces"] = getattr(pred, "hop_traces", [])

                total_score += score
                if has_error:
                    total_format_errors += 1

            except Exception as exc:
                traj["parsing_error"] = True
                traj["error"]         = str(exc)
                total_format_errors  += 1
                logger.warning(f"Example {i} failed: {exc}")

            trajectories.append(traj)

            if (i + 1) % 50 == 0:
                logger.info(
                    f"Progress: {i + 1}/{len(examples)} "
                    f"({(i + 1) / len(examples):.1%})"
                )

        elapsed = time.time() - start_time
        n = len(examples)

        results = {
            "task":              self.task_name,
            "adapter":           self.adapter_name,
            "split":             self.split,
            "model":             self.model.name,
            "total_examples":    n,
            "accuracy":          round(total_score / n, 4) if n > 0 else 0.0,
            "format_error_rate": round(total_format_errors / n, 4) if n > 0 else 0.0,
            "elapsed_seconds":   round(elapsed, 2),
        }

        # ---- Persist outputs ----
        out_dir = Path(self.output_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{self.task_name}_{self.adapter_name}_{self.split}"

        traj_file = out_dir / f"{stem}_trajectories.jsonl"
        with open(traj_file, "w", encoding="utf-8") as f:
            for traj in trajectories:
                f.write(json.dumps(traj, cls=_EnumSafeEncoder, ensure_ascii=False) + "\n")

        results_file = out_dir / f"{stem}_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # ---- Print summary ----
        logger.info("=" * 52)
        logger.info(f"  Task          : {results['task']}")
        logger.info(f"  Adapter       : {results['adapter']}")
        logger.info(f"  Split         : {results['split']}")
        logger.info(f"  Model         : {results['model']}")
        logger.info(f"  Examples      : {results['total_examples']}")
        logger.info(f"  Accuracy      : {results['accuracy']:.2%}")
        logger.info(f"  Format errors : {results['format_error_rate']:.2%}")
        logger.info(f"  Elapsed       : {results['elapsed_seconds']}s")
        logger.info("=" * 52)
        logger.info(f"  Trajectories  -> {traj_file}")
        logger.info(f"  Results       -> {results_file}")

        return results

    def launch_evaluate_with_ray(self, model, evals, output_path, **kwargs):
        raise NotImplementedError(
            "Ray-based evaluation is not supported for DspyEvaluator."
        )
