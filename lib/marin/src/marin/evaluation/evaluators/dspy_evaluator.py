import enum
import json
import logging
import time
from pathlib import Path
from typing import Any

import dspy
import requests
from openai import AsyncOpenAI


class _EnumSafeEncoder(json.JSONEncoder):
    """JSON encoder that converts Enum values to their .value string."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, enum.Enum):
            return obj.value
        return super().default(obj)

from experiments.dspy.adapters.baml import BAMLAdapter
from experiments.dspy.adapters.toon import ToonAdapter
from experiments.dspy.metrics import claim_verification_metric
from experiments.dspy.programs.claim_verification import ClaimVerification
from marin.evaluation.evaluators.evaluator import Evaluator, ModelConfig

logger = logging.getLogger(__name__)

# --- Supported adapters ---
ADAPTER_MAP: dict[str, type[dspy.Adapter]] = {
    "baml": BAMLAdapter,
    "chat": dspy.ChatAdapter,
    "toon": ToonAdapter,
}

# --- Supported tasks ---
# Each entry has:
#   "program"   : DSPy module class
#   "metric"    : callable(example, prediction) -> float
#   "input_keys": fields used as program inputs
TASK_MAP = {
    "hover": {
        "program": ClaimVerification,
        "metric": claim_verification_metric,
        "input_keys": ("claim", "evidence"),
    },
}


def _load_hover(split: str, max_examples: int | None) -> list[dspy.Example]:
    """Load HoVer examples from HuggingFace for the requested split.

    HoVer only has a 'train' split on HF, so we partition it deterministically:
        train : first 80%
        dev   : next 10%
        test  : last 10%
    """
    from dspy.datasets.dataloader import DataLoader

    dl = DataLoader()
    full = dl.from_huggingface(
        "Dzeniks/hover",
        split="train",
        input_keys=("claim", "evidence"),
    )

    n = len(full)
    boundaries = {
        "train": (0, int(0.8 * n)),
        "dev":   (int(0.8 * n), int(0.9 * n)),
        "test":  (int(0.9 * n), n),
    }
    if split not in boundaries:
        raise ValueError(f"Unknown split '{split}'. Choose from {list(boundaries)}")

    lo, hi = boundaries[split]
    examples = full[lo:hi]

    if max_examples is not None:
        examples = examples[:max_examples]

    logger.info(f"Loaded {len(examples)} HoVer examples (split={split})")
    return examples


def _build_bm25s_retriever(examples: list[dspy.Example]):
    """Build an offline BM25S retriever from HoVer evidence passages.

    Collects all unique passages from the dataset examples, indexes them
    with BM25S (no server needed), and returns a callable that DSPy can
    use as dspy.settings.rm.
    """
    import bm25s

    # Collect all unique passages from the dataset
    corpus = []
    seen = set()
    for ex in examples:
        evidence = getattr(ex, "evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        for passage in evidence:
            if passage not in seen:
                corpus.append(passage)
                seen.add(passage)

    if not corpus:
        raise ValueError("No evidence passages found in dataset to build BM25S index.")

    logger.info(f"Building BM25S index over {len(corpus)} passages...")
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus))
    logger.info("BM25S index ready.")

    def rm(query: str, k: int = 3, **kwargs) -> list[str]:
        tokens = bm25s.tokenize([query])
        results, _ = retriever.retrieve(tokens, k=min(k, len(corpus)))
        return [corpus[i] for i in results[0]]

    return rm


def _detect_format_error(pred: dspy.Prediction, task_name: str) -> bool:
    """Return True if the prediction is missing expected output fields."""
    if pred is None:
        return True
    if task_name == "hover":
        return getattr(pred, "label", None) is None
    return False


class DspyEvaluator(Evaluator):
    """Evaluates a DSPy program on a structured task across different adapters.

    Measures accuracy and format-error rate, and saves per-example trajectories
    to a JSONL file for downstream analysis and SFT data generation.

    Args:
        model: Model configuration (name, path, engine kwargs).
        endpoint: URL of a running OpenAI-compatible inference server.
        output_path: Directory where results and trajectories are saved.
        adapter_name: DSPy adapter to use. One of 'baml', 'chat', 'toon'.
        task_name: Task to evaluate. Currently only 'hover' is supported.
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
        endpoint: str,
        output_path: str,
        adapter_name: str = "baml",
        task_name: str = "hover",
        split: str = "test",
        max_eval_instances: int | None = 1000,
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
        self.wandb_tags = wandb_tags

        self.endpoint = self._validate_endpoint(endpoint)
        self.client = AsyncOpenAI(base_url=self.endpoint, api_key="dspy")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_endpoint(self, endpoint: str) -> str:
        """Return endpoint if healthy, otherwise fall back to localhost."""
        try:
            resp = requests.get(endpoint + "/health", timeout=5)
            if resp.status_code == 200:
                return endpoint
        except requests.exceptions.RequestException:
            pass
        logger.warning("Endpoint unreachable — falling back to local server.")
        return self._run_oai_server()

    def _run_oai_server(self) -> str:
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
                  "parsing_error": false
              }

        * ``<task>_<adapter>_<split>_results.json``
          Aggregate metrics (accuracy, format_error_rate, elapsed_seconds, …).
        """
        task_cfg = TASK_MAP[self.task_name]

        # Configure DSPy — temperature=0 and cache=False for reproducibility
        lm = dspy.LM(
            model=f"openai/{self.model.name}",
            base_url=self.endpoint,
            api_key="dspy",
            temperature=0.0,
            cache=False,
        )
        adapter = ADAPTER_MAP[self.adapter_name]()

        # BM25S runs fully offline — no server needed.
        # ColBERT was avoided because it requires a running server and caused
        # pipeline crashes in earlier experiments.
        rm = _build_bm25s_retriever(examples)
        dspy.configure(lm=lm, adapter=adapter, rm=rm)

        examples = _load_hover(self.split, self.max_eval_instances)
        program = task_cfg["program"]()
        metric  = task_cfg["metric"]

        trajectories: list[dict] = []
        total_score        = 0.0
        total_format_errors = 0
        start_time = time.time()

        for i, example in enumerate(examples):
            traj: dict = {
                "sample_id":    i,
                "input":        example.toDict() if hasattr(example, "toDict") else vars(example),
                "output":       None,
                "score":        None,
                "parsing_error": False,
            }

            try:
                pred      = program(**example.inputs())
                has_error = _detect_format_error(pred, self.task_name)
                score     = float(metric(example, pred))

                traj["output"]        = pred.toDict() if hasattr(pred, "toDict") else str(pred)
                traj["score"]         = score
                traj["parsing_error"] = has_error

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
                logger.info(f"Progress: {i + 1}/{len(examples)} ({(i+1)/len(examples):.1%})")

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

        traj_file = out_dir / f"{self.task_name}_{self.adapter_name}_{self.split}_trajectories.jsonl"
        with open(traj_file, "w", encoding="utf-8") as f:
            for traj in trajectories:
                f.write(json.dumps(traj, cls=_EnumSafeEncoder, ensure_ascii=False) + "\n")

        results_file = out_dir / f"{self.task_name}_{self.adapter_name}_{self.split}_results.json"
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
