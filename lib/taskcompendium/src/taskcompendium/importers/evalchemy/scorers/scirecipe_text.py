# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Thoth's text metrics with offline models and explicit infrastructure failures."""

import re
from functools import cache

from nltk.translate.meteor_score import meteor_score
from rouge_score.rouge_scorer import RougeScorer
from sacrebleu.metrics import BLEU
from verifyit.grade import GradingInfraError

from .scirecipe import MINILM_PATH, ROBERTA_PATH, TEXT_METRICS

KEYWORD_LIMIT = 64
STEP_PREFIX = re.compile(r"(?im)^step\s*\d+\s*:\s*")


@cache
def keyword_model():
    from keybert import KeyBERT  # noqa: PLC0415
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    return KeyBERT(model=SentenceTransformer(MINILM_PATH, device="cpu", local_files_only=True))


@cache
def bert_scorer():
    from bert_score import BERTScorer  # noqa: PLC0415

    return BERTScorer(model_type=ROBERTA_PATH, num_layers=24, device="cpu", lang="en", rescale_with_baseline=False)


def lexical_metrics(reference: str, prediction: str) -> dict[str, float]:
    """Reference BLEU precisions, METEOR and ROUGE, without embedding models."""
    bleu = BLEU(effective_order=True).sentence_score(prediction, [reference])
    metrics = {f"bleu{i + 1}": value / 100 for i, value in enumerate(bleu.precisions)}
    metrics["bleu_avg"] = sum(metrics.values()) / 4
    metrics["meteor"] = meteor_score([reference.split()], prediction.split())
    rouge = RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True).score(reference, prediction)
    metrics.update({name: score.fmeasure for name, score in rouge.items()})
    return metrics


def text_metrics(reference: str, prediction: str) -> dict[str, float]:
    reference, prediction = reference.strip(), prediction.strip()
    if not reference or not prediction:
        return dict.fromkeys(TEXT_METRICS, 0.0)
    try:
        metrics = lexical_metrics(reference, prediction)
        ref_keywords = {word for word, _ in keyword_model().extract_keywords(reference, top_n=KEYWORD_LIMIT)}
        pred_keywords = {word for word, _ in keyword_model().extract_keywords(prediction, top_n=KEYWORD_LIMIT)}
        intersection = len(ref_keywords & pred_keywords)
        precision = intersection / len(pred_keywords) if pred_keywords else 0.0
        recall = intersection / len(ref_keywords) if ref_keywords else 0.0
        metrics.update(
            kw_precision=precision,
            kw_recall=recall,
            kw_f1=2 * precision * recall / (precision + recall + 1e-8) if precision + recall else 0.0,
        )
        ref_clean = STEP_PREFIX.sub("", reference)
        pred_clean = STEP_PREFIX.sub("", prediction)
        precision_tensor, recall_tensor, f1_tensor = bert_scorer().score([pred_clean], [ref_clean])
        metrics.update(
            bertscore_p=float(precision_tensor.mean()),
            bertscore_r=float(recall_tensor.mean()),
            bertscore_f1=float(f1_tensor.mean()),
        )
        return metrics
    except (ImportError, OSError, LookupError, RuntimeError) as error:
        # Unlike the reference, missing model/data dependencies must not become scored zeros.
        raise GradingInfraError(f"SciRecipe text metric failed: {error}") from error
