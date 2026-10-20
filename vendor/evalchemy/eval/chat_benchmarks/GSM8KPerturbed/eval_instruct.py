from eval.sources.common import text_specs
from eval.contracts.eval_spec import ReducerSpec, SourceRef
"""GSM8KPerturbed: chat-mode GSM8K under meaning-preserving prompt noise and
irrelevant chat history.

One invocation runs two tasks from checked-in static data (no network access):

    gsm8k-clean      the unperturbed reference (data/gsm8k_clean.jsonl)
    gsm8k-perturbed  one seeded condition per item (data/gsm8k_perturbed.jsonl):
                     noise conditions rewrite only the question text (HELM
                     invariance transforms plus a CMUdict homophone swap);
                     history conditions prepend unrelated off-domain chat
                     exchanges (OpenAssistant/oasst2) as prior chat turns

Reported metrics: clean accuracy, and per-condition PAIRED comparisons --
each perturbed instance is graded against the clean run of the same item
(matched by id), so per-condition deltas are free of subset-composition
noise. Per condition: accuracy, clean-subset accuracy, paired delta,
harmed/helped counts, and no-answer rate, plus changed-only and
spoken-number slices. Data provenance (exact generation
command and parameters) is the meta line of each data file; see
data_prep/generate.py and src/perturbations.py.

Prompting and grading follow lm-eval-harness's gsm8k conventions, zero-shot
through the chat template: the "Question: ... Answer:" prompt and
flexible-extract grading (last number-like token, sanitized numeric match) --
the exact protocol of the prior Marin chat-mode GSM8K measurement
(marin-community/marin#7321). Known caveat of that convention: a correct
answer followed by a restated premise ("... the same distance of 480
miles") loses the last-number race; on a chatty weak model this mis-graded
28% of clean-vs-perturbed verdict flips, concentrated in history
conditions. Anchored extraction ("conclude with: The answer is N") is the
candidate follow-up if that rate matters for a target model.

Design: marin-community/marin#7776 (part of marin-community/marin#7090).
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional


from eval.generation_stops import GSM8K_STOP_SEQUENCES
from eval.task import BaseBenchmark

# lm-eval-harness's gsm8k prompt, sent zero-shot through the chat template --
# the protocol of the prior Marin chat-mode GSM8K measurement
# (marin-community/marin#7321).
PROMPT_TEMPLATE = "Question: {question}\nAnswer:"

# lm-eval-harness gsm8k "flexible-extract" convention: last number-like token.
FLEXIBLE_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CLEAN_TASK = "gsm8k-clean"
PERTURBED_TASK = "gsm8k-perturbed"
TASK_FILES = {CLEAN_TASK: "gsm8k_clean.jsonl", PERTURBED_TASK: "gsm8k_perturbed.jsonl"}
NUMBER_RE = re.compile(r"-?\d+(\.\d+)?")


def load_static_records(filename: str) -> List[Dict[str, Any]]:
    """Read a checked-in data file; line 1 is the provenance meta header."""
    with open(os.path.join(DATA_DIR, filename)) as f:
        header = json.loads(f.readline())
        assert "meta" in header, f"{filename} must start with its provenance meta line"
        return [json.loads(line) for line in f]








class GSM8KPerturbedBenchmark(BaseBenchmark):
    SOURCE = SourceRef("marin-community/gsm8k-perturbed", "test")
    METRICS = ("gsm8k-clean", "gsm8k-perturbed")
    PRIMARY_METRIC = "gsm8k-perturbed"
    def __init__(
        self,
        max_tokens: int = 4096,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        super().__init__(logger=logger, system_instruction=system_instruction)
        self.max_new_tokens = max_tokens
        self.debug = debug










    REDUCER = ReducerSpec(name="paired_conditions")

    def iter_specs(self, options):
        for task, filename in TASK_FILES.items():
            rows = load_static_records(filename)
            if self.debug:
                rows = rows[:10]
            def prompt(row):
                messages = []
                for exchange in row.get("history_exchanges") or []:
                    messages.extend([{"role": "user", "content": exchange["question"]},
                                     {"role": "assistant", "content": exchange["answer"]}])
                return [*messages, {"role": "user", "content": PROMPT_TEMPLATE.format(question=row["question"])}]
            yield from text_specs(
                rows, options, prompt=prompt, target=lambda row: row["answer"], grader="gsm8k_flexible", namespace=task,
                reference=lambda row: {"pair_id": row["id"].split("::")[0]},
                generation_kwargs={"do_sample": False, "temperature": 0, "max_gen_toks": self.max_new_tokens,
                            "until": list(GSM8K_STOP_SEQUENCES)},
                tags=lambda row: {"task": task, "condition": row.get("condition", "baseline"),
                                  "changed": str(bool(row.get("changed"))), "number_words": str(bool(row.get("number_words_affected")))},
            )
