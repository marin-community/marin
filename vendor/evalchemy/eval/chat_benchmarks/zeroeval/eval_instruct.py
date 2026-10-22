from eval.contracts.eval_spec import ReducerSpec, SourceRef
from eval.sources.common import text_specs
from datasets import load_dataset
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import logging

from eval.chat_benchmarks.zeroeval.src.task_configs import prompt_generation
from eval.task import BaseBenchmark

from .src.task_configs import mapping_task_names


@dataclass
class ZeroEvalConfig:
    # Dataset configuration
    start_index: int = 0
    end_index: int = -1


class ZeroEvalBenchmark(BaseBenchmark):
    """
    ZeroEval benchmark for a number of tasks and benchmarks.
    """

    REDUCER = ReducerSpec(trial_metric=None)

    def __init__(
        self,
        tasks: List[str] = ["zebra-grid", "numersense-v2", "crux", "math-l5"],
        config: Optional[ZeroEvalConfig] = None,
        max_tokens: Optional[int] = 4096,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        super().__init__(logger, system_instruction=system_instruction)
        self.tasks = tasks
        self.config = config or ZeroEvalConfig()
        self.max_new_tokens = int(max_tokens) if max_tokens is not None else 4096
        self.debug = debug

    def load_dataset(self, data_name: str) -> Tuple[List[str], List[str], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Load dataset for ZeroEval tasks.
        """
        try:
            chat_history = []
            id_strs = []
            extracted_chats = []
            metadata = {}
            dataset, id_name = mapping_task_names(data_name)

            if self.debug:
                dataset = dataset.select(range(min(10, len(dataset))))
                self.logger.info(f"Debug mode: using {len(dataset)} examples")

            prompt_generation_args = Namespace(run_name="")
            for ind, item in enumerate(dataset):
                id_strs.append(item.get(id_name, f"{data_name}#{ind}"))
                prompt = prompt_generation(data_name, item, prompt_generation_args)
                chat_history.append([prompt])
                extracted_chats.append([{"content": prompt, "role": "user"}])
                for key in item:
                    if key not in metadata:
                        metadata[key] = []
                    metadata[key].append(item[key])

            self.logger.info(f"Finished processing {data_name} dataset.")

            # Apply index limits
            if self.config.end_index < 0:
                self.config.end_index = len(id_strs)

            slice_range = slice(self.config.start_index, self.config.end_index)
            return (
                id_strs[slice_range],
                chat_history[slice_range],
                extracted_chats[slice_range],
                {k: v[slice_range] for k, v in metadata.items()},
            )

        except Exception as e:
            self.logger.error(f"Error loading dataset for task {data_name}: {e}")
            raise e



    def iter_specs(self, options):
        for task in self.tasks:
            identities, _, chats, metadata = self.load_dataset(task)
            solutions = {}
            if task == "zebra-grid":
                solutions = {row["id"]: row["solution"] for row in load_dataset("allenai/ZebraLogicBench-private", "grid_mode", split="test")}
            rows = ({**{key: values[index] for key, values in metadata.items()},
                     "id": identity, "prompt": chats[index]} for index, identity in enumerate(identities))
            yield from text_specs(
                rows, options, namespace=task, prompt=lambda row: row["prompt"],
                target=lambda row: solutions[row["id"]] if task == "zebra-grid" else row["answer"],
                grader="json_answer", reference=lambda row: {"kind": "grid" if task == "zebra-grid" else "numeric" if task in {"numersense-v2", "math-l5"} else "literal"},
                generation_kwargs={"temperature": 0, "do_sample": False, "max_gen_toks": self.max_new_tokens},
                tags=lambda row: {"task": task},
            )
    SOURCE = SourceRef("zeroeval")
    METRICS = ("accuracy",)
    PRIMARY_METRIC = "accuracy"
