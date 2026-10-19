"""Compatibility shim for the upstream EleutherAI ``lm-eval`` API.

Evalchemy historically depended on a fork of ``lm-evaluation-harness`` that
still exposed three symbols which upstream removed/relocated after v0.4.7:

* ``lm_eval.utils.eval_logger`` -- a module-level ``logging.getLogger`` singleton.
  Upstream (>= v0.4.8) no longer defines this at module scope.
* ``lm_eval.__main__.setup_parser`` / ``lm_eval.__main__.parse_eval_args`` --
  upstream (>= v0.4.8) refactored the CLI into a class-based ``HarnessCLI``
  (``lm_eval._cli``) and dropped these helpers, and the modern ``run``
  subcommand's argument set has diverged (renamed/removed flags) from what
  evalchemy's ``eval/eval.py`` consumes.

To keep evalchemy decoupled from lm-eval CLI churn, we maintain the parser
derived from v0.4.7 here and provide the ``eval_logger`` singleton. Import
these from this module instead of from ``lm_eval`` directly.
"""

import argparse
import logging
from functools import partial

from eval.contracts.finestore_output import DEFAULT_SOURCE_PREFIX
# ---------------------------------------------------------------------------
# eval_logger singleton (was lm_eval.utils.eval_logger, removed upstream >=0.4.8)
# ---------------------------------------------------------------------------
eval_logger = logging.getLogger("lm-eval")


# ---------------------------------------------------------------------------
# CLI parser (was lm_eval.__main__.{setup_parser,parse_eval_args}, refactored
# into lm_eval._cli.HarnessCLI upstream >=0.4.8). Vendored verbatim from the
# v0.4.7-era harness, adapted to Evalchemy's FineStore output.
# ---------------------------------------------------------------------------
def _int_or_none_list_arg_type(min_len: int, max_len: int, defaults: str, value: str, split_char: str = ","):
    def parse_value(item):
        item = item.strip().lower()
        if item == "none":
            return None
        try:
            return int(item)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{item} is not an integer or None")

    items = [parse_value(v) for v in value.split(split_char)]
    num_items = len(items)

    if num_items == 1:
        # Makes downstream handling the same for single and multiple values
        items = items * max_len
    elif num_items < min_len or num_items > max_len:
        raise argparse.ArgumentTypeError(f"Argument requires {max_len} integers or None, separated by '{split_char}'")
    elif num_items != max_len:
        logging.warning(
            f"Argument requires {max_len} integers or None, separated by '{split_char}'. "
            "Missing values will be filled with defaults."
        )
        default_items = [parse_value(v) for v in defaults.split(split_char)]
        items.extend(default_items[num_items:])  # extend items list with missing defaults

    return items


def check_argument_types(parser: argparse.ArgumentParser):
    """Check to make sure all CLI args are typed, raises error if not."""
    for action in parser._actions:
        if action.dest != "help" and not action.const:
            if action.type is None:
                raise ValueError(f"Argument '{action.dest}' doesn't have a type specified.")
            else:
                continue


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--model", "-m", type=str, default="local-chat-completions",
        choices=("local-chat-completions", "local-completions"),
        help="OpenAI-compatible endpoint adapter",
    )
    parser.add_argument(
        "--tasks",
        "-t",
        default=None,
        type=str,
        metavar="task1,task2",
        help="Comma-separated list of task names or task groupings to evaluate on.\nTo get full list of tasks, use one of the commands `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above",
    )
    parser.add_argument(
        "--model_args",
        "-a",
        default="",
        type=str,
        help="Comma-separated endpoint arguments, e.g. `model=served,base_url=http://localhost:8000/v1/chat/completions`",
    )
    parser.add_argument(
        "--num_fewshot",
        "-f",
        type=int,
        default=None,
        metavar="N",
        help="Number of examples in few-shot context",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        metavar="N",
        help="Maximal batch size to try with --batch_size auto.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--finestore_output_path",
        "--output_path",
        dest="finestore_output_path",
        required=True,
        type=str,
        metavar="DIR",
        help="FineStore archive for aggregate results, normalized samples, and request resume state.",
    )
    parser.add_argument(
        "--finestore_output_prefix",
        default=DEFAULT_SOURCE_PREFIX,
        type=str,
        metavar="NAME",
        help="Stable task-group name for results and resume state inside FineStore.",
    )
    parser.add_argument(
        "--limit",
        "-L",
        type=float,
        default=None,
        metavar="N|0<N<1",
        help="Limit the number of examples per task. If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        "--system_instruction",
        type=str,
        default=None,
        help="System instruction to be used in the prompt",
    )
    parser.add_argument(
        "--apply_chat_template",
        type=str,
        nargs="?",
        const=True,
        default=False,
        help=(
            "If True, apply chat template to the prompt. "
            "Providing `--apply_chat_template` without an argument will apply the default chat template to the prompt. "
            "To apply a specific template from the available list of templates, provide the template name as an argument. "
            "E.g. `--apply_chat_template template_name`"
        ),
    )
    parser.add_argument(
        "--fewshot_as_multiturn",
        action="store_true",
        default=False,
        help="If True, uses the fewshot as a multi-turn conversation",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        metavar="DIR",
        help="Additional path to include if there are external tasks to include.",
    )
    parser.add_argument(
        "--gen_kwargs",
        type=str,
        default=None,
        help=(
            "String arguments for model generation on greedy_until tasks,"
            " e.g. `temperature=0,top_k=0,top_p=0`."
        ),
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        type=str.upper,
        default="INFO",
        metavar="CRITICAL|ERROR|WARNING|INFO|DEBUG",
        help="Controls the reported logging error level. Set to DEBUG when testing + adding new task configurations for comprehensive log output.",
    )
    parser.add_argument(
        "--predict_only",
        "-x",
        action="store_true",
        default=False,
        help="Only model outputs will be stored; metrics will not be evaluated.",
    )
    default_seed_string = "0,1234,1234,1234"
    parser.add_argument(
        "--seed",
        type=partial(_int_or_none_list_arg_type, 3, 4, default_seed_string),
        default=default_seed_string,  # for backward compatibility
        help=(
            "Set seed for python's random, numpy, torch, and fewshot sampling.\n"
            "Accepts a comma-separated list of 4 values for python's random, numpy, torch, and fewshot sampling seeds, "
            "respectively, or a single integer to set the same seed for all four.\n"
            f"The values are either an integer or 'None' to not set the seed. Default is `{default_seed_string}` "
            "(for backward compatibility).\n"
            "E.g. `--seed 0,None,8,52` sets `random.seed(0)`, `torch.manual_seed(8)`, and fewshot sampling seed to 52. "
            "Here numpy's seed is not set since the second value is `None`.\n"
            "E.g, `--seed 42` sets all four seeds to 42."
        ),
    )
    parser.add_argument(
        "--confirm_run_unsafe_code",
        action="store_true",
        help="Confirm that you understand the risks of running unsafe code for tasks that require it",
    )
    return parser


def parse_eval_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    check_argument_types(parser)
    return parser.parse_args()
