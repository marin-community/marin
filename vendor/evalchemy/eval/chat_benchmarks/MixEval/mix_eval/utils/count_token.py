import argparse
import os
import json

import tiktoken

from mix_eval.prompts.evaluation_prompts import (
    construct_prompt_multichoice,
    construct_prompt_freeform,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        type=str,
        choices=["close_freeform", "close_multichoice", "open", "all"],
        default="all",
        help="Split to evaluate.",
    )
    return parser.parse_args()


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo-0613":  # note: future models may deviate from this
        num_tokens = 0
        for message in messages:
            num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
        num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not presently implemented for model {model}.
    See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )


def count_all_tokens_to_filter(args):
    number_tokens = 0

    if args.split == "all":
        splits = ["close_freeform", "close_multichoice", "open"]
    else:
        splits = [args.split]

    for split in splits:
        if split == "close_freeform":
            data_path = "mix_eval/data/text2text/text2text_closeended/free-form.json"
        elif split == "close_multichoice":
            data_path = "mix_eval/data/text2text/text2text_closeended/multiple-choice.json"
        elif split == "open":
            data_path = "mix_eval/data/text2text/text2text_openended.json"
        with open(data_path, "r") as f:
            data = json.load(f)
            for id, d in data.items():
                if split == "close_multichoice":
                    formated_input = construct_prompt_multichoice(d)
                    number_tokens += num_tokens_from_messages([{"content": formated_input}])
                elif split == "close_freeform":
                    formated_input = construct_prompt_freeform(d)
                    number_tokens += num_tokens_from_messages([{"content": formated_input}])
                else:
                    formated_input = "\n".join(d["turns"])
                    number_tokens += num_tokens_from_messages([{"content": formated_input}]) + 1500

    print(f"Total number of tokens: {number_tokens}")


if __name__ == "__main__":
    args = parse_args()
    count_all_tokens_to_filter(args)
