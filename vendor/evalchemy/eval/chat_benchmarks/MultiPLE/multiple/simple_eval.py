import argparse
import json
from containerized_eval import eval_string_script


def cli():
    args = argparse.ArgumentParser()
    args.add_argument("--lang", required=True, type=str, help="The language of the file")
    return args


if __name__ == "__main__":
    args = cli().parse_args()
    prog_str = ""
    while True:
        try:
            line = input()
            prog_str += line + "\n"
        except EOFError:
            break
    print(json.dumps(eval_string_script(args.lang, prog_str)), end="")
