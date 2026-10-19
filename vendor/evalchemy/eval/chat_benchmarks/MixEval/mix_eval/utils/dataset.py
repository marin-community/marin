import os
import json
import nltk

nltk.download("punkt", quiet=True)

from typing import Any, Dict

from mix_eval.prompts.evaluation_prompts import (
    construct_prompt_multichoice,
    construct_prompt_freeform,
)


def get_eval_dataset(args):
    if (
        args.split == "close_freeform"
        or args.split == "close_multichoice"
        or args.split == "close_freeform_hard"
        or args.split == "close_multichoice_hard"
    ):
        return EvalDatasetCloseended(args)
    else:
        raise ValueError(f"Split {args.split} not supported in {get_eval_dataset.__name__}.")


class EvalDatasetCloseended:
    def __init__(self, args):
        self.args = args

        version_dir = os.path.join(args.data_path, f"mixeval-{args.version}")

        raw_inputs = []
        if args.split == "close_freeform":
            print("Loading close-ended freeform data.")
            data_path_freeform = os.path.join(version_dir, "mixeval/free-form.json")
            with open(data_path_freeform, "r") as f:
                data = json.load(f)
                for id, d in data.items():
                    d["formated_input"] = construct_prompt_freeform(d)
                    d["id"] = id
                    raw_inputs.append(d)
        elif args.split == "close_multichoice":
            print("Loading close-ended multichoice data.")
            data_path_multiplechoice = os.path.join(version_dir, "mixeval/multiple-choice.json")
            with open(data_path_multiplechoice, "r") as f:
                data = json.load(f)
                for id, d in data.items():
                    d["formated_input"] = construct_prompt_multichoice(d)
                    d["id"] = id
                    raw_inputs.append(d)
        elif args.split == "close_freeform_hard":
            print("Loading close-ended freeform hard data.")
            data_path_freeform_hard = os.path.join(version_dir, "mixeval-hard/free-form.json")
            with open(data_path_freeform_hard, "r") as f:
                data = json.load(f)
                for id, d in data.items():
                    d["formated_input"] = construct_prompt_freeform(d)
                    d["id"] = id
                    raw_inputs.append(d)
        elif args.split == "close_multichoice_hard":
            print("Loading close-ended multichoice hard data.")
            data_path_multiplechoice_hard = os.path.join(version_dir, "mixeval-hard/multiple-choice.json")
            with open(data_path_multiplechoice_hard, "r") as f:
                data = json.load(f)
                for id, d in data.items():
                    d["formated_input"] = construct_prompt_multichoice(d)
                    d["id"] = id
                    raw_inputs.append(d)
        else:
            raise ValueError(f"Split {args.split} not supported in {self.__class__.__name__}")

        self.raw_inputs = raw_inputs

    def __len__(self):
        return len(self.raw_inputs)

    def __getitem__(self, i) -> Dict[str, Any]:
        return dict(
            raw_inputs=self.raw_inputs[i],
        )
