"""Construct executable test programs for a single dataset example."""

IMPORT_HELPER = {
    "python": [
        "import math",
        "import re",
        "import sys",
        "import copy",
        "import datetime",
        "import itertools",
        "import collections",
        "import heapq",
        "import functools",
        "import hashlib",
        "import numpy",
        "import numpy as np",
        "import string",
        "from typing import *",
        "from collections import *",
    ],
    "go": [
        "math",
        "strings",
        "fmt",
        "strconv",
        "time",
        "bytes",
        "regexp",
        "sort",
        "math/rand",
        "crypto/md5",
    ],
    "cpp": [
        "#include<stdlib.h>",
        "#include<algorithm>",
        "#include<math.h>",
        "#include<stdio.h>",
        "#include<vector>",
        "#include<string>",
        "#include<climits>",
        "#include<cstring>",
        "#include<iostream>",
        "#include<cassert>",
        "using namespace std;",
    ],
    "cs": [
        "using System.Numerics;",
        "using System.Diagnostics;",
        "using System.Collections.Generic;",
        "using System.Linq;",
        "using System.Text;",
        "using System.Security.Cryptography;",
        "using System.Collections.Generic;",
    ],
}


def process_humaneval_test(sample, problems, example_test=False, is_mbpp=False, language="python"):
    """
    Processes a sample for evaluation.
    """
    task_id = sample["task_id"]
    if is_mbpp:
        return sample["generation"] + "\n" + "\n".join(problems[task_id]["test"])

    prompt = sample["prompt"]
    if example_test and "example_test" in problems[task_id] and problems[task_id]["example_test"] != "":
        test = problems[task_id]["example_test"]
    else:
        test = problems[task_id]["test"]
    code = sample["generation"]

    # Pre-process for different languages
    if language == "python":
        test_setup = "\n".join(IMPORT_HELPER["python"]) + "\n"
        test_string = test_setup + code + "\n" + test + "\n"
    elif language == "cpp":
        test_set_up = ""
        for s in IMPORT_HELPER["cpp"]:
            if s not in prompt:
                test_set_up += s + "\n"
        test_string = test_set_up + "\n" + code + "\n" + test
    elif language == "java":
        num_open_brackets = code.count("{") - code.count("}")
        # if num_open_brackets < 0:
        #     code = code[:num_open_brackets]
        test_string = code + "\n" + test
    elif language == "cs":
        test_set_up = ""
        for s in IMPORT_HELPER["cs"]:
            test_set_up += s + "\n"
        test_string = test_set_up + "\n" + code + "\n" + test
    elif language in ["js", "javascript", "ts", "sh", "go"]:
        test_string = code + "\n" + test
    elif language == "go232":
        import_string = problems[task_id]["import"]
        prompt = prompt.replace(import_string, "")
        if example_test and "example_test" in problems[task_id]:
            test = problems[task_id]["example_test"]
        else:
            test = problems[task_id]["test"]
        test_setup = problems[task_id]["test_setup"]
        other_pkgs = []
        for pkg in IMPORT_HELPER["go"]:
            if pkg not in test_setup:
                p = pkg.split("/")[-1]
                if p + "." in code:
                    other_pkgs.append(f'"{pkg}"')
        if other_pkgs:
            import_other_pkgs = "import (\n" + "    ".join([p + "\n" for p in other_pkgs]) + ")"
            test_string = test_setup + "\n" + import_other_pkgs + "\n" + prompt + code + "\n" + test
        else:
            test_string = test_setup + "\n" + prompt + code + "\n" + test
    elif language == "rust":
        main = "\nfn main(){ \n } \n"
        declaration = problems[task_id]["declaration"]
        test_string = main + declaration + prompt + code + test
    elif language == "php":
        if code[:5] != "<?php":
            code = "<?php\n" + code
        test_string = code + "\n" + test + "?>"
    else:
        test_string = code + "\n" + test
    return test_string
