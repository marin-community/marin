import re
from tree_sitter import Language, Parser

LANUGUAGES = [
    "adb",
    "clj",
    "cpp",
    "cs",
    "dart",
    "dfy",
    "dlang",
    "elixir",
    "fs",
    "go",
    "hs",
    "java",
    "javascript",
    "julia",
    "lean",
    "lua",
    "luau",
    "matlab",
    "ocaml",
    "php",
    "pl",
    "python",
    "r",
    "racket",
    "ruby",
    "rust",
    "scala",
    "sh",
    "swift",
    "ts",
    "v",
]

language_settings = {
    "python": {
        "full_name": "Python",
        "indent": 4,
    },
    "cpp": {
        "full_name": "cpp",
        "indent": 0,
        "main": "int main()",
    },
    "java": {
        "full_name": "Java",
        "indent": 4,
        "main": "public static void main",
    },
    "cs": {
        "full_name": "csharp",
        "indent": 0,
        "main": "public static void Main",
    },
    "php": {
        "full_name": "PHP",
        "indent": 0,
    },
    "ts": {
        "full_name": "TypeScript",
        "indent": 0,
    },
    "javascript": {"full_name": "JavaScript", "indent": 0},
    "js": {"full_name": "JavaScript", "indent": 0},
    "sh": {"full_name": "bash", "indent": 0},
    "go": {"full_name": "Go", "indent": 0},
    "rust": {"full_name": "Rust", "indent": 0},
    "rs": {"full_name": "Rust", "indent": 0},
    "v": {"full_name": "Coq", "indent": 0},
    "r": {"full_name": "R", "indent": 0},
    "racket": {"full_name": "Racket", "indent": 0},
    "ruby": {"full_name": "Ruby", "indent": 0},
    "lua": {"full_name": "Lua", "indent": 0},
    "swift": {"full_name": "Swift", "indent": 0},
    "scala": {"full_name": "Scala", "indent": 0},
    "pl": {"full_name": "Perl", "indent": 0},
    "ocaml": {"full_name": "OCaml", "indent": 0},
    "matlab": {"full_name": "Matlab", "indent": 0},
    "luau": {"full_name": "Luau", "indent": 0},
    "lean": {"full_name": "Lean", "indent": 0},
    "julia": {"full_name": "Julia", "indent": 0},
}


def get_function_name(question: str, lang: str):
    if question.startswith("<?php"):
        question = question[5:]

    # remove comments from the question
    question = re.sub(r"//.*?\n", "", question).strip()

    func_lines = [x for x in question.strip().split("\n") if x.strip()]

    if lang.lower() == "python":
        func_idx = [i for i in range(len(func_lines)) if func_lines[i].startswith("def ")][-1]
        func_name = func_lines[func_idx].split("(")[0].strip()
        func_prefix = "\n".join(func_lines[:func_idx])
        return func_name, func_prefix

    func_name = func_lines[-1].split("{")[0].strip()
    func_prefix = "\n".join(func_lines[:-1])
    return func_name, func_prefix


def match_brackets(code: str, lang: str):
    """
    Matches brackets in the code.
    """

    code = code.strip()
    if lang.lower() == "python":
        return code

    parser = Parser()
    parser.set_language(Language("tree-sitter-{}".format(lang.lower())))
    tree = parser.parse(bytes(code, "utf-8"))
    root_node = tree.root_node

    bracket_pairs = []
    for node in root_node.children:
        if node.type == "function_declaration":
            bracket_pairs.append((node.start_point, node.end_point))

    return bracket_pairs


def extract_generation_code(example: str, lang_code: str, verbose: bool = False):
    task_id = example["task_id"]
    output = example.get("output", example.get("gpt_completion"))
    question = example["prompt"].strip()
    setting = language_settings[lang_code]
    lang = setting["full_name"]
    indent = setting["indent"]

    try:
        code_block: str = re.findall(f"```{lang.lower()}\n(.*?)```", output, re.DOTALL | re.IGNORECASE)[0]

        func_name, func_prefix = get_function_name(question, lang)
        if verbose:
            print(">>> Task: {}\n{}".format(task_id, code_block))

        # Remove main
        if setting.get("main", None) and setting["main"] in code_block:
            main_start = code_block.index(setting["main"])
            code_block = code_block[:main_start]

        try:
            start = code_block.lower().index(func_name.lower())
            indent = 0
            while start - indent >= 0 and code_block[start - indent - 1] == " ":
                indent += 1

            try:
                end = code_block.rindex("\n" + " " * indent + "}")
            except:
                end = len(code_block)
        except:
            start = 0
            try:
                end = code_block.rindex("\n" + " " * indent + "}")
            except:
                end = len(code_block)

        body = code_block[start:end]

        if lang_code.lower() in ["php", "ts", "js", "r"]:
            body += "\n" + " " * indent + "}"

        if lang_code.lower() in ["rs", "rust"]:
            # remove "Main" from the body since tests will have their own main
            body = body.replace("fn main", "fn example")

        if lang_code.lower() == "php":
            func_str = func_name + " " + func_prefix + "{"

            if func_str not in body:
                body = func_str + body
            body = body.replace("<?php", "").replace("?>", "").strip()

        elif lang_code.lower() in ["java", "cpp", "cs"]:
            if lang_code.lower() == "cpp":
                body = body.replace("std::", "")
                func_prefix = func_prefix.replace("std::", "")
                func_name = func_name.replace("std::", "")
            if lang_code.lower() == "cs":
                # remove "Main" from the body since tests will have their own main
                body = body.replace("Main", "Example")
            func_str = func_prefix + "\n" + func_name + " {"
            if func_name in body:
                body = body.replace(func_name + " {", "")
            if func_prefix in body:
                body = body.replace(func_prefix, "")
            if func_str not in body:
                body = func_str + body
        else:
            body = func_prefix + "\n" + body + "\n"

        example["generation"] = body

    except Exception as ex:
        print(
            "Failed to extract code block with error `{}`:\n>>> Task: {}\n>>> Output:\n{}".format(ex, task_id, output)
        )
        if lang.lower() == "php":
            output = output.replace("<?php", "").replace("?>", "").strip()
        output = output.replace(f"```{lang.lower()}", "").strip()
        output = output.replace("```", "").strip()

        example["generation"] = example["prompt"] + "\n" + output

    return example


def cleanup_code(code: str, language_type: str = None, dataset: str = None, issft: bool = False, stop_words=[]):
    """
    Cleans up the generated code.
    """

    if language_type.lower() == "python":
        if issft:
            code = _clean_python_code_for_sft(code)
        stop_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint"]
        code = _truncate_code_at_stopwords(code, stop_words)
    elif language_type.lower() == "ts":
        code = _truncate_code_at_stopwords(
            code, stop_words + ["\nexport", "\nimport", "\nexport default", "\nimport default", "\nconsole.log"]
        )
    else:
        code = _truncate_code_at_stopwords(code, stop_words)

    return code


def _clean_python_code_for_sft(code):
    code = code.replace("\r", "")
    if "```python" in code:
        code_start_idx = code.index("```python")
        code = code[code_start_idx:].replace("```python", "").strip()
        end_idx = code.find("```") if "```" in code else len(code)
        code = code[:end_idx].strip()

    return code


def _truncate_code_at_stopwords(code, stop_words):
    min_stop_idx = len(code)
    for stop_word in stop_words:
        stop_index = code.find(stop_word)
        if 0 <= stop_index < min_stop_idx:
            min_stop_idx = stop_index
    return code[:min_stop_idx]
