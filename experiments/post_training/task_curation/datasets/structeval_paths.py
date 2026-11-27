# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The official StructEval metric-path grammar, ported from TIGER-AI-Lab/StructEval eval_utils.

Pure stdlib so both the converter (fail-closed path validation at conversion) and the shipped
grader script share one implementation of the tokenizer and existence walk.
"""


def tokenize_path(path: str) -> list[str]:
    """The official tokenizer: ``csv::`` stays one token; backticks quote dots; ``[N]`` splits."""
    if path.startswith("csv::"):
        return [path]
    tokens: list[str] = []
    buf = ""
    in_backtick = False
    index = 0
    while index < len(path):
        char = path[index]
        if char == "`":
            in_backtick = not in_backtick
            index += 1
            continue
        if char == "." and not in_backtick:
            if buf:
                tokens.append(buf)
                buf = ""
            index += 1
            continue
        if char == "[" and not in_backtick:
            if buf:
                tokens.append(buf)
                buf = ""
            end = path.find("]", index)
            if end == -1:
                raise ValueError(f"Unclosed '[' in path: {path}")
            tokens.append(path[index : end + 1])
            index = end + 1
            continue
        buf += char
        index += 1
    if buf:
        tokens.append(buf)
    return tokens


def path_exists(data, path: str) -> bool:
    """The official walk: list wildcards, fixed indices, dict keys, ``@attr`` fallback."""

    def walk(node, tokens):
        if not tokens:
            return True
        token, rest = tokens[0], tokens[1:]
        if isinstance(node, dict) and "csv_headers" in node and token.startswith("csv::"):
            header = token[5:]
            return header in node["csv_headers"] and not rest
        if token == "*":
            if isinstance(node, list):
                return any(walk(item, rest) for item in node)
            return False
        if token.startswith("[") and token.endswith("]"):
            try:
                index = int(token[1:-1])
            except ValueError:
                return False
            return isinstance(node, list) and 0 <= index < len(node) and walk(node[index], rest)
        if isinstance(node, dict):
            if token in node:
                return walk(node[token], rest)
            if token.startswith("@"):
                attribute = token[1:]
                if attribute in node:
                    return walk(node[attribute], rest)
        return False

    return walk(data, tokenize_path(path))
