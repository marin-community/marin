# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Locate the tested function's declaration in an MT-MBPP reference solution.

MT-MBPP's prompts state each task in words; the executable variant discloses the signature of the function the tests
call. The signature is copied verbatim from the o4-mini reference solution (``allenai/multilingual_mbpp``): the
declaration header of the function whose name matches the name MBPP's Python asserts call, up to the start of its
body, with whitespace collapsed to single spaces. Haskell uses the type signature line; Ruby, MATLAB and bash use the
declaration line.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from difflib import SequenceMatcher

C_LIKE_KEYWORDS = frozenset(
    "return else if while for foreach in var lock fixed checked unchecked switch case new delete throw sizeof do goto "
    "using typedef catch await yield".split()
)
MODIFIERS = (
    r"(?:static|public|private|protected|internal|override|virtual|async|readonly|unsafe|extern|sealed|abstract|new|"
    r"partial|const|final|synchronized|native|strictfp|return|if|while|for|foreach|switch|using|var)\b"
)
ARROW = r"^[ \t]*(?:export\s+)?(?:const|let|var)\s+(?P<name2>\w+)\s*(?::[^=\n]+)?=\s*(?:async\s+)?(?:\([^)\n]*\)|\w+)\s*(?::[^=\n]+)?=>"  # noqa: E501
DECLARATION = {
    "python": r"^[ \t]*(?:async\s+)?def\s+(?P<name>\w+)\s*\(",
    "ruby": r"^[ \t]*def\s+(?:self\.)?(?P<name>\w+[?!]?)",
    "javascript": r"^[ \t]*(?:export\s+)?(?:async\s+)?function\s*\*?\s*(?P<name>\w+)\s*\(|" + ARROW,
    "typescript": r"^[ \t]*(?:export\s+)?(?:async\s+)?function\s*\*?\s*(?P<name>\w+)\s*[<(]|" + ARROW,
    "php": r"^[ \t]*(?:(?:public|private|protected|static|final)\s+)*function\s+&?(?P<name>\w+)\s*\(",
    "r": r"^[ \t]*(?P<name>[\w.]+)\s*(?:<-|=)\s*function\s*\(",
    "matlab": r"^[ \t]*function\s+(?:(?:\[[^\]\n]*\]|\w+)\s*=\s*)?(?P<name>\w+)\s*(?:\(|$)",
    "bash": r"^[ \t]*(?:function\s+(?P<name>\w+)\s*(?:\(\s*\))?|(?P<name2>\w+)\s*\(\s*\))\s*\{?",
    "go": r"^func\s+(?P<name>\w+)\s*(?:\[[^\]\n]*\])?\s*\(",
    "rust": r"^[ \t]*(?:pub(?:\([^)\n]*\))?\s+)?(?:const\s+)?(?:unsafe\s+)?fn\s+(?P<name>\w+)",
    "swift": r"^[ \t]*(?:(?:public|private|internal|fileprivate|static|final|@\w+)\s+)*func\s+(?P<name>\w+)",
    "scala": r"^[ \t]*(?:(?:private|protected|final|inline)\s+)*def\s+(?P<name>\w+)",
    "haskell": r"^(?P<name>[a-z_][\w']*)\s*::",
    "c": r"^[ \t]*(?P<head>[\w\s\*&:<>,\[\]~]*?)\b(?P<name>\w+)\s*\(",
    "cpp": r"^[ \t]*(?P<head>[\w\s\*&:<>,\[\]~]*?)\b(?P<name>\w+)\s*\(",
    "java": r"^[ \t]*(?P<head>[\w\s\*&:<>,\[\]~?@.]*?)\b(?!" + MODIFIERS + r")(?P<name>\w+)\s*\(",
    "csharp": r"^[ \t]*(?P<head>[\w\s\*&:<>,\[\]~?@.()]*?)\b(?!"
    + MODIFIERS
    + r")(?P<name>\w+)\s*(?:<(?:[\w,?\[\]\s.]|<(?:[\w,?\[\]\s.]|<[\w,?\[\]\s.]*>)*>)*>)?\s*\(",
}
BRACE_LANGUAGES = frozenset(
    {"c", "cpp", "java", "csharp", "go", "rust", "swift", "javascript", "typescript", "php", "r", "bash"}
)
LINE_LANGUAGES = frozenset({"ruby", "matlab"})
FUZZY_THRESHOLD = 0.75
ENTRY_THRESHOLD = 0.5


@dataclass(frozen=True)
class Declaration:
    name: str
    signature: str
    start: int


def normalized(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def tested_name(asserts: str, reference: str) -> str:
    """The function MBPP's asserts call: the one the Python reference solution defines at top level."""
    called = {
        n.func.id for n in ast.walk(ast.parse(asserts)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    defined = [n.name for n in ast.parse(reference).body if isinstance(n, ast.FunctionDef)]
    names = [n for n in defined if n in called]
    if len(names) != 1:
        raise ValueError(f"Expected one tested function, got {names}")
    return names[0]


def matching_close(text: str, open_index: int, pair: str = "()") -> int:
    depth = 0
    for i in range(open_index, len(text)):
        if text[i] == pair[0]:
            depth += 1
        elif text[i] == pair[1]:
            depth -= 1
            if depth == 0:
                return i
    raise ValueError("Unbalanced declaration")


def header_end(language: str, code: str, start: int, name_end: int) -> int | None:
    """Index where the declaration header ends (exclusive), or None for a prototype or non-declaration."""
    if language in LINE_LANGUAGES:
        end = code.find("\n", start)
        return len(code) if end < 0 else end
    if language == "haskell":
        end = start
        while True:
            newline = code.find("\n", end)
            if newline < 0:
                return len(code)
            nxt = code[newline + 1 : newline + 2]
            if nxt not in (" ", "\t") or not code[newline + 1 :].strip():
                return newline
            end = newline + 1
    if language in ("javascript", "typescript") and code[start:name_end].lstrip().startswith(
        ("const", "let", "var", "export")
    ):
        return name_end
    if language == "bash":
        end = code.find("\n", start)
        line = code[start : len(code) if end < 0 else end]
        return start + line.index("{") if "{" in line else start + len(line.rstrip())
    paren = code.find("(", name_end)
    if paren < 0:
        return None
    if language in ("typescript", "rust", "swift") and code[name_end:paren].strip().startswith("<"):
        pass  # generic parameters precede the value parameters
    close = matching_close(code, paren)
    if language == "python":
        depth, i = 0, close + 1
        while i < len(code):
            if code[i] in "([{":
                depth += 1
            elif code[i] in ")]}":
                depth -= 1
            elif code[i] == ":" and depth == 0:
                return i
            i += 1
        return None
    if language == "scala":
        depth, i = 0, close + 1
        while i < len(code):
            c = code[i]
            if c in "([{":
                if c == "{" and depth == 0:
                    return i
                depth += 1
            elif c in ")]}":
                depth -= 1
            elif c == "=" and depth == 0 and code[i + 1 : i + 2] != ">" and code[i - 1 : i] not in "<>!=":
                return i
            i += 1
        return None
    depth, i = 0, close + 1
    while i < len(code):
        c = code[i]
        if c == ";" and depth == 0:
            return None
        if c == "=" and code[i + 1 : i + 2] == ">" and depth == 0 and language == "csharp":
            return i
        if c in "(<[":
            depth += 1
        elif c in ")>]":
            if not (c == ">" and code[i - 1 : i] == "-"):
                depth -= 1
        elif c == "{" and depth <= 0:
            return i
        i += 1
    return None


def declarations(language: str, code: str) -> list[Declaration]:
    found = []
    for match in re.finditer(DECLARATION[language], code, flags=re.MULTILINE):
        name = match.group("name") or (match.groupdict().get("name2") or "")
        if not name or name in C_LIKE_KEYWORDS or name == "main":
            continue
        if language in ("c", "cpp", "java", "csharp"):
            head = match.group("head").split()
            if not head or head[-1] in C_LIKE_KEYWORDS or any(w in C_LIKE_KEYWORDS for w in head):
                continue
        start = match.start()
        if language == "cpp":
            previous = code.rfind("\n", 0, max(start - 1, 0))
            line = code[previous + 1 : start].strip()
            if line.startswith("template"):
                start = previous + 1
        end = header_end(language, code, match.start(), match.end("name") if match.group("name") else match.end())
        if end is None:
            continue
        signature = " ".join(code[start:end].split())
        if signature:
            found.append(Declaration(name, signature, start))
    return found


def tested_declaration(language: str, code: str, name: str) -> Declaration | None:
    """The reference solution's declaration of the tested function, or None when it cannot be identified.

    Prefer the unique declaration whose name matches MBPP's after normalization, then the unique closest name
    (translations fix misspellings such as ``rearange_string``), then the only function no other function calls.
    """
    found = declarations(language, code)
    if language == "haskell":
        first = {}
        for d in found:
            first.setdefault(d.name, d)
        found = list(first.values())
    exact = [d for d in found if normalized(d.name) == normalized(name)]
    if len(exact) == 1:
        return exact[0]
    if exact:
        return None
    scored = sorted(
        ((SequenceMatcher(None, normalized(d.name), normalized(name)).ratio(), d) for d in found), key=lambda x: -x[0]
    )
    if scored and scored[0][0] >= FUZZY_THRESHOLD and (len(scored) == 1 or scored[1][0] < scored[0][0]):
        return scored[0][1]
    ordered = sorted(found, key=lambda d: d.start)
    spans = {
        id(d): (d.start, ordered[i + 1].start if i + 1 < len(ordered) else len(code)) for i, d in enumerate(ordered)
    }

    def called_elsewhere(d: Declaration) -> bool:
        lo, hi = spans[id(d)]
        return any(not lo <= m.start() < hi for m in re.finditer(rf"\b{re.escape(d.name)}\s*\(", code))

    entry = [d for d in found if not called_elsewhere(d)]
    if len(entry) == 1:
        return entry[0]
    ranked = sorted(
        ((SequenceMatcher(None, normalized(d.name), normalized(name)).ratio(), d) for d in entry), key=lambda x: -x[0]
    )
    if len(ranked) > 1 and ranked[0][0] >= ENTRY_THRESHOLD and ranked[1][0] < ranked[0][0]:
        return ranked[0][1]
    return None


def bash_usage(code: str, name: str) -> str:
    """A shell function's interface as a usage line: its name, positional parameters, and ``items...`` for "$@".

    Parameter names are the variables the reference assigns from ``$1``, ``$2``, ... (``argN`` when unnamed); a
    function that reads ``"$@"`` (after any ``shift``) takes the remaining arguments as a list, and one that reads
    standard input is marked ``< stdin``.
    """
    positions = {}
    for var, index in re.findall(r"\b(\w+)=\(?[\"']?\$\{?([1-9])\}?", code):
        positions.setdefault(int(index), var)
    if not re.search(r"\bshift\b", code):
        unquoted = re.sub(r"'[^']*'", "", code)
        for index in re.findall(r"\$\{?([1-9])\}?", unquoted):
            positions.setdefault(int(index), f"arg{index}")
    params = [positions[i] for i in sorted(positions)]
    if re.search(r"\$@|\$\*|\$\{@", code):
        rest = re.search(r'\b(\w+)=\(\s*"?\$[@*]"?\s*\)|\bfor\s+(\w+)\s+in\s+"?\$[@*]', code)
        label = (rest.group(1) or rest.group(2) + "s") if rest else "items"
        params.append(label + "...")
    usage = " ".join([name, *params])
    return usage + " < stdin" if re.search(r"\bread\b", code) else usage
