# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Assemble one executable test program from a solution and a translated test.

A program is the test's ``imports``, then the solution (a reference, a stub, or a model completion cut at its closing
code fence), then the test's ``main``. Per-language fixes keep an otherwise valid solution from breaking the harness:
a solution's own ``main`` is renamed, Java's public top-level types are demoted (the test's ``public class Main``
owns the file) and a solution class named ``Main`` is renamed, Go and Haskell imports are merged into the header,
module headers and ``export`` keywords are dropped, and PHP tags are balanced. ``{CLASS}`` in a Java or C# test is
replaced by the class that declares the tested method.

Script languages print ``SENTINEL`` after the tests; the grader requires it, so a solution that exits early or
swallows the test code (a MATLAB function without ``end``) cannot pass.
"""

import re

SENTINEL = "MT_MBPP_TESTS_PASSED"
SENTINEL_LINE = {
    "python": f'print("{SENTINEL}")',
    "bash": f"echo {SENTINEL}",
    "javascript": f'console.log("{SENTINEL}");',
    "typescript": f'console.log("{SENTINEL}");',
    "php": f'echo "{SENTINEL}\\n";',
    "r": f'cat("{SENTINEL}\\n")',
    "ruby": f'puts "{SENTINEL}"',
    "matlab": f"disp('{SENTINEL}')",
    "swift": f'print("{SENTINEL}")',
}
C_MAIN = re.compile(r"\b(int|void)\s+main\s*\(")
SUCCESS_EXIT = re.compile(
    r"\n?[ \t]*(?:exit(?:\s*\(\s*0?\s*\)|\s+0)?|sys\.exit\(\s*0?\s*\)|process\.exit\(\s*0?\s*\)|quit\((?:\s*status\s*=\s*0\s*)?\)|"  # noqa: E501
    r"q\(\s*\))[ \t]*;?[ \t]*$"
)
PHP_TAG = re.compile(r"<\?php|\?>")
WRAPPER = "SolutionWrapper"
TYPE_DECLARATION = re.compile(r"\b(?:class|struct|record|interface|enum)\s+(\w+)")


def enclosing_class(source: str, function: str) -> str | None:
    """The nearest type declared before the tested method's declaration (prefixed by a block namespace in C#)."""
    method = re.search(rf"^[^\n;=]*\b{re.escape(function)}\s*(?:<[^()\n]*>)?\s*\(", source, flags=re.MULTILINE)
    if method is None:
        return None
    classes = [m for m in TYPE_DECLARATION.finditer(source) if m.start() < method.start()]
    if not classes:
        return None
    name = classes[-1].group(1)
    block_namespace = re.search(r"\bnamespace\s+([\w.]+)\s*\{", source[: classes[-1].start()])
    return f"{block_namespace.group(1)}.{name}" if block_namespace else name


def wrapped(language: str, solution: str) -> str:
    """A bare Java or C# method wrapped in a class, with the file's imports or usings kept above it."""
    head = r"^\s*(?:using|import)\s+[^\n;]+;\s*$"
    imports = "\n".join(re.findall(head, solution, flags=re.MULTILINE))
    body = re.sub(head, "", solution, flags=re.MULTILINE)
    keyword = "public static class" if language == "csharp" else "class"
    return f"{imports}\n{keyword} {WRAPPER} {{\n{body}\n}}"


def go_imports(source: str) -> set[str]:
    paths = set(re.findall(r'^\s*import\s+(?:\w+\s+)?"([^"]+)"', source, flags=re.MULTILINE))
    for block in re.findall(r"^\s*import\s*\((.*?)\)", source, flags=re.MULTILINE | re.DOTALL):
        paths |= set(re.findall(r'"([^"]+)"', block))
    return paths


def split_haskell(source: str) -> tuple[str, str, str]:
    """(pragmas, imports, rest) of a Haskell module, with any ``module ... where`` header dropped."""
    source = re.sub(r"^module\s+[\w.]+(?:\s*\(.*?\))?\s*where\s*$", "", source, count=1, flags=re.MULTILINE | re.DOTALL)
    pragmas, imports, rest = [], [], []
    lines = source.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("{-#") and not rest:
            pragmas.append(line)
        elif line.startswith("import ") and not rest:
            imports.append(line)
            while i + 1 < len(lines) and lines[i + 1].startswith((" ", "\t")) and lines[i + 1].strip():
                i += 1
                imports.append(lines[i])
        elif line.strip() or rest:
            rest.append(line)
        i += 1
    return "\n".join(pragmas), "\n".join(imports), "\n".join(rest)


def program(language: str, solution: str, test: dict, function: str) -> str:
    """The complete source file for ``language``."""
    imports, main = test["imports"].strip(), test["main"].strip()
    solution = solution.strip("\n")
    if language in ("c", "cpp"):
        solution = C_MAIN.sub(r"\1 solution_main_unused(", solution)
    elif language == "rust":
        attributes = "\n".join(re.findall(r"^#!\[.*\]$", solution, flags=re.MULTILINE))
        solution = re.sub(r"^#!\[.*\]$", "", solution, flags=re.MULTILINE)
        solution = re.sub(r"\bfn\s+main\s*\(", "fn solution_main_unused(", solution)
        imports = "\n".join(x for x in (attributes, imports) if x)
    elif language == "go":
        if re.search(r"^\s*package\s+\w+", solution, flags=re.MULTILINE):
            solution = re.sub(r"^\s*package\s+\w+", "package main", solution, count=1, flags=re.MULTILINE)
        else:
            solution = "package main\n\n" + solution
        solution = re.sub(r"\bfunc\s+main\s*\(\s*\)", "func solutionMainUnused()", solution)
        wanted = set(re.findall(r'"([^"]+)"', imports)) or {
            w for w in re.split(r"[\s()]+", imports) if w and w != "import"
        }
        missing = sorted(wanted - go_imports(solution))
        block = "import (\n" + "".join(f'\t"{p}"\n' for p in missing) + ")\n" if missing else ""
        solution = re.sub(
            r"^(package main[^\n]*\n)", lambda m: m.group(1) + block, solution, count=1, flags=re.MULTILINE
        )
        imports = ""
    elif language == "haskell":
        pragmas, own_imports, rest = split_haskell(solution)
        rest = re.sub(r"^main(\s*(?:::|=))", r"solutionMainUnused\1", rest, flags=re.MULTILINE)
        solution = "\n".join(x for x in (own_imports, imports, rest) if x)
        imports = pragmas
    elif language == "java":
        solution = re.sub(r"^\s*package\s+[\w.]+\s*;\s*$", "", solution, flags=re.MULTILINE)
        solution = re.sub(
            r"^public\s+((?:final\s+|abstract\s+|sealed\s+)*(?:class|interface|enum|record)\b)",
            r"\1",
            solution,
            flags=re.MULTILINE,
        )
        if re.search(r"\b(?:class|interface|enum|record)\s+Main\b", solution):
            solution = re.sub(r"\bMain\b", "SolutionMain", solution)
    elif language in ("javascript", "typescript"):
        solution = re.sub(r"^export\s+(?:default\s+)?", "", solution, flags=re.MULTILINE)
    elif language == "php":
        imports, main = PHP_TAG.sub("", imports).strip(), PHP_TAG.sub("", main).strip()
        if not solution.lstrip().startswith("<?php"):
            solution = "<?php\n" + solution
        if imports:
            solution = solution.replace("<?php", "<?php\n" + imports + "\n", 1)
            imports = ""
        if solution.rstrip().endswith("?>"):
            main = "<?php\n" + main
    elif language == "matlab":
        solution = "1;\n" + solution
    elif language == "bash":
        solution = "#!/usr/bin/env bash\n" + solution
    if language in ("java", "csharp") and "{CLASS}" in main:
        owner = enclosing_class(solution, function)
        if owner is None:
            solution, owner = wrapped(language, solution), WRAPPER
        main = main.replace("{CLASS}", owner)
    if language in SENTINEL_LINE:
        while SUCCESS_EXIT.search(main):
            main = SUCCESS_EXIT.sub("", main).rstrip()
    parts = [imports, solution, main]
    if language in SENTINEL_LINE:
        parts.append(SENTINEL_LINE[language])
    return "\n\n".join(p for p in parts if p) + "\n"


def passed(language: str, result: dict) -> bool:
    """A run passes when it exits 0 and, for script languages, prints the sentinel."""
    if not result["passed"]:
        return False
    return language not in SENTINEL_LINE or SENTINEL in result.get("stdout_tail", "")
