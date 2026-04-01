# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Filter Common Pile stackv2 raw data to useful programming language extensions.

Extensions chosen from top ~50 languages by GitHub/Stack Overflow usage (2024),
excluding noisy formats (data files, binaries, configs, lock files, etc.).

Example Usage:
    uv run python experiments/common_pile/filter_stackv2_code.py
"""

from experiments.defaults import default_tokenize
from experiments.marin_models import marin_tokenizer
from marin.datakit.download.common_pile import download_common_pile_filtered_step
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.transform.common_pile.filter_by_extension import (
    FilterByMetadataExtensionConfig,
    filter_dataset_by_metadata_extension,
)

stackv2 = download_common_pile_filtered_step("stackv2").as_executor_step()

# fmt: off
STACKV2_CODE_EXTENSIONS = (
    # Python
    ".py", ".pyw", ".pyi",
    # C / C++
    ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hxx", ".hh",
    # C#
    ".cs",
    # SQL
    ".sql",
    # Java
    ".java",
    # PHP
    ".php",
    # Rust
    ".rs",
    # JavaScript
    ".js", ".jsx", ".mjs",
    # TypeScript
    ".ts", ".tsx",
    # Go
    ".go",
    # Ruby
    ".rb", ".erb",
    # Markdown
    ".md",
    # Swift
    ".swift",
    # Shell
    ".sh", ".bash", ".zsh",
    # Kotlin
    ".kt", ".kts",
    # Scala
    ".scala", ".sc",
    # R
    ".r",
    # Lua
    ".lua",
    # Perl
    ".pl", ".pm",
    # Dart
    ".dart",
    # TeX / LaTeX
    ".tex", ".sty", ".cls",
    # Web
    ".html", ".css",
    # Haskell
    ".hs", ".lhs",
    # PowerShell
    ".ps1",
    # Assembly
    ".asm", ".s",
    # Objective-C / MATLAB
    ".m", ".mm",
    # VBA
    ".bas", ".vba",
    # Groovy
    ".groovy",
    # Elixir
    ".ex", ".exs",
    # GDScript
    ".gd",
    # Delphi / Pascal
    ".pas", ".dpr",
    # Lisp / Emacs Lisp
    ".lisp", ".cl", ".el",
    # Clojure
    ".clj", ".cljs", ".cljc",
    # Julia
    ".jl",
    # Zig
    ".zig",
    # Fortran
    ".f", ".f90", ".f95", ".f03",
    # Erlang
    ".erl", ".hrl",
    # F#
    ".fs", ".fsi", ".fsx",
    # OCaml
    ".ml", ".mli",
    # Nim
    ".nim",
    # HDL (Verilog / VHDL)
    ".v", ".sv", ".vhd",
)
# fmt: on

stackv2_code_filtered = ExecutorStep(
    name="documents/common_pile/stackv2_code_filtered",
    fn=filter_dataset_by_metadata_extension,
    config=FilterByMetadataExtensionConfig(
        input_path=stackv2,
        output_path=this_output_path(),
        allowed_extensions=STACKV2_CODE_EXTENSIONS,
        input_glob="documents/*.jsonl.gz",
    ),
)

stackv2_code_tokenized = default_tokenize(
    name="common_pile/stackv2_code",
    dataset=stackv2_code_filtered,
    tokenizer=marin_tokenizer,
)

if __name__ == "__main__":
    executor_main(steps=[stackv2_code_filtered, stackv2_code_tokenized])
