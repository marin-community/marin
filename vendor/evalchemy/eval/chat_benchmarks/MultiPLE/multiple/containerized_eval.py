"""
NOTE: Nothing containerized about this any more. This is just a helper
for problem_evaluator.py.
"""

from pathlib import Path
from .eval_adb import eval_script as eval_adb_eval_script
from .eval_ruby import eval_script as eval_ruby_eval_script
from .eval_lua import eval_script as eval_lua_eval_script
from .eval_python import eval_script as eval_python_eval_script
from .eval_rust import eval_script as eval_rust_eval_script
from .eval_julia import eval_script as eval_julia_eval_script
from .eval_java import eval_script as eval_java_eval_script
from .eval_racket import eval_script as eval_racket_eval_script
from .eval_javascript import eval_script as eval_javascript_eval_script
from .eval_swift import eval_script as eval_swift_eval_script
from .eval_cpp import eval_script as eval_cpp_eval_script
from .eval_php import eval_script as eval_php_eval_script
from .eval_dlang import eval_script as eval_dlang_eval_script
from .eval_julia import eval_script as eval_julia_eval_script
from .eval_r import eval_script as eval_r_eval_script
from .eval_fs import eval_script as eval_fs_eval_script
from .eval_ocaml import eval_script as eval_ocaml_eval_script
from .eval_matlab import eval_script as eval_matlab_eval_script
from .eval_hs import eval_script as eval_hs_eval_script
from .eval_elixir import eval_script as eval_elixir_eval_script
from .eval_clj import eval_script as eval_clj_eval_script
from .eval_v import eval_script as eval_v_eval_script
from .eval_lean import eval_script as eval_lean_eval_script
from .eval_dart import eval_script as eval_dart_eval_script
from .eval_ts import eval_script as eval_ts_eval_script
from .eval_go import eval_script as eval_go_eval_script
from .eval_sh import eval_script as eval_sh_eval_script
from .eval_cs import eval_script as eval_cs_eval_script
import tempfile
import os


EVALUATORS = {
    "ada": (eval_adb_eval_script, ".adb"),
    "rb": (eval_ruby_eval_script, ".rb"),
    "lua": (eval_lua_eval_script, ".lua"),
    # "python": (eval_python_eval_script, ".py"),
    # "py": (eval_python_eval_script, ".py"),
    # "notypes.py": (eval_python_eval_script, ".py"),
    "julia": (eval_julia_eval_script, ".jl"),
    "java": (eval_java_eval_script, ".java"),
    "rust": (eval_rust_eval_script, ".rs"),
    "rs": (eval_rust_eval_script, ".rs"),
    "swift": (eval_swift_eval_script, ".swift"),
    "racket": (eval_racket_eval_script, ".rkt"),
    "rkt": (eval_racket_eval_script, ".rkt"),
    "javascript": (eval_javascript_eval_script, ".js"),
    "js": (eval_javascript_eval_script, ".js"),
    "cpp": (eval_cpp_eval_script, ".cpp"),
    "php": (eval_php_eval_script, ".php"),
    "humaneval_to_dlang.py": (eval_dlang_eval_script, ".d"),
    "d": (eval_dlang_eval_script, ".d"),
    "r": (eval_r_eval_script, ".r"),
    "humaneval_to_r.py": (eval_r_eval_script, ".r"),
    "jl": (eval_julia_eval_script, ".jl"),
    "fs": (eval_fs_eval_script, ".fsx"),
    "ml": (eval_ocaml_eval_script, ".ml"),
    "m": (eval_matlab_eval_script, ".m"),
    "hs": (eval_hs_eval_script, ".hs"),
    "elixir": (eval_elixir_eval_script, ".exs"),
    "clj": (eval_clj_eval_script, ".clj"),
    "coq": (eval_v_eval_script, ".v"),
    "lean": (eval_lean_eval_script, ".lean"),
    "dart": (eval_dart_eval_script, ".dart"),
    "ts": (eval_ts_eval_script, ".ts"),
    "go": (eval_go_eval_script, ".go"),
    "sh": (eval_sh_eval_script, ".sh"),
    "cs": (eval_cs_eval_script, ".cs"),
}


def eval_string_script(language: str, program: str, tmpdir):
    if language in EVALUATORS:
        (eval_script, file_ext) = EVALUATORS[language]
    else:
        eval_module = __import__(f"eval_{language}" if language != "go_test.go" else "eval_go")
        eval_script = eval_module_eval_script
        file_ext = f".{language}" if language != "go_test.go" else "_test.go"

    with tempfile.NamedTemporaryFile(
        suffix=file_ext,
        delete=True,
    ) as f:
        f.write(program.encode("utf-8"))
        f.flush()
        result = eval_script(Path(f.name))
        # Only save the first 2K of output from the running program. Any futher
        # output is very likely an exceptionally long stack trace or a long
        # series of prints.
        if type(result["stdout"]) == bytes:
            result["stdout"] = result["stdout"].decode("utf-8", errors="ignore")
        if result["stdout"] is None:
            result["stdout"] = ""
        if result["stderr"] is None:
            result["stderr"] = ""
        if type(result["stderr"]) == bytes:
            result["stderr"] = result["stderr"].decode("utf-8", errors="ignore")
        assert type(result["stdout"]) == str
        assert type(result["stderr"]) == str
        return {
            "program": program,
            "stdout": result["stdout"].replace("!!int", "")[:2048],
            "stderr": result["stderr"][:2048],
            "exit_code": result["exit_code"],
            "status": result["status"],
        }
