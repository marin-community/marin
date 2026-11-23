# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Embed self-contained grading into Micro210 TaskSpecs.

Every script-graded Micro210 row (the ``config.json`` family) gains a vendored grading
library as inline verifier resources: the row's ``grade.py`` becomes a standalone driver
that imports only the embedded ``grader_lib`` tree, and the row's ``grader.environment``
drops the ambient ``runtime-v4`` evalchemy and lm_eval wheels. Harbor-trial rows (no
``config.json``) already carry their whole task package inline and pass through unchanged.
Each transformed row records grader provenance in its labels, and the vendored library is
verified against the committed manifest on every use.
"""

import base64
import hashlib
import json
from dataclasses import replace as dataclass_replace
from pathlib import Path

from taskcompendium.models import InlineFile, TaskResource, TaskSpec

GRADER_LIBRARY = Path(__file__).parent / "micro210_grader"
SPEC_GRADE_DRIVER = Path(__file__).parent / "micro210_spec_grade.py"
GRADER_LIB_PREFIX = "grader_lib/"
AMBIENT_PACKAGE_PREFIXES = ("evalchemy[", "lm-eval ")
PROVENANCE_LABEL = "micro210_grader_provenance"

# The IFBench checkers load their tokenizers from the bundled NLTK data directory inside
# the vendored checker package; other benchmarks never import it.
NLTK_DATA_MARKER = "eval/chat_benchmarks/IFBench/.nltk_data/"

# MBPP+ rows embed an evalplus-generated ``is_floats`` guard in their test program; it
# only recognized flat float lists, so complex-returning tasks (polar_rect) fell back to
# exact ``==`` and a 1-ULP libm difference zeroed the canonical answer (#12159). The
# patched guard keeps the same flat-float verdicts and extends the float tolerance to
# complex numbers and nested numeric structures.
MBPP_FLOAT_TOLERANCE_ORIGINAL = """def is_floats(x) -> bool:
    # check if it is float; List[float]; Tuple[float]
    if isinstance(x, float):
        return True
    if isinstance(x, (list, tuple)):
        return all(isinstance(i, float) for i in x)
    if isinstance(x, np.ndarray):
        return x.dtype == np.float64 or x.dtype == np.float32
    return False"""

MBPP_FLOAT_TOLERANCE_PATCHED = """def is_floats(x) -> bool:
    # float; List[float]; Tuple[float]; nested numeric structures and complex numbers,
    # so the assertion's float tolerance covers complex-returning tasks (#12159)
    if isinstance(x, bool):
        return False
    if isinstance(x, (float, complex)):
        return True
    if isinstance(x, (list, tuple)):
        return all(is_floats(item) for item in x)
    if isinstance(x, np.ndarray):
        return x.dtype == np.float64 or x.dtype == np.float32
    return False"""

MBPP_ASSERTION_ORIGINAL = """def assertion(out, exp, atol):
    if atol == 0 and is_floats(exp):
        atol = 1e-6
    if out != exp and atol != 0:
        assert np.allclose(out, exp, rtol=1e-07, atol=atol)
    else:
        assert out == exp, f"out: {out}, exp: {exp}\""""

MBPP_ASSERTION_PATCHED = """def assertion(out, exp, atol):
    if atol == 0 and is_floats(exp):
        atol = 1e-6
    if out == exp:
        return
    if atol == 0:
        assert False, f"out: {out}, exp: {exp}"
    assert _close(out, exp, atol), f"out: {out}, exp: {exp}"


def _close(out, exp, atol):
    # Structure-aware float comparison: np.allclose cannot build an array from ragged
    # numeric structures such as (tuple, complex) task outputs (#12159).
    if isinstance(out, (list, tuple)) and isinstance(exp, (list, tuple)):
        return len(out) == len(exp) and all(
            _close(out_item, exp_item, atol) for out_item, exp_item in zip(out, exp)
        )
    return bool(np.all(np.isclose(out, exp, rtol=1e-07, atol=atol)))"""

MBPP_FLOAT_TOLERANCE_PATCH = {
    "patch": "mbpp-float-tolerance-12159",
    "note": (
        "the embedded MBPP+ test program's is_floats recognizes complex numbers and nested "
        "numeric structures, and its assertion compares them with a structure-aware 1e-6 "
        "tolerance instead of exact equality or ragged np.allclose; flat-float verdicts are "
        "unchanged"
    ),
    "issue": 12159,
}

# Harbor task packages pin npm:pi-mcp-adapter@7.0.0 in their Pi runtime, but the Pi agent
# loads the image-provided pi-mcp-extension and none of these tasks configure MCP servers,
# so the adapter is installed and never used (#12122). The patched package manifests drop
# the unused pin; the pi-coding-agent 0.87.0 pin the images assert is untouched.
HARBOR_PI_ADAPTER_PATCH = {
    "patch": "harbor-pi-adapter-pin-removal-12122",
    "note": (
        "environment/long-loop-pi/package.json and package-lock.json drop the unused "
        "pi-mcp-adapter@7.0.0 dependency; the agent loads the image-provided pi-mcp-extension "
        "and no Micro210 task configures MCP servers"
    ),
    "issue": 12122,
}


class GraderLibrary:
    """The committed vendored grading library, manifest-verified on load."""

    def __init__(self) -> None:
        manifest_path = GRADER_LIBRARY / "MANIFEST.json"
        self.manifest = json.loads(manifest_path.read_text())
        self.files: dict[str, bytes] = {}
        for entry in self.manifest["files"]:
            payload = (GRADER_LIBRARY / entry["path"]).read_bytes()
            if hashlib.sha256(payload).hexdigest() != entry["sha256"] or len(payload) != entry["bytes"]:
                raise ValueError(f"Vendored grader library file does not match the manifest: {entry['path']}")
            self.files[entry["path"]] = payload
        self.manifest_bytes = manifest_path.read_bytes()

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(self.manifest_bytes).hexdigest()

    def row_files(self, benchmark: str) -> dict[str, bytes]:
        """The library files one row embeds: everything, minus NLTK data off IFBench rows."""
        return {
            path: payload
            for path, payload in self.files.items()
            if benchmark == "ifbench" or not path.startswith(NLTK_DATA_MARKER)
        }


def inline_resource(path: str, payload: bytes) -> TaskResource:
    return TaskResource(path=path, source=InlineFile(content_base64=base64.b64encode(payload).decode()))


def driver_bytes() -> bytes:
    return SPEC_GRADE_DRIVER.read_bytes()


def driver_sha256() -> str:
    return hashlib.sha256(driver_bytes()).hexdigest()


def dropped_packages(packages: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Split a package_spec list into kept entries and the ambient wheels grading no longer needs."""
    kept, dropped = [], []
    for package in packages:
        (dropped if package.startswith(AMBIENT_PACKAGE_PREFIXES) else kept).append(package)
    return kept, dropped


def provenance_label(
    library: GraderLibrary, benchmark: str, dropped: list[str], data_patches: tuple[dict, ...] = ()
) -> str:
    return json.dumps(
        {
            "driver_sha256": driver_sha256(),
            "library_manifest_sha256": library.manifest_sha256,
            "source_wheels": library.manifest["source_wheels"],
            "patches": [
                {"path": entry["path"], "patch": entry["patch"], "note": entry["note"]}
                for entry in library.manifest["files"]
                if "patch" in entry
            ],
            "row_data_patches": [dict(patch) for patch in data_patches],
            "bundled_nltk_data": benchmark == "ifbench",
            "removed_ambient_packages": dropped,
        },
        sort_keys=True,
    )


def drop_unused_pi_adapter_pin(task: TaskSpec) -> TaskSpec:
    """Remove the unused pi-mcp-adapter@7.0.0 pin from one Harbor task package (#12122)."""
    files = {resource.path: resource for resource in task.resources.verifier}
    package_path = "environment/long-loop-pi/package.json"
    lock_path = "environment/long-loop-pi/package-lock.json"
    if package_path not in files or lock_path not in files:
        return task
    package = json.loads(base64.b64decode(files[package_path].source.content_base64))
    lock = json.loads(base64.b64decode(files[lock_path].source.content_base64))
    if package.get("dependencies", {}).get("pi-mcp-adapter") != "7.0.0":
        return task
    package["dependencies"].pop("pi-mcp-adapter")
    lock["packages"][""].get("dependencies", {}).pop("pi-mcp-adapter", None)
    removed = lock["packages"].pop("node_modules/pi-mcp-adapter", None)
    if removed is None:
        raise ValueError(f"Harbor task package {task.id} pins pi-mcp-adapter in package.json but not in its lock")
    resources = (
        *(resource for resource in task.resources.verifier if resource.path not in (package_path, lock_path)),
        # package.json is npm-compact, the lock is npm-pretty; keep each file's own shape.
        inline_resource(package_path, (json.dumps(package, separators=(",", ":")) + "\n").encode()),
        inline_resource(lock_path, (json.dumps(lock, indent=2) + "\n").encode()),
    )
    return task.model_copy(update={"resources": task.resources.model_copy(update={"verifier": resources})})


def tolerant_mbpp_float_comparison(task: TaskSpec) -> TaskSpec:
    """Apply the MBPP+ float-tolerance patch to one row's embedded test program (#12159)."""
    files = {resource.path: resource for resource in task.resources.verifier}
    if "config.json" not in files:
        return task
    config = json.loads(base64.b64decode(files["config.json"].source.content_base64))
    spec = config["spec"]
    patched = 0
    for holder, key in ((spec["grader"].get("reference", {}).get("problem", {}), "test"), (spec, "target")):
        value = holder.get(key)
        if isinstance(value, str) and value.count(MBPP_FLOAT_TOLERANCE_ORIGINAL) == 1:
            holder[key] = value.replace(MBPP_FLOAT_TOLERANCE_ORIGINAL, MBPP_FLOAT_TOLERANCE_PATCHED)
        if isinstance(value, str) and value.count(MBPP_ASSERTION_ORIGINAL) == 1:
            holder[key] = holder[key].replace(MBPP_ASSERTION_ORIGINAL, MBPP_ASSERTION_PATCHED)
            patched += 1
    if not patched:
        raise ValueError(f"MBPP+ row {task.id} does not carry the expected is_floats guard")
    content = json.dumps(config).encode()
    resources = (
        *(resource for resource in task.resources.verifier if resource.path != "config.json"),
        inline_resource("config.json", content),
    )
    return task.model_copy(update={"resources": task.resources.model_copy(update={"verifier": resources})})


def embed_spec_grader(
    task: TaskSpec, *, benchmark: str, library: GraderLibrary, data_patches: tuple[dict, ...] = ()
) -> TaskSpec:
    """Replace one row's wheel-importing grade.py with the embedded, self-contained grader."""
    files = {resource.path: resource for resource in task.resources.verifier}
    if "config.json" not in files or "grade.py" not in files:
        # Harbor-trial rows carry their task package inline and grade through the runtime;
        # they still record any row-data patches applied to that package.
        if not data_patches:
            return task
        labels = {
            **task.labels,
            PROVENANCE_LABEL: json.dumps({"row_data_patches": [dict(patch) for patch in data_patches]}, sort_keys=True),
        }
        return task.model_copy(update={"labels": labels})
    grader = task.grader
    if not hasattr(grader, "environment") or grader.environment.package_spec is None:
        raise ValueError(f"Script-graded row {task.id} has no pinned verifier environment")
    kept, dropped = dropped_packages(grader.environment.package_spec.pypi_packages or ())
    if not dropped:
        raise ValueError(f"Row {task.id} has no ambient grading wheel to drop; refusing a no-op embed")

    resources = [resource for resource in task.resources.verifier if resource.path not in ("grade.py",)] + [
        inline_resource("grade.py", driver_bytes())
    ]
    resources += [
        inline_resource(f"{GRADER_LIB_PREFIX}{path}", payload) for path, payload in library.row_files(benchmark).items()
    ]

    environment = grader.environment.model_copy(
        update={"package_spec": dataclass_replace(grader.environment.package_spec, pypi_packages=tuple(kept))}
    )
    grader = grader.model_copy(update={"environment": environment})
    labels = {
        **task.labels,
        PROVENANCE_LABEL: provenance_label(library, benchmark, dropped, data_patches),
    }
    return task.model_copy(
        update={
            "grader": grader,
            "resources": task.resources.model_copy(update={"verifier": tuple(resources)}),
            "labels": labels,
        }
    )
