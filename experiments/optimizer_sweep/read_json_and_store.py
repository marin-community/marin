import ast
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


ROOT = Path(__file__).resolve().parents[2]
OPT_SWEEP_DIR = ROOT / "experiments" / "optimizer_sweep"


def _safe_eval(node: ast.AST) -> Any:
    """Safely evaluate literal-like AST nodes (dicts, lists, numbers, strings, booleans, None).

    Raises ValueError for unsupported nodes.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.List):
        return [_safe_eval(elt) for elt in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        keys = [_safe_eval(k) for k in node.keys]
        values = [_safe_eval(v) for v in node.values]
        return {k: v for k, v in zip(keys, values)}
    # accept unary ops like -1
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        operand = _safe_eval(node.operand)
        if isinstance(operand, (int, float)):
            return -operand if isinstance(node.op, ast.USub) else +operand
    raise ValueError(f"Unsupported AST node for literal eval: {ast.dump(node, include_attributes=False)}")


def _extract_assignments(tree: ast.AST) -> Dict[str, Any]:
    """Extract last assignment to known names: sweep_grids, baseline_config, model_size, target_chinchilla."""
    targets_of_interest = {"sweep_grids", "baseline_config", "model_size", "target_chinchilla"}
    results: Dict[str, Any] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            # we only handle simple Name targets
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in targets_of_interest:
                    try:
                        results[tgt.id] = _safe_eval(node.value)
                    except Exception:
                        # ignore values we cannot safely evaluate
                        pass
    return results


def _extract_template_call(tree: ast.AST) -> Optional[Tuple[Any, Any, Any]]:
    """Returns (model_size, target_chinchilla, optimizer_name) from the template(...) call if parsable.

    Falls back to None if cannot parse.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            is_template = False
            if isinstance(func, ast.Name) and func.id == "template":
                is_template = True
            elif isinstance(func, ast.Attribute) and func.attr == "template":
                is_template = True

            if not is_template:
                continue

            # Expecting positional args: (model_size, target_chinchilla, optimizer_name, ...)
            if len(node.args) >= 3:
                try:
                    model_size = _safe_eval(node.args[0])
                    target_chinchilla = _safe_eval(node.args[1])
                    optimizer_name = _safe_eval(node.args[2])
                    return model_size, target_chinchilla, optimizer_name
                except Exception:
                    continue
    return None


def extract_experiment_info(py_file: Path) -> Optional[Dict[str, Any]]:
    """Parse an experiment Python file and extract desired fields.

    Returns dict with keys: sweep_grids, baseline_config, model_size, target_chinchilla, optimizer_name.
    Returns None if mandatory fields are missing.
    """
    try:
        src = py_file.read_text()
    except Exception:
        return None

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    assigns = _extract_assignments(tree)

    model_size = assigns.get("model_size")
    target_chinchilla = assigns.get("target_chinchilla")
    sweep_grids = assigns.get("sweep_grids")
    baseline_config = assigns.get("baseline_config")
    optimizer_name: Optional[str] = None

    tpl = _extract_template_call(tree)
    if tpl is not None:
        tpl_model_size, tpl_target_chinchilla, tpl_optimizer_name = tpl
        # prefer explicit assignments; fallback to template values
        if model_size is None:
            model_size = tpl_model_size
        if target_chinchilla is None:
            target_chinchilla = tpl_target_chinchilla
        optimizer_name = tpl_optimizer_name

    if not isinstance(optimizer_name, str):
        # attempt to infer optimizer_name from filename (e.g., exp725_adamwsweep_...)
        stem = py_file.stem
        parts = stem.split("_")
        for p in parts:
            if p.endswith("sweep") and len(p) > len("sweep"):
                optimizer_name = p[:-len("sweep")]
                break

    # minimal required fields
    if model_size is None or target_chinchilla is None or optimizer_name is None:
        return None

    # ensure dicts
    if not isinstance(sweep_grids, dict):
        sweep_grids = {}
    if not isinstance(baseline_config, dict):
        baseline_config = {}

    return {
        "sweep_grids": sweep_grids,
        "baseline_config": baseline_config,
        "model_size": model_size,
        "target_chinchilla": target_chinchilla,
        "optimizer_name": optimizer_name,
    }


def _normalize_phase_name(phase_dir: str) -> str:
    # Map PhaseIII variants to PhaseIII, otherwise keep PhaseI/PhaseII
    if phase_dir.startswith("PhaseIII"):
        return "PhaseIII"
    return phase_dir


def _iter_experiment_files() -> Tuple[Path, str]:
    """Yield (file_path, normalized_phase) for experiment files in PhaseI/II/III* directories."""
    for child in OPT_SWEEP_DIR.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("PhaseI") and not child.name.startswith("PhaseII") and not child.name.startswith("PhaseIII"):
            continue
        normalized_phase = _normalize_phase_name(child.name)
        for py_file in child.rglob("*.py"):
            # Skip this script and non-experiment utilities
            if py_file.name == Path(__file__).name:
                continue
            yield py_file, normalized_phase


def _write_json(output_path: Path, data: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def main() -> None:
    baseline_root = OPT_SWEEP_DIR / "baseline_config"
    grids_root = OPT_SWEEP_DIR / "sweep_grids"

    total_files = 0
    written_baseline = 0
    written_grids = 0

    for py_file, phase in _iter_experiment_files():
        total_files += 1
        info = extract_experiment_info(py_file)
        if info is None:
            continue

        optimizer = str(info["optimizer_name"]).lower()
        target = info["target_chinchilla"]
        # Use original filename stem for output
        stem = py_file.stem + ".json"

        baseline_payload = {
            "optimizer_name": info["optimizer_name"],
            "model_size": info["model_size"],
            "target_chinchilla": target,
            "baseline_config": info["baseline_config"],
        }
        grids_payload = {
            "optimizer_name": info["optimizer_name"],
            "model_size": info["model_size"],
            "target_chinchilla": target,
            "sweep_grids": info["sweep_grids"],
        }

        baseline_out = baseline_root / phase / optimizer / str(target) / stem
        grids_out = grids_root / phase / optimizer / str(target) / stem

        _write_json(baseline_out, baseline_payload)
        written_baseline += 1
        _write_json(grids_out, grids_payload)
        written_grids += 1

    print(
        json.dumps(
            {
                "scanned_files": total_files,
                "written_baseline_json": written_baseline,
                "written_sweep_grids_json": written_grids,
                "output_base": str(OPT_SWEEP_DIR),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()


