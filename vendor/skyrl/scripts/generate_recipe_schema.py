"""Generate the author schema from Hydra YAML and generation-only annotations."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import runpy
import subprocess
import tomllib
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "skyrl-train/skyrl_train/config"
OUTPUT_DIR = ROOT / "marinskyrl/recipe_schema"
SIDECAR = ROOT / "scripts/recipe_schema_sidecar.py"


@dataclass(frozen=True)
class SourceDocuments:
    """Hydra source mappings used during author-schema generation."""

    base: dict
    groups: dict
    comments: dict[str, str]


def following_source(value: Any) -> str | None:
    if not isinstance(value, str) or not value.startswith("${") or not value.endswith("}"):
        return None
    source = value[2:-1]
    return source if all(part.isidentifier() for part in source.split(".")) else None


def field_comments(source: Path, mount: str = "") -> dict[str, str]:
    text = source.read_text()
    lines = text.splitlines()
    comments: dict[str, str] = {}

    def visit(node: yaml.Node, prefix: str) -> None:
        if not isinstance(node, yaml.MappingNode):
            return
        for key, value in node.value:
            path = f"{prefix}.{key.value}" if prefix else key.value
            line = key.start_mark.line
            blocks = []
            previous = line - 1
            while previous >= 0:
                candidate = lines[previous]
                if (
                    not candidate.lstrip().startswith("#")
                    or candidate.lstrip().startswith("# @")
                    or len(candidate) - len(candidate.lstrip()) != key.start_mark.column
                ):
                    break
                blocks.insert(0, candidate.lstrip()[1:].strip())
                previous -= 1
            if previous >= 0 and lines[previous].lstrip().startswith("# @package"):
                blocks = []
            # Inline comments begin outside quoted scalar values.
            quote = ""
            escaped = False
            for position, char in enumerate(lines[line]):
                if escaped:
                    escaped = False
                elif char == "\\" and quote == '"':
                    escaped = True
                elif char in "\"'":
                    quote = "" if quote == char else char if not quote else quote
                elif char == "#" and not quote:
                    blocks.append(lines[line][position + 1 :].strip())
                    break
            if blocks:
                comments[path] = "\n".join(blocks)
            visit(value, path)

    visit(yaml.compose(text), mount)
    return comments


def source_documents(config_dir: Path) -> SourceDocuments:
    raw = yaml.safe_load((config_dir / "ppo_base_config.yaml").read_text())
    selections: dict[str, list[str]] = {}
    for selection in raw["defaults"]:
        if isinstance(selection, dict):
            for key in selection:
                selections.setdefault(key.partition("@")[0], []).append(key)
    groups = {}
    comments = field_comments(config_dir / "ppo_base_config.yaml")
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        base = OmegaConf.to_container(compose(config_name="ppo_base_config"), resolve=False)
        for folder in sorted(path for path in config_dir.iterdir() if path.is_dir()):
            for key in selections.get(folder.name, [folder.name]):
                options = {}
                for source in sorted(folder.glob("*.yaml")):
                    document = OmegaConf.to_container(
                        compose(config_name=None, overrides=[f"+{key}={source.stem}"]), resolve=False
                    )
                    # These rule files are data for the algorithm's rule-list field.
                    if folder.name == "off_policy_correction":
                        document = {"trainer": {"algorithm": {"off_policy_correction_rules": document[key]["rules"]}}}
                    if "terminal_bench_config" in document:
                        document["terminal_bench"] = document.pop("terminal_bench_config")
                    options[source.stem] = document
                    _, separator, mount = key.partition("@")
                    if not separator:
                        mount = next(
                            (
                                line.split("@package ", 1)[1]
                                for line in source.read_text().splitlines()
                                if "@package " in line
                            ),
                            folder.name,
                        )
                    if mount == "terminal_bench_config":
                        mount = "terminal_bench"
                    comments.update(field_comments(source, "" if mount == "_global_" else mount))
                if options:
                    groups[key] = options
    return SourceDocuments(base, groups, comments)


def recipe_documents(recipe_dir: Path, config_dir: Path, groups: dict) -> list[dict]:
    """Read authored type observations with the launcher's Hydra inheritance semantics."""
    documents = []
    for path in sorted(recipe_dir.glob("*.yaml")):
        recipe = OmegaConf.load(path)
        if "defaults" in recipe:
            with initialize_config_dir(version_base=None, config_dir=str(recipe_dir.resolve())):
                recipe = compose(
                    config_name=path.name,
                    overrides=[f"hydra.searchpath=[file://{config_dir.resolve()}]"],
                )
        selected = recipe.get("config_groups", {}).get("algorithm_recipe")
        if selected is not None:
            recipe = OmegaConf.merge(groups["algorithm_recipe"][selected], recipe)
        documents.append(OmegaConf.to_container(recipe, resolve=False))
    return documents


def render_sections(
    base: dict,
    sidecar: dict,
    owners: dict,
    comments: dict,
    groups: dict | None = None,
    recipes: list[dict] | None = None,
) -> str:
    types, open_paths, names = sidecar["TYPES"], sidecar["OPEN"], sidecar["NAMES"]
    undeclared = sidecar["UNDECLARED"]
    owned = owners["DERIVED_PATHS"] | owners["LAUNCH_PATHS"]
    class_names: dict[str, str] = {}
    declarations = ast.parse((OUTPUT_DIR / "declared.py").read_text())
    declared_names = [node.name for node in declarations.body if isinstance(node, ast.ClassDef)]
    declared_names += [node.targets[0].id for node in declarations.body if isinstance(node, ast.Assign)]
    used_names: set[str] = {"RecipeSections", "ConfigGroups", *declared_names}
    observed: dict[str, list[Any]] = {}
    schema: dict = {}
    missing = object()

    def inventory(document: dict, target: dict, prefix: str = "") -> None:
        for key, value in document.items():
            path = f"{prefix}.{key}" if prefix else key
            observed.setdefault(path, []).append(value)
            if isinstance(value, dict):
                child = target.setdefault(key, {})
                if isinstance(child, dict):
                    inventory(value, child, path)
            else:
                target.setdefault(key, value)

    inventory(base, schema)
    for options in (groups or {}).values():
        for document in options.values():
            inventory(document, schema)
    for table_name in ("TYPES", "OPEN"):
        stale = set(sidecar[table_name]) - observed.keys()
        if stale:
            raise ValueError(f"{table_name} paths absent from YAML: {', '.join(sorted(stale))}")
    shadowed = undeclared.keys() & observed.keys()
    if shadowed:
        raise ValueError(f"UNDECLARED shadows YAML: {', '.join(sorted(shadowed))}")
    for path, (_, default) in undeclared.items():
        node = schema
        keys = path.split(".")
        for key in keys[:-1]:
            child = node.get(key)
            if child is None:
                child = node[key] = {}
            if not isinstance(child, dict):
                raise ValueError(f"UNDECLARED parent is not a mapping: {path}")
            node = child
        node[keys[-1]] = default

    declared_paths = {
        ".".join(path.split(".")[:index]) for path in undeclared for index in range(1, len(path.split(".")) + 1)
    }
    stale_names = set(names) - observed.keys() - declared_paths
    if stale_names:
        raise ValueError(f"NAMES paths absent from schema: {', '.join(sorted(stale_names))}")

    def observe_recipe(document: dict, prefix: str = "") -> None:
        for key, value in document.items():
            path = f"{prefix}.{key}" if prefix else key
            if path in observed:
                observed[path].append(value)
            if isinstance(value, dict):
                observe_recipe(value, path)

    for document in recipes or ():
        observe_recipe(document)

    def lookup(path: str) -> Any:
        node = base
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                return missing
            node = node[key]
        return node

    def scalar(value: Any, path: str, chain: tuple[str, ...] = ()) -> str:
        if path in chain:
            raise ValueError(f"following-default cycle: {' -> '.join((*chain, path))}")
        if source := following_source(value):
            source_value = lookup(source)
            if source_value is missing:
                raise ValueError(f"following-default source absent: {path} -> {source}")
            annotation = types.get(path, scalar(source_value, source, (*chain, path)))
            return annotation.replace(" | None", "")
        if path in types:
            return types[path]
        if path in open_paths:
            return "OpenMap | None" if value is None else "OpenMap"
        if isinstance(value, bool):
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            # Hydra retains authored integer literals for numeric fields too.
            return "int | float"
        if isinstance(value, str):
            return "str"
        if isinstance(value, list):
            if not value:
                raise ValueError(f"{path}: add a TYPES entry")
            annotations = sorted({scalar(item, path) for item in value})
            return f"tuple[{' | '.join(annotations)}, ...]"
        if isinstance(value, dict) and not value:
            return "OpenMap"
        if value is None:
            return "None"
        raise ValueError(f"{path}: add a TYPES entry")

    def annotation_for(path: str, value: Any) -> str:
        if path in types or path in open_paths or following_source(value):
            return scalar(value, path)
        alternatives = {scalar(item, path) for item in observed.get(path, [value])}
        if "int" in alternatives and "int | float" in alternatives:
            alternatives.remove("int")
        return " | ".join(sorted(alternatives))

    def camel(text: str) -> str:
        return "".join(piece.capitalize() for piece in text.replace(".", "_").split("_"))

    mapping_paths = []

    def collect_mappings(path: str, mapping: dict) -> None:
        mapping_paths.append(path)
        for key, value in mapping.items():
            child = f"{path}.{key}"
            annotation = undeclared[child][0] if child in undeclared else types.get(child)
            if isinstance(value, dict) and value and child not in owned | open_paths and annotation != "NumberMap":
                collect_mappings(child, value)

    for key, value in schema.items():
        if key not in undeclared and types.get(key) not in ("OpenMap", "OpenMap | None"):
            collect_mappings(key, value)
    leaf_counts = Counter(path.rsplit(".", 1)[-1] for path in mapping_paths)

    def class_name(path: str) -> str:
        if path not in class_names:
            leaf = path.rsplit(".", 1)[-1]
            if path not in names and leaf_counts[leaf] > 1:
                raise ValueError(f"NAMES needs {path}: {leaf!r} names multiple mappings")
            name = names.get(path, camel(leaf))
            if name in used_names:
                raise ValueError(f"duplicate class name at {path}: {name}; add a NAMES entry")
            class_names[path] = name
            used_names.add(name)
        return class_names[path]

    emitted: list[str] = []

    def emit(path: str, mapping: dict) -> str:
        name = class_name(path)
        fields = []
        for key, value in mapping.items():
            field_path = f"{path}.{key}"
            if field_path in owned:
                continue
            default = undeclared[field_path][1] if field_path in undeclared else lookup(field_path)
            declared_annotation = undeclared[field_path][0] if field_path in undeclared else types.get(field_path)
            if (
                isinstance(value, dict)
                and value
                and field_path not in open_paths
                and declared_annotation != "NumberMap"
            ):
                section_name = emit(field_path, value)
                annotation = types.get(field_path, section_name)
                if default is None:
                    if field_path not in types:
                        annotation += " | None"
                    expression = "field(None)"
                elif default is missing or default is Ellipsis:
                    expression = "unset_field()"
                else:
                    expression = f"Field(default_factory={section_name})"
                if field_path in undeclared and undeclared[field_path][0] != annotation:
                    raise ValueError(f"UNDECLARED mapping annotation must be {annotation!r}: {field_path}")
            elif field_path in undeclared:
                annotation, default = undeclared[field_path]
                expression = "unset_field()" if default is Ellipsis else f"field({default!r})"
            else:
                annotation = annotation_for(field_path, value)
                if isinstance(value, dict) and not value and field_path not in types:
                    annotation = "OpenMap"
                if default is missing or following_source(default):
                    expression = "unset_field()"
                else:
                    expression = f"field({default!r})"
            fields.append(f"    {key}: {annotation} = {expression}")
            if field_path in comments:
                fields.append(f"    {json.dumps(comments[field_path])}")
        body = "\n".join(fields) if fields else "    pass"
        emitted.append(f"class {name}(Section):\n{body}\n")
        return name

    root_fields = []
    for key, value in schema.items():
        if key in undeclared:
            annotation, default = undeclared[key]
            expression = "unset_field()" if default is Ellipsis else f"field({default!r})"
        elif types.get(key) in ("OpenMap", "OpenMap | None"):
            annotation = types[key]
            expression = f"field({base[key]!r})" if key in base else "unset_field()"
        else:
            section_name = emit(key, value)
            annotation = types.get(key, section_name)
            expression = f"Field(default_factory={section_name})" if key in base else "unset_field()"
        root_fields.append(f"    {key}: {annotation} = {expression}")
    emitted.append("class RecipeSections(Section):\n" + "\n".join(root_fields) + "\n")
    group_fields = []
    for group in ("terminal_bench_config", "algorithm_recipe"):
        options = (groups or {}).get(group)
        if options:
            choices = ", ".join(repr(option) for option in sorted(options))
            group_fields.append(f"    {group}: Literal[{choices}] = unset_field()")
    emitted.append("class ConfigGroups(Section):\n" + ("\n".join(group_fields) or "    pass") + "\n")
    emitted.append(
        "_namespace = {**vars(declared), **globals()}\n"
        "for _section in [value for value in _namespace.values() if isinstance(value, type) and issubclass(value, Section)]:\n"
        "    _section.model_rebuild(_types_namespace=_namespace)\ndel _section, _namespace\n"
    )
    referenced = {node.id for node in ast.walk(ast.parse("\n\n".join(emitted))) if isinstance(node, ast.Name)}
    imports = ", ".join(name for name in declared_names if name in referenced)
    header = f'''"""Generated by scripts/generate_recipe_schema.py; edit YAML or scripts/recipe_schema_sidecar.py."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, NonNegativeInt, PositiveInt

from . import declared
from .declared import {imports}
from .model import NumberMap, OpenMap, Section, SectionMap, field, unset_field

'''
    return header + "\n\n".join(emitted)


def generate(
    config_dir: Path = CONFIG_DIR, output_dir: Path = OUTPUT_DIR, recipe_dir: Path = ROOT / "cloud/iris/configs"
) -> dict[str, str]:
    """Return repository-formatted author code; the CLI writes or checks it."""
    sidecar = runpy.run_path(str(SIDECAR))
    owners = runpy.run_path(str(output_dir / "ownership.py"))
    sources = source_documents(config_dir)
    sections = render_sections(
        sources.base,
        sidecar,
        owners,
        sources.comments,
        sources.groups,
        recipe_documents(recipe_dir, config_dir, sources.groups),
    )
    ruff_version = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["marin-style"]["ruff_version"]
    formatted = subprocess.run(
        [
            "uv",
            "tool",
            "run",
            "--from",
            f"ruff=={ruff_version}",
            "ruff",
            "format",
            "--config",
            str(ROOT / "pyproject.toml"),
            "--stdin-filename",
            str(output_dir / "sections.py"),
            "-",
        ],
        input=sections,
        text=True,
        capture_output=True,
        check=True,
        cwd=ROOT,
    ).stdout
    generated = {"sections.py": formatted}
    public = output_dir / "__init__.py"
    if public.exists():
        exports = []
        for node in ast.parse(formatted).body:
            if isinstance(node, ast.ClassDef) and node.name != "RecipeSections":
                exports.append(node.name)
        declarations = "\n".join(
            line for line in public.read_text().splitlines() if not line.startswith("from .sections import ")
        ).rstrip()
        generated["__init__.py"] = (
            declarations + "\n" + "".join(f"from .sections import {name} as {name}\n" for name in sorted(exports))
        )
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--config-dir", type=Path, default=CONFIG_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--recipe-dir", type=Path, default=ROOT / "cloud/iris/configs")
    args = parser.parse_args()
    changed = []
    for name, text in generate(args.config_dir, args.output_dir, args.recipe_dir).items():
        path = args.output_dir / name
        if args.check:
            if not path.exists() or path.read_text() != text:
                changed.append(name)
        else:
            path.write_text(text)
    if changed:
        parser.exit(
            1,
            f"recipe schema needs regeneration: {', '.join(changed)}; run uv run python scripts/generate_recipe_schema.py\n",
        )


if __name__ == "__main__":
    main()
