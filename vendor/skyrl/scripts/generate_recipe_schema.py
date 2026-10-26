"""Generate the author schema from Hydra YAML and generation-only annotations."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
import runpy
import subprocess
import textwrap
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import yaml

from skyrlrecipe.documents import MISSING, get_path, set_path
from skyrlrecipe.ownership import DERIVED_PATHS, LAUNCH_PATHS

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "skyrl-train/skyrl_train/config"
WORKSPACE = ROOT.parents[1]
OUTPUT_DIR = WORKSPACE / "lib/skyrlrecipe/src/skyrlrecipe"
SIDECAR = ROOT / "scripts/recipe_schema_sidecar.py"
INLINE_COMMENT = re.compile(r"""^(?:[^"\'#]|"(?:\\.|[^"\\])*"|'[^']*')*#(.*)$""")


@dataclass(frozen=True)
class SourceDocuments:
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
            for previous in range(line - 1, -1, -1):
                candidate = lines[previous]
                if not candidate.startswith(" " * key.start_mark.column + "#") or candidate.lstrip().startswith("# @"):
                    if candidate.lstrip().startswith("# @package"):
                        blocks = []
                    break
                blocks.insert(0, candidate.lstrip()[1:].strip())
            if inline := INLINE_COMMENT.match(lines[line]):
                blocks.append(inline.group(1).strip())
            if blocks:
                comments[path] = "\n".join(blocks)
            visit(value, path)

    visit(yaml.compose(text), mount)
    return comments


def source_documents(config_dir: Path) -> SourceDocuments:
    selections: dict[str, list[str]] = {}
    for selection in yaml.safe_load((config_dir / "ppo_base_config.yaml").read_text())["defaults"]:
        for key in selection if isinstance(selection, dict) else ():
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
                        package = re.search(r"@package (.*)", source.read_text())
                        mount = package[1] if package else folder.name
                    mount = {"terminal_bench_config": "terminal_bench", "_global_": ""}.get(mount, mount)
                    comments.update(field_comments(source, mount))
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


def walk_fields(document: dict, prefix: str = ""):
    for key, value in document.items():
        path = f"{prefix}.{key}".lstrip(".")
        yield path, value
        if isinstance(value, dict):
            yield from walk_fields(value, path)


def inventory(document: dict, tree: dict, observed: dict, prefix: str = "") -> None:
    for key, value in document.items():
        path = f"{prefix}.{key}".lstrip(".")
        observed.setdefault(path, []).append(value)
        child = tree.setdefault(key, {} if isinstance(value, dict) else value)
        if isinstance(value, dict) and isinstance(child, dict):
            inventory(value, child, observed, path)


def schema_inventory(base: dict, sidecar: dict, groups: dict, recipes: list[dict]) -> tuple[dict, dict]:
    tree, observed = {}, {}
    for document in [base, *(value for options in groups.values() for value in options.values())]:
        inventory(document, tree, observed)
    declared = sidecar["UNDECLARED"]
    prefixes = {".".join(path.split(".")[:index]) for path in declared for index in range(1, len(path.split(".")) + 1)}
    invalid = {
        **{f"{key} paths absent from YAML": set(sidecar[key]) - observed.keys() for key in ("TYPES", "OPEN")},
        "UNDECLARED shadows YAML": declared.keys() & observed.keys(),
        "NAMES paths absent from schema": set(sidecar["NAMES"]) - observed.keys() - prefixes,
    }
    for label, paths in invalid.items():
        if paths:
            raise ValueError(f"{label}: {', '.join(sorted(paths))}")
    for path, (_, default) in declared.items():
        for index in range(1, len(path.split("."))):
            parent = get_path(tree, ".".join(path.split(".")[:index]))
            if parent is not MISSING and parent is not None and not isinstance(parent, dict):
                raise ValueError(f"UNDECLARED parent is not a mapping: {path}")
        set_path(tree, path, default)
    for document in recipes:
        for path, value in walk_fields(document):
            if path in observed:
                observed[path].append(value)
    return tree, observed


def generated_module(body: str, description: str, imports: str = "") -> str:
    return f'''# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""{description}"""

from __future__ import annotations
from typing import Annotated, Literal
from pydantic import Field, NonNegativeInt, PositiveInt
from .model import NumberMap, OpenMap, Section, SectionMap, field, unset_field
{imports}
{body}
'''


@dataclass(frozen=True)
class RenderSchema:
    source: SourceDocuments
    observed: dict
    names: dict
    sidecar: dict


def section_paths(mapping: dict, sidecar: dict, owned: frozenset, prefix: str = ""):
    for key, value in mapping.items():
        path = f"{prefix}.{key}".lstrip(".")
        annotation = sidecar["UNDECLARED"][path][0] if path in sidecar["UNDECLARED"] else sidecar["TYPES"].get(path)
        section = (
            isinstance(value, dict) and value and path not in owned | sidecar["OPEN"] and annotation != "NumberMap"
            if prefix
            else path not in sidecar["UNDECLARED"] and annotation not in ("OpenMap", "OpenMap | None")
        )
        if section:
            yield from section_paths(value, sidecar, owned, path)
    yield prefix, mapping


def class_names(paths: list, overrides: dict, declared: list[str]) -> dict[str, str]:
    counts = Counter(path.rsplit(".", 1)[-1] for path, _ in paths if path)
    names = {"": "RecipeSections"}
    used = {"RecipeSections", "ConfigGroups", *declared}
    for path, _ in paths[:-1]:
        leaf = path.rsplit(".", 1)[-1]
        if counts[leaf] > 1 and path not in overrides:
            raise ValueError(f"NAMES needs {path}: {leaf!r} names multiple mappings")
        name = overrides.get(path, "".join(piece.capitalize() for piece in leaf.split("_")))
        if name in used:
            raise ValueError(f"duplicate class name at {path}: {name}; add a NAMES entry")
        names[path] = name
        used.add(name)
    return names


def scalar_type(schema: RenderSchema, value: Any, path: str, chain: tuple[str, ...] = ()) -> str:
    if path in chain:
        raise ValueError(f"following-default cycle: {' -> '.join((*chain, path))}")
    if source := following_source(value):
        source_value = get_path(schema.source.base, source)
        if source_value is MISSING:
            raise ValueError(f"following-default source absent: {path} -> {source}")
        annotation = schema.sidecar["TYPES"].get(path, scalar_type(schema, source_value, source, (*chain, path)))
        return annotation.replace(" | None", "")
    if path in schema.sidecar["TYPES"]:
        return schema.sidecar["TYPES"][path]
    if path in schema.sidecar["OPEN"]:
        return "OpenMap | None" if value is None else "OpenMap"
    for kind, annotation in ((bool, "bool"), (int, "int"), (float, "int | float"), (str, "str"), (type(None), "None")):
        if isinstance(value, kind):
            return annotation
    if isinstance(value, list) and value:
        return f"tuple[{' | '.join(sorted({scalar_type(schema, item, path) for item in value}))}, ...]"
    if isinstance(value, dict) and not value:
        return "OpenMap"
    raise ValueError(f"{path}: add a TYPES entry")


def field_annotation(schema: RenderSchema, path: str, value: Any) -> str:
    if path in schema.sidecar["TYPES"] or path in schema.sidecar["OPEN"] or following_source(value):
        return scalar_type(schema, value, path)
    choices = {scalar_type(schema, item, path) for item in schema.observed.get(path, [value])}
    if "int | float" in choices:
        choices.discard("int")
    return "OpenMap" if isinstance(value, dict) and not value else " | ".join(sorted(choices))


def render_field(schema: RenderSchema, path: str, value: Any) -> str:
    declared, types = schema.sidecar["UNDECLARED"], schema.sidecar["TYPES"]
    default = declared[path][1] if path in declared else get_path(schema.source.base, path)
    if path in schema.names:
        annotation = types.get(path, schema.names[path])
        if default is None and path not in types:
            annotation += " | None"
        if path in declared and declared[path][0] != annotation:
            raise ValueError(f"UNDECLARED mapping annotation must be {annotation!r}: {path}")
    else:
        annotation = declared[path][0] if path in declared else field_annotation(schema, path, value)
    following = path not in declared and path not in schema.names and "." in path and following_source(default)
    unset = default is MISSING or default is Ellipsis or following or ("OpenMap" in annotation and bool(default))
    expression = "unset_field()" if unset else f"field({default!r})"
    if path in schema.names and default is not None and not unset:
        expression = f"Field(default_factory={schema.names[path]})"
    rendered = f"    {path.rsplit('.', 1)[-1]}: {annotation} = {expression}"
    if "." in path and path in schema.source.comments:
        comment = schema.source.comments[path]
        docstring = json.dumps(comment)
        if len(docstring) > 115:
            docstring = '"""' + "\n    ".join(textwrap.wrap(comment, width=110)) + '"""'
        rendered += f"\n    {docstring}"
    return rendered


def render_sources(source: SourceDocuments, sidecar: dict, owners: dict, recipes: list[dict] = ()) -> dict[str, str]:
    tree, observed = schema_inventory(source.base, sidecar, source.groups, recipes)
    owned = owners["DERIVED_PATHS"] | owners["LAUNCH_PATHS"]
    paths = list(section_paths(tree, sidecar, owned))
    declarations = ast.parse((OUTPUT_DIR / "declared.py").read_text()).body
    declared = [node.name for node in declarations if isinstance(node, ast.ClassDef)]
    declared += [node.targets[0].id for node in declarations if isinstance(node, ast.Assign)]
    schema = RenderSchema(source, observed, class_names(paths, sidecar["NAMES"], declared), sidecar)
    classes = {}
    for path, mapping in paths:
        fields = [
            render_field(schema, child, value)
            for key, value in mapping.items()
            if (child := f"{path}.{key}".lstrip(".")) not in owned
        ]
        name = schema.names[path]
        classes[name] = f"class {name}(Section):\n" + ("\n".join(fields) or "    pass") + "\n"
    fields = []
    for group in ("terminal_bench_config", "algorithm_recipe"):
        if options := source.groups.get(group):
            choices = ", ".join(repr(option) for option in sorted(options))
            fields.append(f"    {group}: Literal[{choices}] = unset_field()")
    classes["ConfigGroups"] = "class ConfigGroups(Section):\n" + ("\n".join(fields) or "    pass") + "\n"
    sampling = classes.pop("EvalSamplingParams", None)
    body = "\n\n".join(classes.values())
    body += """

_namespace = {**vars(declared), **globals()}
for _section in [value for value in _namespace.values() if isinstance(value, type) and issubclass(value, Section)]:
    _section.model_rebuild(_types_namespace=_namespace)
del _section, _namespace
"""
    referenced = {node.id for node in ast.walk(ast.parse(body)) if isinstance(node, ast.Name)}
    names = ", ".join(name for name in declared if name in referenced)
    imports = f"from . import declared\nfrom .declared import {names}\n"
    if sampling is not None:
        imports = (
            sampling + "\n\n# Callback annotations use the evaluation sampling class above.\n"
            "from . import declared  # noqa: E402\n"
            f"from .declared import {names}  # noqa: E402\n"
        )
    description = "Generated by scripts/generate_recipe_schema.py; edit YAML or scripts/recipe_schema_sidecar.py."
    modules = {"sections.py": generated_module(body, description, imports)}
    return modules


def format_generated(source: str, filename: Path) -> str:
    """Format generated Python with the repository's pinned tools."""
    config = ["--config", str(WORKSPACE / "pyproject.toml")]
    black = ["uvx", "black@25.9.0", *config, "-"]
    ruff = ["uvx", "ruff@0.14.3", "check", "--fix", *config]
    ruff += ["--stdin-filename", str(filename), "-"]
    for command in (black, ruff, black):
        source = subprocess.run(command, input=source, text=True, capture_output=True, check=True, cwd=WORKSPACE).stdout
    return source


def generate() -> dict[str, str]:
    """Return repository-formatted author code."""
    source = source_documents(CONFIG_DIR)
    modules = render_sources(
        source,
        runpy.run_path(str(SIDECAR)),
        {"DERIVED_PATHS": DERIVED_PATHS, "LAUNCH_PATHS": LAUNCH_PATHS},
        recipe_documents(ROOT / "cloud/iris/configs", CONFIG_DIR, source.groups),
    )
    return {name: format_generated(code, OUTPUT_DIR / name) for name, code in modules.items()}


if __name__ == "__main__":
    for name, text in generate().items():
        (OUTPUT_DIR / name).write_text(text)
