from pathlib import Path
from textwrap import dedent
from urllib.error import URLError

import pytest
from mkdocs.commands.build import build
from mkdocs.config import load_config
from mkdocs.exceptions import Abort
from mkdocstrings import Inventory
from mkdocstrings._internal.handlers import base as inventory_loader


_VALID_PAGE = "# Inventory test\n\n::: sample.use_external\n"
_BROKEN_LINK_PAGE = "# Inventory test\n\n[Missing page](missing.md)\n\n::: sample.use_external\n"


def _build_docs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inventory_error: Exception,
    index_markdown: str,
) -> str:
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    docs_dir = tmp_path / "docs"
    source_dir = tmp_path / "src"
    docs_dir.mkdir()
    source_dir.mkdir()

    (docs_dir / "index.md").write_text(
        index_markdown,
        encoding="utf-8",
    )
    (source_dir / "sample.py").write_text(
        dedent(
            """
            def use_external(value: external.Type) -> None:
                \"\"\"Use an external type.\"\"\"
            """,
        ),
        encoding="utf-8",
    )

    inventory = Inventory(project="external", version="1")
    inventory.register(
        "external.Type",
        domain="py",
        role="class",
        uri="type.html",
    )
    available_url = f"https://inventory.example/{tmp_path.name}/objects.inv"
    unavailable_url = f"https://unavailable.example/{tmp_path.name}/objects.inv"

    def download_inventory(url: str) -> bytes:
        if url == available_url:
            return inventory.format_sphinx()
        raise inventory_error

    monkeypatch.setattr(inventory_loader, "_download_url_with_gz", download_inventory)

    repository_root = Path(__file__).parents[1]
    config_path = tmp_path / "mkdocs.yml"
    config_path.write_text(
        dedent(
            f"""
            site_name: Inventory test
            docs_dir: {docs_dir}
            site_dir: {tmp_path / "site"}
            strict: true
            hooks:
              - {repository_root / "infra/mkdocs_hooks.py"}
            plugins:
              - mkdocstrings:
                  handlers:
                    python:
                      paths: [{source_dir}]
                      inventories:
                        - url: {available_url}
                          base_url: https://external.example
                        - {unavailable_url}
                      options:
                        separate_signature: true
                        signature_crossrefs: true
                        show_signature_annotations: true
                        show_root_heading: true
            """,
        ),
        encoding="utf-8",
    )

    config = load_config(config_file=str(config_path))
    build(config)
    return (tmp_path / "site/index.html").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "inventory_error",
    [URLError("inventory unavailable"), TimeoutError("inventory timed out")],
    ids=["unavailable", "timeout"],
)
def test_docs_build_keeps_available_crossrefs_when_an_inventory_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    inventory_error: Exception,
) -> None:
    html = _build_docs(tmp_path, monkeypatch, inventory_error, _VALID_PAGE)

    assert 'href="https://external.example/type.html"' in html


def test_docs_build_still_fails_strict_mode_for_broken_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(Abort, match="Aborted with 1 warning"):
        _build_docs(tmp_path, monkeypatch, URLError("inventory unavailable"), _BROKEN_LINK_PAGE)
