import shutil
from pathlib import Path
from typing import Annotated

from rich.console import Console
from typer import Option, Typer

cache_app = Typer(
    no_args_is_help=True, context_settings={"help_option_names": ["-h", "--help"]}
)
console = Console()


@cache_app.command()
def clean(
    force: Annotated[
        bool,
        Option("-f", "--force", help="Skip confirmation and remove the cache."),
    ] = False,
    dry: Annotated[
        bool,
        Option("--dry", help="Show the cache without removing it."),
    ] = False,
):
    """Remove Harbor's downloaded task cache."""
    cache_dir = Path("~/.cache/harbor").expanduser()
    if not cache_dir.exists():
        console.print(f"Cache directory not found: {cache_dir}")
        return
    total_size = sum(
        path.stat().st_size for path in cache_dir.rglob("*") if path.is_file()
    )
    console.print(f"Cache directory: {cache_dir} ({total_size / (1024 * 1024):.2f} MB)")
    if dry:
        return
    if not force and console.input("Remove this cache? (y/N): ").lower() not in {
        "y",
        "yes",
    }:
        return
    shutil.rmtree(cache_dir)
    console.print(f"Removed cache directory: {cache_dir}")
