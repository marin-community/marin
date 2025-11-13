#!/usr/bin/env -S uv run --quiet

# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Repl for examining Hugging Face tokenizers."""

import ast
import shlex

import click
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from transformers import AutoTokenizer

console = Console()

current_tokenizer: str | None = None


def load_tokenizer(model_name: str) -> bool:
    """Load a tokenizer and set it as current."""
    global current_tokenizer
    console.print(f"[blue]Loading {model_name}...[/blue]")
    current_tokenizer = AutoTokenizer.from_pretrained(model_name)
    console.print(f"[green]✓ Loaded {model_name}[/green]")


@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Token Decoder REPL."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(repl_cmd)


@cli.command("repl")
@click.pass_context
def repl_cmd(ctx):
    """Start the interactive REPL."""
    console.print(
        Panel.fit(
            "[bold blue]Token Decoder REPL[/bold blue]\n" "Type [bold]/help[/bold] for commands", border_style="blue"
        )
    )

    # Build command list for completion
    commands = [cmd for cmd in cli.list_commands(None)]
    completions = ["/" + cmd for cmd in commands] + ["/quit", "/exit"]
    completer = WordCompleter(completions, ignore_case=True)
    history = InMemoryHistory()

    while True:
        user_input = prompt(">>> ", completer=completer, history=history).strip()

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            cmd_parts = shlex.split(user_input[1:])
            if not cmd_parts:
                continue

            cmd_name = cmd_parts[0]
            args = cmd_parts[1:]

            if cmd_name in ["quit", "exit"]:
                break

            if cmd_name == "help":
                show_help()
                continue

            if cmd_name == "load" and args:
                load_tokenizer(args[0])
                continue
        else:
            # Check if it looks like a token list
            if user_input.startswith("[") and user_input.endswith("]"):
                try:
                    tokens = ast.literal_eval(user_input)
                    if isinstance(tokens, list) and all(isinstance(t, int) for t in tokens):
                        decode_tokens(tokens)
                        continue
                except Exception:
                    # Failed to parse as token list, fall through to encode as text
                    pass

            # Otherwise encode as text
            encode_text(user_input)


def encode_text(text: str):
    tokens = current_tokenizer.encode(text)
    for token in tokens:
        console.print(f"{token} ", end="")
    console.print()


@cli.command("load")
@click.argument("model_name")
def load_tokenizer_cmd(model_name: str):
    """Load a tokenizer and start REPL."""
    if load_tokenizer(model_name):
        click.Context(cli).invoke(repl_cmd)


@cli.command("encode")
@click.argument("text")
@click.option("--tokenizer", "-t", help="Tokenizer to use")
def encode_cmd(text: str, tokenizer: str | None):
    """Encode text to tokens."""
    if tokenizer:
        load_tokenizer(tokenizer)
    encode_text(text)


@cli.command("decode")
@click.argument("tokens")
@click.option("--tokenizer", "-t", help="Tokenizer to use")
def decode_cmd(tokens: str, tokenizer: str | None):
    """Decode tokens to text."""
    if tokenizer:
        load_tokenizer(tokenizer)
    try:
        token_list = ast.literal_eval(tokens)
        decode_tokens(token_list)
    except Exception as e:
        console.print(f"[red]Error parsing tokens: {e}[/red]")


def decode_tokens(token_list: list[int]):
    """Helper to decode tokens."""
    decoded = current_tokenizer.decode(token_list)
    console.print(f"[green]Decoded:[/green] {decoded!r}")


@cli.command("vocab")
@click.argument("search_term", required=False)
@click.option("--tokenizer", "-t", help="Tokenizer to use")
def vocab_cmd(search_term: str | None, tokenizer: str | None):
    """Search vocabulary."""
    if tokenizer:
        load_tokenizer(tokenizer)

    console.print(f"[blue]Vocabulary size:[/blue] {current_tokenizer.vocab_size:,}")

    if search_term:
        vocab_dict = current_tokenizer.get_vocab()
        matches = [(t, i) for t, i in vocab_dict.items() if search_term.lower() in t.lower()][:20]

        if matches:
            table = Table(title=f"Matches for '{search_term}'")
            table.add_column("Token", style="cyan")
            table.add_column("ID", style="magenta")
            for token, token_id in matches:
                table.add_row(repr(token), str(token_id))
            console.print(table)
        else:
            console.print("[yellow]No matches found[/yellow]")


@cli.command("special")
@click.option("--tokenizer", "-t", help="Tokenizer to use")
def special_cmd(tokenizer: str | None):
    """Show special tokens."""
    if tokenizer:
        load_tokenizer(tokenizer)
    table = Table(title="Special Tokens")
    table.add_column("Name", style="cyan")
    table.add_column("Token", style="green")
    table.add_column("ID", style="magenta")

    special_tokens = {
        "PAD": getattr(current_tokenizer, "pad_token", None),
        "UNK": getattr(current_tokenizer, "unk_token", None),
        "BOS": getattr(current_tokenizer, "bos_token", None),
        "EOS": getattr(current_tokenizer, "eos_token", None),
    }

    for name, token in special_tokens.items():
        if token:
            token_id = current_tokenizer.convert_tokens_to_ids(token)
            table.add_row(name, repr(token), str(token_id))

    console.print(table)


@cli.command("info")
def info_cmd():
    """Display current tokenizer information."""
    if not current_tokenizer:
        console.print("[red]No tokenizer loaded. Use /tokenizer <model> first.[/red]")
        return

    if current_tokenizer:
        max_length = getattr(current_tokenizer, "model_max_length", "N/A")
        vocab_size_fmt = f"{current_tokenizer.vocab_size:,}"
        info_panel = (
            f"[bold blue]Current Tokenizer:[/bold blue] {current_tokenizer}\n"
            f"[bold blue]Vocabulary Size:[/bold blue] {vocab_size_fmt}\n"
            f"[bold blue]Model Max Length:[/bold blue] {max_length}"
        )
        console.print(Panel(info_panel, title="Tokenizer Info", border_style="blue"))


@cli.command("clear")
def clear_cmd():
    """Clear the screen."""
    import os

    os.system("clear" if os.name == "posix" else "cls")


def show_help():
    """Show help text."""
    help_text = """
[bold cyan]Commands:[/bold cyan]
  /load <model>   Load tokenizer (e.g., gpt2, meta-llama/Llama-3.2-1B)
  /encode <text>       Encode text to tokens
  /decode <tokens>     Decode token list (e.g., [15496, 995, 0])
  /vocab [search]      Search vocabulary
  /special             Show special tokens
  /stats <text>        Show tokenization statistics
  /chunk <text>        Show text chunking
  /info                Show current tokenizer info
  /clear               Clear screen
  /help                Show this help
  /quit                Exit

[bold yellow]Direct Input:[/bold yellow]
  Type text to encode it
  Type [tokens] to decode them
    """
    console.print(Panel(help_text, border_style="blue"))


if __name__ == "__main__":
    cli()
