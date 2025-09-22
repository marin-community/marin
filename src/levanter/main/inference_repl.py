# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""
Interactive REPL and CLI for Levanter model inference.

Allows loading/unloading models and submitting chat/text completion requests.

Interactive REPL Usage:
    uv run src/levanter/main/inference_repl.py
    uv run src/levanter/main/inference_repl.py --checkpoint /path/to/checkpoint

CLI Usage:
    uv run src/levanter/main/inference_repl.py --command=complete --args="The chicken liked to eat" --checkpoint=meta-llama/Llama-3.2-1B-Instruct
"""

import asyncio
import logging
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional

import equinox as eqx
import haliax as hax
import jax.random as jrandom
from haliax import Axis
from haliax.partitioning import round_axis_for_partitioning
from prompt_toolkit import prompt
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import levanter
from levanter.checkpoint import load_checkpoint
from levanter.compat.hf_checkpoints import RepoRef
from levanter.inference.engine import InferenceEngineConfig
from levanter.inference.openai import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    InferenceServer,
    InferenceServerConfig,
    _create_chat_completion,
    _create_completion,
)
from levanter.models.llama import LlamaConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device

logger = logging.getLogger(__name__)
console = Console()

# Global state
server: Optional[InferenceServer] = None
server_config: InferenceServerConfig = InferenceServerConfig()
model_name: Optional[str] = None


def weight_loader(server, server_config, current_model):

    with (
        server_config.trainer.device_mesh,
        hax.axis_mapping(server_config.trainer.compute_axis_mapping),
    ):
        with use_cpu_device():
            key = jrandom.PRNGKey(server_config.seed)
            vocab_size = len(server.inference_context.tokenizer)
            Vocab = round_axis_for_partitioning(Axis("vocab", vocab_size), server_config.trainer.compute_axis_mapping)
            model = eqx.filter_eval_shape(server_config.model.build, Vocab, key=key)
            model = load_checkpoint(model, model, subpath="model")
            model = server_config.trainer.mp.cast_to_compute(model)
        return model


@dataclass
class InferenceReplConfig:
    """Configuration for the inference REPL."""

    checkpoint: Optional[str] = None
    tokenizer: Optional[str] = None

    # Model and training configuration
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LlamaConfig = field(default_factory=LlamaConfig)
    service: InferenceEngineConfig = field(
        default_factory=lambda: InferenceEngineConfig(
            max_seqs=2, page_size=8, max_pages_per_seq=16, max_queued_tokens=8
        )
    )

    # Generation parameters
    temperature: float = 0.7
    seed: int = 42
    max_tokens: int = 64

    # CLI mode parameters
    command: Optional[str] = None
    args: str = ""


class Commands:
    """Command handler for both REPL and CLI modes."""

    def __init__(self, config: InferenceReplConfig):
        self.config = config
        self.commands: Dict[str, Callable] = {
            "load": self.load_model,
            "unload": self.unload_model,
            "chat": self.chat,
            "complete": self.complete,
            "reload": self.reload,
            "config": self.show_config,
            "info": self.info,
            "help": self.show_help,
        }

    def execute(self, cmd_name: str, *args, **kwargs):
        """Execute a command by name."""
        if cmd_name in self.commands:
            return self.commands[cmd_name](*args, **kwargs)
        else:
            console.print(f"[red]Unknown command: {cmd_name}[/red]")

    def load_model(self, path: str, tokenizer: Optional[str] = None, **kwargs):
        """Load a model from checkpoint or HuggingFace."""
        global server, server_config, model_name

        # Unload existing model first
        if server:
            self.unload_model()

        console.print(f"[blue]Loading {path}...[/blue]")

        # Use provided tokenizer or fall back to config
        tokenizer = tokenizer or self.config.tokenizer

        # Determine if HF model
        is_hf_model = not ("://" in path or path.startswith("/") or path.startswith("./") or path.startswith("../"))

        if is_hf_model:
            server_config = InferenceServerConfig(
                hf_checkpoint=RepoRef.from_string(path),
                tokenizer=path,
                model=self.config.model,
                temperature=self.config.temperature,
                seed=self.config.seed,
                trainer=self.config.trainer,
                service=self.config.service,
            )
            model_name = path
        else:
            if not tokenizer:
                console.print("[red]Must specify --tokenizer for local checkpoints[/red]")
                return
            server_config = InferenceServerConfig(
                checkpoint_path=path,
                tokenizer=tokenizer,
                model=self.config.model,
                temperature=self.config.temperature,
                seed=self.config.seed,
                trainer=self.config.trainer,
                service=self.config.service,
            )
            model_name = Path(path).name

        server = InferenceServer.create(server_config)
        console.print(f"[green]✓ Loaded {model_name}[/green]")

    def unload_model(self):
        """Unload the current model."""
        global server, server_config, model_name

        if server:
            console.print(f"[blue]Unloading {model_name}...[/blue]")
            server.shutdown()
            server = None
            server_config = None
            model_name = None
            console.print("[green]✓ Model unloaded[/green]")
        else:
            console.print("[yellow]No model loaded[/yellow]")

    def chat(self, message: Optional[str] = None):
        """Chat with the model."""
        if not server:
            console.print("[red]No model loaded. Use 'load' command first[/red]")
            return

        if message:
            # Single message mode
            messages = [ChatMessage(role="user", content=message)]
            response = self._run_chat_completion(messages)
            self._print_completion_response(response)
        else:
            # Interactive chat session
            self._run_chat_session()

    def _run_chat_session(self):
        """Run interactive chat session with prompt_toolkit."""
        console.print("[cyan]Chat mode. Commands: /exit, /clear[/cyan]")

        messages = []
        chat_history_path = Path("~/.cache/levanter/chat_history").expanduser()
        chat_history_path.parent.mkdir(parents=True, exist_ok=True)
        chat_history = FileHistory(str(chat_history_path))
        chat_completer = WordCompleter(["/exit", "/clear"], ignore_case=True)

        while True:
            try:
                user_input = prompt("You: ", history=chat_history, completer=chat_completer).strip()

                if not user_input:
                    continue

                if user_input == "/exit":
                    break
                elif user_input == "/clear":
                    messages = []
                    console.print("[green]Conversation cleared[/green]")
                    continue

                messages.append(ChatMessage(role="user", content=user_input))
                response = self._run_chat_completion(messages, print_response=False)

                if response:
                    assistant_msg = response.choices[0].message.content
                    messages.append(ChatMessage(role="assistant", content=assistant_msg))
                    console.print(f"[green]Assistant:[/green] {assistant_msg}")

            except (KeyboardInterrupt, EOFError):
                break

    def complete(self, prompt_text: str):
        """Complete a text prompt."""
        if not server:
            console.print("[red]No model loaded[/red]")
            return

        request = CompletionRequest(
            model=model_name or "model",
            prompt=prompt_text,
            max_tokens=self.config.max_tokens,
            temperature=server_config.temperature,
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(_create_completion(server.inference_context, request))

            self._print_completion_response(response)
        finally:
            loop.close()

    def reload(self, model: str):
        """Reload model from a new checkpoint."""
        global server_config

        if not server:
            console.print("[red]No model loaded to reload[/red]")
            return

        # Update checkpoint path in config
        server_config = InferenceServerConfig(
            hf_checkpoint=None,  # Clear HF checkpoint if reloading from path
            tokenizer=server_config.tokenizer,
            model=server_config.model,
            checkpoint_path=model,
            temperature=server_config.temperature,
            seed=server_config.seed,
            trainer=server_config.trainer,
            service=server_config.service,
        )

        console.print(f"[blue]Reloading from {model}...[/blue]")

        server.reload(lambda current_model: weight_loader(server, server_config, current_model))
        console.print("[green]✓ Model reloaded successfully[/green]")

    def info(self):
        """Show current model information."""
        if not server:
            console.print("[yellow]No model loaded[/yellow]")
            return

        table = Table(title="Model Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Model", model_name or "Unknown")
        table.add_row("Tokenizer", str(server_config.tokenizer))
        table.add_row("Vocab Size", str(len(server.inference_context.tokenizer)))
        table.add_row("Max Sequences", str(server_config.service.max_seqs))
        table.add_row("Temperature", str(server_config.temperature))
        table.add_row("Seed", str(server_config.seed))
        table.add_row("Max Tokens", str(self.config.max_tokens))

        console.print(table)

    def show_help(self):
        """Show help text."""
        help_text = """
[bold cyan]Commands:[/bold cyan]
  load <path|hf:model>      Load model (e.g., load meta-llama/Llama-3.2-1B)
  unload                    Unload current model
  chat [text]               Chat with model (interactive if no text)
  complete <text>           Complete text prompt
  reload <checkpoint>       Reload from new checkpoint
  info                      Show model information
  help                      Show this help

[bold yellow]Direct Input:[/bold yellow]
  Type text to chat with the model
        """
        console.print(Panel(help_text, border_style="blue"))

    def show_config(self, key_value: Optional[str] = None):
        """Show current configuration."""
        console.print("[cyan]Current Config:[/cyan]")
        console.print(f"Checkpoint: {self.config.checkpoint}")
        console.print(f"Tokenizer: {self.config.tokenizer}")
        console.print(f"Temperature: {self.config.temperature}")
        console.print(f"Max Tokens: {self.config.max_tokens}")

    def _run_chat_completion(self, messages, print_response=True):
        """Run async chat completion."""
        request = ChatCompletionRequest(
            model=model_name or "model",
            messages=messages,
            stop=[server.inference_context.tokenizer.eos_token],
            max_tokens=self.config.max_tokens,
            temperature=server_config.temperature,
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(_create_chat_completion(server.inference_context, request))

            if print_response:
                self._print_completion_response(response)
            return response
        finally:
            loop.close()

    def _print_completion_response(self, response):
        """Pretty print completion response."""
        for i, choice in enumerate(response.choices):
            if hasattr(choice, "message"):  # Chat completion
                content = choice.message.content
            else:  # Text completion
                content = choice.text

            panel = Panel(content, title=f"Response {i + 1}/{len(response.choices)}", border_style="green")
            console.print(panel)

        # Show usage stats
        if response.usage:
            console.print(
                f"[dim]Tokens - Prompt: {response.usage.prompt_tokens}, "
                f"Completion: {response.usage.completion_tokens}, "
                f"Total: {response.usage.total_tokens}[/dim]"
            )


def repl_mode(config: InferenceReplConfig, commands: Commands):
    """Run interactive REPL."""
    console.print(
        Panel.fit(
            "[bold blue]Levanter Inference REPL[/bold blue]\n" "Type [bold]help[/bold] for commands",
            border_style="blue",
        )
    )

    # Auto-load model if specified
    if config.checkpoint:
        commands.load_model(config.checkpoint, config.tokenizer)

    # Setup prompt_toolkit
    command_names = list(commands.commands.keys()) + ["quit", "exit"]
    completer = WordCompleter(command_names, ignore_case=True)

    history_path = Path("~/.cache/levanter/inference_repl_history").expanduser()
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history = FileHistory(str(history_path))

    # REPL loop
    while True:
        try:
            user_input = prompt("🤖 > ", completer=completer, history=history).strip()

            if not user_input:
                continue

            # Parse command
            if user_input in ["quit", "exit"]:
                break

            parts = shlex.split(user_input)
            cmd = parts[0]
            args = parts[1:]

            # Handle commands
            if cmd == "load":
                # Parse: load path [--tokenizer tok]
                if not args:
                    console.print("[red]Usage: load <path> [--tokenizer <name>][/red]")
                    continue

                path = args[0]
                tokenizer = None
                if "--tokenizer" in args:
                    idx = args.index("--tokenizer")
                    tokenizer = args[idx + 1] if idx + 1 < len(args) else None
                commands.execute("load", path, tokenizer=tokenizer)
            elif cmd == "chat":
                message = " ".join(args) if args else None
                commands.execute("chat", message)
            elif cmd == "complete":
                if not args:
                    console.print("[red]Usage: complete <prompt>[/red]")
                    continue
                prompt_text = " ".join(args)
                commands.execute("complete", prompt_text)
            elif cmd == "reload":
                if not args:
                    console.print("[red]Usage: reload <model>[/red]")
                    continue
                commands.execute("reload", args[0])
            elif cmd == "config":
                key_value = args[0] if args else None
                commands.execute("config", key_value)
            elif cmd in commands.commands:
                commands.execute(cmd)
            else:
                # Direct input = chat
                commands.execute("chat", user_input)

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    # Cleanup
    if server:
        server.shutdown()
    console.print("[blue]Goodbye![/blue]")


def cli_mode(config: InferenceReplConfig, commands: Commands):
    """Execute single command from CLI."""
    if not config.command:
        return

    if config.checkpoint:
        commands.load_model(config.checkpoint, config.tokenizer)
    else:
        console.print("[red]No model specified. Use --checkpoint to specify a model.[/red]")
        return

    if config.command == "chat":
        message = config.args if config.args else None
        commands.execute("chat", message)
    elif config.command == "complete":
        if not config.args:
            console.print("[red]Usage: complete <prompt>[/red]")
            return
        prompt_text = config.args
        commands.execute("complete", prompt_text)
    else:
        commands.execute(config.command, *config.args)

    # Cleanup
    if server:
        server.shutdown()


def main(config: InferenceReplConfig):
    """Main entry point."""
    commands = Commands(config)

    # Determine mode
    if config.command:
        cli_mode(config, commands)
    else:
        repl_mode(config, commands)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    levanter.config.main(main)()
