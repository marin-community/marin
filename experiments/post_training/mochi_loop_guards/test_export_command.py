# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from shlex import split

from marin.rl.skyrl import ResolvedModelLocator

from experiments.post_training.mochi_loop_guards.presets import export_command


def test_export_command_carries_the_replayable_model_locator_to_the_native_cli():
    model = ResolvedModelLocator(
        uri="s3://models/sealed-sft",
        identity="sealed-sft@v1:abcd",
        local_path="/tmp/model metadata",
        tokenizer_uri=None,
        tokenizer_revision=None,
    )
    command = split(
        export_command(
            "s3://checkpoints/run",
            4,
            "/tmp/saved launch.yaml",
            model=model,
            export_path="s3://permanent/exports",
        )
    )
    for option, value in (
        ("--model_path", model.local_path),
        ("--model-source-uri", model.uri),
        ("--model-source-identity", model.identity),
        ("--launch-config", "/tmp/saved launch.yaml"),
        ("--export_path", "s3://permanent/exports"),
    ):
        assert command[command.index(option) + 1] == value
    assert command[-1] == "--dry-run"
