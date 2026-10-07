# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import shutil
from pathlib import Path

import haliax as hax
import jax
import numpy as np
import pytest

from levanter.data.dataset import ListAsyncDataset
from levanter.data.mixture import MixtureDataset, rescale_mixture_schedule_for_batch_schedule
from levanter.data.text.datasets import (
    BlockShuffleConfig,
    DatasetComponent,
    LmDataConfig,
    MixturePhase,
    PriorDataPhase,
    skip_to_window_offsets,
)
from levanter.data.text.formats import TextLmDatasetFormat, preprocessor_for_format
from levanter.schedule import BatchSchedule
from levanter.store.cache import write_levanter_cache
from levanter.tokenizers import load_tokenizer

# Token counts per source; "b" leaves a partial shuffle block and window at the end.
TOKENS = {"a": 4096, "b": 3000, "c": 2048}
OLD_SEQ, NEW_SEQ = 2, 8
BLOCK_TOKENS, WINDOW_BLOCKS = 16, 4
WINDOW_TOKENS = BLOCK_TOKENS * WINDOW_BLOCKS
MIXTURE_BLOCK = 16
OLD_BATCH, NEW_BATCH = 8, 2  # 16 tokens per step in both phases
SWITCH_STEP = 37  # mid mixture block at both batch sizes
NEW_STEPS = 60
WEIGHTS_BEFORE = {"a": 0.5, "b": 0.3, "c": 0.2}
WEIGHTS_AFTER = {"a": 0.2, "b": 0.3, "c": 0.5}


def _sources(seq_len: int, token_counts: dict[str, int] = TOKENS):
    """Shuffled sources whose items are (source, token ids), with the same shuffle keys at every length."""
    datasets = {}
    for i, (name, tokens) in enumerate(token_counts.items()):
        items = [(name, tuple(range(start, start + seq_len))) for start in range(0, tokens - seq_len + 1, seq_len)]
        datasets[name] = ListAsyncDataset(items).block_shuffle(
            io_block_size=BLOCK_TOKENS // seq_len,
            window_blocks=WINDOW_BLOCKS,
            key=jax.random.PRNGKey(10 + i),
            perm_type="feistel",
        )
    return datasets


def _stages(stage_step: int, batch_size: int):
    return rescale_mixture_schedule_for_batch_schedule(
        [(0, WEIGHTS_BEFORE), (stage_step, WEIGHTS_AFTER)], BatchSchedule(batch_size)
    )


def _read(mixture: MixtureDataset, start: int, stop: int) -> list[tuple[str, int]]:
    """Return (source, token) pairs for mixture indices [start, stop), in order."""
    items = mixture.as_sync_dataset().get_batch(list(range(start, stop)))
    return [(name, token) for name, tokens in items for token in tokens]


def _window_of_token(name: str) -> dict[int, int]:
    """Map each token of a source to the shuffle window that serves it."""
    dataset = _sources(OLD_SEQ)[name].as_sync_dataset()
    window_sequences = WINDOW_TOKENS // OLD_SEQ
    return {
        token: position // window_sequences
        for position, (_, tokens) in enumerate(dataset.get_batch(list(range(len(dataset)))))
        for token in tokens
    }


@pytest.fixture(scope="module")
def switch_reads():
    key = jax.random.PRNGKey(0)
    old_stages = _stages(4, OLD_BATCH)
    # At the new batch a stage must start on a multiple of 8 steps, so the boundary moves from 4 to 8.
    new_stages = _stages(8, NEW_BATCH)
    old_mixture = MixtureDataset(_sources(OLD_SEQ), old_stages, MIXTURE_BLOCK, key=key)
    new_sources = _sources(NEW_SEQ)
    offsets = skip_to_window_offsets(
        datasets=new_sources,
        token_counts=TOKENS,
        phases=[
            MixturePhase(start_step=0, seq_len=OLD_SEQ, batch_schedule=BatchSchedule(OLD_BATCH), weights=old_stages),
            MixturePhase(
                start_step=SWITCH_STEP, seq_len=NEW_SEQ, batch_schedule=BatchSchedule(NEW_BATCH), weights=new_stages
            ),
        ],
        block_size=MIXTURE_BLOCK,
        window_tokens=WINDOW_TOKENS,
        key=key,
    )
    new_start = SWITCH_STEP * NEW_BATCH
    new_stop = new_start + NEW_STEPS * NEW_BATCH
    resumed = MixtureDataset(new_sources, new_stages, MIXTURE_BLOCK, key=key, start_offsets=offsets)
    return {
        "before": _read(old_mixture, 0, SWITCH_STEP * OLD_BATCH),
        "after": _read(resumed, new_start, new_stop),
    }


@pytest.mark.parametrize("name", sorted(TOKENS))
def test_context_switch_resumes_each_source_at_its_next_shuffle_window(switch_reads, name):
    windows = _window_of_token(name)
    before = [token for source, token in switch_reads["before"] if source == name]
    after = [token for source, token in switch_reads["after"] if source == name]
    next_window = -(-len(before) // WINDOW_TOKENS)
    following_window_tokens = {token for token, window in windows.items() if window == next_window + 1}

    # The source resumes at the window after the one it was reading, and touches no earlier window.
    assert min(windows[token] for token in after) == next_window
    # Only the transition skips data: the next window after that is served in full.
    assert following_window_tokens <= set(after)


def _phase(start_step: int, seq_len: int, batch_size: int, weights: dict[str, float]) -> MixturePhase:
    return MixturePhase(
        start_step=start_step, seq_len=seq_len, batch_schedule=BatchSchedule(batch_size), weights=[(0, weights)]
    )


def _offsets(phases: list[MixturePhase], token_counts: dict[str, int]) -> dict[str, int]:
    return skip_to_window_offsets(
        datasets=_sources(phases[-1].seq_len, token_counts),
        token_counts=token_counts,
        phases=phases,
        block_size=MIXTURE_BLOCK,
        window_tokens=WINDOW_TOKENS,
        key=jax.random.PRNGKey(0),
    )


def test_second_context_switch_continues_past_the_first_switch_skip():
    # One source at 16 tokens per step: 2 -> 4 at step 1, then 4 -> 8 at step 8.
    token_counts = {"a": TOKENS["a"]}
    phases = [
        _phase(0, 2, 8, {"a": 1.0}),
        _phase(1, 4, 4, {"a": 1.0}),
        _phase(8, 8, 2, {"a": 1.0}),
    ]
    reads = []
    for i, phase in enumerate(phases):
        offsets = _offsets(phases[: i + 1], token_counts) if i else None
        mixture = MixtureDataset(
            _sources(phase.seq_len, token_counts),
            phase.weights,
            MIXTURE_BLOCK,
            key=jax.random.PRNGKey(0),
            start_offsets=offsets,
        )
        stop_step = phases[i + 1].start_step if i + 1 < len(phases) else phase.start_step + NEW_STEPS
        batch_size = phase.batch_schedule.batch_size_at_step(0)
        reads.append(set(_read(mixture, phase.start_step * batch_size, stop_step * batch_size)))

    assert not reads[0] & reads[1]
    # The middle phase started past its own skip, so the last phase must start past the middle phase's reads.
    assert not (reads[0] | reads[1]) & reads[2]


def test_context_switch_in_the_final_partial_window_resumes_at_the_next_epoch():
    # 3000 tokens are 1500 old sequences; the final 32-sequence shuffle window starts at 1472 and is partial.
    token_counts = {"b": TOKENS["b"]}
    switch_step = 185  # the old phase has served through sequence 1488, rounded up to its mixture block
    phases = [_phase(0, OLD_SEQ, OLD_BATCH, {"b": 1.0}), _phase(switch_step, NEW_SEQ, NEW_BATCH, {"b": 1.0})]
    new_sources = _sources(NEW_SEQ, token_counts)
    resumed = MixtureDataset(
        new_sources,
        phases[1].weights,
        MIXTURE_BLOCK,
        key=jax.random.PRNGKey(0),
        start_offsets=_offsets(phases, token_counts),
    )
    block_start = switch_step * NEW_BATCH // MIXTURE_BLOCK * MIXTURE_BLOCK
    epoch_start = new_sources["b"].as_sync_dataset().get_batch(list(range(MIXTURE_BLOCK)))

    # The mixture block at the switch reads the first sequences of the next epoch, skipping none of them.
    assert set(_read(resumed, block_start, block_start + MIXTURE_BLOCK)) == {
        (name, token) for name, tokens in epoch_start for token in tokens
    }


def test_context_switch_rejects_a_change_in_active_sources():
    before, after = {"a": 0.5, "b": 0.3, "c": 0.2}, {"a": 0.0, "b": 0.5, "c": 0.5}
    # A training config drops sources with zero weight in every stage, so only the current ones arrive here.
    current = {name: dataset for name, dataset in _sources(NEW_SEQ).items() if after[name] > 0}
    with pytest.raises(ValueError, match="same sources"):
        skip_to_window_offsets(
            datasets=current,
            token_counts=TOKENS,
            phases=[_phase(0, OLD_SEQ, OLD_BATCH, before), _phase(SWITCH_STEP, NEW_SEQ, NEW_BATCH, after)],
            block_size=MIXTURE_BLOCK,
            window_tokens=WINDOW_TOKENS,
            key=jax.random.PRNGKey(0),
        )


def _cached_config(tmp_path: Path, seq_len: int, stage_step: int) -> LmDataConfig:
    """An LmDataConfig over token caches whose ids record (source, position)."""
    tokenizer_dir = tmp_path / "tokenizer"
    if not tokenizer_dir.exists():
        tokenizer_dir.mkdir()
        tests_dir = Path(__file__).parent
        shutil.copy(tests_dir / "gpt2_tokenizer.json", tokenizer_dir / "tokenizer.json")
        shutil.copy(tests_dir / "gpt2_tokenizer_config.json", tokenizer_dir / "tokenizer_config.json")
        metadata = preprocessor_for_format(
            TextLmDatasetFormat(), load_tokenizer(str(tokenizer_dir)), enforce_bos=True, enforce_eos=True
        ).metadata
        # Ids stay below GPT-2's EOS id (50256) so no sequence splits into documents.
        for i, (name, tokens) in enumerate(TOKENS.items()):
            ids = np.arange(tokens, dtype=np.int32) + 10_000 * i
            docs = ({"input_ids": ids[start : start + 100]} for start in range(0, tokens, 100))
            write_levanter_cache(docs, str(tmp_path / name / "train"), metadata=metadata)
    return LmDataConfig(
        tokenizer=str(tokenizer_dir),
        cache_dir=None,
        auto_build_caches=False,
        components={
            name: DatasetComponent(source=None, cache_dir=str(tmp_path / name), format=TextLmDatasetFormat())
            for name in TOKENS
        },
        train_weights=[(0, WEIGHTS_BEFORE), (stage_step, WEIGHTS_AFTER)],
        shuffle=BlockShuffleConfig(io_block_size=BLOCK_TOKENS // seq_len, window_blocks=WINDOW_BLOCKS),
        mixture_block_size=MIXTURE_BLOCK,
    )


def _tokens(config: LmDataConfig, seq_len: int, batch_size: int, start: int, stop: int) -> set[int]:
    dataset = config.train_set(hax.Axis("position", seq_len), BatchSchedule(batch_size), key=jax.random.PRNGKey(0))
    batch = dataset.as_sync_dataset().get_batch(list(range(start, stop)))
    return {int(token) for example in batch for token in np.asarray(example.tokens.array)}


def test_lm_data_config_prior_phase_resumes_without_repeating_tokens(tmp_path):
    old = _cached_config(tmp_path, OLD_SEQ, stage_step=4)
    before = _tokens(old, OLD_SEQ, OLD_BATCH, 0, SWITCH_STEP * OLD_BATCH)
    naive = _cached_config(tmp_path, NEW_SEQ, stage_step=8)
    resumed = dataclasses.replace(
        naive,
        prior_phases=[
            PriorDataPhase(
                end_step=SWITCH_STEP,
                seq_len=OLD_SEQ,
                batch_size=OLD_BATCH,
                train_weights=[(0, WEIGHTS_BEFORE), (4, WEIGHTS_AFTER)],
                shuffle=old.shuffle,
            )
        ],
    )
    new_start = SWITCH_STEP * NEW_BATCH
    new_stop = new_start + NEW_STEPS * NEW_BATCH

    assert not before & _tokens(resumed, NEW_SEQ, NEW_BATCH, new_start, new_stop)
    assert before & _tokens(naive, NEW_SEQ, NEW_BATCH, new_start, new_stop)
