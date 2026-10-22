# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared setup for the d512 Belay interpretability study: the exact Ckpt 1 d512 model config, its checkpoint, a
one-GPU mesh, the token sources (training cache and Paloma validation sets) and result upload."""

import concurrent.futures
import dataclasses
import functools
import io
import itertools
import json
import logging
from collections.abc import Iterator

import fsspec
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from haliax import Axis
from haliax.partitioning import set_mesh
from jax.sharding import Mesh
from levanter.checkpoint import load_checkpoint
from levanter.data.text.datasets import DatasetComponent, LmDataConfig, TextLmDatasetFormat
from levanter.data.text.examples import grug_lm_example_from_named
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.lm_model import LmExample
from levanter.tokenizers import load_tokenizer

from experiments.grug.fast_track import belay_ckpt1, launch
from experiments.grug.fast_track.model import GrugModelConfig, Transformer
from experiments.grug.fast_track.train import restore_template_from
from experiments.grug.moe.launch_datakit_moe_mix import _val_component

logger = logging.getLogger(__name__)

SIZE = "d512"
ONSET_ARMS = ("ctl", "last-half", "first-half", "stagger", "rstagger", "last-half-wu")
# The plain-residual (attn_res_plain_sum) onset arms.
PLAIN_SUM_ARMS = ("ctl-ps", "last-half-ps", "stagger-ps")
# The 8-minute Belay d512 (Ckpt 1 at 2760 steps) and the 8821-step 171-TPP ladder rung, by study tag.
CHECKPOINTS = {
    "tn-d512-ckpt1-rno": "s3://marin-us-east-02a/marin/grug/tn-d512-ckpt1-rno/2026.09.25/checkpoints/step-2760",
    "lc1-l171-d512": "s3://marin-us-east-02a/marin/grug/lc1-l171-d512/2026.09.25/checkpoints/step-8821",
    # The layer onset study (Ckpt 1 d512, 2760 steps, seed 0; one arm per layer_onset_steps schedule).
    **{
        f"lc1-onset-{arm}-s0": f"s3://marin-us-east-02a/marin/grug/lc1-onset-{arm}-s0/2026.09.25/checkpoints/step-2760"
        for arm in (*ONSET_ARMS, *PLAIN_SUM_ARMS)
    },
    # The deeper 171-TPP ladder rungs (8, 12 and 14 layers), for the depth comparison of the skip study.
    "lc1-l171-d768": "s3://marin-us-east-02a/marin/grug/lc1-l171-d768/2026.09.25/checkpoints/step-24995",
    "lc1-l171-d1024": "s3://marin-us-east-02a/marin/grug/lc1-l171-d1024/2026.09.25/checkpoints/step-32323",
    "lc1-l171-d1280": "s3://marin-us-east-02a/marin/grug/lc1-l171-d1280/2026.09.25/checkpoints/step-57578",
}
# Study tags of other ladder sizes (every other tag is d512).
TAG_SIZES = {"lc1-l171-d768": "d768", "lc1-l171-d1024": "d1024", "lc1-l171-d1280": "d1280"}
OUTPUT_ROOT = "s3://marin-us-east-02a/tmp/ttl=30d/interp-belay-d512"
SEQ_LEN = launch.SEQ_LEN
TRAIN_CACHE = launch.V16384_CACHE_DIR
TOKENIZER = launch.V16384_TOKENIZER
PALOMA_ROOT = "s3://marin-us-east-02a/marin/paloma"
PALOMA_SETS = (
    "4chan",
    "c4_100_domains",
    "c4_en",
    "dolma-v1_5",
    "dolma_100_programing_languages",
    "dolma_100_subreddits",
    "falcon-refinedweb",
    "gab",
    "m2d2_s2orc_unsplit",
    "m2d2_wikipedia_unsplit",
    "manosphere_meta_sep",
    "mc4",
    "ptb",
    "redpajama",
    "twitterAAE_HELM_fixed",
    "wikitext_103",
)
# Sequences the run trained on: 8821 steps x 128. Analysis train samples come from beyond them (unseen tokens of the
# same document-shuffled cache).
TRAINED_SEQUENCES = 8821 * 128  # the longer run's; the 2760-step run saw a prefix of it
MP = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")


PLAIN_SUM_TAGS = frozenset(f"lc1-onset-{arm}-s0" for arm in PLAIN_SUM_ARMS)


def model_config(tag: str) -> GrugModelConfig:
    """The Ckpt 1 model of study tag ``tag`` (d512 unless ``TAG_SIZES`` says otherwise), built the way
    ``build_h100_ladder_run`` builds it from ``belay_ckpt1.recipe_args`` (with ``attn_res_plain_sum`` for the
    plain-residual onset arms)."""
    size = TAG_SIZES.get(tag, SIZE)
    args = belay_ckpt1.recipe_args(size)
    settings = {}
    for flag, value in itertools.pairwise(args):
        if flag == "--model-set":
            key, _, val = value.partition("=")
            settings[key] = val
    rung = launch._resolved_rung(size, None, None)
    model = dataclasses.replace(launch._h100_ladder_model(rung), vocab_size=launch.V16384_VOCAB)
    model = launch._kma_model(model)
    model = launch._apply_settings(model, settings)
    return dataclasses.replace(model, attn_res_plain_sum=tag in PLAIN_SUM_TAGS)


def one_gpu_mesh() -> Mesh:
    return compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)


def load_params(cfg: GrugModelConfig, mesh: Mesh, path: str) -> Transformer:
    """The checkpoint's float32 params (the weights the run's final eval scored)."""
    with set_mesh(mesh):
        template = jax.jit(lambda key: MP.cast_to_param(Transformer.init(cfg, key=key)))(jax.random.PRNGKey(0))
        template = restore_template_from(template)
        loaded = load_checkpoint({"params": template}, path, axis_mapping=None, mesh=mesh, allow_partial=False)
    return loaded["params"]


def compute_model(params: Transformer) -> Transformer:
    return MP.cast_to_compute(params)


def data_config() -> LmDataConfig:
    components = {
        "train": DatasetComponent(
            source=None, cache_dir=TRAIN_CACHE, format=TextLmDatasetFormat(), tags=["train"], flat_cache=True
        ),
        **{
            f"paloma/{name}": _val_component(f"{PALOMA_ROOT}/{name}-{TOKENIZER}/{launch.PALOMA_DETOK_VERSION}")
            for name in PALOMA_SETS
        },
    }
    return LmDataConfig(
        tokenizer=TOKENIZER,
        cache_dir=None,
        components=components,
        train_weights={name: 1.0 if name == "train" else 0.0 for name in components},
        auto_build_caches=False,
        shuffle=False,
    )


@dataclasses.dataclass(frozen=True)
class TokenBatch:
    """``[B, S]`` tokens, segment ids and loss weights of one source."""

    source: str
    tokens: np.ndarray
    segment_ids: np.ndarray
    loss_weight: np.ndarray


def examples_to_batch(source: str, examples) -> TokenBatch:
    grug = [grug_lm_example_from_named(ex) if isinstance(ex, LmExample) else ex for ex in examples]
    tokens = np.stack([np.asarray(ex.tokens, np.int32) for ex in grug])
    weights = np.stack([np.asarray(ex.loss_weight, np.float32) for ex in grug])
    segs = []
    for ex in grug:
        seg = ex.attn_mask.segment_ids
        segs.append(np.zeros(tokens.shape[1], np.int32) if seg is None else np.asarray(seg[0], np.int32))
    return TokenBatch(source, tokens, np.stack(segs), weights)


def off_mesh(fn):
    """Run ``fn`` on a worker thread: the datasets' per-example jit pins outputs to a CPU device, and JAX's mesh
    context (thread-local) would otherwise make it a GPU-mesh computation."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs) -> list:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: list(fn(*args, **kwargs))).result()

    return wrapper


@off_mesh
def paloma_batches(batch_size: int, max_sequences: int | None = None) -> Iterator[TokenBatch]:
    """Every Paloma validation sequence, set by set, in batches of ``batch_size`` (the last batch of a set padded with
    zero-weight copies)."""
    sets = data_config().validation_sets(Axis("position", SEQ_LEN))
    for name in PALOMA_SETS:
        ds = sets[f"paloma/{name}"].as_sync_dataset()
        n = len(ds) if max_sequences is None else min(len(ds), max_sequences)
        for start in range(0, n, batch_size):
            idx = list(range(start, min(n, start + batch_size)))
            batch = examples_to_batch(name, ds.get_batch(idx))
            yield _pad_batch(batch, batch_size)


def _pad_batch(batch: TokenBatch, batch_size: int) -> TokenBatch:
    pad = batch_size - batch.tokens.shape[0]
    if pad == 0:
        return batch

    def rep(x, fill):
        return np.concatenate([x, np.full((pad, x.shape[1]), fill, x.dtype)], axis=0)

    return TokenBatch(batch.source, rep(batch.tokens, 0), rep(batch.segment_ids, 0), rep(batch.loss_weight, 0.0))


@off_mesh
def train_batches(batch_size: int, num_sequences: int, *, seed: int = 0) -> Iterator[TokenBatch]:
    """``num_sequences`` random unseen training-cache sequences (indices past the trained prefix)."""
    sets = data_config().train_sets(Axis("position", SEQ_LEN), key=jax.random.PRNGKey(0))
    ds = sets["train"].as_sync_dataset()
    total = len(ds)
    rng = np.random.default_rng(seed)
    picks = np.sort(rng.choice(np.arange(TRAINED_SEQUENCES, total), size=num_sequences, replace=False))
    for start in range(0, num_sequences, batch_size):
        idx = [int(i) for i in picks[start : start + batch_size]]
        batch = examples_to_batch("train", ds.get_batch(idx))
        yield _pad_batch(batch, batch_size)


def vocab_strings() -> list[str]:
    """Each token id's decoded text."""
    tok = load_tokenizer(TOKENIZER)
    return [tok.decode([i]) for i in range(launch.V16384_VOCAB)]


def save_npz(tag: str, name: str, **arrays: np.ndarray) -> str:
    path = f"{OUTPUT_ROOT}/{tag}/{name}.npz"
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    with fsspec.open(path, "wb") as f:
        f.write(buffer.getvalue())
    logger.info("wrote %s (%.1f MB)", path, buffer.tell() / 1e6)
    return path


def save_json(tag: str, name: str, obj) -> str:
    path = f"{OUTPUT_ROOT}/{tag}/{name}.json"
    with fsspec.open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    logger.info("wrote %s", path)
    return path


def device_batch(batch: TokenBatch, mesh: Mesh) -> tuple[jax.Array, jax.Array, jax.Array]:
    with set_mesh(mesh):
        return (
            jnp.asarray(batch.tokens),
            jnp.asarray(batch.segment_ids),
            jnp.asarray(batch.loss_weight),
        )
