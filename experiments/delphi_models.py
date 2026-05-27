# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical Delphi base model registry.

This module exists to keep Delphi base selection boring. Use the named
constants here instead of pasting raw GCS paths into experiment files.

Critical incident context:
issue #4547 used the deprecated v5 isoflop ablation
``isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5`` as the "1e20 Delphi base"
for 12 sweep cells. That checkpoint is not a Delphi scaling-law point. See
``.agents/ops/2026-05-14-wrong-1e20-base-v5-vs-v6.md``.

Path policy:
- ``hf_repo`` identifies the public HF weights. Use it for inference, eval,
  and MODEL_ONLY continued-pretraining warm starts. HF weights do not include
  Levanter optimizer or trainer state.
- ``gcs_run_root`` is the native Levanter run root on GCS. Use it only when a
  workflow needs native full-state checkpoints, especially true midtraining
  resume/staging.
- ``verified_checkpoint_path`` is a concrete native Levanter checkpoint path.
  Use it with ``initialize_from_checkpoint_path`` only when the train config
  explicitly uses ``CheckpointInitMode.MODEL_ONLY``.

There is no "1e20 Delphi" model. Delphi proper starts at 1e21. The closest
sub-1e21 anchor is ``DELPHI_3E20``, one of the seven isoflop-bucket winners
used to fit the Delphi scaling law.

The v5/v6 trap:
- ``adamh_scaling_v5`` suffix on ``isoflop-...`` step names is a deprecated
  heuristic generation. It is not Delphi.
- ``adamh_scaling_v6`` suffix is the current canonical isoflop heuristic.
- ``-v5-XXXXXX`` suffix on
  ``adamh-scaling-ladder-nemotron-optimal-...`` step names is an unrelated
  experiment-iteration tag. Those headline runs still use the v6 heuristic.
"""

from dataclasses import dataclass, field

from marin.execution.executor import ExecutorStep

from experiments.models import ModelConfig, download_model_step

DELPHI_SEQ_LEN = 4096
UNPINNED_HF_REVISION = "main"

DELPHI_POSTMORTEM_PATH = ".agents/ops/2026-05-14-wrong-1e20-base-v5-vs-v6.md"

# Resolved from HuggingFace refs/heads/main with `git ls-remote` on 2026-05-15.
DELPHI_HF_REVISIONS: dict[str, str] = {
    "marin-community/delphi-3e18-447Mparams-1.2Btokens": "d39dd26109500de83cb3a7b79d73420785139246",
    "marin-community/delphi-9e18-550Mparams-2.9Btokens": "24722a6b925deaa9b51d9cb59373a2edf42e959e",
    "marin-community/delphi-2e19-837Mparams-3.6Btokens": "513d4b7333233fa3a63f24430d7ac78237d9db36",
    "marin-community/delphi-3e19-998Mparams-5Btokens": "b1648396f93f50daf561208b3e4cfe080b9d5e2c",
    "marin-community/delphi-9e19-1.4Bparams-10.6Btokens": "d5712937fe6d6666d7205107539b9ec6dc801fa7",
    "marin-community/delphi-2e20-1.9Bparams-14.8Btokens": "6c7305b56385327c60eecbab6482201477c8172d",
    "marin-community/delphi-3e20-2.5Bparams-18.6Btokens": "b585d913d6f72cf86482a7adc71cc77af4b910f3",
    "marin-community/delphi-1e21-3.4Bparams-46.3Btokens": "586f00015fed3a11a72db392c1236d9947e8b652",
    "marin-community/delphi-1e22-9.7Bparams-160Btokens": "ca7b0e7c0a6b9ea8e3a4bbe847efa8b53f793902",
    "marin-community/delphi-1e23-25Bparams-628Btokens": "e175dd1b9f30639b476c7b6c41972f62a4064c31",
}


# ---------------------------------------------------------------------------
# Banned paths - known-wrong checkpoints prior sessions used by mistake.
# ---------------------------------------------------------------------------

DELPHI_BANNED_SUBSTRINGS: frozenset[str] = frozenset(
    {
        # Deprecated v5 isoflop ablation point used as a "1e20 Delphi" stand-in
        # in issue #4547. Use DELPHI_3E20, the 3e20 v6 bucket winner.
        "isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5",
        # Same family, broader: any explicit v5 isoflop suffix in production code.
        "adamh_scaling_v5",
    }
)


def assert_not_banned(path_or_name: str) -> None:
    """Raise if ``path_or_name`` matches a known-wrong base checkpoint."""
    for banned in DELPHI_BANNED_SUBSTRINGS:
        if banned in path_or_name:
            raise ValueError(
                f"Refusing to use banned Delphi base substring {banned!r} in {path_or_name!r}. "
                f"See {DELPHI_POSTMORTEM_PATH} for context. "
                "Use one of the named constants in experiments.delphi_models instead of pasting a GCS path."
            )


def _join_gcs_path(base: str, *parts: str) -> str:
    return "/".join((base.rstrip("/"), *(part.strip("/") for part in parts)))


def _validate_positive_int(field_name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} must be positive, got {value!r}")


@dataclass(frozen=True)
class DelphiModel:
    """A canonical Delphi compute-optimal base model.

    The HF repo is for weights-only use. ``gcs_run_root`` is the matching
    native Levanter run root for full-state checkpoint workflows.

    The optimizer fields below (``peak_lr``, ``peak_adam_lr``, ``beta2``,
    ``epsilon``, etc.) are read verbatim from each pretrain run's
    ``.executor_info`` (v6 isoflop buckets) or the legacy hardcoded
    ``MidtrainingBaseConfig`` (v5 ladder runs at 1e21/1e22). Per the
    canonical comment in ``experiments/exp_delphi_math_10b_midtrain.py``:
    "DO NOT recompute via the heuristic; the W&B config is the canonical
    source of truth for what the weights were optimized against (G4)."
    These runs are finished — the values do not change.
    """

    flops_key: str
    flops_budget: float
    hf_repo: str
    params: int
    hidden_dim: int
    num_layers: int
    batch_size: int
    num_train_steps: int
    gcs_run_root: str
    verified_checkpoint_step: int
    hf_revision: str = ""
    seed: int = 0
    # Pretrain-time AdamH optimizer hparams (keyword-only; verbatim from W&B).
    peak_lr: float = field(kw_only=True)
    peak_adam_lr: float = field(kw_only=True)
    beta2: float = field(kw_only=True)
    epsilon: float = field(kw_only=True)
    beta1: float = field(default=0.9, kw_only=True)
    max_grad_norm: float = field(default=0.1, kw_only=True)
    weight_decay: float = field(default=0.1, kw_only=True)
    z_loss_weight: float = field(default=1e-7, kw_only=True)
    warmup_fraction: float = field(default=0.1, kw_only=True)
    decay_fraction: float = field(default=0.2, kw_only=True)
    min_lr_ratio: float = field(default=0.0, kw_only=True)
    lr_schedule: str = field(default="linear", kw_only=True)
    nesterov: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        if self.flops_key in {"1e20", "1e+20"}:
            raise ValueError("There is no 1e20 Delphi model. Use DELPHI_3E20 if you mean the 3e20 bucket winner.")
        if "+" in self.flops_key:
            raise ValueError(f"Use canonical flops keys without '+', got {self.flops_key!r}")
        if self.flops_budget <= 0:
            raise ValueError(f"flops_budget must be positive, got {self.flops_budget!r}")
        if not self.hf_repo.startswith("marin-community/delphi-"):
            raise ValueError(f"hf_repo must point at the marin-community Delphi namespace, got {self.hf_repo!r}")
        hf_revision = self.hf_revision or DELPHI_HF_REVISIONS.get(self.hf_repo)
        if not hf_revision:
            raise ValueError(f"No pinned HF revision registered for {self.hf_repo!r}")
        object.__setattr__(self, "hf_revision", hf_revision)
        if not self.gcs_run_root.startswith("gs://"):
            raise ValueError(f"gcs_run_root must be a GCS URI, got {self.gcs_run_root!r}")
        if "/checkpoints/step-" in self.gcs_run_root:
            raise ValueError(f"gcs_run_root must be a run root, not a concrete checkpoint: {self.gcs_run_root!r}")

        for field_name in ("params", "hidden_dim", "num_layers", "batch_size", "num_train_steps"):
            _validate_positive_int(field_name, getattr(self, field_name))
        _validate_positive_int("verified_checkpoint_step", self.verified_checkpoint_step)
        if self.verified_checkpoint_step > self.num_train_steps:
            raise ValueError(
                "verified_checkpoint_step cannot exceed num_train_steps: "
                f"{self.verified_checkpoint_step} > {self.num_train_steps}"
            )
        for optim_field in ("peak_lr", "peak_adam_lr", "epsilon", "beta2", "weight_decay"):
            value = getattr(self, optim_field)
            if not (0 < value <= 1):
                raise ValueError(f"{optim_field}={value!r} must be in (0, 1]")
        if not (0 <= self.min_lr_ratio < 1):
            raise ValueError(f"min_lr_ratio={self.min_lr_ratio!r} must be in [0, 1)")
        if self.max_grad_norm <= 0:
            raise ValueError(f"max_grad_norm={self.max_grad_norm!r} must be positive")
        if not (0 < self.warmup_fraction < 1) or not (0 < self.decay_fraction <= 1):
            raise ValueError(
                f"warmup_fraction/decay_fraction out of range: {self.warmup_fraction}, {self.decay_fraction}"
            )
        assert_not_banned(self.gcs_run_root)

    @property
    def tokens(self) -> int:
        return self.num_train_steps * self.batch_size * DELPHI_SEQ_LEN

    @property
    def hf_revision_is_pinned(self) -> bool:
        return self.hf_revision != UNPINNED_HF_REVISION

    @property
    def model_config(self) -> ModelConfig:
        return ModelConfig(hf_repo_id=self.hf_repo, hf_revision=self.hf_revision)

    @property
    def gcs_hf_export_path(self) -> str:
        """GCS mirror of the HF-format export, not a native full-state checkpoint."""
        return _join_gcs_path(self.gcs_run_root, "hf")

    @property
    def gcs_checkpoint_root(self) -> str:
        """Permanent native Levanter checkpoint directory under ``gcs_run_root``."""
        return _join_gcs_path(self.gcs_run_root, "checkpoints")

    @property
    def verified_checkpoint_path(self) -> str:
        """Concrete native checkpoint path verified for this registry entry."""
        return self.levanter_checkpoint_path()

    def levanter_checkpoint_path(self, step: int | None = None) -> str:
        """Return ``gcs_run_root/checkpoints/step-N`` for native Levanter loading."""
        checkpoint_step = self.verified_checkpoint_step if step is None else step
        _validate_positive_int("step", checkpoint_step)
        return _join_gcs_path(self.gcs_checkpoint_root, f"step-{checkpoint_step}")

    def download_step(self) -> ExecutorStep:
        """Mirror HF weights to GCS for eval/inference/MODEL_ONLY warm starts."""
        return download_model_step(self.model_config)


# ---------------------------------------------------------------------------
# Headline compute-optimal models (held-out forecast targets)
# Source: experiments/exp1337_delphi_suite.py + exp1337_eval_suite.py:183-186
# ---------------------------------------------------------------------------

DELPHI_1E21 = DelphiModel(
    flops_key="1e21",
    flops_budget=1e21,
    hf_repo="marin-community/delphi-1e21-3.4Bparams-46.3Btokens",
    params=3_400_000_000,
    hidden_dim=2560,
    num_layers=26,
    batch_size=512,
    num_train_steps=22_057,
    gcs_run_root="gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021",
    verified_checkpoint_step=21_979,
    # v5 ladder hparams — lifted verbatim from `experiments/exp_delphi_math_10b_midtrain.py:228`,
    # which itself read them from the pretrain W&B config (v5 ladder runs do not write a
    # resolved train_config to `.executor_info`). Framework-default weight_decay/z_loss/etc.
    peak_lr=7.425e-3,
    peak_adam_lr=4.314e-4,
    beta2=0.99920,
    epsilon=2.81e-8,
)

DELPHI_1E22 = DelphiModel(
    flops_key="1e22",
    flops_budget=1e22,
    hf_repo="marin-community/delphi-1e22-9.7Bparams-160Btokens",
    params=9_700_000_000,
    hidden_dim=3840,
    num_layers=37,
    batch_size=1024,
    num_train_steps=38_235,
    gcs_run_root="gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e",
    verified_checkpoint_step=38_206,
    # v5 ladder hparams — from `experiments/exp_delphi_math_10b_midtrain.py:253`.
    peak_lr=7.231797280729413e-3,
    peak_adam_lr=3.276222099351447e-4,
    beta2=0.9984011994401821,
    epsilon=3.70426657045089e-8,
)

# 1e23 is intentionally not registered yet. We know the public identity and run
# root, but not the architecture and concrete checkpoint step from a verified
# source. Do not add DELPHI_1E23 until those fields are filled with real values.
DELPHI_1E23_HF_REPO = "marin-community/delphi-1e23-25Bparams-628Btokens"
DELPHI_1E23_HF_REVISION = DELPHI_HF_REVISIONS[DELPHI_1E23_HF_REPO]
DELPHI_1E23_GCS_RUN_ROOT = "gs://marin-us-central2/adamh-scaling-ladder-nemotron-optimal-1e+23-v5-27f2fb"

# ---------------------------------------------------------------------------
# ISOFlop-bucket winners - the seven scaling-law fit points (3e18 to 3e20)
# Architecture verified from experiments/exp1337_eval_suite.py:174-180.
# `num_train_steps` is the original trainer schedule length from each run's
# `.executor_info`. `verified_checkpoint_step` is the latest observed native
# checkpoint under the canonical pretraining root.
# ---------------------------------------------------------------------------

# **The CANONICAL 3e20 anchor - use this for 1e20-region midtraining.**
# (There is no 1e20 Delphi; the smallest bucket winner is 3e20.)
DELPHI_3E20 = DelphiModel(
    flops_key="3e20",
    flops_budget=3e20,
    hf_repo="marin-community/delphi-3e20-2.5Bparams-18.6Btokens",
    params=2_500_000_000,
    hidden_dim=2304,
    num_layers=23,
    batch_size=128,
    num_train_steps=35_510,
    gcs_run_root="gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d2304-L23-B128-adamh_scaling_v6",
    verified_checkpoint_step=35_408,
    # v6 isoflop bucket winner — verbatim from .executor_info.config.train_config.optimizer.
    peak_lr=0.004878221521523273,
    peak_adam_lr=0.00033996075743610645,
    beta2=0.9998000100000001,
    epsilon=3.569823791288878e-08,
)

DELPHI_2E20 = DelphiModel(
    flops_key="2e20",
    flops_budget=2e20,
    hf_repo="marin-community/delphi-2e20-1.9Bparams-14.8Btokens",
    params=1_900_000_000,
    hidden_dim=2048,
    num_layers=21,
    batch_size=64,
    num_train_steps=56_392,
    gcs_run_root="gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+20-d2048-L21-B64-adamh_scaling_v6",
    verified_checkpoint_step=56_392,
    peak_lr=0.003694876216643079,
    peak_adam_lr=0.0002695686680789442,
    beta2=0.9999,
    epsilon=4.502006886217921e-08,
)

DELPHI_9E19 = DelphiModel(
    flops_key="9e19",
    flops_budget=9e19,
    hf_repo="marin-community/delphi-9e19-1.4Bparams-10.6Btokens",
    params=1_400_000_000,
    hidden_dim=1792,
    num_layers=18,
    batch_size=64,
    num_train_steps=40_163,
    gcs_run_root="gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+19-d1792-L18-B64-adamh_scaling_v6",
    verified_checkpoint_step=40_163,
    peak_lr=0.004089070809779834,
    peak_adam_lr=0.0003191861039276992,
    beta2=0.9999,
    epsilon=3.802170536455748e-08,
)

DELPHI_3E19 = DelphiModel(
    flops_key="3e19",
    flops_budget=3e19,
    hf_repo="marin-community/delphi-3e19-998Mparams-5Btokens",
    params=998_000_000,
    hidden_dim=1536,
    num_layers=16,
    batch_size=32,
    num_train_steps=38_014,
    gcs_run_root="gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+19-d1536-L16-B32-adamh_scaling_v6",
    verified_checkpoint_step=37_870,
    peak_lr=0.0036221806669679214,
    peak_adam_lr=0.0003285714083231954,
    beta2=0.9999,
    epsilon=3.693565445007487e-08,
)

DELPHI_2E19 = DelphiModel(
    flops_key="2e19",
    flops_budget=2e19,
    hf_repo="marin-community/delphi-2e19-837Mparams-3.6Btokens",
    params=837_000_000,
    hidden_dim=1408,
    num_layers=15,
    batch_size=16,
    num_train_steps=54_915,
    gcs_run_root="gs://marin-us-central2/checkpoints/isoflop/isoflop-2e+19-d1408-L15-B16-adamh_scaling_v6",
    verified_checkpoint_step=54_915,
    peak_lr=0.002820610414485248,
    peak_adam_lr=0.0002728525996258053,
    beta2=0.9999,
    epsilon=4.447822749954927e-08,
)

DELPHI_9E18 = DelphiModel(
    flops_key="9e18",
    flops_budget=9e18,
    hf_repo="marin-community/delphi-9e18-550Mparams-2.9Btokens",
    params=550_000_000,
    hidden_dim=1152,
    num_layers=12,
    batch_size=16,
    num_train_steps=44_096,
    gcs_run_root="gs://marin-us-central2/checkpoints/isoflop/isoflop-9e+18-d1152-L12-B16-adamh_scaling_v6",
    verified_checkpoint_step=44_096,
    peak_lr=0.003011461785505323,
    peak_adam_lr=0.000304311605183897,
    beta2=0.9999,
    epsilon=3.988017477238883e-08,
)

DELPHI_3E18 = DelphiModel(
    flops_key="3e18",
    flops_budget=3e18,
    hf_repo="marin-community/delphi-3e18-447Mparams-1.2Btokens",
    params=447_000_000,
    hidden_dim=1024,
    num_layers=11,
    batch_size=8,
    num_train_steps=37_335,
    gcs_run_root="gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+18-d1024-L11-B8-adamh_scaling_v6",
    verified_checkpoint_step=37_001,
    peak_lr=0.0027599905274620106,
    peak_adam_lr=0.00033154735825338737,
    beta2=0.9999,
    epsilon=3.6604122149949323e-08,
)

# ---------------------------------------------------------------------------
# Lookup
# ---------------------------------------------------------------------------

ALL_DELPHI_MODELS: tuple[DelphiModel, ...] = (
    DELPHI_3E18,
    DELPHI_9E18,
    DELPHI_2E19,
    DELPHI_3E19,
    DELPHI_9E19,
    DELPHI_2E20,
    DELPHI_3E20,
    DELPHI_1E21,
    DELPHI_1E22,
)

DELPHI_BY_FLOPS_KEY: dict[str, DelphiModel] = {m.flops_key: m for m in ALL_DELPHI_MODELS}
DELPHI_BY_HF_REPO: dict[str, DelphiModel] = {m.hf_repo: m for m in ALL_DELPHI_MODELS}
DELPHI_FLOPS_KEY_ALIASES: dict[str, str] = {m.flops_key.replace("e", "e+"): m.flops_key for m in ALL_DELPHI_MODELS}


def get_delphi_model(flops_key: str) -> DelphiModel:
    """Return a verified Delphi model by canonical string key.

    Accepts keys like ``"3e20"`` or ``"3e+20"``. ``"1e20"`` fails loudly
    because no such Delphi model exists.
    """
    normalized = flops_key.strip().lower().replace(" ", "")
    if normalized in {"1e20", "1e+20"}:
        raise ValueError("There is no 1e20 Delphi model. Use DELPHI_3E20 only if you mean the 3e20 bucket winner.")

    normalized = DELPHI_FLOPS_KEY_ALIASES.get(normalized, normalized)
    model = DELPHI_BY_FLOPS_KEY.get(normalized)
    if model is None:
        allowed = ", ".join(DELPHI_BY_FLOPS_KEY)
        raise ValueError(f"Unknown Delphi flops key {flops_key!r}. Verified keys: {allowed}.")
    return model


if __name__ == "__main__":
    print(f"{len(ALL_DELPHI_MODELS)} verified Delphi models registered:\n")
    for m in ALL_DELPHI_MODELS:
        hf_pin = "pinned" if m.hf_revision_is_pinned else "UNPINNED"
        print(
            f"  {m.flops_key:<5} d={m.hidden_dim:<5} L={m.num_layers:<3} "
            f"B={m.batch_size:<5} N={m.num_train_steps:<7} "
            f"params={m.params/1e9:.2f}B tokens={m.tokens/1e9:.2f}B"
        )
        print(f"        HF weights:        {m.hf_repo}@{m.hf_revision} ({hf_pin})")
        print(f"        GCS HF export:     {m.gcs_hf_export_path}")
        print(f"        Levanter ckpt:     {m.verified_checkpoint_path}")
