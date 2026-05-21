# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Additional Delphi-family base checkpoints not in `delphi.py`.

`1e20_iso` is the 1.9B Delphi stand-in base that Ahmed's midtraining sweep
(issue #4547) initialized from for all `1e20_*` runs. The architecture is
`d2048-L21-B128` trained for 3e+20 FLOPs under the `adamh_scaling_v5` sweep —
distinct from Rohith's `delphi.py` 3e20 (which is `d2304-L23-B128-v6`).

Verified path (resolved by reading `.executor_info` on the midtrain run
`delphi-1e20-p33m67-4p94b-lr0.67-7c32da`):

    initialize_from_checkpoint_path =
        mirror://checkpoints/isoflop/isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5/checkpoints/step-46915

The HF version of this checkpoint lives in `gs://marin-us-central2` (NOT
`gs://marin-us-east5`). Final HF step is 47063. With `MARIN_PREFIX=gs://marin-us-east5`,
loading this base triggers a cross-region read at SFT init time. That's a
one-time ~few-GB model-weight download per SFT job — slow but tractable.
"""

DELPHI_EXTRA_BASE_CHECKPOINTS: dict[str, str] = {
    # Absolute path: this base's HF version only exists in central2. Using a full
    # gs:// URI here bypasses MARIN_PREFIX so the launcher can run with
    # MARIN_PREFIX=east5 (matching Delphi 1e21/1e22 + midtrained) without losing
    # access to 1e20_iso. os.path.join treats an absolute path as overriding the
    # prefix, so _resolve_latest_hf_checkpoint handles this transparently.
    "1e20_iso": "gs://marin-us-central2/checkpoints/isoflop/isoflop-3e+20-d2048-L21-B128-adamh_scaling_v5/hf",
}
