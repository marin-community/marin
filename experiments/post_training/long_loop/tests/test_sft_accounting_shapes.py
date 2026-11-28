# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The accounting step must handle the consumed-config shapes real rounds produce.

The Exp7 SFT stage was stopped externally: its adopted artifact preserves the launcher's
submitted request config (no ``format`` key) instead of an SFTSpec consumed config. The
committed fixture is the real consumed-config.json of that stage
(sha256 d6567a95ec0dd24120dce47de53b08f1964ff28f34fb58c716adf4f652eb5283).
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from levanter.main.train_lm import TrainLmConfig

from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig  # registers the MuonH draccus choice
from experiments.post_training.long_loop.sft_accounting import _native_train_config, accounting_train_config

FIXTURE = Path(__file__).parent / "data" / "sft655-consumed-config.json"


def test_real_launcher_request_consumed_config_decodes():
    """The real externally stopped SFT's request config decodes as the launcher-request shape."""
    document = json.loads(FIXTURE.read_text())
    assert "format" not in document
    config, shape = accounting_train_config(document)
    assert isinstance(config.optimizer, GrugMoeMuonHConfig)  # the restored MuonH discriminator
    assert shape == "launcher-request"
    assert isinstance(config, TrainLmConfig)
    assert config.trainer.train_batch_size == 256
    assert config.train_seq_len == 262144


def test_sftspec_consumed_config_decodes():
    """The SFTSpec shape keeps decoding as before."""
    request = json.loads(FIXTURE.read_text())["train_config"]
    # A genuine SFTSpec consumed config carries the trainer's own canonical encodings: the
    # policy string, the encoded timedelta strings, the choice discriminators.
    canonical = _native_train_config({"train_config": request})
    document = {"format": "SFTSpec", "train_config": canonical}
    config, shape = accounting_train_config(document)
    assert shape == "SFTSpec"
    assert isinstance(config, TrainLmConfig)


def test_legacy_consumed_config_is_rejected():
    """The legacy Grug shape still cannot drive native accounting."""
    with pytest.raises(ValueError, match="native SFTSpec producer"):
        accounting_train_config({"trainer": {"trainer": {}}})


def test_real_config_decodes_in_fresh_subprocess(tmp_path):
    """The accounting step registers the MuonH choice itself: a fresh interpreter decodes.

    In-region the step runs without the test module's imports; the decode must not depend on
    any prior import of the registering module.
    """
    script = "\n".join(
        [
            "import json",
            "from experiments.post_training.long_loop.sft_accounting import accounting_train_config",
            f"document = json.loads(open({str(FIXTURE)!r}).read())",
            "config, shape = accounting_train_config(document)",
            "assert shape == 'launcher-request'",
            "assert config.trainer.train_batch_size == 256",
            "assert config.train_seq_len == 262144",
            "print('DECODE-OK')",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[4],
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "DECODE-OK" in result.stdout
