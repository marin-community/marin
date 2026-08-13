# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-off diagnostic: which holdout sources go flat under bigger_fused_v1.

For every source in the shared seed-0 holdout, reports holdout rows, prediction
std and within-source Spearman for the fusion candidate, the deployed v3
baseline, and the embedding-only MLP probe (to test whether the doc embedding
itself is what goes flat). Also prints per-content-type train/holdout counts.
Grep FLATSRC / TYPECOUNT to harvest.
"""

import logging

import numpy as np
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging
from scipy import stats

from experiments.datakit.cluster.quality.fast_transformer.data import encode_texts, pack
from experiments.datakit.cluster.quality.fast_transformer.embed_exp import (
    DEFAULT_LABELS,
    FLAT_STD,
    _mlp_probe_preds,
    holdout_id_set,
)
from experiments.datakit.cluster.quality.fast_transformer.inference import predict
from experiments.datakit.cluster.quality.fast_transformer.joined_labels import (
    DEFAULT_JOINED,
    embedding_matrix,
    load_joined,
)
from experiments.datakit.cluster.quality.fast_transformer.scorer import load_pooled_scorer, score_bme

logger = logging.getLogger(__name__)

CANDIDATE_DIR = "s3://marin-us-east-02a/marin/user/muchanem/quality_exp/bigger_fused"
BASELINE_DIR = "s3://marin-us-east-02a/marin/user/rav/quality_v2/models/pooled_glm52_v3/"


def main() -> None:
    configure_logging(logging.INFO)
    configure_coreweave_s3()

    holdout_ids = holdout_id_set(DEFAULT_LABELS)
    joined = load_joined(DEFAULT_JOINED)
    is_eval = np.array([doc_id in holdout_ids for doc_id in joined["id"]])
    quality = np.array(joined["glm52_quality"], dtype=float)
    target = np.array(joined["glm52_score_normalized"], dtype=np.float32)
    types = np.array(joined["glm52_content_type"])
    sources = np.array(joined["glm52_source"])
    texts = [t or "" for t in joined["text"]]
    emb = embedding_matrix(joined["embedding"])
    tr, ev = ~is_eval, is_eval
    ev_idx = np.flatnonzero(ev)

    for name in sorted(set(types.tolist())):
        mask = types == name
        logger.info(
            "TYPECOUNT %-14s train=%-6d holdout=%-5d", name, int((mask & tr).sum()), int((mask & ev).sum())
        )

    ev_texts = [texts[i] for i in ev_idx]
    candidate = load_pooled_scorer(CANDIDATE_DIR)
    raw = encode_texts(candidate.tokenizer_name, ev_texts, candidate.max_tokens)
    ids = pack(raw, candidate.remap, target[ev_idx], candidate.max_tokens).ids
    cand_preds = predict(candidate.model, ids, doc_embed=emb[ev_idx])
    base_preds = score_bme(load_pooled_scorer(BASELINE_DIR), ev_texts)
    probe_preds = _mlp_probe_preds(emb, target, tr, ev)

    q, s = quality[ev], sources[ev]

    def rho(preds, mask):
        r = stats.spearmanr(preds[mask], q[mask]).statistic
        return float(r) if np.isfinite(r) else float("nan")

    logger.info("FLATSRC %-34s %5s | %8s %8s %8s | %7s %7s %7s", "source", "n_ev", "std_v1", "std_v3", "std_pr", "rho_v1", "rho_v3", "rho_pr")
    rows = []
    for name in sorted(set(s.tolist())):
        mask = s == name
        rows.append(
            (
                name,
                int(mask.sum()),
                float(cand_preds[mask].std()),
                float(base_preds[mask].std()),
                float(probe_preds[mask].std()),
                rho(cand_preds, mask),
                rho(base_preds, mask),
                rho(probe_preds, mask),
            )
        )
    rows.sort(key=lambda r: -r[1])
    for name, n, s1, s3, sp, r1, r3, rp in rows:
        flag = " FLAT_V1" if s1 < FLAT_STD else ""
        flag += " FLAT_V3" if s3 < FLAT_STD else ""
        if flag or True:
            logger.info(
                "FLATSRC %-34s %5d | %8.4f %8.4f %8.4f | %+7.3f %+7.3f %+7.3f%s",
                name, n, s1, s3, sp, r1, r3, rp, flag,
            )


if __name__ == "__main__":
    main()
