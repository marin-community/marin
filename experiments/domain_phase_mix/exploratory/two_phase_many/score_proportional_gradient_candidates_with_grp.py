# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score proportional gradient-step candidates with the 60M GRP no-L2 model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px

from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import grp_no_l2_exact as grp

ROOT = Path("experiments/domain_phase_mix/exploratory/two_phase_many")
GRP_DATA_DIR = ROOT / "collaborator_scaling_data_packet_20260430/data/grp_no_l2"
CANDIDATE_DIR = (
    ROOT / "reference_outputs/proportional_perturbation_scale_transfer_20260507/gradient_step_candidates_domain_only"
)
OUTPUT_CSV = CANDIDATE_DIR / "candidate_summary_with_grp_no_l2.csv"


def candidate_weight_tensor(weights_frame: pd.DataFrame, domain_names: list[str]) -> np.ndarray:
    weights = np.zeros((len(weights_frame), 2, len(domain_names)), dtype=float)
    for phase_idx, phase_name in enumerate(("phase_0", "phase_1")):
        for domain_idx, domain in enumerate(domain_names):
            column = f"{phase_name}_{domain}"
            if column not in weights_frame:
                raise ValueError(f"Missing candidate weight column {column}")
            weights[:, phase_idx, domain_idx] = weights_frame[column].to_numpy(dtype=float)
    return grp.normalize_rows(weights)


def load_grp_model() -> tuple[grp.GenericFamilyPacket, grp.GenericFamilyPenaltyCalibrationSurrogate]:
    packet = grp.load_packet(GRP_DATA_DIR, target=grp.DEFAULT_TARGET)
    params = grp.included_no_l2_best_params(GRP_DATA_DIR)
    model = grp.build_model(packet, params)
    model.fit(packet.base.w, packet.base.y)
    return packet, model


def proportional_prediction(
    packet: grp.GenericFamilyPacket, model: grp.GenericFamilyPenaltyCalibrationSurrogate
) -> tuple[float, float]:
    matches = packet.base.frame["run_name"].astype(str) == "baseline_proportional"
    if not matches.any():
        raise ValueError("GRP fit panel is missing baseline_proportional")
    index = int(np.nonzero(matches.to_numpy())[0][0])
    predicted = float(model.predict(packet.base.w[index : index + 1])[0])
    actual = float(packet.base.y[index])
    return predicted, actual


def write_plot(frame: pd.DataFrame) -> None:
    figure = px.scatter(
        frame,
        x="actual_tv",
        y="grp_no_l2_predicted_delta_vs_proportional",
        color="construction",
        symbol="source_scale",
        hover_data=[
            "candidate_id",
            "predicted_60m_bpb_effect",
            "predicted_100m_bpb_effect",
            "grp_no_l2_predicted_bpb",
            "max_domain",
            "max_weight",
        ],
        title="GRP no-L2 60M prediction for proportional gradient-step candidates",
        labels={
            "actual_tv": "TV from proportional",
            "grp_no_l2_predicted_delta_vs_proportional": "GRP predicted BPB delta vs proportional",
        },
    )
    figure.add_hline(y=0.0, line_dash="dash", line_color="black")
    figure.write_html(CANDIDATE_DIR / "grp_no_l2_candidate_predictions.html")


def main() -> None:
    summary = pd.read_csv(CANDIDATE_DIR / "candidate_summary.csv")
    weights = pd.read_csv(CANDIDATE_DIR / "candidate_weights.csv")
    packet, model = load_grp_model()

    candidate_weights = candidate_weight_tensor(weights, packet.base.domain_names)
    predicted_bpb = model.predict(candidate_weights)
    proportional_predicted, proportional_actual = proportional_prediction(packet, model)

    scored = summary.merge(
        weights[["candidate_id", "source_scale", "construction"]],
        on=["candidate_id", "source_scale", "construction"],
        how="inner",
        validate="one_to_one",
    )
    if len(scored) != len(summary):
        raise ValueError(f"Expected {len(summary)} scored rows, got {len(scored)}")
    scored["grp_no_l2_predicted_bpb"] = predicted_bpb
    scored["grp_no_l2_proportional_predicted_bpb"] = proportional_predicted
    scored["grp_no_l2_proportional_actual_bpb"] = proportional_actual
    scored["grp_no_l2_predicted_delta_vs_proportional"] = predicted_bpb - proportional_predicted
    scored["grp_no_l2_rank"] = scored["grp_no_l2_predicted_bpb"].rank(method="min", ascending=True).astype(int)
    scored = scored.sort_values(["grp_no_l2_predicted_bpb", "actual_tv", "candidate_id"])
    scored.to_csv(OUTPUT_CSV, index=False)
    write_plot(scored)

    print(f"Wrote GRP-scored candidates to {OUTPUT_CSV}")
    display_columns = [
        "candidate_id",
        "construction",
        "actual_tv",
        "grp_no_l2_predicted_delta_vs_proportional",
        "predicted_60m_bpb_effect",
        "predicted_100m_bpb_effect",
        "max_domain",
        "max_weight",
    ]
    print(scored[display_columns].head(12).to_string(index=False))
    print("\nTV summary:")
    scored["rounded_tv"] = scored["actual_tv"].round(3)
    print(
        scored.groupby("rounded_tv", as_index=False)["grp_no_l2_predicted_delta_vs_proportional"]
        .agg(["min", "median", "max"])
        .reset_index()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
