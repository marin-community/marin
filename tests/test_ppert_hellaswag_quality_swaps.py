from experiments.domain_phase_mix.exploratory.two_phase_many import plot_ppert_hellaswag_quality_swaps as plotter


def test_quality_swap_metric_effects_exclude_redundant_metric_leaves():
    effects = plotter.build_metric_effects()

    assert "logprob" not in set(effects["metric_leaf"])
    assert "choice_prob_norm" not in set(effects["metric_leaf"])


def test_quality_swap_plot_downloads_use_high_resolution_png():
    options = plotter.PLOTLY_CONFIG["toImageButtonOptions"]

    assert options["format"] == "png"
    assert options["scale"] >= 3.0
