/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
window.AtlasDifficulty = (() => {
  const titles = {small: "Small", large: "Large", hosted: "Hosted", followup: "Follow-up"};
  function modelLabel(model) {
    const name = model.model?.split("/").at(-1) || model.display_name || "Native verifier recheck";
    return model.measurement_status === "current" ? `${titles[model.size]} · ${name}` : name;
  }
  function compactModelLabel(model) {
    const names = {
      "Qwen/Qwen3-Coder-30B-A3B-Instruct": "Qwen 30B",
      "Qwen/Qwen3.5-122B-A10B": "Qwen 122B",
      "zai-org/GLM-5.3": "GLM 5.3",
    };
    return names[model.model] || model.display_name || titles[model.size] || model.model?.split("/").at(-1) || "Model";
  }
  function currentLarge(summary) {
    return summary?.status === "current" ? summary.models.find(model => model.size === "large" && model.measurement_status === "current") : undefined;
  }
  function comparison(models, summary, compact = false) {
    const chart = document.createElement("div");
    chart.className = `difficulty-comparison${compact ? " difficulty-comparison-compact" : ""}`;
    if (summary && summary.status !== "current") {
      const status = document.createElement("span");
      status.className = "difficulty-evidence-status";
      status.textContent = summary.status === "invalid" ? "Protocol mismatch" : "Historical";
      status.title = summary.status_note;
      chart.append(status);
    }
    if (summary?.ordering_warning) {
      const warning = document.createElement("span");
      warning.className = compact ? "difficulty-warning-icon" : "difficulty-ordering-warning";
      warning.textContent = compact ? "⚠" : `⚠ ${summary.ordering_warning}`;
      warning.title = compact ? "Ordering warning; open the review for details" : summary.ordering_warning;
      warning.setAttribute("aria-label", warning.title);
      chart.append(warning);
    }
    for (const model of models) {
      const row = document.createElement("div");
      row.className = `difficulty-model difficulty-${model.size}`;
      const name = document.createElement("span");
      name.className = "difficulty-model-label";
      name.textContent = compact ? compactModelLabel(model) : modelLabel(model);
      const meter = document.createElement("span");
      meter.className = "difficulty-track";
      meter.setAttribute("aria-hidden", "true");
      const fill = document.createElement("span");
      fill.className = "difficulty-fill";
      fill.style.width = `${model.verified ? 100 * model.solved / model.verified : 0}%`;
      meter.append(fill);
      const score = document.createElement("span");
      score.className = "difficulty-score";
      score.textContent = model.verified ? `${Math.round(100 * model.solved / model.verified)}%${compact ? "" : ` · ${model.solved}/${model.verified}`}` : "No scores";
      const interval = model.wilson_95 ? ` · 95% interval ${(100 * model.wilson_95[0]).toFixed(1)}–${(100 * model.wilson_95[1]).toFixed(1)}%` : "";
      row.title = compact
        ? `${model.model || model.display_name || "Model"} · ${score.textContent}; open the review for attempts and verifier results`
        : `${model.model || model.display_name || "Native verifier recheck"} · ${model.provider || "Provider recorded in run settings"} · ${model.measurement_status || "historical"} · ${model.solved}/${model.verified} solved · ${model.unverified} unverified${interval}`;
      row.setAttribute("aria-label", `${name.textContent}: ${row.title}`);
      row.append(name, meter, score);
      chart.append(row);
    }
    return chart;
  }
  return {comparison, modelLabel, currentLarge};
})();
