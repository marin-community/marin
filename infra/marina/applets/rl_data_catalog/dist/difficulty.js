/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
window.AtlasDifficulty = (() => {
  const titles = {small: "Small", large: "Large", hosted: "Hosted", followup: "Follow-up"};
  function comparison(models) {
    const chart = document.createElement("div");
    chart.className = "difficulty-comparison";
    for (const model of models) {
      const row = document.createElement("div");
      row.className = `difficulty-model difficulty-${model.size}`;
      const name = document.createElement("span");
      name.className = "difficulty-model-label";
      name.textContent = titles[model.size] || model.size;
      const meter = document.createElement("span");
      meter.className = "difficulty-track";
      meter.setAttribute("aria-hidden", "true");
      const fill = document.createElement("span");
      fill.className = "difficulty-fill";
      fill.style.width = `${model.verified ? 100 * model.solved / model.verified : 0}%`;
      meter.append(fill);
      const score = document.createElement("span");
      score.className = "difficulty-score";
      score.textContent = model.verified ? `${Math.round(100 * model.solved / model.verified)}% · ${model.solved}/${model.verified}` : "No valid scores";
      const interval = model.wilson_95 ? ` · 95% interval ${(100 * model.wilson_95[0]).toFixed(1)}–${(100 * model.wilson_95[1]).toFixed(1)}%` : "";
      row.title = `${model.model || titles[model.size]} · ${model.solved}/${model.verified} solved · ${model.unverified} unverified${interval}`;
      row.setAttribute("aria-label", `${name.textContent}: ${row.title}`);
      row.append(name, meter, score);
      chart.append(row);
    }
    return chart;
  }
  return {comparison};
})();
