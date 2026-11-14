/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
// Keep solver rows in protocol order so missing measurements do not shift model positions.
window.AtlasDifficulty = (() => {
  const titles = {small: 'Small', large: 'Large', hosted: 'Hosted', snowball: 'Snowball v0.1'};
  const solvers = Object.keys(titles);
  function selected(summary, solver) {
    return summary?.models.find(model => model.solver === solver);
  }
  function tooltip(model, title) {
    if (!model || model.status !== 'measured') return `${title}: ${model?.reason || 'Not measured for this pinned release.'}`;
    const interval = model.wilson_95;
    return `${title}: ${(100 * model.solve_rate).toFixed(1)}% solved · ${model.solved}/${model.n} · n=${model.n}` +
      ` · Wilson 95% CI ${(100 * interval.low).toFixed(1)}–${(100 * interval.high).toFixed(1)}%` +
      ` · ${model.model.model} @ ${model.model.revision} · ${model.protocol_id}` +
      (model.sample.pre_oracle_filter ? ' · Pre-oracle-filter sample; resampling required after filtering.' : '');
  }
  function comparison(summary) {
    const chart = document.createElement('div'); chart.className = 'difficulty-comparison';
    for (const solver of solvers) {
      const model = selected(summary, solver), measured = model?.status === 'measured';
      const row = document.createElement('div'); row.className = `difficulty-model difficulty-${solver}`;
      const name = document.createElement('span'); name.className = 'difficulty-model-label'; name.textContent = titles[solver];
      const meter = document.createElement('span'); meter.className = `difficulty-track${measured ? '' : ' difficulty-unmeasured'}`;
      meter.setAttribute('aria-hidden', 'true');
      const fill = document.createElement('span'); fill.className = 'difficulty-fill';
      fill.style.width = `${measured ? 100 * model.solve_rate : 0}%`; meter.append(fill);
      const score = document.createElement('span'); score.className = 'difficulty-score';
      score.textContent = measured ? `${Math.round(100 * model.solve_rate)}%` : '—';
      row.title = tooltip(model, titles[solver]); row.setAttribute('aria-label', row.title); row.tabIndex = 0;
      row.append(name, meter, score); chart.append(row);
    }
    return chart;
  }
  return {comparison, selected, solvers, titles};
})();
