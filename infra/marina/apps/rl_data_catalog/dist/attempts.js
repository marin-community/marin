/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
(async () => {
  const parameters = new URLSearchParams(location.search), source = parameters.get('source');
  let measurement = parameters.get('measurement'), offset = 0;
  const PAGE = 50;
  try {
    const evidence = await json(`api/source?${new URLSearchParams({source_id: source})}`);
    $('title').textContent = `Difficulty attempts · ${evidence.declaration.display_name}`;
    if (contentWithheld(evidence)) {
      $('message').textContent = contentWithheldMessage(evidence); return;
    }
    const models = [...evidence.difficulty.models, ...evidence.difficulty.historical].filter(model => model.measurement_id);
    const label = node('label', 'Solver measurement ', 'measurement-picker'), select = node('select');
    for (const model of models) {
      const option = node('option', `${model.solver} · ${model.created_at} · ${model.status}`);
      option.value = model.measurement_id; select.append(option);
    }
    measurement ||= models[0]?.measurement_id;
    if (!measurement) { $('message').textContent = 'No recorded difficulty attempts for this source.'; return; }
    select.value = measurement; label.append(select); $('content').append(label);
    const pager = node('nav', undefined, 'review-pager'), previous = node('button', '← Previous'), next = node('button', 'Next →');
    previous.type = next.type = 'button'; pager.append(previous, next);
    const results = node('section'); $('content').append(pager, results);
    async function render() {
      const data = await json(`api/difficulty-attempts?${new URLSearchParams({source_id: source, measurement_id: measurement, limit: PAGE, offset})}`);
      if (contentWithheld(data)) { $('message').textContent = contentWithheldMessage(data); results.replaceChildren(); return; }
      $('message').textContent = data.status === 'not_loaded' ? 'Attempt artifacts have not been loaded for this measurement.'
        : `${data.current_pin ? 'Current pinned input' : 'Historical input'} · ${data.total.toLocaleString()} attempt/outcome records · ${data.measurement.solver} · ${data.measurement.status}${data.measurement.reason ? ` · ${data.measurement.reason}` : ''}`;
      previous.disabled = offset === 0; next.disabled = offset + PAGE >= data.total;
      results.replaceChildren();
      for (const item of data.items) {
        const card = node('section', undefined, 'assessment-card'), link = node(item.has_rollout ? 'a' : 'span', item.task_id);
        if (item.has_rollout) link.href = `rollout.html?${new URLSearchParams({source, measurement, attempt: item.attempt_id})}`;
        card.append(link, node('p', `Recorded attempt score ${item.score ?? 'unscored'} · ${item.solver} · attempt ${item.attempt} · ${item.status}`),
          node('p', `Duration ${item.duration == null ? 'not recorded' : `${item.duration.toFixed(2)} seconds`} · Exception ${item.exception || 'none recorded'}`));
        if (item.reason) card.append(node('p', item.reason));
        if (item.task_outcomes) card.append(node('p', 'Task outcome: ' + item.task_outcomes.map(outcome => `${outcome.status} · ${outcome.reward ?? 'unscored'}${outcome.reason ? ` · ${outcome.reason}` : ''}`).join('; ')));
        results.append(card);
      }
      if (!data.items.length) results.append(node('p', 'No stored attempts on this page.'));
    }
    select.addEventListener('change', () => { measurement = select.value; offset = 0; render().catch(showError); });
    previous.addEventListener('click', () => { offset = Math.max(0, offset - PAGE); render().catch(showError); });
    next.addEventListener('click', () => { offset += PAGE; render().catch(showError); });
    await render();
  } catch (error) { showError(error); }
  function showError(error) { $('message').textContent = error.message; $('message').className = 'error'; }
})();
