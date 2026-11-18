/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
// This fallback must not depend on any other page bundle.
window.AtlasSourceBootstrap = (() => {
  const title = document.getElementById('title');
  const message = document.getElementById('message');
  const sections = document.getElementById('sections');
  const reload = document.getElementById('source-reload');
  let started = false, failed = false;
  reload.href = location.href;
  reload.addEventListener('click', event => { event.preventDefault(); location.reload(); });
  function fail(error) {
    failed = true;
    title.textContent = 'Source evidence';
    message.textContent = error.message;
    message.classList.add('error'); message.setAttribute('role', 'alert');
    reload.hidden = false; sections.setAttribute('aria-busy', 'false');
  }
  window.addEventListener('error', event => {
    if (event.target instanceof HTMLScriptElement) {
      fail(new Error(`Required page asset did not load: ${new URL(event.target.src).pathname.split('/').pop()}`));
    }
  }, true);
  document.addEventListener('DOMContentLoaded', () => {
    if (!started && !failed) fail(new Error('Source page scripts did not start. Reload this page to try again.'));
  });
  function run(start) {
    started = true;
    if (failed) return;
    if (typeof window.node !== 'function') { fail(new Error('Required page asset did not load: common.js')); return; }
    if (!window.AtlasDifficulty) { fail(new Error('Required page asset did not load: difficulty.js')); return; }
    message.textContent = ''; message.classList.remove('error');
    message.setAttribute('role', 'status'); reload.hidden = true;
    try { start(); } catch (error) { fail(error); }
  }
  return {run, fail};
})();
