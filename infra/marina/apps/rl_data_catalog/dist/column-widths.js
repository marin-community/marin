/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
window.AtlasColumns = (() => {
  const storageKey = 'atlas.source-column-widths.v1';
  const maximumWidth = 2000;
  let saved = {};
  try {
    const record = JSON.parse(localStorage.getItem(storageKey) || '{}');
    if (record && typeof record === 'object' && !Array.isArray(record)) {
      saved = Object.fromEntries(Object.entries(record).filter(([, width]) => Number.isFinite(width) && width > 0 && width <= maximumWidth));
    }
  } catch { saved = {}; }
  function persist() {
    try { localStorage.setItem(storageKey, JSON.stringify(saved)); } catch { /* Resizing still works without storage. */ }
  }
  function minimumWidth(header, cells) {
    const canvas = document.createElement('canvas'), context = canvas.getContext('2d');
    const content = header.querySelector('button,.header-label'), style = getComputedStyle(content);
    context.font = style.font;
    let width = context.measureText(content.textContent).width + parseFloat(style.paddingLeft) + parseFloat(style.paddingRight) + 18;
    for (const cell of cells) {
      const cellStyle = getComputedStyle(cell), padding = parseFloat(cellStyle.paddingLeft) + parseFloat(cellStyle.paddingRight);
      for (const chip of cell.querySelectorAll('.chip,.difficulty-comparison,.backends-link')) {
        width = Math.max(width, chip.getBoundingClientRect().width + padding + 2);
      }
      if (cell.classList.contains('num') || cellStyle.whiteSpace === 'nowrap') {
        context.font = cellStyle.font;
        width = Math.max(width, context.measureText(cell.textContent).width + padding + 2);
      }
    }
    return Math.ceil(Math.max(72, width));
  }
  function attach(table, columns) {
    table.querySelector('colgroup')?.remove();
    table.classList.remove('resized-columns'); table.style.width = '';
    const headers = [...table.tHead.rows[0].cells];
    const defaults = headers.map(header => header.getBoundingClientRect().width);
    const minimum = headers.map((header, index) => minimumWidth(header, [...table.tBodies[0].rows].map(row => row.cells[index])));
    const widths = columns.map(([key], index) => saved[key] == null ? defaults[index] : Math.max(minimum[index], saved[key]));
    const group = document.createElement('colgroup');
    for (const width of widths) { const col = document.createElement('col'); col.style.width = `${width}px`; group.append(col); }
    function apply() {
      table.classList.add('resized-columns');
      table.style.width = `${widths.reduce((sum, width) => sum + width, 0)}px`;
      [...group.children].forEach((col, index) => { col.style.width = `${widths[index]}px`; });
    }
    if (columns.some(([key]) => saved[key] != null)) { table.insertBefore(group, table.tHead); apply(); }
    for (const [index, header] of headers.entries()) {
      header.querySelector('.column-resize-handle')?.remove();
      const [key, title] = columns[index], handle = document.createElement('span');
      handle.className = 'column-resize-handle'; handle.tabIndex = 0;
      handle.setAttribute('role', 'separator'); handle.setAttribute('aria-orientation', 'vertical');
      handle.setAttribute('aria-label', `Resize ${title} column`);
      handle.setAttribute('aria-controls', table.id);
      handle.setAttribute('aria-valuemin', minimum[index]);
      handle.setAttribute('aria-valuemax', Math.max(maximumWidth, minimum[index]));
      handle.setAttribute('aria-valuenow', Math.round(widths[index]));
      handle.setAttribute('aria-valuetext', `${Math.round(widths[index])} pixels`);
      handle.title = `Drag to resize ${title}. Arrow keys resize; Home uses the minimum; double-click resets this column.`;
      function resize(width) {
        widths[index] = Math.max(minimum[index], Math.min(Math.max(maximumWidth, minimum[index]), Math.round(width)));
        if (!group.isConnected) table.insertBefore(group, table.tHead);
        saved[key] = widths[index]; apply();
        handle.setAttribute('aria-valuenow', widths[index]);
        handle.setAttribute('aria-valuetext', `${widths[index]} pixels`);
      }
      handle.addEventListener('pointerdown', event => {
        if (event.button !== 0) return;
        event.preventDefault(); event.stopPropagation();
        const start = event.clientX, width = header.getBoundingClientRect().width;
        handle.setPointerCapture(event.pointerId); handle.focus();
        table.classList.add('resizing-column');
        function move(next) { resize(width + next.clientX - start); }
        function finish(next) {
          handle.removeEventListener('pointermove', move);
          handle.removeEventListener('pointerup', finish); handle.removeEventListener('pointercancel', finish);
          if (handle.hasPointerCapture(next.pointerId)) handle.releasePointerCapture(next.pointerId);
          table.classList.remove('resizing-column'); persist();
        }
        handle.addEventListener('pointermove', move);
        handle.addEventListener('pointerup', finish); handle.addEventListener('pointercancel', finish);
      });
      handle.addEventListener('keydown', event => {
        if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
        event.preventDefault(); event.stopPropagation();
        const step = event.shiftKey ? 64 : 16;
        resize(event.key === 'Home' ? minimum[index] : event.key === 'End' ? maximumWidth : widths[index] + (event.key === 'ArrowRight' ? step : -step));
        persist();
      });
      handle.addEventListener('dblclick', event => {
        event.preventDefault(); event.stopPropagation(); delete saved[key]; persist(); attach(table, columns);
      });
      header.append(handle);
    }
  }
  function reset(table, columns) {
    saved = {};
    try { localStorage.removeItem(storageKey); } catch { /* No persisted widths to clear. */ }
    attach(table, columns);
  }
  return {attach, reset};
})();
