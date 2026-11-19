/* Copyright The Marin Authors. SPDX-License-Identifier: Apache-2.0 */
// Executed by Playwright, never shipped as application code. Measure unrounded WCAG ratios.
({mode = 'content', target = null} = {}) => {
  const findings = [], measurements = [], textOverflow = [];
  const controls = 'a[href],button,input,select,textarea,summary,[tabindex]:not([tabindex="-1"])';
  const modal = document.querySelector('dialog[open]');
  const visible = e => e.checkVisibility({checkVisibilityCSS: true}) && !e.closest('.sr-only,[hidden]') &&
    (!modal || modal.contains(e)) && e.getBoundingClientRect().width > 0;
  const selector = e => {
    if (e === document.documentElement) return 'html';
    if (e.id) return '#' + CSS.escape(e.id);
    const parts = [];
    while (e && e !== document.body) {
      if (e.id) { parts.unshift('#' + CSS.escape(e.id)); break; }
      parts.unshift(e.tagName.toLowerCase() + ':nth-child(' + (Array.from(e.parentElement.children).indexOf(e) + 1) + ')');
      e = e.parentElement;
    }
    return parts.join(' > ');
  };
  const color = value => {
    // Chromium serializes computed sRGB colors as rgb()/rgba(). Reject unhandled paint.
    const channels = value.match(/^rgba?\(([^)]+)\)$/);
    if (!channels) throw Error('Unsupported computed color: ' + value);
    const values = channels[1].split(',').map(Number);
    return [...values.slice(0, 3), values[3] ?? 1];
  };
  const over = (a, b) => {
    const alpha = a[3] + b[3] * (1 - a[3]);
    return [...a.slice(0, 3).map((v, i) => alpha ? (v * a[3] + b[i] * b[3] * (1 - a[3])) / alpha : 0), alpha];
  };
  const luminance = c => c.slice(0, 3).reduce((sum, v, i) => {
    const s = v / 255;
    return sum + [0.2126, 0.7152, 0.0722][i] * (s <= .04045 ? s / 12.92 : ((s + .055) / 1.055) ** 2.4);
  }, 0);
  const ratio = (a, b) => { const x = luminance(a), y = luminance(b); return (Math.max(x, y) + .05) / (Math.min(x, y) + .05); };
  const pair = (e, ink) => {
    let fg = ink, bg = [0, 0, 0, 0];
    for (let p = e; p; p = p.parentElement) {
      const s = getComputedStyle(p);
      if (s.backgroundImage !== 'none') throw Error('Background image needs sampled paint: ' + selector(p));
      let layer = color(s.backgroundColor);
      if (p instanceof SVGGElement) {
        const rect = p.querySelector(':scope > rect');
        if (rect) layer = over(color(getComputedStyle(rect).fill), layer);
      }
      fg = over(fg, layer); bg = over(bg, layer);
      const opacity = Number(s.opacity); fg[3] *= opacity; bg[3] *= opacity;
    }
    return [over(fg, [255,255,255,1]), over(bg, [255,255,255,1])];
  };
  const check = (e, rule, measured, required, detail = {}) => {
    const row = {selector: selector(e), rule, measured, required, ...detail};
    measurements.push(row);
    if (detail.maximum ? measured > required : measured < required) findings.push(row);
  };
  const contrast = (e, ink, required, rule, detail = {}) => {
    try { const [fg, bg] = pair(e, ink); check(e, rule, ratio(fg, bg), required, {foreground: fg, background: bg, ...detail}); }
    catch (error) { findings.push({selector: selector(e), rule: 'unmeasured-paint', detail: error.message}); }
  };
  if (mode === 'focus') {
    const e = document.activeElement;
    if (!e || !e.matches(controls) || !visible(e)) return {findings, measurements};
    const s = getComputedStyle(e), width = s.outlineStyle === 'none' ? 0 : parseFloat(s.outlineWidth);
    check(e, 'focus-width', width, 2);
    // The offset ring sits on the parent backdrop. Check both that and the component fill.
    contrast(e, color(s.outlineColor), 3, 'focus-element-contrast');
    contrast(e.parentElement, color(s.outlineColor), 3, 'focus-backdrop-contrast', {control: selector(e)});
    const r = e.getBoundingClientRect();
    let exposed = false;
    // A wrapped inline link's union box includes gaps between its actual fragments.
    for (const fragment of e.getClientRects()) {
      const left = Math.max(0, fragment.left), right = Math.min(innerWidth, fragment.right);
      const top = Math.max(0, fragment.top), bottom = Math.min(innerHeight, fragment.bottom);
      if (right > left && bottom > top) {
        for (const x of [.1,.5,.9]) for (const y of [.1,.5,.9]) {
          const hit = document.elementFromPoint(left+(right-left)*x, top+(bottom-top)*y);
          exposed ||= !!hit && e.contains(hit);
        }
      }
    }
    check(e, 'focus-obscured', exposed ? 1 : 0, 1, {bounds: {left:r.left,top:r.top,right:r.right,bottom:r.bottom}});
    return {findings, measurements};
  }
  const root = target ? document.querySelector(target) : (modal || document.body);
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  for (let n; (n = walker.nextNode());) {
    const e = n.parentElement;
    if (!n.textContent.trim() || !visible(e) || e.matches('script,style,option') || e.closest('button:disabled')) continue;
    const range = document.createRange(); range.selectNodeContents(n);
    if (!range.getBoundingClientRect().width) continue;
    const s = getComputedStyle(e), size = parseFloat(s.fontSize);
    const textBounds = range.getBoundingClientRect();
    if (textBounds.right + scrollX > innerWidth) textOverflow.push({selector: selector(e), right: textBounds.right + scrollX, text: n.textContent.trim().slice(0,80)});
    if (e.closest('p,pre,li,dd') && !e.closest('.source-name,.task-description')) {
      for (let parent=e; parent && parent!==root; parent=parent.parentElement) {
        const paint=getComputedStyle(parent), box=parent.getBoundingClientRect();
        if (paint.overflowX === 'hidden' || paint.overflowY === 'hidden') {
          check(e, 'text-clipping', Math.max(0, textBounds.right-box.right, box.left-textBounds.left, textBounds.bottom-box.bottom, box.top-textBounds.top), 1, {maximum: true});
          break;
        }
      }
    }
    // Apply the stricter house contrast to all text, including text in controls and charts.
    contrast(e, color(e instanceof SVGElement ? s.fill : s.color), 7, 'text-contrast', {text: n.textContent.trim().slice(0,160)});
    check(e, 'font-minimum', size, 12);
    if (e.closest('p,li,dd,pre,.intro,.description,.scope-note,.task-description')) {
      check(e, 'prose-font', size, innerWidth <= 480 ? 16 : 14);
      check(e, 'line-height', s.lineHeight === 'normal' ? 1.2 : parseFloat(s.lineHeight) / size, 1.4);
      if (innerWidth > 480 && e.matches('p,pre') && !e.closest('table')) {
        const canvas = document.createElement('canvas'), ctx = canvas.getContext('2d'); ctx.font = s.font;
        check(e, 'prose-line-length', e.clientWidth / ctx.measureText('0').width, 82, {maximum: true});
      }
    }
    if (e.matches('a') && e.closest('p,li,dd') && !e.matches('.secondary'))
      check(e, 'prose-link-underline', s.textDecorationLine.includes('underline') ? 1 : 0, 1);
  }
  const targets = [...(root.matches(controls) ? [root] : []), ...root.querySelectorAll(controls)].filter(visible).map(e => ({e, r:e.getBoundingClientRect()}));
  for (const {e, r} of targets) {
    if (!visible(e) || e.matches(':disabled')) continue;
    const s = getComputedStyle(e);
    if (e.matches('input:not([type="checkbox"]),select,textarea')) {
      contrast(e, color(s.color), 7, 'control-text-contrast');
      check(e, 'font-minimum', parseFloat(s.fontSize), 12);
    }
    if (e.matches('input[placeholder],textarea[placeholder]') && !e.value) {
      const ps = getComputedStyle(e, '::placeholder'), ink = color(ps.color); ink[3] *= Number(ps.opacity);
      contrast(e, ink, 7, 'placeholder-contrast');
    }
    // Standalone controls need 24 px; inline links use WCAG's inline exception.
    const inline = e.matches('a') && !!e.closest('p,li,dd,td');
    const mobilePrimary = innerWidth <= 480 && e.matches('button,select,input:not([type="checkbox"]),summary,.atlas-tabs a');
    const required = mobilePrimary ? 44 : 24;
    if (!inline && (r.width < required || r.height < required)) {
      // WCAG spacing exception: no other target intersects a 24 px circle centered here.
      const neighbors = targets.filter(o => o.e !== e && !o.e.contains(e) && !e.contains(o.e));
      const spacing = required === 24 && neighbors.every(o => {
        const b = o.r;
        if (b.width < 24 || b.height < 24) return Math.hypot(b.x+b.width/2-r.x-r.width/2, b.y+b.height/2-r.y-r.height/2) >= 24;
        return Math.hypot(Math.max(b.left - (r.x+r.width/2), (r.x+r.width/2)-b.right, 0),
          Math.max(b.top - (r.y+r.height/2), (r.y+r.height/2)-b.bottom, 0)) >= 12;
      });
      if (!spacing) { check(e, 'target-width', r.width, required); check(e, 'target-height', r.height, required); }
    }
    // Text buttons/links identify themselves; form fields and filled controls need a boundary.
    if (e.matches('input:not([type="checkbox"]),select,textarea,.secondary,.outcome-button,.domain-count')) {
      const boundary = parseFloat(s.borderTopWidth) > 0 ? e : (e.closest('.search') || e);
      const boundaryStyle = getComputedStyle(boundary);
      check(e, 'control-boundary-width', parseFloat(boundaryStyle.borderTopWidth), 1);
      contrast(boundary.parentElement, color(boundaryStyle.borderTopColor), 3, 'control-boundary', {control: selector(e)});
    }
  }
  for (const e of root.querySelectorAll('.map-area,.difficulty-fill,.difficulty-unmeasured,.column-resize-handle')) {
    if (!visible(e)) continue;
    const pseudo = e.matches('.column-resize-handle') ? '::after' : null;
    const s = getComputedStyle(e,pseudo);
    const ink = e.matches('.map-area,.difficulty-unmeasured') ? color(s.borderTopColor) : color(s.backgroundColor);
    contrast(e.parentElement, ink, 3, 'graphical-contrast', {graphic: selector(e)});
    if (e.matches('.map-area')) contrast(e, ink, 3, 'graphical-inner-contrast');
  }
  for (const e of root.querySelectorAll('.flow .edge path,.flow .node rect,.flow .pill rect,.flow .arrowhead')) {
    if (!visible(e)) continue;
    const s = getComputedStyle(e), ink = color(e.matches('.arrowhead') ? s.fill : s.stroke);
    contrast(e.parentElement, ink, 3, 'svg-graphical-contrast', {graphic: selector(e)});
  }
  for (const e of root.querySelectorAll('input[type="checkbox"]:checked,[aria-current="page"],.tab.selected')) {
    if (!visible(e)) continue;
    const s = getComputedStyle(e);
    if (e.matches('input')) contrast(e.parentElement, color(s.accentColor), 3, 'checkbox-contrast');
    else if (parseFloat(s.borderBottomWidth) > 0) contrast(e, color(s.borderBottomColor), 3, 'selected-indicator-contrast');
  }
  const overflow = document.documentElement.scrollWidth - innerWidth;
  check(document.documentElement, 'page-reflow', overflow, 1, {maximum: true, textOverflow: textOverflow.slice(0,12),
    overflowing: overflow > 1 ? Array.from(document.querySelectorAll('body *')).filter(e=>visible(e) && e.getBoundingClientRect().right + scrollX > innerWidth && !e.closest('table,.flow')).slice(0,12).map(e=>({selector:selector(e),right:e.getBoundingClientRect().right + scrollX})) : []});
  return {findings, measurements};
}
