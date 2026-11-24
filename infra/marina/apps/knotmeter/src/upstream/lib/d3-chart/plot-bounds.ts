// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.
export interface PlotBounds {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

const TRANSLATE = /translate\(\s*(?<x>-?[\d.]+)[\s,]+(?<y>-?[\d.]+)\s*\)/u;
const CLIP_URL = /url\(\s*#(?<id>[^)\s]+)\s*\)/u;

export interface PlotSize {
  width: number;
  height: number;
}

export function plotClipSize(
  zoomGroup: Element,
  svg: Element | null = (zoomGroup as SVGElement).ownerSVGElement,
): PlotSize | null {
  if (!svg) return null;
  const clipId = CLIP_URL.exec(zoomGroup.getAttribute("clip-path") ?? "")
    ?.groups?.id;
  if (!clipId) return null;
  const clip = [...svg.querySelectorAll("clipPath")]
    .find((candidate) => candidate.id === clipId)
    ?.querySelector("rect");
  if (!clip) return null;

  const width = Number(clip.getAttribute("width"));
  const height = Number(clip.getAttribute("height"));
  if (!(width > 0) || !(height > 0)) return null;
  return { width, height };
}

export function plotBounds(svg: Element): PlotBounds | null {
  const root = svg.querySelector(".chart-root");
  const zoomGroup = svg.querySelector(".zoom-group");
  if (!root || !zoomGroup) return null;

  const size = plotClipSize(zoomGroup, svg);
  if (!size) return null;
  const { width, height } = size;

  const translate = TRANSLATE.exec(
    root.getAttribute("transform") ?? "",
  )?.groups;
  if (!translate) return null;

  const box = svg.getBoundingClientRect();
  const left = box.left + Number(translate.x);
  const top = box.top + Number(translate.y);
  return { left, top, right: left + width, bottom: top + height };
}
