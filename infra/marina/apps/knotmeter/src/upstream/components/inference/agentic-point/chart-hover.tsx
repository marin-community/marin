// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.

import { useId, useState, type ReactNode } from "react";

export interface HoverItem {
  color: string;
  label: string;
  value: string;
  hint?: string;
}

interface ChartHoverProps {
  pad: { top: number; right: number; bottom: number; left: number };
  width: number;
  height: number;
  resolve: (xFraction: number) => { items: HoverItem[]; title?: string } | null;
  ariaLabel?: string;
  keyboardSteps?: number;
  formatAriaValueText?: (items: HoverItem[]) => string;
  children: ReactNode;
}

export function ChartHover({
  pad,
  width,
  height,
  resolve,
  ariaLabel,
  keyboardSteps,
  formatAriaValueText,
  children,
}: ChartHoverProps) {
  const tooltipId = useId();
  const [keyboardIndex, setKeyboardIndex] = useState(0);
  const [hover, setHover] = useState<{
    xPx: number;
    yPx: number;
    fraction: number;
    items: HoverItem[];
    title?: string;
  } | null>(null);

  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const accessible = Boolean(ariaLabel && keyboardSteps && keyboardSteps > 0);
  const steps = accessible ? keyboardSteps! : 0;
  const keyboardFraction = steps > 0 ? (keyboardIndex + 0.5) / steps : 0;
  const keyboardResolved = accessible ? resolve(keyboardFraction) : null;
  const ariaValueText = keyboardResolved
    ? (formatAriaValueText?.(keyboardResolved.items) ??
      keyboardResolved.items
        .map((item) => `${item.label}: ${item.value}`)
        .join("; "))
    : undefined;

  const showKeyboardStep = (index: number) => {
    if (!accessible) return;
    const nextIndex = Math.max(0, Math.min(steps - 1, index));
    const fraction = (nextIndex + 0.5) / steps;
    const resolved = resolve(fraction);
    setKeyboardIndex(nextIndex);
    if (!resolved) {
      setHover(null);
      return;
    }
    setHover({
      xPx: pad.left + fraction * innerW,
      yPx: pad.top + innerH / 2,
      fraction,
      items: resolved.items,
      title: resolved.title,
    });
  };

  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const svg = e.currentTarget.ownerSVGElement;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const sx = ((e.clientX - rect.left) * width) / rect.width;
    const sy = ((e.clientY - rect.top) * height) / rect.height;
    const fraction = Math.max(0, Math.min(1, (sx - pad.left) / innerW));
    const resolved = resolve(fraction);
    if (!resolved) {
      setHover(null);
      return;
    }
    if (accessible) {
      setKeyboardIndex(Math.min(steps - 1, Math.floor(fraction * steps)));
    }
    setHover({
      xPx: sx,
      yPx: sy,
      fraction,
      items: resolved.items,
      title: resolved.title,
    });
  };

  const onLeave = (e: React.MouseEvent<SVGRectElement>) => {
    if (e.currentTarget.ownerDocument.activeElement !== e.currentTarget)
      setHover(null);
  };

  const onKeyDown = (e: React.KeyboardEvent<SVGRectElement>) => {
    if (!accessible) return;
    let nextIndex: number | null = null;
    if (e.key === "ArrowLeft") nextIndex = keyboardIndex - 1;
    if (e.key === "ArrowRight") nextIndex = keyboardIndex + 1;
    if (e.key === "Home") nextIndex = 0;
    if (e.key === "End") nextIndex = steps - 1;
    if (nextIndex === null) return;
    e.preventDefault();
    showKeyboardStep(nextIndex);
  };

  return (
    <div className="relative w-full">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="xMidYMid meet"
        className="w-full h-auto text-foreground"
      >
        {children}
        {hover && (
          <line
            x1={hover.xPx}
            x2={hover.xPx}
            y1={pad.top}
            y2={pad.top + innerH}
            stroke="currentColor"
            strokeWidth={1}
            strokeDasharray="3 3"
            opacity={0.4}
            pointerEvents="none"
          />
        )}
        <rect
          x={pad.left}
          y={pad.top}
          width={innerW}
          height={innerH}
          fill="transparent"
          onMouseMove={onMove}
          onMouseLeave={onLeave}
          {...(accessible
            ? {
                role: "slider",
                tabIndex: 0,
                "aria-label": ariaLabel,
                "aria-orientation": "horizontal" as const,
                "aria-valuemin": 1,
                "aria-valuemax": steps,
                "aria-valuenow": keyboardIndex + 1,
                "aria-valuetext": ariaValueText,
                "aria-describedby": hover ? tooltipId : undefined,
                onFocus: () => showKeyboardStep(keyboardIndex),
                onBlur: () => setHover(null),
                onKeyDown,
              }
            : {})}
        />
      </svg>
      {hover && hover.items.length > 0 && (
        <HoverTooltip
          xFraction={hover.fraction}
          containerWidth={width}
          padLeft={pad.left}
          innerW={innerW}
          title={hover.title}
          items={hover.items}
          id={tooltipId}
        />
      )}
    </div>
  );
}

function HoverTooltip({
  xFraction,
  containerWidth,
  padLeft,
  innerW,
  title,
  items,
  id,
}: {
  xFraction: number;
  containerWidth: number;
  padLeft: number;
  innerW: number;
  title?: string;
  items: HoverItem[];
  id: string;
}) {
  const xPx = padLeft + xFraction * innerW;
  const onRight = xPx < containerWidth * 0.55;
  const left = onRight ? `${(xPx / containerWidth) * 100}%` : "auto";
  const right = onRight
    ? "auto"
    : `${((containerWidth - xPx) / containerWidth) * 100}%`;
  return (
    <div
      id={id}
      role="tooltip"
      className="pointer-events-none absolute top-2 z-10 rounded-md border border-border bg-popover px-2 py-1.5 text-xs shadow-md"
      style={{
        left,
        right,
        marginLeft: onRight ? 8 : 0,
        marginRight: onRight ? 0 : 8,
      }}
    >
      {title && <div className="font-medium text-foreground mb-1">{title}</div>}
      {items.map((it, i) => (
        <div key={i} className="flex items-center gap-1.5 leading-tight">
          <span
            className="inline-block w-2 h-2 rounded-sm"
            style={{ backgroundColor: it.color }}
          />
          <span className="text-muted-foreground">{it.label}</span>
          <span className="ml-auto font-medium text-foreground tabular-nums">
            {it.value}
          </span>
        </div>
      ))}
    </div>
  );
}
