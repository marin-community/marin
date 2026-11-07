// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.

import { type ReactNode } from "react";

import { cn } from "@/lib/shared/utils";
import { SEGMENTED_CONTAINER_STYLE } from "@/components/ui/control-styles";

interface SegmentedToggleOptionBase<TValue extends string> {
  value: TValue;
  icon?: ReactNode;
  title?: string;
  testId?: string;
  className?: string;
  disabled?: boolean;
}

type SegmentedToggleOptionContent =
  | {
      label: string;
      ariaLabel?: string;
    }
  | {
      label?: undefined;
      ariaLabel: string;
    };

export type SegmentedToggleOption<TValue extends string> =
  SegmentedToggleOptionBase<TValue> & SegmentedToggleOptionContent;

interface SegmentedToggleProps<TValue extends string> {
  value: TValue;
  options: SegmentedToggleOption<TValue>[];
  onValueChange: (value: TValue) => void;
  ariaLabel: string;
  testId?: string;
  className?: string;
  buttonClassName?: string;
  activeButtonClassName?: string;
  inactiveButtonClassName?: string;
  size?: "sm" | "default";
  role?: "tablist" | "group";
}

export function SegmentedToggle<TValue extends string>({
  value,
  options,
  onValueChange,
  ariaLabel,
  testId,
  className,
  buttonClassName,
  activeButtonClassName = "bg-muted text-foreground",
  inactiveButtonClassName = "text-muted-foreground hover:text-foreground",
  size = "sm",
  role = "tablist",
}: SegmentedToggleProps<TValue>) {
  return (
    <div
      className={cn(
        SEGMENTED_CONTAINER_STYLE,
        size === "sm" ? "md:min-h-8" : "md:min-h-9",
        className,
      )}
      role={role}
      aria-label={ariaLabel}
      data-testid={testId}
    >
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          role={role === "tablist" ? "tab" : undefined}
          aria-selected={
            role === "tablist" ? value === option.value : undefined
          }
          aria-pressed={role === "group" ? value === option.value : undefined}
          aria-label={option.ariaLabel}
          title={option.title}
          disabled={option.disabled}
          className={cn(
            "inline-flex min-h-11 min-w-0 max-w-full items-center gap-1.5 rounded-[max(0px,calc(var(--radius)-var(--segmented-inset)))] px-2 py-1 text-sm font-medium transition-colors disabled:pointer-events-none disabled:opacity-50 md:py-0.5",
            size === "sm" ? "md:min-h-6" : "md:min-h-7",
            buttonClassName,
            value === option.value
              ? activeButtonClassName
              : inactiveButtonClassName,
            option.className,
          )}
          onClick={() => onValueChange(option.value)}
          data-testid={option.testId}
        >
          {option.icon}
          {option.label}
        </button>
      ))}
    </div>
  );
}
