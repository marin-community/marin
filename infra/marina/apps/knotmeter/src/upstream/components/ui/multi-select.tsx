// SPDX-License-Identifier: GPL-3.0-only
// Adapted from InferenceX 315128ce; see THIRD_PARTY.md.

import { CheckIcon, ChevronDownIcon, SearchIcon, XIcon } from "lucide-react";
import * as React from "react";

import { cn } from "@/lib/shared/utils";
import {
  CONTROL_MIN_HEIGHT,
  CONTROL_FOCUS,
  CONTROL_OPTION_STYLE,
  SELECT_TRIGGER_STYLE,
  CONTROL_SEARCH_STYLE,
  CONTROL_SEARCH_CLEAR_STYLE,
} from "./control-styles";
import { useLocale } from "@/lib/i18n/use-locale";

const STRINGS = {
  en: {
    placeholder: "Select items...",
    searchPlaceholder: "Search...",
    searchAriaLabel: "Search options",
    noResultsLabel: "No results",
    clearSearchLabel: "Clear search",
    selectedSuffix: " selected",
    minimumPrefix: "Minimum: ",
    removePrefix: "Remove ",
    clearAllSelections: "Clear all selections",
  },
  zh: {
    placeholder: "选择项目...",
    searchPlaceholder: "搜索...",
    searchAriaLabel: "搜索选项",
    noResultsLabel: "没有结果",
    clearSearchLabel: "清除搜索",
    selectedSuffix: " 项已选择",
    minimumPrefix: "最少：",
    removePrefix: "移除 ",
    clearAllSelections: "清除所有选择",
  },
} as const;

interface MultiSelectOption {
  value: string;
  label: string;
  disabled?: boolean;
  title?: string;
  testId?: string;
  icon?: React.ReactNode;
  badge?: React.ReactNode;
}

export interface MultiSelectSection {
  id: string;
  header?: React.ReactNode;
  options: MultiSelectOption[];
}

interface MultiSelectProps {
  options?: MultiSelectOption[];
  sections?: MultiSelectSection[];
  value?: string[];
  onChange?: (value: string[]) => void;
  triggerId?: string;
  triggerTestId?: string;
  ariaLabel?: string;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  placeholder?: string;
  size?: "sm" | "default";
  className?: string;
  disabled?: boolean;
  maxSelections?: number;
  minSelections?: number;
  showClearAll?: boolean;
  searchable?: boolean;
  plainSelectedText?: boolean;
  wrapSelectedChips?: boolean;
  showSelectionSummary?: boolean;
  searchPlaceholder?: string;
  searchAriaLabel?: string;
  noResultsLabel?: string;
  clearSearchLabel?: string;
  selectedSuffix?: string;
  minimumPrefix?: string;
}

function MultiSelect({
  options,
  sections,
  value = [],
  onChange,
  triggerId,
  triggerTestId,
  ariaLabel,
  open,
  onOpenChange,
  placeholder,
  size = "default",
  className,
  disabled = false,
  maxSelections,
  minSelections,
  showClearAll = true,
  searchable = true,
  plainSelectedText = false,
  wrapSelectedChips = true,
  showSelectionSummary = true,
  searchPlaceholder,
  searchAriaLabel,
  noResultsLabel,
  clearSearchLabel,
  selectedSuffix,
  minimumPrefix,
}: MultiSelectProps) {
  const t = STRINGS[useLocale()];
  const resolvedPlaceholder = placeholder ?? t.placeholder;
  const resolvedSearchPlaceholder = searchPlaceholder ?? t.searchPlaceholder;
  const resolvedSearchAriaLabel = searchAriaLabel ?? t.searchAriaLabel;
  const resolvedNoResultsLabel = noResultsLabel ?? t.noResultsLabel;
  const resolvedClearSearchLabel = clearSearchLabel ?? t.clearSearchLabel;
  const resolvedSelectedSuffix = selectedSuffix ?? t.selectedSuffix;
  const resolvedMinimumPrefix = minimumPrefix ?? t.minimumPrefix;
  const [internalIsOpen, setInternalIsOpen] = React.useState(false);
  const [search, setSearch] = React.useState("");
  const listboxId = React.useId();
  const resolvedTriggerId = triggerId ?? `${listboxId}-trigger`;
  const triggerRef = React.useRef<HTMLButtonElement>(null);
  const searchStateRef = React.useRef(search);
  searchStateRef.current = search;
  const searchableRef = React.useRef(searchable);
  searchableRef.current = searchable;
  const containerRef = React.useRef<HTMLDivElement>(null);
  const searchRef = React.useRef<HTMLInputElement>(null);
  const contentRef = React.useRef<HTMLDivElement>(null);
  const searchUsedRef = React.useRef(false);
  const isControlledOpen = open !== undefined;
  const isOpen = isControlledOpen ? open : internalIsOpen;
  const setIsOpen = React.useCallback(
    (nextOpen: boolean) => {
      if (!nextOpen && isOpen) {
        if (searchUsedRef.current) {
          searchUsedRef.current = false;
        }
        setSearch("");
      }
      if (!isControlledOpen) {
        setInternalIsOpen(nextOpen);
      }
      onOpenChange?.(nextOpen);
    },
    [isControlledOpen, isOpen, onOpenChange],
  );

  const isMaxReached =
    maxSelections !== undefined && value.length >= maxSelections;
  const isMinReached =
    minSelections !== undefined && value.length <= minSelections;

  React.useEffect(() => {
    const handleEscape = (event: KeyboardEvent) => {
      if (
        event.key === "Escape" &&
        event.target instanceof Node &&
        (containerRef.current?.contains(event.target) ||
          event.target === document.body ||
          event.target === document.documentElement)
      ) {
        event.preventDefault();
        event.stopPropagation();
        setIsOpen(false);
        triggerRef.current?.focus();
      }
    };
    const handlePointerDownOutside = (event: PointerEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    };
    const handleFocusOutside = (event: FocusEvent) => {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    if (isOpen) {
      window.addEventListener("keydown", handleEscape, true);
      document.addEventListener("pointerdown", handlePointerDownOutside, true);
      document.addEventListener("focusin", handleFocusOutside);
    }

    return () => {
      window.removeEventListener("keydown", handleEscape, true);
      document.removeEventListener(
        "pointerdown",
        handlePointerDownOutside,
        true,
      );
      document.removeEventListener("focusin", handleFocusOutside);
    };
  }, [isOpen, setIsOpen]);

  React.useEffect(() => {
    if (!isOpen) return;
    if (searchableRef.current) searchRef.current?.focus();
    else {
      const firstOption = contentRef.current?.querySelector<HTMLButtonElement>(
        '[role="option"]:not(:disabled)',
      );
      (firstOption ?? contentRef.current)?.focus();
    }
  }, [isOpen]);

  const flatOptions = React.useMemo(() => {
    if (sections?.length) {
      return sections.flatMap((s) => s.options);
    }
    return options ?? [];
  }, [options, sections]);

  const filteredSections = React.useMemo(() => {
    if (!sections?.length) return null;
    const lower = search.toLowerCase();
    const filterOpts = (opts: MultiSelectOption[]) =>
      search
        ? opts.filter((opt) => opt.label.toLowerCase().includes(lower))
        : opts;

    return sections.map((section) => ({
      ...section,
      options: filterOpts(section.options),
    }));
  }, [sections, search]);

  const filteredOptions = React.useMemo(() => {
    if (filteredSections) {
      return filteredSections.flatMap((s) => s.options);
    }
    const opts = flatOptions;
    if (!search) return opts;
    const lower = search.toLowerCase();
    return opts.filter((opt) => opt.label.toLowerCase().includes(lower));
  }, [filteredSections, flatOptions, search]);

  const handleToggle = (optionValue: string) => {
    if (
      disabled ||
      flatOptions.find((option) => option.value === optionValue)?.disabled
    ) {
      return;
    }

    const isSelected = value.includes(optionValue);

    if (isSelected) {
      if (minSelections !== undefined && value.length <= minSelections) {
        return;
      }
      const newValue = value.filter((v) => v !== optionValue);

      onChange?.(newValue);
      setIsOpen(false);
      triggerRef.current?.focus();
      return;
    }

    if (maxSelections !== undefined && value.length >= maxSelections) {
      if (maxSelections === 1) {
        const newValue = [optionValue];

        onChange?.(newValue);
        setIsOpen(false);
        triggerRef.current?.focus();
        return;
      }
      return;
    }

    const newValue = [...value, optionValue];

    onChange?.(newValue);
    setIsOpen(false);
    triggerRef.current?.focus();
  };

  const handleRemove = (optionValue: string, e: React.SyntheticEvent) => {
    e.stopPropagation();
    if (disabled) {
      return;
    }

    if (minSelections !== undefined && value.length <= minSelections) {
      return;
    }

    const newValue = value.filter((v) => v !== optionValue);
    onChange?.(newValue);
  };

  const handleClearAll = (e: React.SyntheticEvent) => {
    e.stopPropagation();
    if (disabled) {
      return;
    }
    if (minSelections !== undefined && minSelections > 0) {
      return;
    }

    onChange?.([]);
  };

  const selectedOptions = value.map(
    (val) =>
      flatOptions.find((opt) => opt.value === val) ?? {
        value: val,
        label: val,
      },
  );
  const selectedLabels = selectedOptions.map((opt) => opt.label);

  const focusOption = (direction: "first" | "last" | "next" | "previous") => {
    const enabled = [
      ...(contentRef.current?.querySelectorAll<HTMLButtonElement>(
        '[role="option"]:not(:disabled)',
      ) ?? []),
    ];
    if (enabled.length === 0) return;
    const current = enabled.indexOf(
      document.activeElement as HTMLButtonElement,
    );
    const index =
      direction === "first"
        ? 0
        : direction === "last"
          ? enabled.length - 1
          : (current + (direction === "next" ? 1 : -1) + enabled.length) %
            enabled.length;
    enabled[index].focus();
  };

  const renderOption = (option: MultiSelectOption) => {
    const isSelected = value.includes(option.value);
    const isDisabledOption = Boolean(
      option.disabled ||
        (isSelected && isMinReached) ||
        (!isSelected && isMaxReached && maxSelections !== 1),
    );
    return (
      <button
        key={option.value}
        type="button"
        role="option"
        tabIndex={-1}
        aria-selected={isSelected}
        aria-disabled={isDisabledOption}
        disabled={isDisabledOption}
        title={option.title}
        data-testid={option.testId}
        data-slot="select-item"
        onClick={() => handleToggle(option.value)}
        onKeyDown={(event) => {
          const direction =
            event.key === "ArrowDown"
              ? "next"
              : event.key === "ArrowUp"
                ? "previous"
                : event.key === "Home"
                  ? "first"
                  : event.key === "End"
                    ? "last"
                    : undefined;
          if (direction) {
            event.preventDefault();
            focusOption(direction);
          } else if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            handleToggle(option.value);
          }
        }}
        className={cn(
          "[&_svg:not([class*='text-'])]:text-muted-foreground relative flex w-full cursor-pointer items-center gap-2 rounded-sm py-1.5 pr-8 pl-2 text-left text-sm outline-none select-none transition-colors",
          CONTROL_OPTION_STYLE,
          "hover:bg-accent",
          isSelected && "bg-primary/10 font-medium",
          isDisabledOption &&
            "opacity-50 cursor-not-allowed hover:bg-transparent hover:shadow-none",
        )}
      >
        <span className="absolute right-2 flex size-3.5 items-center justify-center">
          {isSelected && <CheckIcon className="size-4 text-primary" />}
        </span>
        <span className="flex min-w-0 items-center gap-2">
          {option.icon}
          {option.label}
          {option.badge}
        </span>
      </button>
    );
  };

  return (
    <div ref={containerRef} className="relative">
      <button
        type="button"
        ref={triggerRef}
        id={resolvedTriggerId}
        aria-label={ariaLabel}
        data-testid={triggerTestId}
        role="combobox"
        aria-expanded={isOpen}
        aria-haspopup="listbox"
        aria-controls={listboxId}
        onClick={() => !disabled && setIsOpen(!isOpen)}
        onKeyDown={(event) => {
          if (
            !disabled &&
            (event.key === "ArrowDown" || event.key === "ArrowUp")
          ) {
            event.preventDefault();
            setIsOpen(true);
          }
        }}
        disabled={disabled}
        data-slot="select-trigger"
        data-size={size}
        className={cn(
          SELECT_TRIGGER_STYLE,
          CONTROL_MIN_HEIGHT[size],
          CONTROL_FOCUS,
          "w-full",
          "py-1",
          className,
        )}
      >
        <div
          data-slot="select-values"
          title={
            !plainSelectedText && !wrapSelectedChips
              ? selectedLabels.join(", ")
              : undefined
          }
          className={cn(
            "flex gap-1 flex-1 min-w-0 items-center min-h-5",
            wrapSelectedChips
              ? "flex-wrap"
              : "flex-nowrap overflow-x-auto [scrollbar-width:none]",
          )}
        >
          {value.length > 0 ? (
            plainSelectedText ? (
              <span className="text-foreground flex min-w-0 items-center gap-1.5">
                {selectedOptions.length === 1 && selectedOptions[0].icon}
                <span
                  className="block min-w-0 truncate"
                  title={selectedLabels.join(", ")}
                >
                  {selectedLabels.join(", ")}
                </span>
              </span>
            ) : (
              selectedLabels.map((label, index) => (
                <span
                  key={value[index]}
                  data-slot="select-chip"
                  className="bg-transparent text-foreground border border-border dark:bg-[#0a6ca8] dark:border-border inline-flex min-w-0 max-w-full items-center gap-1 rounded px-1.5 py-0.5 text-xs font-medium transition-colors shrink-0"
                >
                  {selectedOptions[index]?.icon}
                  <span className="min-w-0 truncate" title={label}>
                    {label}
                  </span>
                  <span
                    role="button"
                    tabIndex={0}
                    onClick={(e) => handleRemove(value[index], e)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        handleRemove(value[index], e);
                      }
                    }}
                    className={cn(
                      "hover:bg-primary/20 rounded-sm cursor-pointer transition-colors",
                      (disabled || isMinReached) && "hidden",
                    )}
                    aria-label={`${t.removePrefix}${label}`}
                    aria-disabled={disabled || isMinReached}
                  >
                    <XIcon className="size-4 text-foreground" />
                  </span>
                </span>
              ))
            )
          ) : (
            <span className="text-muted-foreground">{resolvedPlaceholder}</span>
          )}
        </div>
        {value.length > 0 && showClearAll && (
          <span
            role="button"
            tabIndex={0}
            onClick={handleClearAll}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                handleClearAll(e);
              }
            }}
            className={cn(
              "hover:bg-destructive/10 hover:text-destructive text-muted-foreground shrink-0 rounded-sm p-1 transition-colors",
              (disabled ||
                (minSelections !== undefined && minSelections > 0)) &&
                "cursor-not-allowed opacity-50 pointer-events-none",
            )}
            aria-label={t.clearAllSelections}
            aria-disabled={
              disabled || (minSelections !== undefined && minSelections > 0)
            }
          >
            <XIcon className="size-4" />
          </span>
        )}
        <ChevronDownIcon
          className={cn(
            "size-4 opacity-90 shrink-0 transition-transform",
            isOpen && "transform rotate-180",
          )}
        />
      </button>

      {isOpen && (
        <div
          ref={contentRef}
          tabIndex={-1}
          data-slot="select-content"
          className={cn(
            "bg-popover text-popover-foreground animate-in fade-in-0 zoom-in-95 slide-in-from-top-2 absolute z-[120] mt-1 flex max-h-72 w-full origin-top flex-col overflow-hidden rounded-md border shadow-md",
          )}
        >
          <div className="min-h-0 p-1 space-y-1 overflow-y-auto custom-scrollbar">
            {searchable && (
              <div className="flex items-center gap-1.5 px-2 border-b mb-1">
                <SearchIcon className="size-3.5 shrink-0 text-muted-foreground mr-2" />
                <input
                  ref={searchRef}
                  onKeyDown={(event) => {
                    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
                      event.preventDefault();
                      focusOption(event.key === "ArrowDown" ? "first" : "last");
                    }
                  }}
                  type="text"
                  value={search}
                  onChange={(e) => {
                    setSearch(e.target.value);
                    if (e.target.value) searchUsedRef.current = true;
                  }}
                  placeholder={resolvedSearchPlaceholder}
                  aria-label={resolvedSearchAriaLabel}
                  className={CONTROL_SEARCH_STYLE}
                />
                {search && (
                  <button
                    type="button"
                    onClick={() => {
                      setSearch("");
                      searchRef.current?.focus();
                    }}
                    className={cn(CONTROL_SEARCH_CLEAR_STYLE, CONTROL_FOCUS)}
                    aria-label={resolvedClearSearchLabel}
                  >
                    <XIcon className="size-3.5" />
                  </button>
                )}
              </div>
            )}
            {showSelectionSummary &&
              (maxSelections !== undefined || minSelections !== undefined) && (
                <div
                  data-slot="select-summary"
                  className="text-muted-foreground flex flex-wrap items-center justify-between gap-x-3 gap-y-1 px-2 py-1.5 text-xs border-b mb-1"
                >
                  <span>
                    {value.length}
                    {maxSelections !== undefined && ` / ${maxSelections}`}
                    {resolvedSelectedSuffix}
                  </span>
                  {minSelections !== undefined && minSelections > 0 && (
                    <span>
                      {resolvedMinimumPrefix}
                      {minSelections}
                    </span>
                  )}
                </div>
              )}
            {filteredOptions.length === 0 && (
              <div className="text-muted-foreground px-2 py-1.5 text-sm text-center">
                {resolvedNoResultsLabel}
              </div>
            )}
            <div
              id={listboxId}
              role="listbox"
              aria-label={ariaLabel}
              aria-labelledby={ariaLabel ? undefined : resolvedTriggerId}
              aria-multiselectable={maxSelections !== 1}
            >
              {filteredSections
                ? filteredSections.map(
                    (section) =>
                      section.options.length > 0 && (
                        <div key={section.id} className="space-y-0.5">
                          {section.header && (
                            <div className="text-muted-foreground px-2 py-1.5 text-xs font-medium">
                              {section.header}
                            </div>
                          )}
                          {section.options.map(renderOption)}
                        </div>
                      ),
                  )
                : filteredOptions.map(renderOption)}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export { MultiSelect };
export type { MultiSelectOption, MultiSelectProps };
